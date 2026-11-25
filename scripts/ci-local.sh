#!/usr/bin/env bash
# Runs the same checks as the Linux PR CI jobs locally, so they pass before anything is pushed.
#
# Mirrors:
#   .github/workflows/repo-hygiene.yml    -> agent parity, gitleaks
#   .github/workflows/ci.yml (Ubuntu)     -> ggml + whisper.cpp + llama.cpp pins, host build + CTest (runtime,
#                                            whisper and llama), host linkage audits, :kggml-runtime + :kggml-whisper + :kggml-llama jvmTest,
#                                            Android cross-compile + audits, :samples:whisper-only-minimal check +
#                                            release APK binary symbol audit; with --full also `./gradlew check`
#                                            and the sample assemble tasks
# The macOS/iOS jobs (ci.yml macOS, runtime-native-macos.yml) cannot run on a Linux host;
# on a macOS host the host build, CTest, linkage audit and jvmTest steps below cover the macOS job.
#
# Usage: ./scripts/ci-local.sh [--only <steps>] [--skip <steps>] [--strict] [--clean] [--full]
#   steps (comma-separated): parity,gitleaks,pin,native,jvm,android,sample,check
#   --full     run every step regardless of what changed, including `check` (./gradlew check + sample
#              assembles, which need an Android SDK). This is the merge gate when GitHub Actions cannot
#              run (quota exhausted); see AGENTS.md "When the Actions quota is exhausted".
#   --strict   fail instead of skipping when a tool (gitleaks, Android NDK/SDK) is missing
#   --clean    pass --clean to the host native build
#   The Android NDK comes from ANDROID_NDK_HOME / ANDROID_NDK_ROOT, else the newest NDK under
#   ANDROID_HOME, ANDROID_SDK_ROOT, local.properties sdk.dir, or /opt/android-sdk.
# Native steps only run when files they cover changed vs origin/main, unless named with --only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Make sure commit attribution hooks (.githooks/) are active in this clone.
./scripts/install-git-hooks.sh

all_steps=(parity gitleaks pin native jvm android sample check)
only=""
skip=""
strict=0
clean_flag=()
full=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) only="${2:?--only needs a step list}"; shift ;;
    --skip) skip="${2:?--skip needs a step list}"; shift ;;
    --strict) strict=1 ;;
    --clean) clean_flag=(--clean) ;;
    --full) full=1 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

in_list() { [[ ",$2," == *",$1,"* ]]; }

# Native-relevant paths (what used to be runtime-native.yml's PR path filter, plus ci.yml).
native_paths=(kggml-runtime kggml-whisper/native kggml-whisper/src kggml-whisper/build.gradle.kts kggml-llama/native kggml-llama/src kggml-llama/build.gradle.kts kggml-core-api/src/commonMain build-logic gradle gradle.properties .gitmodules NOTICE
  scripts/ci .github/workflows/ci.yml .github/workflows/runtime-native.yml .github/workflows/runtime-native-macos.yml)
native_changed() {
  local base
  base="$(git merge-base HEAD origin/main 2>/dev/null || true)"
  [[ -n "${base}" ]] || return 0   # no origin/main: assume changed
  # Committed changes since the merge base, plus staged/unstaged/untracked changes.
  [[ -n "$(git diff --name-only "${base}" -- "${native_paths[@]}")" ]] \
    || [[ -n "$(git ls-files --others --exclude-standard -- "${native_paths[@]}")" ]]
}

# Paths that affect :samples:whisper-only-minimal's binaries or its audit (in addition to native_paths).
sample_paths=(samples/whisper-only-minimal scripts/audit-binary-symbols.sh scripts/tests .github/workflows/ci.yml)
sample_changed() {
  native_changed && return 0
  local base
  base="$(git merge-base HEAD origin/main 2>/dev/null || true)"
  [[ -n "${base}" ]] || return 0
  [[ -n "$(git diff --name-only "${base}" -- "${sample_paths[@]}")" ]] \
    || [[ -n "$(git ls-files --others --exclude-standard -- "${sample_paths[@]}")" ]]
}

selected() {
  local step="$1"
  if [[ -n "${only}" ]]; then in_list "${step}" "${only}"; return; fi
  in_list "${step}" "${skip}" && return 1
  [[ ${full} -eq 1 ]] && return 0
  case "${step}" in
    check) return 1 ;;   # full Gradle check only with --full or --only check
    pin|native|jvm|android) native_changed ;;
    sample) sample_changed ;;
    *) return 0 ;;
  esac
}

# Prints the Android NDK to use, or fails. Explicit NDK variables win (in CI's order of precedence);
# otherwise the newest NDK under an SDK root is used: $ANDROID_HOME, $ANDROID_SDK_ROOT,
# sdk.dir from local.properties, then /opt/android-sdk (where the cloud setup script installs it).
# Only directories that actually contain an NDK toolchain file are accepted.
is_ndk() { [[ -n "$1" && -f "$1/build/cmake/android.toolchain.cmake" ]]; }
find_ndk() {
  local candidate sdk sdk_dir=""
  for candidate in "${ANDROID_NDK_LATEST_HOME:-}" "${ANDROID_NDK_HOME:-}" "${ANDROID_NDK_ROOT:-}"; do
    if is_ndk "${candidate}"; then echo "${candidate}"; return 0; fi
  done
  if [[ -f local.properties ]]; then
    sdk_dir="$(sed -n 's/^sdk\.dir=//p' local.properties | tail -n 1)"
  fi
  for sdk in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "${sdk_dir}" /opt/android-sdk; do
    [[ -n "${sdk}" && -d "${sdk}/ndk" ]] || continue
    candidate="$(find "${sdk}/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
    if is_ndk "${candidate}"; then echo "${candidate}"; return 0; fi
  done
  return 1
}

# Prints the Android SDK root for Gradle, or fails.
find_sdk() {
  local sdk sdk_dir=""
  if [[ -f local.properties ]]; then
    sdk_dir="$(sed -n 's/^sdk\.dir=//p' local.properties | tail -n 1)"
  fi
  for sdk in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "${sdk_dir}" /opt/android-sdk; do
    if [[ -n "${sdk}" && -d "${sdk}/platforms" ]]; then echo "${sdk}"; return 0; fi
  done
  return 1
}

# The Android audits call readelf, which macOS does not ship. Without one on PATH, use the NDK's
# llvm-readelf (GNU-compatible output when invoked as `readelf`).
if ! command -v readelf >/dev/null 2>&1 && ndk="$(find_ndk)"; then
  for tool in "${ndk}"/toolchains/llvm/prebuilt/*/bin/llvm-readelf; do
    [[ -x "${tool}" ]] || continue
    mkdir -p build/ci-tools
    ln -sf "${tool}" build/ci-tools/readelf
    export PATH="${REPO_ROOT}/build/ci-tools:${PATH}"
    break
  done
fi

# This gate mirrors the Ubuntu job, where iOS targets are disabled. On macOS `check` would also publish iOS
# klibs to the BOM verification repo (which fails for modules without sources, e.g. :kggml-llama) and run the
# simulator tests; both belong to the ci.yml macOS job, so leave them out here.
mac_check_args=()
if [[ "$(uname -s)" == Darwin ]]; then
  for target in IosArm64 IosSimulatorArm64; do
    mac_check_args+=(-x "publish${target}PublicationToKggmlVerificationRepository")
  done
  mac_check_args+=(-x iosSimulatorArm64Test)
fi

results=()
record() { results+=("$(printf '%-9s %s' "$1" "$2")"); }
missing() {
  if [[ ${strict} -eq 1 ]]; then echo "FAIL: $2" >&2; record "$1" "FAIL (missing tool)"; summary; exit 1; fi
  echo "SKIP: $2" >&2; record "$1" "SKIPPED ($2)"
}
summary() {
  echo
  echo "==== ci-local summary ($(git rev-parse --short HEAD), $(uname -s)) ===="
  printf '%s\n' "${results[@]}"
}
run_step() {
  local step="$1"; shift
  echo
  echo "==== [${step}] $* ===="
  if "$@"; then record "${step}" "PASS"; else record "${step}" "FAIL"; summary; exit 1; fi
}

for step in "${all_steps[@]}"; do
  if ! selected "${step}"; then record "${step}" "skipped (not selected / no relevant changes)"; continue; fi
  case "${step}" in
    parity)
      run_step parity ./scripts/verify-agent-parity.sh ;;
    gitleaks)
      if command -v gitleaks >/dev/null 2>&1; then
        run_step gitleaks gitleaks git --redact --no-banner --config .gitleaks.toml
      else
        missing gitleaks "gitleaks not installed"
      fi ;;
    pin)
      run_step pin ./scripts/ci/verify-ggml-pin.sh
      run_step w-pin ./scripts/ci/verify-whisper-pin.sh
      run_step l-pin ./scripts/ci/verify-llama-pin.sh ;;
    native)
      run_step native ./kggml-runtime/native/scripts/build-host.sh ${clean_flag[@]+"${clean_flag[@]}"}
      run_step audit ./scripts/ci/audit-host-linkage.sh
      run_step whisper ./kggml-whisper/native/scripts/build-host.sh ${clean_flag[@]+"${clean_flag[@]}"}
      run_step w-audit ./scripts/ci/audit-whisper-linkage.sh host
      run_step llama ./kggml-llama/native/scripts/build-host.sh ${clean_flag[@]+"${clean_flag[@]}"}
      run_step l-audit ./scripts/ci/audit-llama-linkage.sh host ;;
    jvm)
      run_step jvm ./gradlew :kggml-runtime:jvmTest :kggml-whisper:jvmTest :kggml-llama:jvmTest -Pkggml.requireNative=true --console=plain ;;
    android)
      if ndk="$(find_ndk)"; then
        echo "Using Android NDK: ${ndk}"
        export ANDROID_NDK_HOME="${ndk}"
        run_step android ./kggml-runtime/native/scripts/build-android.sh arm64-v8a x86_64
        run_step a-audit ./scripts/ci/audit-android-linkage.sh arm64-v8a x86_64
        run_step w-android ./kggml-whisper/native/scripts/build-android.sh arm64-v8a x86_64
        run_step wa-audit ./scripts/ci/audit-whisper-linkage.sh android arm64-v8a x86_64
        run_step l-android ./kggml-llama/native/scripts/build-android.sh arm64-v8a x86_64
        run_step la-audit ./scripts/ci/audit-llama-linkage.sh android arm64-v8a x86_64
      else
        missing android "Android NDK not found (set ANDROID_NDK_HOME or ANDROID_HOME)"
      fi ;;
    sample)
      run_step s-check ./gradlew :samples:whisper-only-minimal:check --console=plain
      if ndk="$(find_ndk)"; then
        export ANDROID_NDK_HOME="${ndk}"
        run_step s-audit ./scripts/ci/audit-minimal-sample.sh android
      else
        run_step s-audit-t ./scripts/tests/test-audit-binary-symbols.sh
        missing s-audit "Android NDK not found (APK binary symbol audit)"
      fi ;;
    check)
      if sdk="$(find_sdk)"; then
        export ANDROID_HOME="${sdk}"
        run_step g-check ./gradlew check --console=plain ${mac_check_args[@]+"${mac_check_args[@]}"}
        run_step g-assemble ./gradlew --console=plain :samples:kggml-showcase:assembleDebug \
          :samples:kggml-showcase:desktopJar :samples:whisper-only-minimal:assembleDebug
      else
        missing check "Android SDK not found (./gradlew check needs ANDROID_HOME)"
      fi ;;
  esac
done

summary
