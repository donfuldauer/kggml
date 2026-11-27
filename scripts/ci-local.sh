#!/usr/bin/env bash
# Runs the same checks as the Linux PR CI jobs locally, so they pass before anything is pushed.
#
# Mirrors:
#   .github/workflows/repo-hygiene.yml    -> agent parity, CI-waiter test, hosted-runner audit, release-audit test,
#                                            license/NOTICE/fixture audit, docs audit (links, README snippets,
#                                            coordinates, KDoc), gitleaks, full-history audit
#   .github/workflows/ci.yml (Ubuntu)     -> ggml + whisper.cpp + llama.cpp + stable-diffusion.cpp pins, host build +
#                                            CTest (runtime, whisper, llama and diffusion), host linkage audits,
#                                            :kggml-runtime + :kggml-whisper + :kggml-llama + :kggml-diffusion
#                                            jvmTest,
#                                            Android cross-compile + audits, :samples:whisper-only-minimal check +
#                                            release APK binary symbol audit, :docs-snippets compile (README code
#                                            against the published artifacts), Maven Central release dry-run
#                                            (signed staging + audit; iOS components only on macOS); with --full
#                                            also `./gradlew check` and the sample assemble tasks
# The macOS/iOS job (ci.yml macOS) cannot run on a Linux host;
# on a macOS host the host build, CTest, linkage audit and jvmTest steps below cover the macOS job.
#
# Usage: ./scripts/ci-local.sh [--only <steps>] [--skip <steps>] [--strict] [--clean] [--full]
#   steps (comma-separated): parity,gitleaks,pin,native,jvm,android,sample,snippets,release,check,coverage,instrumented,
#                            ios-apps
#   coverage      (only with --only) merged Kover report of every JVM test (scripts/ci/coverage.sh)
#   instrumented  (only with --only) every Android instrumented suite on a connected device / KGGML_AVD emulator
#                 with the real native libraries (scripts/ci/run-android-instrumented.sh)
#   ios-apps      (only with --only, macOS) the samples' Xcode host apps for the simulator and devices, unsigned
#                 (scripts/ios/build-xcframeworks.sh + scripts/ios/build-sample-apps.sh)
#   --full     run every step regardless of what changed, including `check` (./gradlew check + sample
#              assembles, which need an Android SDK). Stands in for the Linux CI job when a runner fails or hangs
#              (AGENTS.md "When a runner fails or hangs").
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

all_steps=(parity gitleaks pin native jvm android sample snippets release check coverage instrumented ios-apps)
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
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

in_list() { [[ ",$2," == *",$1,"* ]]; }

# Native-relevant paths.
native_paths=(kggml-runtime kggml-whisper/native kggml-whisper/src kggml-whisper/build.gradle.kts kggml-llama/native kggml-llama/src kggml-llama/build.gradle.kts kggml-diffusion/native kggml-diffusion/src kggml-diffusion/build.gradle.kts kggml-core-api/src/commonMain integration-tests build-logic gradle gradle.properties .gitmodules NOTICE
  scripts/ci .github/workflows/ci.yml)
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

# Paths that affect the README snippet compile (:docs-snippets against the published library artifacts).
snippet_paths=(docs-snippets README.md kggml-bom kggml-core-api kggml-runtime/src kggml-runtime/build.gradle.kts
  kggml-runtime-vulkan/src kggml-runtime-vulkan/build.gradle.kts kggml-runtime-opencl/src kggml-runtime-opencl/build.gradle.kts
  kggml-whisper/src kggml-whisper/build.gradle.kts kggml-llama/src kggml-llama/build.gradle.kts kggml-diffusion/src
  kggml-diffusion/build.gradle.kts kggml-model-manager settings.gradle.kts build-logic gradle gradle.properties)
snippets_changed() {
  local base
  base="$(git merge-base HEAD origin/main 2>/dev/null || true)"
  [[ -n "${base}" ]] || return 0
  [[ -n "$(git diff --name-only "${base}" -- "${snippet_paths[@]}")" ]] \
    || [[ -n "$(git ls-files --others --exclude-standard -- "${snippet_paths[@]}")" ]]
}

# Paths that affect the published artifacts or the release staging/audit.
release_paths=(build-logic gradle gradle.properties settings.gradle.kts kggml-bom kggml-core-api/build.gradle.kts
  kggml-runtime/build.gradle.kts kggml-runtime-vulkan/build.gradle.kts kggml-runtime-opencl/build.gradle.kts kggml-whisper/build.gradle.kts kggml-llama/build.gradle.kts
  kggml-diffusion/build.gradle.kts kggml-model-manager/build.gradle.kts scripts/ci/stage-release.sh
  scripts/ci/verify_release.py .github/workflows/release.yml)
release_changed() {
  local base
  base="$(git merge-base HEAD origin/main 2>/dev/null || true)"
  [[ -n "${base}" ]] || return 0
  [[ -n "$(git diff --name-only "${base}" -- "${release_paths[@]}")" ]] \
    || [[ -n "$(git ls-files --others --exclude-standard -- "${release_paths[@]}")" ]]
}

selected() {
  local step="$1"
  if [[ -n "${only}" ]]; then in_list "${step}" "${only}"; return; fi
  in_list "${step}" "${skip}" && return 1
  # Opt-in steps: coverage re-runs every JVM test with the Kover agent; instrumented needs a device; ios-apps builds
  # the iOS xcframeworks and both Xcode apps.
  case "${step}" in coverage|instrumented|ios-apps) return 1 ;; esac
  [[ ${full} -eq 1 ]] && return 0
  case "${step}" in
    check) return 1 ;;   # full Gradle check only with --full or --only check
    pin|native|jvm|android) native_changed ;;
    sample) sample_changed ;;
    snippets) snippets_changed ;;
    release) release_changed ;;
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
      run_step parity ./scripts/verify-agent-parity.sh
      run_step waiter ./scripts/tests/test-wait-pr-checks.sh
      run_step runners ./scripts/ci/verify-hosted-runners.sh
      run_step rn-test ./scripts/tests/test-verify-hosted-runners.sh
      run_step rel-test ./scripts/tests/test-verify-release.sh
      run_step arch-test ./scripts/tests/test-check-native-arch.sh
      run_step license ./scripts/ci/verify-licenses.sh
      run_step l-test ./scripts/tests/test-verify-licenses.sh
      run_step docs ./scripts/ci/verify-docs.sh
      run_step docs-test ./scripts/tests/test-verify-docs.sh
      run_step h-test ./scripts/tests/test-audit-history.sh
      run_step cls-test ./scripts/tests/test-classify-changes.sh ;;
    gitleaks)
      if command -v gitleaks >/dev/null 2>&1; then
        run_step gitleaks gitleaks git --redact --no-banner --config .gitleaks.toml
        # Full history of every ref: gitleaks plus blob-size, credential-file, binary, key and local-path audit.
        run_step history ./scripts/ci/audit-history.sh
      else
        missing gitleaks "gitleaks not installed"
      fi ;;
    pin)
      run_step pin ./scripts/ci/verify-ggml-pin.sh
      run_step w-pin ./scripts/ci/verify-whisper-pin.sh
      run_step l-pin ./scripts/ci/verify-llama-pin.sh
      run_step d-pin ./scripts/ci/verify-diffusion-pin.sh ;;
    native)
      run_step native ./kggml-runtime/native/scripts/build-host.sh ${clean_flag[@]+"${clean_flag[@]}"}
      run_step audit ./scripts/ci/audit-host-linkage.sh
      run_step whisper ./kggml-whisper/native/scripts/build-host.sh ${clean_flag[@]+"${clean_flag[@]}"}
      run_step w-audit ./scripts/ci/audit-whisper-linkage.sh host
      run_step llama ./kggml-llama/native/scripts/build-host.sh ${clean_flag[@]+"${clean_flag[@]}"}
      run_step l-audit ./scripts/ci/audit-llama-linkage.sh host
      run_step diffusion ./kggml-diffusion/native/scripts/build-host.sh ${clean_flag[@]+"${clean_flag[@]}"}
      run_step d-audit ./scripts/ci/audit-diffusion-linkage.sh host ;;
    jvm)
      run_step jvm ./gradlew :kggml-runtime:jvmTest :kggml-whisper:jvmTest :kggml-llama:jvmTest :kggml-diffusion:jvmTest :integration-tests:jvmTest -Pkggml.requireNative=true --console=plain ;;
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
        run_step d-android ./kggml-diffusion/native/scripts/build-android.sh arm64-v8a x86_64
        run_step da-audit ./scripts/ci/audit-diffusion-linkage.sh android arm64-v8a x86_64
        run_step v-android ./kggml-runtime-vulkan/native/scripts/build-android.sh arm64-v8a x86_64
        run_step va-audit ./scripts/ci/audit-gpu-addon.sh vulkan arm64-v8a x86_64
        run_step o-android ./kggml-runtime-opencl/native/scripts/build-android.sh arm64-v8a x86_64
        run_step oa-audit ./scripts/ci/audit-gpu-addon.sh opencl arm64-v8a x86_64
      else
        missing android "Android NDK not found (set ANDROID_NDK_HOME or ANDROID_HOME)"
      fi ;;
    sample)
      run_step s-check ./gradlew :samples:whisper-only-minimal:check :samples:whisper-only-minimal:androidApp:check --console=plain
      run_step sd-audit ./scripts/ci/audit-minimal-sample.sh desktop
      if ndk="$(find_ndk)"; then
        export ANDROID_NDK_HOME="${ndk}"
        run_step s-audit ./scripts/ci/audit-minimal-sample.sh android
      else
        run_step s-audit-t ./scripts/tests/test-audit-binary-symbols.sh
        missing s-audit "Android NDK not found (APK binary symbol audit)"
      fi ;;
    snippets)
      run_step snippets ./gradlew :docs-snippets:compileKotlinJvm --console=plain ;;
    release)
      # On Linux the iOS components cannot be built; the ci.yml macOS job stages them.
      release_skip=(--skip-targets ios,macos)
      [[ "$(uname -s)" == Darwin ]] && release_skip=()
      run_step release ./scripts/ci/stage-release.sh --dry-run ${release_skip[@]+"${release_skip[@]}"} ;;
    check)
      if sdk="$(find_sdk)"; then
        export ANDROID_HOME="${sdk}"
        run_step g-check ./gradlew check --console=plain ${mac_check_args[@]+"${mac_check_args[@]}"}
        run_step g-assemble ./gradlew --console=plain :samples:kggml-showcase:androidApp:assembleDebug \
          :samples:kggml-showcase:desktopJar :samples:whisper-only-minimal:androidApp:assembleDebug
      else
        missing check "Android SDK not found (./gradlew check needs ANDROID_HOME)"
      fi ;;
    coverage)
      run_step coverage ./scripts/ci/coverage.sh ;;
    instrumented)
      if sdk="$(find_sdk)"; then
        export ANDROID_HOME="${sdk}"
        if ndk="$(find_ndk)"; then export ANDROID_NDK_HOME="${ndk}"; fi
        run_step instrumented ./scripts/ci/run-android-instrumented.sh
      else
        missing instrumented "Android SDK not found (instrumented tests need ANDROID_HOME and a device)"
      fi ;;
    ios-apps)
      if [[ "$(uname -s)" == Darwin ]] && command -v xcodebuild >/dev/null 2>&1; then
        run_step ios-xcf ./scripts/ios/build-xcframeworks.sh all
        run_step ios-apps ./scripts/ios/build-sample-apps.sh all
      else
        missing ios-apps "Xcode not found (the iOS apps build on macOS only)"
      fi ;;
  esac
done

summary
