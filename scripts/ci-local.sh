#!/usr/bin/env bash
# Runs the same checks as the Linux PR CI jobs locally, so they pass before anything is pushed.
#
# Mirrors:
#   .github/workflows/repo-hygiene.yml    -> agent parity, gitleaks
#   .github/workflows/runtime-native.yml  -> ggml pin, host build + CTest, host linkage audit,
#                                            :kggml-runtime:jvmTest, Android cross-compile + audit
# The macOS/iOS jobs (.github/workflows/runtime-native-macos.yml) cannot run on a Linux host;
# on a macOS host the host build, CTest, linkage audit and jvmTest steps below cover the macOS job.
#
# Usage: ./scripts/ci-local.sh [--only <steps>] [--skip <steps>] [--strict] [--clean]
#   steps (comma-separated): parity,gitleaks,pin,native,jvm,android
#   --strict   fail instead of skipping when a tool (gitleaks, Android NDK) is missing
#   --clean    pass --clean to the host native build
# Native steps only run when files they cover changed vs origin/main, unless named with --only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

all_steps=(parity gitleaks pin native jvm android)
only=""
skip=""
strict=0
clean_flag=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) only="${2:?--only needs a step list}"; shift ;;
    --skip) skip="${2:?--skip needs a step list}"; shift ;;
    --strict) strict=1 ;;
    --clean) clean_flag=(--clean) ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

in_list() { [[ ",$2," == *",$1,"* ]]; }

# Same path filter as runtime-native.yml.
native_paths=(kggml-runtime kggml-core-api/src/commonMain build-logic gradle gradle.properties .gitmodules NOTICE
  scripts/ci .github/workflows/runtime-native.yml .github/workflows/runtime-native-macos.yml)
native_changed() {
  local base
  base="$(git merge-base HEAD origin/main 2>/dev/null || true)"
  [[ -n "${base}" ]] || return 0   # no origin/main: assume changed
  # Committed changes since the merge base, plus staged/unstaged/untracked changes.
  [[ -n "$(git diff --name-only "${base}" -- "${native_paths[@]}")" ]] \
    || [[ -n "$(git ls-files --others --exclude-standard -- "${native_paths[@]}")" ]]
}

selected() {
  local step="$1"
  if [[ -n "${only}" ]]; then in_list "${step}" "${only}"; return; fi
  in_list "${step}" "${skip}" && return 1
  case "${step}" in
    pin|native|jvm|android) native_changed ;;
    *) return 0 ;;
  esac
}

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
      run_step pin ./scripts/ci/verify-ggml-pin.sh ;;
    native)
      run_step native ./kggml-runtime/native/scripts/build-host.sh ${clean_flag[@]+"${clean_flag[@]}"}
      run_step audit ./scripts/ci/audit-host-linkage.sh ;;
    jvm)
      run_step jvm ./gradlew :kggml-runtime:jvmTest -Pkggml.requireNative=true --console=plain ;;
    android)
      ndk="${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
      if [[ -z "${ndk}" && -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}/ndk" ]]; then ndk="${ANDROID_HOME}/ndk"; fi
      if [[ -n "${ndk}" ]]; then
        [[ -z "${ANDROID_NDK_LATEST_HOME:-}" ]] || export ANDROID_NDK_HOME="${ANDROID_NDK_LATEST_HOME}"
        run_step android ./kggml-runtime/native/scripts/build-android.sh arm64-v8a x86_64
        run_step a-audit ./scripts/ci/audit-android-linkage.sh arm64-v8a x86_64
      else
        missing android "Android NDK not found (set ANDROID_NDK_HOME)"
      fi ;;
  esac
done

summary
