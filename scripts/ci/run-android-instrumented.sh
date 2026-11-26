#!/usr/bin/env bash
# Runs every Android instrumented test suite (libraries + samples) on a connected device or emulator, with the real
# native libraries packaged into the APKs (-Pkggml.bundleNative=true) and required (-Pkggml.requireNative=true):
#   :kggml-{core-api,runtime,whisper,llama,diffusion,model-manager}  commonTest + the native engine contracts on ART
#   :samples:kggml-showcase        Compose UI tests (fakes) + real-engine end-to-end UI tests
#   :samples:whisper-only-minimal  real WhisperEngine session tests + MainActivity Espresso tests
# Shared by ./scripts/ci-local.sh (step `instrumented`) and CI. Works on macOS, Linux and Windows (Git Bash) hosts.
#
# Usage: scripts/ci/run-android-instrumented.sh [extra Gradle args...]
#   A device must be connected (adb devices), or set KGGML_AVD=<avd name> to boot that emulator headless first
#   (it is shut down again at the end). The Android NDK is found like ci-local.sh does (ANDROID_NDK_HOME, else the
#   newest NDK under the SDK). The device ABI must be arm64-v8a or x86_64 (the ABIs build-android.sh produces).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
if [[ -z "${sdk}" && -f local.properties ]]; then sdk="$(sed -n 's/^sdk\.dir=//p' local.properties | tail -n 1)"; fi
[[ -n "${sdk}" && -d "${sdk}" ]] || { echo "Android SDK not found (set ANDROID_HOME)" >&2; exit 2; }
export ANDROID_HOME="${sdk}"
exe=""
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) exe=".exe" ;; esac
adb="${sdk}/platform-tools/adb${exe}"
[[ -x "${adb}" ]] || adb="$(command -v adb || true)"
[[ -n "${adb}" ]] || { echo "adb not found (install platform-tools)" >&2; exit 2; }

if [[ -z "${ANDROID_NDK_HOME:-}" && -d "${sdk}/ndk" ]]; then
  ANDROID_NDK_HOME="$(find "${sdk}/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
  export ANDROID_NDK_HOME
fi

booted_pid=""
cleanup() {
  if [[ -n "${booted_pid}" ]]; then
    "${adb}" emu kill >/dev/null 2>&1 || true
    wait "${booted_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

device_count() { "${adb}" devices | awk 'NR > 1 && $2 == "device"' | wc -l | tr -d ' '; }

if [[ "$(device_count)" == 0 ]]; then
  if [[ -z "${KGGML_AVD:-}" ]]; then
    echo "No Android device connected. Connect one or set KGGML_AVD=<avd name> to boot an emulator." >&2
    exit 2
  fi
  emulator="${sdk}/emulator/emulator${exe}"
  [[ -x "${emulator}" ]] || { echo "emulator not found under ${sdk}/emulator" >&2; exit 2; }
  echo "Booting emulator ${KGGML_AVD} (headless)…"
  "${emulator}" -avd "${KGGML_AVD}" -no-window -no-audio -no-snapshot -no-boot-anim -gpu swiftshader_indirect &
  booted_pid=$!
  "${adb}" wait-for-device
  for _ in $(seq 1 120); do
    [[ "$("${adb}" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == 1 ]] && break
    sleep 5
  done
  [[ "$("${adb}" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == 1 ]] || { echo "emulator did not boot" >&2; exit 1; }
fi

abi="$("${adb}" shell getprop ro.product.cpu.abi | tr -d '\r')"
case "${abi}" in
  arm64-v8a|x86_64) echo "Device ABI: ${abi}" ;;
  *) echo "Unsupported device ABI '${abi}' (native libraries are built for arm64-v8a and x86_64)" >&2; exit 2 ;;
esac

./gradlew --console=plain --continue \
  :kggml-core-api:connectedDebugAndroidTest \
  :kggml-runtime:connectedDebugAndroidTest \
  :kggml-whisper:connectedDebugAndroidTest \
  :kggml-llama:connectedDebugAndroidTest \
  :kggml-diffusion:connectedDebugAndroidTest \
  :kggml-model-manager:connectedDebugAndroidTest \
  :samples:whisper-only-minimal:connectedDebugAndroidTest \
  :samples:kggml-showcase:connectedDebugAndroidTest \
  -Pkggml.bundleNative=true -Pkggml.requireNative=true "$@"
