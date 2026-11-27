#!/usr/bin/env bash
# Cross-compiles the Android GPU add-on: ggml's Vulkan backend as a loadable backend module (libggml-vulkan.so) per ABI
# under <repo>/build/native-vulkan-android-<abi>/bin, and stages it in <repo>/build/native-android-vulkan-jniLibs/<abi>/
# for the :kggml-runtime-vulkan AAR. Only libggml-vulkan.so is staged: it links the libggml-base.so of :kggml-runtime.
#
# Requires ANDROID_NDK_HOME (or ANDROID_NDK_ROOT / ANDROID_HOME/ndk/<latest>), a host C/C++ compiler (ggml builds its
# vulkan-shaders-gen tool for the host) and the ggml + Vulkan-Headers submodules.
#
# Usage: kggml-runtime-vulkan/native/scripts/build-android.sh [abi ...]   (default: arm64-v8a x86_64)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NATIVE_DIR}/../.." && pwd)"

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  if [[ -n "${ANDROID_NDK_ROOT:-}" ]]; then
    export ANDROID_NDK_HOME="${ANDROID_NDK_ROOT}"
  elif [[ -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}/ndk" ]]; then
    ANDROID_NDK_HOME="$(find "${ANDROID_HOME}/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
    export ANDROID_NDK_HOME
  fi
fi
if [[ ! -f "${ANDROID_NDK_HOME:-}/build/cmake/android.toolchain.cmake" ]]; then
  echo "ANDROID_NDK_HOME does not point to an NDK (missing build/cmake/android.toolchain.cmake)" >&2
  exit 1
fi

abis=("$@")
[[ ${#abis[@]} -gt 0 ]] || abis=(arm64-v8a x86_64)
jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

for abi in "${abis[@]}"; do
  preset="android-${abi}"
  echo "==> kggml-runtime-vulkan ${preset}"
  cmake -S "${NATIVE_DIR}" --preset "${preset}"
  cmake --build "${REPO_ROOT}/build/native-vulkan-${preset}" -j "${jobs}"

  # ggml puts its shared libraries / modules in bin/ (CMAKE_RUNTIME_OUTPUT_DIRECTORY).
  module="${REPO_ROOT}/build/native-vulkan-${preset}/bin/libggml-vulkan.so"
  [[ -f "${module}" ]] || { echo "${module} was not built" >&2; exit 1; }
  out="${REPO_ROOT}/build/native-android-vulkan-jniLibs/${abi}"
  mkdir -p "${out}"
  cp "${module}" "${out}/"
done

echo "==> libggml-vulkan.so staged under ${REPO_ROOT}/build/native-android-vulkan-jniLibs"
