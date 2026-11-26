#!/usr/bin/env bash
# Cross-compiles the Android OpenCL add-on: ggml's OpenCL backend as a loadable backend module (libggml-opencl.so) per ABI
# under <repo>/build/native-opencl-android-<abi>/bin, and stages it in <repo>/build/native-android-opencl-jniLibs/<abi>/
# for the :kggml-runtime-opencl AAR. Only libggml-opencl.so is staged: it links the libggml-base.so of :kggml-runtime.
#
# Requires ANDROID_NDK_HOME (or ANDROID_NDK_ROOT / ANDROID_HOME/ndk/<latest>), Python 3 (ggml embeds its OpenCL kernels)
# and the ggml + OpenCL-Headers + OpenCL-ICD-Loader submodules.
#
# Usage: kggml-runtime-opencl/native/scripts/build-android.sh [abi ...]   (default: arm64-v8a x86_64)
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
  echo "==> kggml-runtime-opencl ${preset}"
  cmake -S "${NATIVE_DIR}" --preset "${preset}"
  cmake --build "${REPO_ROOT}/build/native-opencl-${preset}" -j "${jobs}"

  # ggml puts its shared libraries / modules in bin/ (CMAKE_RUNTIME_OUTPUT_DIRECTORY).
  module="${REPO_ROOT}/build/native-opencl-${preset}/bin/libggml-opencl.so"
  [[ -f "${module}" ]] || { echo "${module} was not built" >&2; exit 1; }
  out="${REPO_ROOT}/build/native-android-opencl-jniLibs/${abi}"
  mkdir -p "${out}"
  cp "${module}" "${out}/"
done

echo "==> libggml-opencl.so staged under ${REPO_ROOT}/build/native-android-opencl-jniLibs"
