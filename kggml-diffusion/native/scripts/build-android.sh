#!/usr/bin/env bash
# Cross-compiles the :kggml-diffusion native layer (stable-diffusion.cpp + Kggml C ABI) for Android
# (arm64-v8a, x86_64).
#
# Links against the :kggml-runtime Android builds in <repo>/build/native-android-<abi> (built first with
# kggml-runtime/native/scripts/build-android.sh when missing). Outputs an unversioned libkggml_diffusion.so per ABI
# under <repo>/build/native-diffusion-android-<abi>/lib and stages it in
# <repo>/build/native-android-diffusion-jniLibs/<abi>/ (the :kggml-diffusion AAR jniLibs; the runtime libraries stay
# in build/native-android-jniLibs so no .so is packaged by two AARs). The JNI bridge is added by issue #17.
#
# Requires ANDROID_NDK_HOME (or ANDROID_NDK_ROOT / ANDROID_HOME/ndk/<latest>).
# Usage: kggml-diffusion/native/scripts/build-android.sh [abi ...]   (default: arm64-v8a x86_64)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NATIVE_DIR}/../.." && pwd)"
SUBMODULE="kggml-diffusion/native/third_party/stable-diffusion.cpp"

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

if [[ ! -f "${REPO_ROOT}/${SUBMODULE}/src/stable-diffusion.cpp" ]]; then
  echo "stable-diffusion.cpp submodule missing; running: git submodule update --init ${SUBMODULE}" >&2
  git -C "${REPO_ROOT}" submodule update --init "${SUBMODULE}"
fi

abis=("$@")
[[ ${#abis[@]} -gt 0 ]] || abis=(arm64-v8a x86_64)
jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

missing=()
for abi in "${abis[@]}"; do
  [[ -f "${REPO_ROOT}/build/native-android-${abi}/kggml-runtime-config.cmake" ]] || missing+=("${abi}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "==> :kggml-runtime Android build missing for ${missing[*]}; building it first"
  "${REPO_ROOT}/kggml-runtime/native/scripts/build-android.sh" "${missing[@]}"
fi

for abi in "${abis[@]}"; do
  preset="android-${abi}"
  echo "==> ${preset}"
  cmake -S "${NATIVE_DIR}" --preset "${preset}"
  cmake --build "${REPO_ROOT}/build/native-diffusion-${preset}" -j "${jobs}"
  out="${REPO_ROOT}/build/native-android-diffusion-jniLibs/${abi}"
  mkdir -p "${out}"
  cp "${REPO_ROOT}/build/native-diffusion-${preset}/lib/libkggml_diffusion.so" "${out}/"
done

echo "==> libkggml_diffusion.so staged under ${REPO_ROOT}/build/native-android-diffusion-jniLibs"
