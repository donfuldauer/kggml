#!/usr/bin/env bash
# Cross-compiles the :kggml-runtime native layer for Android (arm64-v8a, x86_64).
#
# Requires ANDROID_NDK_HOME (or ANDROID_NDK_ROOT / ANDROID_HOME/ndk/<latest>). Outputs unversioned
# libggml.so, libggml-base.so, libggml-cpu.so and libkggml_runtime.so per ABI under
# <repo>/build/native-android-<abi>/lib, and copies them (plus libc++_shared.so) to
# <repo>/build/native-android-jniLibs/<abi>/ ready for an AGP jniLibs source set.
#
# Usage: kggml-runtime/native/scripts/build-android.sh [abi ...]   (default: arm64-v8a x86_64)
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
host_tag="$(ls "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt" | head -n 1)"
sysroot_lib="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/${host_tag}/sysroot/usr/lib"

for abi in "${abis[@]}"; do
  preset="android-${abi}"
  echo "==> ${preset}"
  cmake -S "${NATIVE_DIR}" --preset "${preset}"
  cmake --build "${REPO_ROOT}/build/native-${preset}" -j "${jobs}"

  out="${REPO_ROOT}/build/native-android-jniLibs/${abi}"
  mkdir -p "${out}"
  cp "${REPO_ROOT}/build/native-${preset}/lib/"*.so "${out}/"
  case "${abi}" in
    arm64-v8a) triple="aarch64-linux-android" ;;
    x86_64)    triple="x86_64-linux-android" ;;
    *) echo "unsupported ABI ${abi}" >&2; exit 1 ;;
  esac
  cp "${sysroot_lib}/${triple}/libc++_shared.so" "${out}/"
done

echo "==> jniLibs staged under ${REPO_ROOT}/build/native-android-jniLibs"
