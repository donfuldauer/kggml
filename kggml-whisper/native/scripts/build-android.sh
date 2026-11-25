#!/usr/bin/env bash
# Cross-compiles the :kggml-whisper native layer for Android (arm64-v8a, x86_64).
#
# Links against the :kggml-runtime Android builds in <repo>/build/native-android-<abi> (built first with
# kggml-runtime/native/scripts/build-android.sh when missing). Outputs unversioned libkggml_whisper.so and
# libkggml_whisper_jni.so (JNI bridge of WhisperEngine) per ABI under <repo>/build/native-whisper-android-<abi>/lib
# and stages them in <repo>/build/native-android-whisper-jniLibs/<abi>/ (the :kggml-whisper AAR jniLibs; the
# runtime libraries stay in build/native-android-jniLibs so no .so is packaged by two AARs).
#
# Requires ANDROID_NDK_HOME (or ANDROID_NDK_ROOT / ANDROID_HOME/ndk/<latest>).
# Usage: kggml-whisper/native/scripts/build-android.sh [abi ...]   (default: arm64-v8a x86_64)
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
  cmake --build "${REPO_ROOT}/build/native-whisper-${preset}" -j "${jobs}"
  out="${REPO_ROOT}/build/native-android-whisper-jniLibs/${abi}"
  mkdir -p "${out}"
  cp "${REPO_ROOT}/build/native-whisper-${preset}/lib/libkggml_whisper.so" "${out}/"
  cp "${REPO_ROOT}/build/native-whisper-${preset}/lib/libkggml_whisper_jni.so" "${out}/"
done

echo "==> libkggml_whisper.so + libkggml_whisper_jni.so staged under ${REPO_ROOT}/build/native-android-whisper-jniLibs"
