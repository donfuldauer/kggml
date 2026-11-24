#!/usr/bin/env bash
# Audits unversioned sonames and dynamic ggml linkage of the Android cross-compiled libraries.
# Expects ./kggml-runtime/native/scripts/build-android.sh to have produced build/native-android-<abi>/lib.
# Shared by CI (.github/workflows/runtime-native.yml) and ./scripts/ci-local.sh.
#
# Usage: scripts/ci/audit-android-linkage.sh [abi ...]   (default: arm64-v8a x86_64)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
abis=("$@")
[[ ${#abis[@]} -gt 0 ]] || abis=(arm64-v8a x86_64)

fail() { echo "FAIL: $*" >&2; exit 1; }
contains() { grep -Eq -- "$2" <<<"$1"; }

for abi in "${abis[@]}"; do
  lib="${REPO_ROOT}/build/native-android-${abi}/lib"
  for so in libggml-base.so libggml-cpu.so libggml.so libkggml_runtime.so libkggml_runtime_jni.so; do
    contains "$(readelf -d "${lib}/${so}")" "SONAME.*\[${so//./\\.}\]" || fail "${abi}: ${so} has wrong SONAME"
  done
  contains "$(readelf -d "${lib}/libkggml_runtime_jni.so")" 'NEEDED.*\[libkggml_runtime\.so\]' \
    || fail "${abi}: libkggml_runtime_jni.so does not link libkggml_runtime.so"
  contains "$(readelf -d "${lib}/libkggml_runtime.so")" 'NEEDED.*\[libggml\.so\]' \
    || fail "${abi}: libkggml_runtime.so does not link libggml.so"
  for so in libkggml_runtime.so libkggml_runtime_jni.so; do
    defs="$(nm -D --defined-only "${lib}/${so}")"
    if contains "${defs}" ' (ggml|gguf)_'; then
      grep -E ' (ggml|gguf)_' <<<"${defs}" >&2
      fail "${abi}: ${so} embeds ggml symbols"
    fi
  done
  echo "${abi}: OK"
done
