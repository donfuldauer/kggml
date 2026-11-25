#!/usr/bin/env bash
# Verifies the llama.cpp submodule HEAD matches KGGML_LLAMA_UPSTREAM_COMMIT in kggml-llama/native/CMakeLists.txt
# and NOTICE (libmtmd is built from the same commit). Shared by CI and ./scripts/ci-local.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

actual="$(git -C kggml-llama/native/third_party/llama.cpp rev-parse HEAD)"
pinned="$(sed -n 's/^set(KGGML_LLAMA_UPSTREAM_COMMIT "\([0-9a-f]\{40\}\)")$/\1/p' kggml-llama/native/CMakeLists.txt)"
echo "submodule=${actual} pinned=${pinned}"
if [[ -z "${pinned}" ]]; then
  echo "KGGML_LLAMA_UPSTREAM_COMMIT not found in kggml-llama/native/CMakeLists.txt" >&2; exit 1
fi
if [[ "${actual}" != "${pinned}" ]]; then
  echo "llama.cpp submodule (${actual}) does not match pinned commit (${pinned})" >&2; exit 1
fi
if ! grep -q "${pinned}" NOTICE; then
  echo "pinned llama.cpp commit ${pinned} is not recorded in NOTICE" >&2; exit 1
fi
echo "llama.cpp pin: OK"
