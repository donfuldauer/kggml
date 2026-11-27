#!/usr/bin/env bash
# Verifies the ggml submodule HEAD matches KGGML_GGML_PINNED_COMMIT in CMakeLists.txt and NOTICE.
# Shared by CI (.github/workflows/ci.yml) and ./scripts/ci-local.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

actual="$(git -C kggml-runtime/native/third_party/ggml rev-parse HEAD)"
pinned="$(sed -n 's/^set(KGGML_GGML_PINNED_COMMIT "\([0-9a-f]\{40\}\)")$/\1/p' kggml-runtime/native/CMakeLists.txt)"
echo "submodule=${actual} pinned=${pinned}"
if [[ -z "${pinned}" ]]; then
  echo "KGGML_GGML_PINNED_COMMIT not found in kggml-runtime/native/CMakeLists.txt" >&2; exit 1
fi
if [[ "${actual}" != "${pinned}" ]]; then
  echo "ggml submodule (${actual}) does not match pinned commit (${pinned})" >&2; exit 1
fi
if ! grep -q "${pinned}" NOTICE; then
  echo "pinned ggml commit ${pinned} is not recorded in NOTICE" >&2; exit 1
fi
echo "ggml pin: OK"
