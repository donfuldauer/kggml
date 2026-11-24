#!/usr/bin/env bash
# Verifies the whisper.cpp submodule HEAD matches KGGML_WHISPER_UPSTREAM_COMMIT in
# kggml-whisper/native/CMakeLists.txt and NOTICE. Shared by CI and ./scripts/ci-local.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

actual="$(git -C kggml-whisper/native/third_party/whisper.cpp rev-parse HEAD)"
pinned="$(sed -n 's/^set(KGGML_WHISPER_UPSTREAM_COMMIT "\([0-9a-f]\{40\}\)")$/\1/p' kggml-whisper/native/CMakeLists.txt)"
echo "submodule=${actual} pinned=${pinned}"
if [[ -z "${pinned}" ]]; then
  echo "KGGML_WHISPER_UPSTREAM_COMMIT not found in kggml-whisper/native/CMakeLists.txt" >&2; exit 1
fi
if [[ "${actual}" != "${pinned}" ]]; then
  echo "whisper.cpp submodule (${actual}) does not match pinned commit (${pinned})" >&2; exit 1
fi
if ! grep -q "${pinned}" NOTICE; then
  echo "pinned whisper.cpp commit ${pinned} is not recorded in NOTICE" >&2; exit 1
fi
echo "whisper.cpp pin: OK"
