#!/usr/bin/env bash
# Verifies the stable-diffusion.cpp submodule HEAD matches KGGML_DIFFUSION_UPSTREAM_COMMIT in
# kggml-diffusion/native/CMakeLists.txt and NOTICE. Shared by CI and ./scripts/ci-local.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

actual="$(git -C kggml-diffusion/native/third_party/stable-diffusion.cpp rev-parse HEAD)"
pinned="$(sed -n 's/^set(KGGML_DIFFUSION_UPSTREAM_COMMIT "\([0-9a-f]\{40\}\)")$/\1/p' kggml-diffusion/native/CMakeLists.txt)"
echo "submodule=${actual} pinned=${pinned}"
if [[ -z "${pinned}" ]]; then
  echo "KGGML_DIFFUSION_UPSTREAM_COMMIT not found in kggml-diffusion/native/CMakeLists.txt" >&2; exit 1
fi
if [[ "${actual}" != "${pinned}" ]]; then
  echo "stable-diffusion.cpp submodule (${actual}) does not match pinned commit (${pinned})" >&2; exit 1
fi
if ! grep -q "${pinned}" NOTICE; then
  echo "pinned stable-diffusion.cpp commit ${pinned} is not recorded in NOTICE" >&2; exit 1
fi
# The gitlink recorded in the superproject index must be the pin too (not just the checked-out HEAD).
gitlink="$(git ls-files -s kggml-diffusion/native/third_party/stable-diffusion.cpp | awk '{print $2}')"
if [[ "${gitlink}" != "${pinned}" ]]; then
  echo "superproject gitlink (${gitlink:-none}) does not match pinned commit (${pinned})" >&2; exit 1
fi
echo "stable-diffusion.cpp pin: OK"
