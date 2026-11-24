#!/usr/bin/env bash
# Builds the :kggml-runtime native layer for the current desktop host and runs its CTest suite.
#
# Output: <repo>/build/native (so `ctest --test-dir build/native --output-on-failure` works from the root).
#
# Usage: kggml-runtime/native/scripts/build-host.sh [--clean] [--no-tests] [--skip-ctest] [--vulkan]
#                                                  [--install <prefix>] [--build-dir <dir>]
#   --no-tests    configure with KGGML_RUNTIME_BUILD_TESTS=OFF (no GoogleTest download, no test hooks)
#   --skip-ctest  build the tests but do not run them
#   --build-dir   build tree to use instead of <repo>/build/native (e.g. a release tree for packaging)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NATIVE_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/native"

clean=0
build_tests=1
run_tests=1
extra_args=()
install_prefix=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) clean=1 ;;
    --no-tests) build_tests=0; run_tests=0 ;;
    --skip-ctest) run_tests=0 ;;
    --vulkan) extra_args+=("-DKGGML_ENABLE_VULKAN=ON") ;;
    --install) install_prefix="${2:?--install needs a prefix}"; shift ;;
    --build-dir) BUILD_DIR="${2:?--build-dir needs a directory}"; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$(uname -s)" in
  Linux)  preset="linux-x64" ;;
  Darwin) preset="macos-arm64" ;;
  MINGW*|MSYS*|CYGWIN*) preset="windows-x64" ;;
  *) echo "unsupported host: $(uname -s)" >&2; exit 1 ;;
esac

if [[ ! -f "${NATIVE_DIR}/third_party/ggml/CMakeLists.txt" ]]; then
  echo "ggml submodule missing; running: git submodule update --init --recursive" >&2
  git -C "${REPO_ROOT}" submodule update --init --recursive
fi

if [[ ${clean} -eq 1 ]]; then
  rm -rf "${BUILD_DIR}"
fi

if command -v nproc >/dev/null 2>&1; then
  jobs="$(nproc)"
else
  jobs="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

tests_flag="ON"
[[ ${build_tests} -eq 1 ]] || tests_flag="OFF"

echo "==> Configuring preset ${preset} -> ${BUILD_DIR}"
cmake -S "${NATIVE_DIR}" --preset "${preset}" -B "${BUILD_DIR}" -DKGGML_RUNTIME_BUILD_TESTS="${tests_flag}" ${extra_args[@]+"${extra_args[@]}"}

echo "==> Building (-j${jobs})"
cmake --build "${BUILD_DIR}" --config Release -j "${jobs}"

if [[ ${run_tests} -eq 1 ]]; then
  echo "==> Running CTest"
  ctest --test-dir "${BUILD_DIR}" --build-config Release --output-on-failure -j "${jobs}"
fi

if [[ -n "${install_prefix}" ]]; then
  echo "==> Installing to ${install_prefix}"
  cmake --install "${BUILD_DIR}" --config Release --prefix "${install_prefix}"
fi
