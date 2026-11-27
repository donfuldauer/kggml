#!/usr/bin/env bash
# Builds GgmlRuntime.xcframework (dynamic) for iOS device (arm64), iOS simulator (arm64) and macOS (arm64; the
# Kotlin/Native macosArm64 target, issue #93).
#
# Each slice is one dynamic framework (GgmlRuntime.framework) produced by the KGGML_RUNTIME_MONOLITHIC
# layout: ggml-base + ggml-cpu (NEON/Accelerate) + ggml-blas (Accelerate) + ggml-metal (embedded Metal
# library) + kggml_runtime, all exported from a single Mach-O so every engine framework links the same
# ggml symbols dynamically.
#
# Requirements: macOS host with Xcode (xcodebuild, xcrun) and CMake >= 3.24.
# Output: <repo>/build/native-ios/GgmlRuntime.xcframework
#
# Usage: kggml-runtime/native/scripts/build-ios-xcframework.sh [--coreml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NATIVE_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/build/native-ios"
XCFRAMEWORK="${OUT_DIR}/GgmlRuntime.xcframework"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "build-ios-xcframework.sh must run on macOS with Xcode installed" >&2
  exit 1
fi
command -v xcodebuild >/dev/null || { echo "xcodebuild not found" >&2; exit 1; }

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --coreml) extra_args+=("-DKGGML_ENABLE_COREML=ON") ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

frameworks=()
for preset in ios-device ios-simulator-arm64 macos-framework; do
  build_dir="${REPO_ROOT}/build/native-${preset}"
  echo "==> ${preset}"
  cmake -S "${NATIVE_DIR}" --preset "${preset}" ${extra_args[@]+"${extra_args[@]}"}
  cmake --build "${build_dir}" --config Release -- -quiet

  # Each preset has its own build dir, so any GgmlRuntime.framework in it is this platform's. Newer Xcode also
  # emits a binary-less stub under build/EagerLinkingTBDs/, so only accept a framework that holds its Mach-O.
  # (A macOS framework is versioned: its top-level GgmlRuntime is a symlink into Versions/Current.)
  fw=""
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/GgmlRuntime" ]]; then fw="${candidate}"; break; fi
  done < <(find "${build_dir}" -name GgmlRuntime.framework -not -path '*/EagerLinkingTBDs/*' -prune)
  if [[ -z "${fw}" ]]; then
    echo "GgmlRuntime.framework (with its binary) not found under ${build_dir}; candidates:" >&2
    find "${build_dir}" -name 'GgmlRuntime*' -not -path '*/CMakeFiles/*' -exec ls -ld {} + >&2 || true
    exit 1
  fi
  echo "    framework: ${fw}"
  # Sanity: a dynamic Mach-O that exports both the kggml C ABI and ggml.
  # (Captured to variables first: `grep -q` closing a pipe early would trip `set -o pipefail`.)
  file_info="$(file -L "${fw}/GgmlRuntime")"
  symbols="$(nm -gU "${fw}/GgmlRuntime")"
  grep -q "dynamically linked shared library" <<<"${file_info}"
  grep -q " _kggml_runtime_init$" <<<"${symbols}"
  grep -q " _ggml_init$" <<<"${symbols}"
  frameworks+=(-framework "${fw}")
done

rm -rf "${XCFRAMEWORK}"
mkdir -p "${OUT_DIR}"
xcodebuild -create-xcframework "${frameworks[@]}" -output "${XCFRAMEWORK}"
echo "==> ${XCFRAMEWORK}"
