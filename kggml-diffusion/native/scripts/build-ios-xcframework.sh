#!/usr/bin/env bash
# Builds KggmlDiffusion.xcframework (dynamic) for iOS device (arm64) and iOS simulator (arm64).
#
# Each slice is KggmlDiffusion.framework: stable-diffusion.cpp + the Kggml C ABI, linked dynamically against
# GgmlRuntime.framework from :kggml-runtime (which also exports the GGML_MAX_NAME=128 ABI definition). The runtime
# slices (build/native-ios-device, build/native-ios-simulator-arm64) are built first with
# kggml-runtime/native/scripts/build-ios-xcframework.sh when missing.
#
# Requirements: macOS host with Xcode (xcodebuild, xcrun) and CMake >= 3.24.
# Output: <repo>/build/native-ios/KggmlDiffusion.xcframework
#
# Usage: kggml-diffusion/native/scripts/build-ios-xcframework.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NATIVE_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/build/native-ios"
XCFRAMEWORK="${OUT_DIR}/KggmlDiffusion.xcframework"
SUBMODULE="kggml-diffusion/native/third_party/stable-diffusion.cpp"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "build-ios-xcframework.sh must run on macOS with Xcode installed" >&2
  exit 1
fi
command -v xcodebuild >/dev/null || { echo "xcodebuild not found" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "${REPO_ROOT}/${SUBMODULE}/src/stable-diffusion.cpp" ]]; then
  echo "stable-diffusion.cpp submodule missing; running: git submodule update --init ${SUBMODULE}" >&2
  git -C "${REPO_ROOT}" submodule update --init "${SUBMODULE}"
fi

for preset in ios-device ios-simulator-arm64; do
  if [[ ! -f "${REPO_ROOT}/build/native-${preset}/kggml-runtime-config.cmake" ]]; then
    echo "==> :kggml-runtime iOS slices missing; building GgmlRuntime.xcframework first"
    "${REPO_ROOT}/kggml-runtime/native/scripts/build-ios-xcframework.sh"
    break
  fi
done

# Finds <name>.framework that actually holds its Mach-O (newer Xcode also emits binary-less stubs).
find_framework() {
  local dir="$1" name="$2" candidate
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/${name}" ]]; then echo "${candidate}"; return 0; fi
  done < <(find "${dir}" -name "${name}.framework" -not -path '*/EagerLinkingTBDs/*' -prune)
  return 1
}

frameworks=()
for preset in ios-device ios-simulator-arm64; do
  build_dir="${REPO_ROOT}/build/native-diffusion-${preset}"
  echo "==> ${preset}"
  cmake -S "${NATIVE_DIR}" --preset "${preset}"
  cmake --build "${build_dir}" --config Release -- -quiet

  fw="$(find_framework "${build_dir}" KggmlDiffusion)" || {
    echo "KggmlDiffusion.framework (with its binary) not found under ${build_dir}; candidates:" >&2
    find "${build_dir}" -name 'KggmlDiffusion*' -not -path '*/CMakeFiles/*' -exec ls -ld {} + >&2 || true
    exit 1
  }
  runtime_fw="$(find_framework "${REPO_ROOT}/build/native-${preset}" GgmlRuntime)"
  echo "    framework: ${fw}"

  # Audit: dynamic Mach-O, links GgmlRuntime.framework, exports the C ABI, defines none of ggml's symbols and no
  # whisper.cpp / llama.cpp symbols, ships no test hooks.
  # (Captured to variables first: `grep -q` closing a pipe early would trip `set -o pipefail`.)
  file_info="$(file "${fw}/KggmlDiffusion")"
  links="$(otool -L "${fw}/KggmlDiffusion")"
  exports="$(nm -gU "${fw}/KggmlDiffusion")"
  defined="$(nm -U "${fw}/KggmlDiffusion" | awk 'NF >= 3 {print $3}' | sort -u)"
  ggml_api="$(nm -gU "${runtime_fw}/GgmlRuntime" | awk '{print $3}' | grep -E '^_(ggml|gguf)_' | sort -u)"
  grep -q "dynamically linked shared library" <<<"${file_info}"
  grep -q "GgmlRuntime.framework/GgmlRuntime" <<<"${links}" || { echo "${links}" >&2; echo "does not link GgmlRuntime" >&2; exit 1; }
  for sym in _kggml_diffusion_generate _kggml_diffusion_context_create _kggml_diffusion_request_abort _generate_image _new_sd_ctx; do
    grep -q " ${sym}$" <<<"${exports}" || { echo "${sym} is not exported" >&2; exit 1; }
  done
  if grep -q " _kggml_diffusion_test_" <<<"${exports}"; then echo "release framework exports test hooks" >&2; exit 1; fi
  if grep -Eq '^_(whisper|llama|mtmd)_' <<<"${defined}"; then echo "KggmlDiffusion embeds whisper / llama symbols" >&2; exit 1; fi
  dups="$(comm -12 <(echo "${defined}") <(echo "${ggml_api}"))"
  if [[ -n "${dups}" ]]; then
    echo "${dups}" | head -n 20 >&2
    echo "KggmlDiffusion embeds ggml symbols" >&2
    exit 1
  fi
  frameworks+=(-framework "${fw}")
done

rm -rf "${XCFRAMEWORK}"
mkdir -p "${OUT_DIR}"
xcodebuild -create-xcframework "${frameworks[@]}" -output "${XCFRAMEWORK}"
echo "==> ${XCFRAMEWORK}"
