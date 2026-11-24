#!/usr/bin/env bash
# Builds KggmlWhisper.xcframework (dynamic) for iOS device (arm64) and iOS simulator (arm64).
#
# Each slice is KggmlWhisper.framework: whisper.cpp + the Kggml C ABI + the CoreML encoder (WHISPER_COREML,
# runtime-gated per context), linked dynamically against GgmlRuntime.framework from :kggml-runtime. The
# runtime slices (build/native-ios-device, build/native-ios-simulator-arm64) are built first with
# kggml-runtime/native/scripts/build-ios-xcframework.sh when missing.
#
# Requirements: macOS host with Xcode (xcodebuild, xcrun) and CMake >= 3.24.
# Output: <repo>/build/native-ios/KggmlWhisper.xcframework
#
# Usage: kggml-whisper/native/scripts/build-ios-xcframework.sh [--no-coreml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NATIVE_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/build/native-ios"
XCFRAMEWORK="${OUT_DIR}/KggmlWhisper.xcframework"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "build-ios-xcframework.sh must run on macOS with Xcode installed" >&2
  exit 1
fi
command -v xcodebuild >/dev/null || { echo "xcodebuild not found" >&2; exit 1; }

extra_args=()
coreml=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-coreml) extra_args+=("-DKGGML_WHISPER_COREML=OFF"); coreml=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

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
  build_dir="${REPO_ROOT}/build/native-whisper-${preset}"
  echo "==> ${preset}"
  cmake -S "${NATIVE_DIR}" --preset "${preset}" ${extra_args[@]+"${extra_args[@]}"}
  cmake --build "${build_dir}" --config Release -- -quiet

  fw="$(find_framework "${build_dir}" KggmlWhisper)" || {
    echo "KggmlWhisper.framework (with its binary) not found under ${build_dir}; candidates:" >&2
    find "${build_dir}" -name 'KggmlWhisper*' -not -path '*/CMakeFiles/*' -exec ls -ld {} + >&2 || true
    exit 1
  }
  runtime_fw="$(find_framework "${REPO_ROOT}/build/native-${preset}" GgmlRuntime)"
  echo "    framework: ${fw}"

  # Audit: dynamic Mach-O, links GgmlRuntime.framework, exports the C ABI, defines none of ggml's symbols.
  # (Captured to variables first: `grep -q` closing a pipe early would trip `set -o pipefail`.)
  file_info="$(file "${fw}/KggmlWhisper")"
  links="$(otool -L "${fw}/KggmlWhisper")"
  exports="$(nm -gU "${fw}/KggmlWhisper")"
  defined="$(nm -U "${fw}/KggmlWhisper" | awk 'NF >= 3 {print $3}' | sort -u)"
  ggml_api="$(nm -gU "${runtime_fw}/GgmlRuntime" | awk '{print $3}' | grep -E '^_(ggml|gguf)_' | sort -u)"
  grep -q "dynamically linked shared library" <<<"${file_info}"
  grep -q "GgmlRuntime.framework/GgmlRuntime" <<<"${links}" || { echo "${links}" >&2; echo "does not link GgmlRuntime" >&2; exit 1; }
  grep -q " _kggml_whisper_transcribe$" <<<"${exports}"
  grep -q " _whisper_full_with_state$" <<<"${exports}"
  dups="$(comm -12 <(echo "${defined}") <(echo "${ggml_api}"))"
  if [[ -n "${dups}" ]]; then
    echo "${dups}" | head -n 20 >&2
    echo "KggmlWhisper embeds ggml symbols" >&2
    exit 1
  fi
  if [[ ${coreml} -eq 1 ]]; then
    grep -q "CoreML.framework" <<<"${links}" || { echo "CoreML.framework not linked" >&2; exit 1; }
  fi
  frameworks+=(-framework "${fw}")
done

rm -rf "${XCFRAMEWORK}"
mkdir -p "${OUT_DIR}"
xcodebuild -create-xcframework "${frameworks[@]}" -output "${XCFRAMEWORK}"
echo "==> ${XCFRAMEWORK}"
