#!/usr/bin/env bash
# Builds the JVM desktop GPU add-on (issue #92): ggml's Vulkan backend as a loadable backend module for this Linux
# host (libggml-vulkan.so, linked against the same ggml as :kggml-runtime), and stages it for the
# :kggml-runtime-vulkan jar at <repo>/build/native-vulkan-jvm/kggml/native-addons/<platform>/ggml-vulkan/.
#
# Requires the Vulkan loader development library and glslc (Debian/Ubuntu: apt install libvulkan-dev glslc; without a
# glslc on PATH, the Android NDK's shader-tools/<host>/glslc from ANDROID_NDK_HOME is used), a C/C++
# compiler, CMake + Ninja and the ggml + Vulkan-Headers + SPIRV-Headers submodules. Windows: build the windows-x64 preset from an MSVC
# developer prompt with the Vulkan SDK installed, then run this script with --stage-only from Git Bash.
#
# Usage: kggml-runtime-vulkan/native/scripts/build-host.sh [--stage-only]
#   --stage-only  do not build; only stage an existing build (the Windows flow)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NATIVE_DIR}/../.." && pwd)"

build=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-only) build=0 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$(uname -s)" in
  Linux) preset="linux-x64"; os="linux"; module="libggml-vulkan.so" ;;
  MINGW*|MSYS*|CYGWIN*) preset="windows-x64"; os="windows"; module="ggml-vulkan.dll" ;;
  *) echo "kggml-runtime-vulkan desktop module: Linux and Windows only (macOS uses Metal)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64|amd64) arch="x86_64" ;;
  *) echo "unsupported host architecture $(uname -m) (the desktop Vulkan add-on is built for x86_64)" >&2; exit 1 ;;
esac
build_dir="${REPO_ROOT}/build/native-vulkan-${preset}"

if [[ ${build} -eq 1 ]]; then
  jobs="$(nproc 2>/dev/null || echo 4)"
  extra=()
  if ! command -v glslc >/dev/null 2>&1 && [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
    ndk_glslc="$(find "${ANDROID_NDK_HOME}/shader-tools" -maxdepth 2 -name 'glslc*' -type f 2>/dev/null | head -n 1)"
    [[ -z "${ndk_glslc}" ]] || { echo "==> glslc from the NDK: ${ndk_glslc}"; extra+=("-DVulkan_GLSLC_EXECUTABLE=${ndk_glslc}"); }
  fi
  cmake -S "${NATIVE_DIR}" --preset "${preset}" ${extra[@]+"${extra[@]}"}
  cmake --build "${build_dir}" -j "${jobs}"
fi

built="${build_dir}/bin/${module}"
[[ -f "${built}" ]] || { echo "${built} was not built" >&2; exit 1; }
out="${REPO_ROOT}/build/native-vulkan-jvm/kggml/native-addons/${os}-${arch}/ggml-vulkan"
rm -rf "${out}"
mkdir -p "${out}"
# Staged under the name the JVM loader looks for (System.mapLibraryName("ggml-vulkan")); the module is dlopen()ed by
# path, and its dependency libggml-base.so.0 / ggml-base.dll is the runtime's, already loaded in the process.
cp "${built}" "${out}/${module}"
echo "==> ${module} staged under ${out}"
