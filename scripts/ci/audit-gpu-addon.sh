#!/usr/bin/env bash
# Audits an Android GPU add-on module, kggml-runtime-vulkan (libggml-vulkan.so) or kggml-runtime-opencl
# (libggml-opencl.so). Per ABI the module must
#   - be a 16 KB-aligned module with the add-on's file name (jniLibs are packaged by file name; no versioned SONAME),
#   - export the ggml backend module entry point ggml_backend_init (GGML_BACKEND_DL),
#   - link ggml only dynamically through libggml-base.so (resolved to :kggml-runtime's copy at run time) and need
#     nothing beyond the device's API loader (libvulkan.so / the vendor's libOpenCL.so) and the libraries
#     :kggml-runtime already ships (no second ggml, no CPU backend, no libggml.so registry),
#   - be compiled with the runtime's GGML_MAX_NAME (kggml-runtime/native/cmake/kggml-ggml-abi.cmake), checked in the
#     build's compile_commands.json: struct ggml_tensor must have the runtime's layout.
# Expects kggml-runtime-<kind>/native/scripts/build-android.sh to have staged build/native-android-<kind>-jniLibs/<abi>.
# Shared by CI (.github/workflows/ci.yml Ubuntu job) and ./scripts/ci-local.sh.
#
# Usage: scripts/ci/audit-gpu-addon.sh <vulkan|opencl> [abi ...]   (default ABIs: arm64-v8a x86_64)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/ci/python.sh
kggml_require_python 3.9
max_name="$(sed -n 's/^set(KGGML_GGML_MAX_NAME \([0-9]*\)).*/\1/p' kggml-runtime/native/cmake/kggml-ggml-abi.cmake)"
[[ -n "${max_name}" ]] || { echo "FAIL: KGGML_GGML_MAX_NAME not found in kggml-runtime/native/cmake/kggml-ggml-abi.cmake" >&2; exit 1; }
readelf_bin="$(command -v readelf || command -v llvm-readelf || true)"
[[ -n "${readelf_bin}" ]] || { echo "FAIL: readelf not found" >&2; exit 1; }

kind="${1:-}"
case "${kind}" in
  vulkan) api_loader=libvulkan.so ;;
  opencl) api_loader=libOpenCL.so ;;
  *) echo "usage: $0 <vulkan|opencl> [abi ...]" >&2; exit 2 ;;
esac
shift
module_name="libggml-${kind}.so"
abis=("$@")
[[ ${#abis[@]} -gt 0 ]] || abis=(arm64-v8a x86_64)
allowed_needed=(libggml-base.so "${api_loader}" libc++_shared.so libc.so libm.so libdl.so liblog.so)
problems=0
fail() { echo "FAIL: $*" >&2; problems=$((problems + 1)); }

for abi in "${abis[@]}"; do
  so="build/native-android-${kind}-jniLibs/${abi}/${module_name}"
  if [[ ! -f "${so}" ]]; then fail "${abi}: ${so} missing"; continue; fi
  dynamic="$("${readelf_bin}" -d "${so}")"
  needed="$(sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p' <<<"${dynamic}")"
  # A CMake MODULE has no SONAME (it is dlopen()ed by path, never linked against); any SONAME must be unversioned.
  if grep -q 'SONAME' <<<"${dynamic}" && ! grep -qF "[${module_name}]" <<<"$(grep SONAME <<<"${dynamic}")"; then
    fail "${abi}: SONAME is not ${module_name}"
  fi
  grep -qx 'libggml-base.so' <<<"${needed}" || fail "${abi}: does not link libggml-base.so (ggml must be shared with :kggml-runtime)"
  grep -qx "${api_loader}" <<<"${needed}" || fail "${abi}: does not link the device's ${api_loader}"
  while IFS= read -r dep; do
    [[ -z "${dep}" ]] && continue
    printf '%s\n' "${allowed_needed[@]}" | grep -qx "${dep}" || fail "${abi}: unexpected dependency ${dep}"
  done <<<"${needed}"
  symbols="$("${readelf_bin}" -Ws --dyn-syms "${so}")"
  grep -Eq ' FUNC +GLOBAL +DEFAULT +[0-9]+ ggml_backend_init$' <<<"${symbols}" || fail "${abi}: ggml_backend_init is not exported"
  # A defined ggml core / CPU-backend symbol would mean a second ggml inside the module.
  if grep -Eq ' (FUNC|OBJECT) +GLOBAL +DEFAULT +[0-9]+ (ggml_init|ggml_backend_cpu_init|ggml_backend_reg_count)$' <<<"${symbols}"; then
    fail "${abi}: defines ggml core / CPU backend / registry symbols (must come from :kggml-runtime)"
  fi
  "${readelf_bin}" -lW "${so}" | awk '$1 == "LOAD" { if ($NF != "0x4000") bad = 1 } END { exit bad }' \
    || fail "${abi}: LOAD segments are not 16 KB aligned"
  commands="build/native-${kind}-android-${abi}/compile_commands.json"
  if [[ ! -f "${commands}" ]]; then
    fail "${abi}: ${commands} missing (the preset exports compile commands)"
  elif ! abi_report="$(kggml_python - "${commands}" "${max_name}" <<'PY'
import json, sys
commands, want = sys.argv[1], sys.argv[2]
ggml = [c for c in json.load(open(commands)) if "/ggml/src/" in c["file"].replace("\\", "/")]
bad = [c["file"] for c in ggml if f"-DGGML_MAX_NAME={want}" not in c.get("command", " ".join(c.get("arguments", [])))]
if not ggml or bad:
    print(f"{len(bad)} of {len(ggml)} ggml sources lack -DGGML_MAX_NAME={want}: {[c.rsplit('/', 1)[-1] for c in bad[:3]]}")
    sys.exit(1)
print(f"{len(ggml)} ggml sources built with GGML_MAX_NAME={want}")
PY
  )"; then
    fail "${abi}: ${abi_report}"
  fi
  echo "${abi}: ${module_name} OK (needs: $(tr '\n' ' ' <<<"${needed}"); ${abi_report:-})"
done

[[ ${problems} -eq 0 ]] || exit 1
echo "kggml-runtime-${kind} audit: OK"
