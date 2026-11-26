#!/usr/bin/env bash
# Audits the Android GPU add-on module (kggml-runtime-vulkan): libggml-vulkan.so per ABI must
#   - be a 16 KB-aligned module named libggml-vulkan.so (jniLibs are packaged by file name; no versioned SONAME),
#   - export the ggml backend module entry point ggml_backend_init (GGML_BACKEND_DL),
#   - link ggml only dynamically through libggml-base.so (resolved to :kggml-runtime's copy at run time) and need
#     nothing beyond the system Vulkan loader and the libraries :kggml-runtime already ships (no second ggml, no
#     CPU backend, no libggml.so registry).
# Expects kggml-runtime-vulkan/native/scripts/build-android.sh to have staged build/native-android-vulkan-jniLibs/<abi>.
# Shared by CI (.github/workflows/ci.yml Ubuntu job) and local runs.
#
# Usage: scripts/ci/audit-vulkan-addon.sh [abi ...]   (default: arm64-v8a x86_64)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
readelf_bin="$(command -v readelf || command -v llvm-readelf || true)"
[[ -n "${readelf_bin}" ]] || { echo "FAIL: readelf not found" >&2; exit 1; }

abis=("$@")
[[ ${#abis[@]} -gt 0 ]] || abis=(arm64-v8a x86_64)
allowed_needed=(libggml-base.so libvulkan.so libc++_shared.so libc.so libm.so libdl.so liblog.so)
problems=0
fail() { echo "FAIL: $*" >&2; problems=$((problems + 1)); }

for abi in "${abis[@]}"; do
  so="build/native-android-vulkan-jniLibs/${abi}/libggml-vulkan.so"
  if [[ ! -f "${so}" ]]; then fail "${abi}: ${so} missing"; continue; fi
  dynamic="$("${readelf_bin}" -d "${so}")"
  needed="$(sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p' <<<"${dynamic}")"
  # A CMake MODULE has no SONAME (it is dlopen()ed by path, never linked against); any SONAME must be unversioned.
  if grep -q 'SONAME' <<<"${dynamic}" && ! grep -q 'SONAME.*\[libggml-vulkan\.so\]' <<<"${dynamic}"; then
    fail "${abi}: SONAME is not libggml-vulkan.so"
  fi
  grep -qx 'libggml-base.so' <<<"${needed}" || fail "${abi}: does not link libggml-base.so (ggml must be shared with :kggml-runtime)"
  grep -qx 'libvulkan.so' <<<"${needed}" || fail "${abi}: does not link the system libvulkan.so"
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
  echo "${abi}: libggml-vulkan.so OK (needs: $(tr '\n' ' ' <<<"${needed}"))"
done

[[ ${problems} -eq 0 ]] || exit 1
echo "kggml-runtime-vulkan audit: OK"
