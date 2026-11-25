#!/usr/bin/env bash
# Audits that libkggml_diffusion (stable-diffusion.cpp + Kggml C ABI) links the :kggml-runtime ggml dynamically and
# contains no copy of ggml.
#
#   host               Linux / macOS desktop build in build/native-diffusion (runtime in build/native).
#   android [abi ...]  Android build in build/native-diffusion-android-<abi> (runtime in build/native-android-<abi>);
#                      default ABIs: arm64-v8a x86_64.
#
# Checks: NEEDED / LC_LOAD_DYLIB entries on libggml, libggml-base and libkggml_runtime; no symbol that ggml exports
# is *defined* in libkggml_diffusion (any binding, so a static copy hidden by visibility is caught too); the
# stable-diffusion.cpp + kggml C ABI is exported; no whisper.cpp / llama.cpp / libmtmd symbol is defined; Android
# sonames are unversioned and test hooks are absent from release (Android) builds. Shared by CI and
# ./scripts/ci-local.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fail() { echo "FAIL: $*" >&2; exit 1; }
contains() { grep -Eq -- "$2" <<<"$1"; }

# ggml C API names (ggml_* / gguf_*) exported by the runtime's ggml libraries vs names defined anywhere in the
# diffusion library. (C++ std:: template instantiations legitimately appear in both and are ignored.)
check_no_duplicate_ggml() {
  local label="$1" sd_defs="$2" ggml_exports
  ggml_exports="$(grep -E '^_?(ggml|gguf)_' <<<"$3" || true)"
  local dups
  dups="$(comm -12 <(sort -u <<<"${sd_defs}") <(sort -u <<<"${ggml_exports}") || true)"
  if [[ -n "${dups}" ]]; then
    echo "${dups}" | head -n 20 >&2
    fail "${label}: libkggml_diffusion defines $(wc -l <<<"${dups}") symbol(s) that ggml exports (embedded ggml copy)"
  fi
  contains "${ggml_exports}" '^_?ggml_init$' || fail "${label}: could not read ggml exports (ggml_init missing)"
  if grep -Eq '^_?(whisper|llama|mtmd)_' <<<"${sd_defs}"; then
    fail "${label}: libkggml_diffusion embeds whisper.cpp / llama.cpp / libmtmd symbols"
  fi
}

REQUIRED_EXPORTS=(new_sd_ctx free_sd_ctx generate_image sd_set_progress_callback sd_get_system_info
  kggml_diffusion_context_create kggml_diffusion_context_free kggml_diffusion_generate kggml_diffusion_image_free
  kggml_diffusion_request_abort kggml_diffusion_clear_abort kggml_diffusion_last_error kggml_diffusion_upstream_commit)

elf_audit() {
  local label="$1" sd_so="$2" ggml_dir="$3" expect_versioned="$4"
  [[ -f "${sd_so}" ]] || fail "${label}: ${sd_so} not found"
  local dyn
  dyn="$(readelf -d "${sd_so}")"
  echo "${dyn}" | grep -E 'NEEDED|SONAME' || true
  for lib in libggml libggml-base libkggml_runtime; do
    contains "${dyn}" "NEEDED.*\[${lib}\.so" || fail "${label}: libkggml_diffusion does not link ${lib}.so dynamically"
  done
  if contains "${dyn}" 'NEEDED.*\[lib(whisper|llama|kggml_whisper|kggml_llama)'; then
    fail "${label}: libkggml_diffusion depends on another engine"
  fi
  if [[ "${expect_versioned}" == "no" ]]; then
    contains "${dyn}" 'SONAME.*\[libkggml_diffusion\.so\]' || fail "${label}: libkggml_diffusion.so has a versioned SONAME"
    if contains "${dyn}" 'NEEDED.*\.so\.[0-9]'; then fail "${label}: versioned NEEDED entry (Android loads by file name)"; fi
  fi
  local sd_defs ggml_exports exports
  sd_defs="$(nm --defined-only "${sd_so}" 2>/dev/null | awk 'NF >= 3 {print $3}')"
  [[ -n "${sd_defs}" ]] || sd_defs="$(nm -D --defined-only "${sd_so}" | awk '{print $3}')"
  ggml_exports="$(for f in "${ggml_dir}"/libggml*.so; do nm -D --defined-only "${f}" | awk '{print $3}'; done)"
  check_no_duplicate_ggml "${label}" "${sd_defs}" "${ggml_exports}"
  exports="$(nm -D --defined-only "${sd_so}" | awk '{print $3}')"
  for sym in "${REQUIRED_EXPORTS[@]}"; do
    contains "${exports}" "^${sym}$" || fail "${label}: ${sym} is not exported"
  done
  # The vendored zip / miniz helpers are internal (hidden), never part of the ABI.
  if contains "${exports}" '^(zip_|mz_|tdefl_|tinfl_)'; then fail "${label}: vendored zip / miniz symbols are exported"; fi
  if [[ "${expect_versioned}" == "no" ]] && contains "${exports}" '^kggml_diffusion_test_'; then
    fail "${label}: release build exports test hooks"
  fi
  echo "${label}: OK"
}

macho_audit() {
  local label="$1" sd_dylib="$2" ggml_dir="$3"
  [[ -f "${sd_dylib}" ]] || fail "${label}: ${sd_dylib} not found"
  local libs
  libs="$(otool -L "${sd_dylib}")"
  echo "${libs}"
  for lib in libggml libggml-base libkggml_runtime; do
    contains "${libs}" "${lib}\." || fail "${label}: libkggml_diffusion does not link ${lib} dynamically"
  done
  local sd_defs ggml_exports exports
  sd_defs="$(nm -U "${sd_dylib}" | awk 'NF >= 3 {print $3}')"
  ggml_exports="$(for f in "${ggml_dir}"/libggml*.dylib; do [[ -L "${f}" ]] || nm -gU "${f}" | awk '{print $3}'; done)"
  check_no_duplicate_ggml "${label}" "${sd_defs}" "${ggml_exports}"
  exports="$(nm -gU "${sd_dylib}" | awk '{print $3}')"
  for sym in "${REQUIRED_EXPORTS[@]}"; do
    contains "${exports}" "^_${sym}$" || fail "${label}: _${sym} is not exported"
  done
  if contains "${exports}" '^_(zip_|mz_|tdefl_|tinfl_)'; then fail "${label}: vendored zip / miniz symbols are exported"; fi
  echo "${label}: OK"
}

mode="${1:-host}"
shift || true
case "${mode}" in
  host)
    case "$(uname -s)" in
      Linux)  elf_audit "linux-x64" "${REPO_ROOT}/build/native-diffusion/lib/libkggml_diffusion.so" \
                "${REPO_ROOT}/build/native/lib" "yes" ;;
      Darwin) macho_audit "macos-arm64" "${REPO_ROOT}/build/native-diffusion/lib/libkggml_diffusion.dylib" \
                "${REPO_ROOT}/build/native/lib" ;;
      *) fail "unsupported host for diffusion linkage audit: $(uname -s)" ;;
    esac
    ;;
  android)
    abis=("$@")
    [[ ${#abis[@]} -gt 0 ]] || abis=(arm64-v8a x86_64)
    for abi in "${abis[@]}"; do
      elf_audit "android-${abi}" "${REPO_ROOT}/build/native-diffusion-android-${abi}/lib/libkggml_diffusion.so" \
        "${REPO_ROOT}/build/native-android-${abi}/lib" "no"
    done
    ;;
  *) fail "unknown mode '${mode}' (host | android [abi ...])" ;;
esac
echo "diffusion linkage audit: OK"
