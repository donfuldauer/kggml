#!/usr/bin/env bash
# Audits that libkggml_llama (llama.cpp + libmtmd + Kggml C ABI) links the :kggml-runtime ggml dynamically and
# contains no copy of ggml.
#
#   host               Linux / macOS desktop build in build/native-llama (runtime in build/native).
#   android [abi ...]  Android build in build/native-llama-android-<abi> (runtime in build/native-android-<abi>);
#                      default ABIs: arm64-v8a x86_64.
#
# Checks: NEEDED / LC_LOAD_DYLIB entries on libggml, libggml-base and libkggml_runtime; no symbol that ggml
# exports is *defined* in libkggml_llama (any binding, so a static copy hidden by visibility is caught too);
# the llama.cpp + libmtmd + kggml C ABI is exported; no whisper.cpp symbol is defined; Android sonames are
# unversioned and test hooks are absent from release (Android) builds. Shared by CI and ./scripts/ci-local.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fail() { echo "FAIL: $*" >&2; exit 1; }
contains() { grep -Eq -- "$2" <<<"$1"; }

# ggml C API names (ggml_* / gguf_*) exported by the runtime's ggml libraries vs names defined anywhere in the
# llama library. (C++ std:: template instantiations legitimately appear in both and are ignored.)
check_no_duplicate_ggml() {
  local label="$1" llama_defs="$2" ggml_exports
  ggml_exports="$(grep -E '^_?(ggml|gguf)_' <<<"$3" || true)"
  local dups
  dups="$(comm -12 <(sort -u <<<"${llama_defs}") <(sort -u <<<"${ggml_exports}") || true)"
  if [[ -n "${dups}" ]]; then
    echo "${dups}" | head -n 20 >&2
    fail "${label}: libkggml_llama defines $(wc -l <<<"${dups}") symbol(s) that ggml exports (embedded ggml copy)"
  fi
  contains "${ggml_exports}" '^_?ggml_init$' || fail "${label}: could not read ggml exports (ggml_init missing)"
  if grep -Eq '^_?whisper_' <<<"${llama_defs}"; then fail "${label}: libkggml_llama embeds whisper.cpp symbols"; fi
}

REQUIRED_EXPORTS=(llama_decode llama_model_load_from_file llama_set_abort_callback llama_sampler_init_grammar
  llama_state_seq_get_data mtmd_init_from_file mtmd_tokenize mtmd_helper_bitmap_init_from_buf
  kggml_llama_model_load kggml_llama_decode kggml_llama_request_abort kggml_llama_state_save
  kggml_llama_sampler_create kggml_llama_get_embeddings kggml_llama_mtmd_eval)

elf_audit() {
  local label="$1" llama_so="$2" ggml_dir="$3" expect_versioned="$4"
  [[ -f "${llama_so}" ]] || fail "${label}: ${llama_so} not found"
  local dyn
  dyn="$(readelf -d "${llama_so}")"
  echo "${dyn}" | grep -E 'NEEDED|SONAME' || true
  for lib in libggml libggml-base libkggml_runtime; do
    contains "${dyn}" "NEEDED.*\[${lib}\.so" || fail "${label}: libkggml_llama does not link ${lib}.so dynamically"
  done
  if [[ "${expect_versioned}" == "no" ]]; then
    contains "${dyn}" 'SONAME.*\[libkggml_llama\.so\]' || fail "${label}: libkggml_llama.so has a versioned SONAME"
    if contains "${dyn}" 'NEEDED.*\.so\.[0-9]'; then fail "${label}: versioned NEEDED entry (Android loads by file name)"; fi
  fi
  local llama_defs ggml_exports exports
  llama_defs="$(nm --defined-only "${llama_so}" 2>/dev/null | awk 'NF >= 3 {print $3}')"
  [[ -n "${llama_defs}" ]] || llama_defs="$(nm -D --defined-only "${llama_so}" | awk '{print $3}')"
  ggml_exports="$(for f in "${ggml_dir}"/libggml*.so; do nm -D --defined-only "${f}" | awk '{print $3}'; done)"
  check_no_duplicate_ggml "${label}" "${llama_defs}" "${ggml_exports}"
  exports="$(nm -D --defined-only "${llama_so}" | awk '{print $3}')"
  for sym in "${REQUIRED_EXPORTS[@]}"; do
    contains "${exports}" "^${sym}$" || fail "${label}: ${sym} is not exported"
  done
  if [[ "${expect_versioned}" == "no" ]] && contains "${exports}" '^kggml_llama_test_'; then
    fail "${label}: release build exports test hooks"
  fi
  echo "${label}: OK"
}

macho_audit() {
  local label="$1" llama_dylib="$2" ggml_dir="$3"
  [[ -f "${llama_dylib}" ]] || fail "${label}: ${llama_dylib} not found"
  local libs
  libs="$(otool -L "${llama_dylib}")"
  echo "${libs}"
  for lib in libggml libggml-base libkggml_runtime; do
    contains "${libs}" "${lib}\." || fail "${label}: libkggml_llama does not link ${lib} dynamically"
  done
  local llama_defs ggml_exports exports
  llama_defs="$(nm -U "${llama_dylib}" | awk 'NF >= 3 {print $3}')"
  ggml_exports="$(for f in "${ggml_dir}"/libggml*.dylib; do [[ -L "${f}" ]] || nm -gU "${f}" | awk '{print $3}'; done)"
  check_no_duplicate_ggml "${label}" "${llama_defs}" "${ggml_exports}"
  exports="$(nm -gU "${llama_dylib}" | awk '{print $3}')"
  for sym in "${REQUIRED_EXPORTS[@]}"; do
    contains "${exports}" "^_${sym}$" || fail "${label}: _${sym} is not exported"
  done
  echo "${label}: OK"
}

mode="${1:-host}"
shift || true
case "${mode}" in
  host)
    case "$(uname -s)" in
      Linux)  elf_audit "linux-x64" "${REPO_ROOT}/build/native-llama/lib/libkggml_llama.so" \
                "${REPO_ROOT}/build/native/lib" "yes" ;;
      Darwin) macho_audit "macos-arm64" "${REPO_ROOT}/build/native-llama/lib/libkggml_llama.dylib" \
                "${REPO_ROOT}/build/native/lib" ;;
      *) fail "unsupported host for llama linkage audit: $(uname -s)" ;;
    esac
    ;;
  android)
    abis=("$@")
    [[ ${#abis[@]} -gt 0 ]] || abis=(arm64-v8a x86_64)
    for abi in "${abis[@]}"; do
      elf_audit "android-${abi}" "${REPO_ROOT}/build/native-llama-android-${abi}/lib/libkggml_llama.so" \
        "${REPO_ROOT}/build/native-android-${abi}/lib" "no"
    done
    ;;
  *) fail "unknown mode '${mode}' (host | android [abi ...])" ;;
esac
echo "llama linkage audit: OK"
