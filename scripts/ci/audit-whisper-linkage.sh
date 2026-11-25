#!/usr/bin/env bash
# Audits that libkggml_whisper links the :kggml-runtime ggml dynamically and contains no copy of ggml.
#
#   host               Linux / macOS desktop build in build/native-whisper (runtime in build/native).
#   android [abi ...]  Android build in build/native-whisper-android-<abi> (runtime in build/native-android-<abi>);
#                      default ABIs: arm64-v8a x86_64.
#
# Checks: NEEDED / LC_LOAD_DYLIB entries on libggml, libggml-base and libkggml_runtime; no symbol that ggml
# exports is *defined* in libkggml_whisper (any binding, so a static copy hidden by visibility is caught too);
# the whisper.cpp + kggml C ABI is exported; Android sonames are unversioned and test hooks are absent from
# release (Android) builds. Shared by CI and ./scripts/ci-local.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fail() { echo "FAIL: $*" >&2; exit 1; }
contains() { grep -Eq -- "$2" <<<"$1"; }

# ggml C API names (ggml_* / gguf_*) exported by the runtime's ggml libraries vs names defined anywhere in the
# whisper library. (C++ std:: template instantiations legitimately appear in both and are ignored.)
check_no_duplicate_ggml() {
  local label="$1" whisper_defs="$2" ggml_exports
  ggml_exports="$(grep -E '^_?(ggml|gguf)_' <<<"$3" || true)"
  local dups
  dups="$(comm -12 <(sort -u <<<"${whisper_defs}") <(sort -u <<<"${ggml_exports}") || true)"
  if [[ -n "${dups}" ]]; then
    echo "${dups}" | head -n 20 >&2
    fail "${label}: libkggml_whisper defines $(wc -l <<<"${dups}") symbol(s) that ggml exports (embedded ggml copy)"
  fi
  contains "${ggml_exports}" '^_?ggml_init$' || fail "${label}: could not read ggml exports (ggml_init missing)"
}

elf_audit() {
  local label="$1" whisper_so="$2" ggml_dir="$3" expect_versioned="$4"
  [[ -f "${whisper_so}" ]] || fail "${label}: ${whisper_so} not found"
  local dyn
  dyn="$(readelf -d "${whisper_so}")"
  echo "${dyn}" | grep -E 'NEEDED|SONAME' || true
  for lib in libggml libggml-base libkggml_runtime; do
    contains "${dyn}" "NEEDED.*\[${lib}\.so" || fail "${label}: libkggml_whisper does not link ${lib}.so dynamically"
  done
  if [[ "${expect_versioned}" == "no" ]]; then
    contains "${dyn}" 'SONAME.*\[libkggml_whisper\.so\]' || fail "${label}: libkggml_whisper.so has a versioned SONAME"
    if contains "${dyn}" 'NEEDED.*\.so\.[0-9]'; then fail "${label}: versioned NEEDED entry (Android loads by file name)"; fi
  fi
  local whisper_defs ggml_exports exports
  whisper_defs="$(nm --defined-only "${whisper_so}" 2>/dev/null | awk 'NF >= 3 {print $3}')"
  [[ -n "${whisper_defs}" ]] || whisper_defs="$(nm -D --defined-only "${whisper_so}" | awk '{print $3}')"
  ggml_exports="$(for f in "${ggml_dir}"/libggml*.so; do nm -D --defined-only "${f}" | awk '{print $3}'; done)"
  check_no_duplicate_ggml "${label}" "${whisper_defs}" "${ggml_exports}"
  exports="$(nm -D --defined-only "${whisper_so}" | awk '{print $3}')"
  for sym in whisper_full_with_state whisper_init_from_file_with_params_no_state kggml_whisper_transcribe \
             kggml_whisper_request_abort kggml_whisper_pcm_to_whisper; do
    contains "${exports}" "^${sym}$" || fail "${label}: ${sym} is not exported"
  done
  if [[ "${expect_versioned}" == "no" ]] && contains "${exports}" '^kggml_whisper_test_'; then
    fail "${label}: release build exports test hooks"
  fi
  jni_elf_audit "${label}" "$(dirname "${whisper_so}")/libkggml_whisper_jni.so" "${ggml_exports}"
  echo "${label}: OK"
}

# The WhisperEngine JNI bridge must link libkggml_whisper dynamically, embed neither ggml nor whisper.cpp, and
# export only Java_* entry points.
jni_elf_audit() {
  local label="$1" jni_so="$2" ggml_exports="$3"
  [[ -f "${jni_so}" ]] || fail "${label}: ${jni_so} not found (KggmlWhisper JNI bridge)"
  local dyn defs exports
  dyn="$(readelf -d "${jni_so}")"
  contains "${dyn}" 'NEEDED.*\[libkggml_whisper\.so' || fail "${label}: libkggml_whisper_jni does not link libkggml_whisper.so"
  defs="$(nm --defined-only "${jni_so}" 2>/dev/null | awk 'NF >= 3 {print $3}')"
  check_no_duplicate_ggml "${label} (jni)" "${defs}" "${ggml_exports}"
  if grep -Eq '^_?whisper_(full|init)' <<<"${defs}"; then fail "${label}: libkggml_whisper_jni embeds whisper.cpp"; fi
  exports="$(nm -D --defined-only "${jni_so}" | awk '{print $3}')"
  contains "${exports}" '^Java_com_donfuldauer_kggml_whisper_KggmlWhisperJni_nativeTranscribe$' \
    || fail "${label}: libkggml_whisper_jni does not export nativeTranscribe"
  if grep -Ev '^(Java_|JNI_On|_init$|_fini$|_edata$|_end$|__bss_start)' <<<"${exports}" | grep -q .; then
    grep -Ev '^(Java_|JNI_On|_init$|_fini$|_edata$|_end$|__bss_start)' <<<"${exports}" | head -n 10 >&2
    fail "${label}: libkggml_whisper_jni exports non-JNI symbols"
  fi
  echo "${label}: libkggml_whisper_jni OK"
}

macho_audit() {
  local label="$1" whisper_dylib="$2" ggml_dir="$3"
  [[ -f "${whisper_dylib}" ]] || fail "${label}: ${whisper_dylib} not found"
  local libs
  libs="$(otool -L "${whisper_dylib}")"
  echo "${libs}"
  for lib in libggml libggml-base libkggml_runtime; do
    contains "${libs}" "${lib}\." || fail "${label}: libkggml_whisper does not link ${lib} dynamically"
  done
  local whisper_defs ggml_exports exports
  whisper_defs="$(nm -U "${whisper_dylib}" | awk 'NF >= 3 {print $3}')"
  ggml_exports="$(for f in "${ggml_dir}"/libggml*.dylib; do [[ -L "${f}" ]] || nm -gU "${f}" | awk '{print $3}'; done)"
  check_no_duplicate_ggml "${label}" "${whisper_defs}" "${ggml_exports}"
  exports="$(nm -gU "${whisper_dylib}" | awk '{print $3}')"
  for sym in _whisper_full_with_state _kggml_whisper_transcribe _kggml_whisper_request_abort; do
    contains "${exports}" "^${sym}$" || fail "${label}: ${sym} is not exported"
  done
  echo "${label}: OK"
}

mode="${1:-host}"
shift || true
case "${mode}" in
  host)
    case "$(uname -s)" in
      Linux)  elf_audit "linux-x64" "${REPO_ROOT}/build/native-whisper/lib/libkggml_whisper.so" \
                "${REPO_ROOT}/build/native/lib" "yes" ;;
      Darwin) macho_audit "macos-arm64" "${REPO_ROOT}/build/native-whisper/lib/libkggml_whisper.dylib" \
                "${REPO_ROOT}/build/native/lib" ;;
      *) fail "unsupported host for whisper linkage audit: $(uname -s)" ;;
    esac
    ;;
  android)
    abis=("$@")
    [[ ${#abis[@]} -gt 0 ]] || abis=(arm64-v8a x86_64)
    for abi in "${abis[@]}"; do
      elf_audit "android-${abi}" "${REPO_ROOT}/build/native-whisper-android-${abi}/lib/libkggml_whisper.so" \
        "${REPO_ROOT}/build/native-android-${abi}/lib" "no"
    done
    ;;
  *) fail "unknown mode '${mode}' (host | android [abi ...])" ;;
esac
echo "whisper linkage audit: OK"
