#!/usr/bin/env bash
# Binary symbol leakage audit for :samples:whisper-only-minimal (AGENTS.md §2.1).
#
# Asserts that a shipped artifact contains
#   1. zero llama.cpp / libmtmd / stable-diffusion.cpp symbols, native libraries or Kotlin classes, and
#   2. single-symbol ggml linkage: every ggml_* / gguf_* symbol is defined by exactly one binary (the
#      :kggml-runtime ggml libraries), and every Kggml engine binary links those libraries dynamically.
#
# Usage:
#   scripts/audit-binary-symbols.sh apk <app.apk> [abi ...]
#       Android APK: extracts lib/<abi>/*.so (default: every ABI in the APK; apkanalyzer lists the files when
#       it is on PATH, unzip otherwise) and the dex files. ELF checks use readelf + nm (llvm-* fallbacks).
#   scripts/audit-binary-symbols.sh ios <path> [path ...]
#       iOS: each path is a .framework, an .xcframework (every slice is audited separately, the slices of all
#       given xcframeworks being matched by name) or a bare Mach-O file. Uses nm + otool -L
#       (llvm-nm / llvm-otool / llvm-objdump fallbacks, so it also runs on Linux).
#   scripts/audit-binary-symbols.sh elf <dir> [dir ...]
#       Every ELF *.so / *.so.<version> file (not symlinks) in the given directories is audited as one set
#       (host builds, tests).
#
# Exit status: 0 when the audit passes, 1 on any violation (the offending symbols are printed), 2 on usage
# errors. Shared by .github/workflows/ci.yml and ./scripts/ci-local.sh; tested by
# scripts/tests/test-audit-binary-symbols.sh.
set -euo pipefail
# Portable to the macOS system bash 3.2 (no mapfile / associative arrays; empty "${a[@]}" is guarded).

# Symbols / names that must never appear in a whisper-only artifact. ggml_* / gguf_* symbols are exempt (the
# ggml CPU backend exports e.g. ggml_cpu_has_llamafile); their uniqueness is checked separately. ggml's own
# llamafile sgemm kernels (GGML_LLAMAFILE) are not llama.cpp and are allowed as well.
FORBIDDEN_SYMBOL_RE='llama|mtmd|stable_?diffusion|diffusion_?engine|kggml_diffusion|(^|_)(new|free)_sd_ctx|(^|_)sd_(ctx|image|img_gen|vid_gen|get_system_info)|(^|_)(txt2img|img2img|upscaler_ctx)'
FORBIDDEN_LIB_RE='(llama|mtmd|stable-diffusion|stable_diffusion|diffusion)'
FORBIDDEN_CLASS_RE='com/donfuldauer/kggml/(llama|diffusion)/|com\.donfuldauer\.kggml\.(llama|diffusion)\.'
GGML_SYMBOL_RE='^_?(ggml|gguf)_'
ALLOWED_SYMBOL_RE='llamafile'

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
die() { echo "FAIL: $*" >&2; exit 1; }
usage() { sed -n '2,25p' "$0" >&2; exit 2; }

pick_tool() {
  local t
  for t in "$@"; do
    if command -v "${t}" >/dev/null 2>&1; then echo "${t}"; return 0; fi
  done
  return 1
}

# llvm-nm reads ELF and Mach-O; GNU nm (Linux) reads only ELF, Apple's nm (macOS) both. $NM overrides.
NM="${NM:-$(pick_tool llvm-nm nm)}" || die "nm not found"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# Defined symbol names of a binary (static + dynamic tables, any binding, so a hidden static copy is caught).
# Dies when nm cannot read the file at all (e.g. GNU nm on Mach-O), so an unreadable binary never passes.
defined_symbols() {
  local f="$1" static dynamic static_rc=0 dynamic_rc=0
  static="$("${NM}" --defined-only "${f}" 2>/dev/null)" || static_rc=$?
  dynamic="$("${NM}" -D --defined-only "${f}" 2>/dev/null)" || dynamic_rc=$?
  if [[ ${static_rc} -ne 0 && ${dynamic_rc} -ne 0 ]]; then
    die "${NM} cannot read $(basename "${f}") (${f}); a Mach-O audit needs llvm-nm or the macOS nm"
  fi
  printf '%s\n%s\n' "${static}" "${dynamic}" | awk 'NF >= 3 {print $NF}' | sort -u
}

# Prints forbidden (llama / mtmd / stable-diffusion) symbol names found in the symbol list on stdin.
forbidden_in() { grep -Ev "${GGML_SYMBOL_RE}" | grep -Eiv "${ALLOWED_SYMBOL_RE}" | grep -Ei "${FORBIDDEN_SYMBOL_RE}" || true; }

# Checks one set of binaries that are loaded together (one Android ABI / one iOS slice).
#   $1 label, $2 file listing the binaries (one path per line), $3 linkage dump command (readelf | otool).
audit_set() {
  local label="$1" list="$2" mode="$3"
  local f base defs exported bad
  local -a bins=()
  local line
  while IFS= read -r line; do
    if [[ -n "${line}" ]]; then bins+=("${line}"); fi
  done <"${list}"
  [[ ${#bins[@]} -gt 0 ]] || { fail "${label}: no native binaries found"; return; }

  local ggml_owners="" have_ggml=0 have_whisper=0 tmp_exports="${WORK}/ggml-defs.tsv"
  : >"${tmp_exports}"
  for f in "${bins[@]}"; do
    base="$(basename "${f}")"
    if grep -Eqi "${FORBIDDEN_LIB_RE}" <<<"${base}"; then fail "${label}: forbidden native library ${base}"; fi
    defs="$(defined_symbols "${f}")"
    bad="$(forbidden_in <<<"${defs}")"
    if [[ -n "${bad}" ]]; then
      head -n 20 <<<"${bad}" >&2
      fail "${label}: ${base} defines $(wc -l <<<"${bad}") llama / mtmd / stable-diffusion symbol(s)"
    fi
    grep -Eq '^_?whisper_full_with_state$' <<<"${defs}" && have_whisper=1
    # Every ggml_* / gguf_* definition, tagged with its owner, for the single-definition check.
    grep -E "${GGML_SYMBOL_RE}" <<<"${defs}" | sed "s|\$|	${base}|" >>"${tmp_exports}" || true
    if grep -Eq '^_?ggml_init$' <<<"${defs}"; then ggml_owners+="${base} "; have_ggml=1; fi
  done

  # Single-symbol ggml linkage: each ggml/gguf symbol name is defined by exactly one binary.
  local dups
  dups="$(sort -u "${tmp_exports}" | awk -F'\t' '{n[$1]++; o[$1]=o[$1] " " $2} END {for (s in n) if (n[s] > 1) print s ":" o[s]}' | sort)"
  if [[ -n "${dups}" ]]; then
    head -n 20 <<<"${dups}" >&2
    fail "${label}: $(wc -l <<<"${dups}") ggml symbol(s) defined in more than one binary (duplicate ggml copy)"
  fi
  [[ ${have_ggml} -eq 1 ]] || fail "${label}: no binary defines ggml_init (ggml runtime missing)"
  [[ ${have_whisper} -eq 1 ]] || fail "${label}: no binary defines whisper_full_with_state (whisper engine missing)"

  # Engine binaries (anything that is not a ggml library itself) must load ggml dynamically.
  for f in "${bins[@]}"; do
    base="$(basename "${f}")"
    case "${base}" in
      libkggml_whisper.so|libkggml_whisper.so.*|libkggml_whisper*.dylib|KggmlWhisper)
        local deps
        deps="$(linked_libraries "${f}" "${mode}")"
        grep -Eq '(^|/)(lib)?ggml(\.|$)|GgmlRuntime' <<<"${deps}" \
          || fail "${label}: ${base} does not link the :kggml-runtime ggml library dynamically" ;;
    esac
  done
  echo "${label}: $(echo "${ggml_owners}" | xargs) own ggml; ${#bins[@]} binaries checked"
}

linked_libraries() {
  local f="$1" mode="$2" tool
  if [[ "${mode}" == "elf" ]]; then
    tool="$(pick_tool readelf llvm-readelf)" || die "readelf not found"
    "${tool}" -d "${f}" | sed -n 's/.*NEEDED.*\[\(.*\)\].*/\1/p'
  else
    if tool="$(pick_tool otool llvm-otool)"; then
      "${tool}" -L "${f}" | tail -n +2 | awk '{print $1}'
    else
      tool="$(pick_tool llvm-objdump)" || die "otool / llvm-objdump not found"
      "${tool}" --macho --dylibs-used "${f}" | tail -n +2 | awk '{print $1}'
    fi
  fi
}

is_elf() { [[ "$(head -c 4 "$1" | od -An -c | tr -d ' ')" == '177ELF' ]]; }

audit_apk() {
  local apk="$1"; shift
  [[ -f "${apk}" ]] || die "APK not found: ${apk}"
  local work="${WORK}"

  local entries
  if command -v apkanalyzer >/dev/null 2>&1 && apkanalyzer files list "${apk}" >"${work}/files.txt" 2>/dev/null; then
    entries="$(sed 's|^/||' "${work}/files.txt")"
  else
    entries="$(unzip -Z1 "${apk}")"
  fi
  local -a abis=()
  if [[ $# -gt 0 ]]; then abis=("$@"); fi
  if [[ ${#abis[@]} -eq 0 ]]; then
    local abi_line
    while IFS= read -r abi_line; do
      if [[ -n "${abi_line}" ]]; then abis+=("${abi_line}"); fi
    done < <(sed -n 's|^lib/\([^/]*\)/[^/]*\.so$|\1|p' <<<"${entries}" | sort -u)
  fi
  [[ ${#abis[@]} -gt 0 ]] || die "${apk}: contains no native libraries (build with -Pkggml.bundleNative=true)"

  local lib
  while read -r lib; do
    [[ -n "${lib}" ]] || continue
    if grep -Eqi "${FORBIDDEN_LIB_RE}" <<<"$(basename "${lib}")"; then fail "apk: forbidden native library ${lib}"; fi
  done < <(grep -E '^lib/.*\.so$' <<<"${entries}" || true)

  unzip -q -o "${apk}" 'lib/*' 'classes*.dex' -d "${work}" 2>/dev/null || unzip -q -o "${apk}" 'lib/*' -d "${work}"
  local abi
  for abi in "${abis[@]}"; do
    [[ -d "${work}/lib/${abi}" ]] || { fail "apk: no lib/${abi}/ in ${apk}"; continue; }
    find "${work}/lib/${abi}" -maxdepth 1 -name '*.so' | sort >"${work}/${abi}.list"
    audit_set "apk/${abi}" "${work}/${abi}.list" elf
  done

  # Kotlin/Java code: no llama / diffusion engine classes in the dex files (and the whisper engine is present,
  # so the scan is not vacuous).
  local dex bad have_whisper_classes=0
  for dex in "${work}"/classes*.dex; do
    [[ -f "${dex}" ]] || continue
    LC_ALL=C grep -aq 'Lcom/donfuldauer/kggml/whisper/' "${dex}" && have_whisper_classes=1
    bad="$(LC_ALL=C grep -aoE "${FORBIDDEN_CLASS_RE}[A-Za-z0-9_$/]*" "${dex}" | sort -u || true)"
    if [[ -n "${bad}" ]]; then
      head -n 20 <<<"${bad}" >&2
      fail "apk: $(basename "${dex}") contains Kggml llama / diffusion classes"
    fi
  done
  [[ ${have_whisper_classes} -eq 1 ]] || fail "apk: no dex file contains com.donfuldauer.kggml.whisper classes"
}

# Prints the Mach-O binaries for one iOS path: "<slice>\t<binary>" lines.
ios_binaries() {
  local path="$1" slice dir name
  path="${path%/}"
  case "${path}" in
    *.xcframework)
      for dir in "${path}"/*/; do
        slice="$(basename "${dir}")"
        for fw in "${dir}"*.framework; do
          [[ -d "${fw}" ]] || continue
          name="$(basename "${fw}" .framework)"
          printf '%s\t%s\n' "${slice}" "${fw}/${name}"
        done
        for lib in "${dir}"*.a "${dir}"*.dylib; do
          if [[ -f "${lib}" ]]; then printf '%s\t%s\n' "${slice}" "${lib}"; fi
        done
      done ;;
    *.framework)
      name="$(basename "${path}" .framework)"
      printf '%s\t%s\n' "all" "${path}/${name}" ;;
    *)
      [[ -f "${path}" ]] || die "not found: ${path}"
      printf '%s\t%s\n' "all" "${path}" ;;
  esac
}

audit_ios() {
  [[ $# -gt 0 ]] || usage
  local work="${WORK}"
  local p
  for p in "$@"; do
    [[ -e "${p}" ]] || die "not found: ${p}"
    ios_binaries "${p}"
  done >"${work}/all.tsv"
  # "all" entries (single frameworks / binaries) belong to every slice.
  local -a slices=()
  local slice_line
  while IFS= read -r slice_line; do
    if [[ -n "${slice_line}" ]]; then slices+=("${slice_line}"); fi
  done < <(awk -F'\t' '$1 != "all" {print $1}' "${work}/all.tsv" | sort -u)
  [[ ${#slices[@]} -gt 0 ]] || slices=(all)
  local slice
  for slice in "${slices[@]}"; do
    awk -F'\t' -v s="${slice}" '$1 == s || $1 == "all" {print $2}' "${work}/all.tsv" >"${work}/${slice}.list"
    audit_set "ios/${slice}" "${work}/${slice}.list" macho
  done
}

audit_elf_dirs() {
  [[ $# -gt 0 ]] || usage
  local list="${WORK}/elf.list"
  local d f
  for d in "$@"; do
    [[ -d "${d}" ]] || die "not a directory: ${d}"
    for f in "${d}"/*.so "${d}"/*.so.*; do
      [[ -f "${f}" && ! -L "${f}" ]] || continue
      if is_elf "${f}"; then echo "${f}"; fi
    done
  done | sort >"${list}"
  audit_set "elf" "${list}" elf
}

mode="${1:-}"
[[ -n "${mode}" ]] || usage
shift
case "${mode}" in
  apk) [[ $# -ge 1 ]] || usage; audit_apk "$@" ;;
  ios) audit_ios "$@" ;;
  elf) audit_elf_dirs "$@" ;;
  -h|--help) usage ;;
  *) echo "unknown mode '${mode}'" >&2; usage ;;
esac

if [[ ${failures} -gt 0 ]]; then
  echo "binary symbol audit: FAILED (${failures} violation(s))" >&2
  exit 1
fi
echo "binary symbol audit: OK (no llama / stable-diffusion symbols; single ggml definition)"
