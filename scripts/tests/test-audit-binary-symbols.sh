#!/usr/bin/env bash
# Tests for scripts/audit-binary-symbols.sh with tiny synthetic binaries (no real ggml / whisper build needed).
#
# Builds fake ELF shared libraries (host cc; the NDK's clang on macOS) and, when clang can target iOS, fake Mach-O objects, packs them the
# way an APK / framework would, and asserts the audit passes for a clean whisper-only set and FAILS for:
# a forbidden llama / stable-diffusion / mtmd symbol, a hidden (local) llama symbol, a forbidden native library
# file, a duplicate ggml definition (statically embedded ggml), an engine library without a dynamic ggml link,
# Kotlin llama / diffusion classes in the dex, and a missing ggml runtime / whisper engine / whisper classes.
#
# Usage: scripts/tests/test-audit-binary-symbols.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT="${REPO_ROOT}/scripts/audit-binary-symbols.sh"
# The fake libraries must be ELF; the macOS host cc emits Mach-O, so use the NDK's clang there.
if [[ -z "${CC:-}" && "$(uname -s)" == Darwin && -n "${ANDROID_NDK_HOME:-}" ]]; then
  for ndk_cc in "${ANDROID_NDK_HOME}"/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android26-clang; do
    if [[ -x "${ndk_cc}" ]]; then CC="${ndk_cc}"; fi
  done
fi
CC="${CC:-cc}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

passed=0
failed=0
ok() { echo "  ok   $1"; passed=$((passed + 1)); }
ko() { echo "  FAIL $1" >&2; failed=$((failed + 1)); }

# expect_pass|expect_fail <name> <expected-output-regex or ''> <audit args...>
expect() {
  local want="$1" name="$2" pattern="$3"; shift 3
  local out rc=0
  out="$("${AUDIT}" "$@" 2>&1)" || rc=$?
  if [[ "${want}" == pass && ${rc} -eq 0 ]] || [[ "${want}" == fail && ${rc} -eq 1 ]]; then
    if [[ -z "${pattern}" ]] || grep -Eq -- "${pattern}" <<<"${out}"; then ok "${name}"; return; fi
    ko "${name}: output does not match /${pattern}/"
  else
    ko "${name}: expected ${want}, exit code ${rc}"
  fi
  sed 's/^/       | /' <<<"${out}" >&2
}

# so <dir> <name> <c source> [cc args...]: builds <dir>/<name> as a shared library.
so() {
  local dir="$1" name="$2" src="$3"; shift 3
  mkdir -p "${dir}"
  printf '%s\n' "${src}" >"${WORK}/src.c"
  "${CC}" -shared -fPIC -o "${dir}/${name}" "${WORK}/src.c" "$@"
}

GGML_SRC='int ggml_init(void){return 1;} int gguf_init_empty(void){return 2;} int ggml_cpu_has_llamafile(void){return 0;}'
WHISPER_SRC='extern int ggml_init(void); int whisper_full_with_state(void){return ggml_init();}'

# A clean whisper-only set: ggml runtime + whisper engine (dynamic) + JNI bridge.
clean_set() {
  local dir="$1"
  so "${dir}" libggml.so "${GGML_SRC}"
  so "${dir}" libkggml_whisper.so "${WHISPER_SRC}" -L"${dir}" -lggml
  so "${dir}" libkggml_whisper_jni.so 'int Java_com_donfuldauer_kggml_whisper_KggmlWhisperJni_nativeTranscribe(void){return 0;}'
}

# make_apk <out.apk> <dir with lib/<abi>/*.so> [dex text]
make_apk() {
  local apk="$1" root="$2" dex="${3:-Lcom/donfuldauer/kggml/whisper/WhisperEngine;}"
  printf 'dex\n035\0%s\0' "${dex}" >"${root}/classes.dex"
  (cd "${root}" && rm -f "${apk}" && zip -q -r "${apk}" lib classes.dex)
}

echo "== elf mode"
clean_set "${WORK}/elf-clean"
expect pass "clean whisper-only set passes" 'binary symbol audit: OK' elf "${WORK}/elf-clean"

cp -r "${WORK}/elf-clean" "${WORK}/elf-llama"
so "${WORK}/elf-llama" libextra.so 'int llama_decode(void){return 0;}'
expect fail "exported llama_ symbol fails" 'llama_decode' elf "${WORK}/elf-llama"

cp -r "${WORK}/elf-clean" "${WORK}/elf-hidden"
so "${WORK}/elf-hidden" libextra.so '__attribute__((visibility("hidden"))) int llama_model_load(void){return 0;} int f(void){return llama_model_load();}'
expect fail "hidden (non-exported) llama symbol fails" 'llama_model_load' elf "${WORK}/elf-hidden"

cp -r "${WORK}/elf-clean" "${WORK}/elf-sd"
so "${WORK}/elf-sd" libextra.so 'int new_sd_ctx(void){return 0;} int txt2img(void){return 0;}'
expect fail "stable-diffusion symbols fail" 'new_sd_ctx' elf "${WORK}/elf-sd"

cp -r "${WORK}/elf-clean" "${WORK}/elf-mtmd"
so "${WORK}/elf-mtmd" libextra.so 'int mtmd_init_from_file(void){return 0;}'
expect fail "libmtmd symbols fail" 'mtmd_init_from_file' elf "${WORK}/elf-mtmd"

cp -r "${WORK}/elf-clean" "${WORK}/elf-libname"
so "${WORK}/elf-libname" libllama.so 'int unrelated(void){return 0;}'
expect fail "forbidden library file name fails" 'forbidden native library libllama.so' elf "${WORK}/elf-libname"

cp -r "${WORK}/elf-clean" "${WORK}/elf-gpu-addon"
so "${WORK}/elf-gpu-addon" libggml-vulkan.so 'int unrelated_gpu(void){return 0;}'
expect fail "GPU backend add-on library fails" 'forbidden native library libggml-vulkan.so' elf "${WORK}/elf-gpu-addon"

mkdir -p "${WORK}/elf-dup"
so "${WORK}/elf-dup" libggml.so "${GGML_SRC}"
so "${WORK}/elf-dup" libkggml_whisper.so "${GGML_SRC/ggml_init/ggml_init_copy} int ggml_init(void){return 3;} int whisper_full_with_state(void){return 0;}" -L"${WORK}/elf-dup" -lggml
expect fail "duplicate (embedded) ggml definition fails" 'ggml_init: .*libggml.so|defined in more than one binary' elf "${WORK}/elf-dup"

mkdir -p "${WORK}/elf-static"
so "${WORK}/elf-static" libggml.so "${GGML_SRC}"
so "${WORK}/elf-static" libkggml_whisper.so 'int whisper_full_with_state(void){return 0;}'
expect fail "engine without dynamic ggml link fails" 'does not link the :kggml-runtime ggml library dynamically' elf "${WORK}/elf-static"

mkdir -p "${WORK}/elf-noggml"
so "${WORK}/elf-noggml" libkggml_whisper_jni.so 'int whisper_full_with_state(void){return 0;}'
expect fail "missing ggml runtime fails" 'ggml runtime missing' elf "${WORK}/elf-noggml"

mkdir -p "${WORK}/elf-nowhisper"
so "${WORK}/elf-nowhisper" libggml.so "${GGML_SRC}"
expect fail "missing whisper engine fails" 'whisper engine missing' elf "${WORK}/elf-nowhisper"

cp -r "${WORK}/elf-clean" "${WORK}/elf-llamafile"
so "${WORK}/elf-llamafile" libggml-cpu.so 'int llamafile_sgemm(void){return 0;}'
printf 'not an ELF file\n' >"${WORK}/elf-llamafile/libnotes.so.txt"
expect pass "ggml llamafile kernels and non-ELF *.so.* files are allowed" 'binary symbol audit: OK' elf "${WORK}/elf-llamafile"

echo "== apk mode"
for abi in arm64-v8a x86_64; do clean_set "${WORK}/apk-clean/lib/${abi}"; done
make_apk "${WORK}/clean.apk" "${WORK}/apk-clean"
expect pass "clean APK passes (all ABIs)" 'apk/arm64-v8a.*' apk "${WORK}/clean.apk"
expect pass "clean APK passes (one ABI)" 'apk/x86_64' apk "${WORK}/clean.apk" x86_64

cp -r "${WORK}/apk-clean" "${WORK}/apk-llama"
so "${WORK}/apk-llama/lib/x86_64" libkggml_llama.so 'int llama_decode(void){return 0;}'
make_apk "${WORK}/llama.apk" "${WORK}/apk-llama"
expect fail "APK with libkggml_llama.so fails" 'forbidden native library' apk "${WORK}/llama.apk"

make_apk "${WORK}/dex.apk" "${WORK}/apk-clean" 'Lcom/donfuldauer/kggml/llama/LlamaEngine;'
expect fail "APK with Kotlin llama classes fails" 'llama / diffusion classes' apk "${WORK}/dex.apk"

make_apk "${WORK}/dex-diff.apk" "${WORK}/apk-clean" 'Lcom/donfuldauer/kggml/diffusion/DiffusionEngine;'
expect fail "APK with Kotlin diffusion classes fails" 'llama / diffusion classes' apk "${WORK}/dex-diff.apk"

make_apk "${WORK}/nodex.apk" "${WORK}/apk-clean" 'Lcom/example/Other;'
expect fail "APK without whisper classes fails" 'no dex file contains' apk "${WORK}/nodex.apk"

mkdir -p "${WORK}/apk-nolib/lib"
make_apk "${WORK}/nolib.apk" "${WORK}/apk-nolib"
expect fail "APK without native libraries fails" 'contains no native libraries' apk "${WORK}/nolib.apk"

expect fail "missing ABI fails" 'no lib/armeabi-v7a/' apk "${WORK}/clean.apk" armeabi-v7a

echo "== ios mode (Mach-O objects)"
macho() {
  local out="$1" src="$2"
  mkdir -p "$(dirname "${out}")"
  printf '%s\n' "${src}" >"${WORK}/m.c"
  clang -target arm64-apple-ios14 -c "${WORK}/m.c" -o "${out}"
}
# The Mach-O cases need clang targeting iOS and an nm that reads Mach-O (llvm-nm, or Apple's nm on macOS).
macho_nm=""
if command -v llvm-nm >/dev/null 2>&1; then macho_nm=llvm-nm; elif [[ "$(uname -s)" == Darwin ]]; then macho_nm=nm; fi
if [[ -n "${NM:-}" ]]; then macho_nm="${NM}"; fi
if command -v clang >/dev/null 2>&1 && macho "${WORK}/probe.o" 'int f(void){return 0;}' 2>/dev/null \
   && [[ -n "${macho_nm}" ]] && "${macho_nm}" --defined-only "${WORK}/probe.o" >/dev/null 2>&1; then
  for slice in ios-arm64 ios-arm64-simulator; do
    macho "${WORK}/ios/GgmlRuntime.xcframework/${slice}/GgmlRuntime.framework/GgmlRuntime" "${GGML_SRC}"
    macho "${WORK}/ios/Whisper.xcframework/${slice}/Whisper.framework/Whisper" "${WHISPER_SRC}"
  done
  macho "${WORK}/ios/App.framework/App" 'extern int whisper_full_with_state(void); int kfun_main(void){return whisper_full_with_state();}'
  expect pass "clean iOS slices pass" 'ios/ios-arm64-simulator' ios \
    "${WORK}/ios/App.framework" "${WORK}/ios/GgmlRuntime.xcframework" "${WORK}/ios/Whisper.xcframework"

  macho "${WORK}/ios/LlamaApp.framework/LlamaApp" 'int llama_decode(void){return 0;}'
  expect fail "iOS framework with llama symbols fails" 'llama_decode' ios \
    "${WORK}/ios/LlamaApp.framework" "${WORK}/ios/GgmlRuntime.xcframework" "${WORK}/ios/Whisper.xcframework"

  macho "${WORK}/ios/DupApp.framework/DupApp" 'int ggml_init(void){return 0;}'
  expect fail "iOS framework embedding ggml fails" 'defined in more than one binary' ios \
    "${WORK}/ios/DupApp.framework" "${WORK}/ios/GgmlRuntime.xcframework" "${WORK}/ios/Whisper.xcframework"
else
  echo "  skip no clang targeting arm64-apple-ios or no Mach-O capable nm"
  if command -v clang >/dev/null 2>&1 && macho "${WORK}/probe.o" 'int f(void){return 0;}' 2>/dev/null; then
    # With an ELF-only nm the audit must refuse Mach-O input instead of passing it with no symbols.
    mkdir -p "${WORK}/ios-elfnm/App.framework" && cp "${WORK}/probe.o" "${WORK}/ios-elfnm/App.framework/App"
    if nm --defined-only "${WORK}/probe.o" >/dev/null 2>&1; then :; else
      NM=nm expect fail "ELF-only nm refuses Mach-O input" 'cannot read App' ios "${WORK}/ios-elfnm/App.framework"
    fi
  fi
fi

echo "== jars (JVM desktop)"
# jarof <jar> <entry...>: a jar (zip) holding empty files at the given entry paths.
jarof() {
  local jar="$1"; shift
  local stage; stage="$(mktemp -d "${WORK}/stage.XXXX")"
  local e
  for e in "$@"; do mkdir -p "${stage}/$(dirname "${e}")"; : >"${stage}/${e}"; done
  (cd "${stage}" && zip -q -r "${jar}" .)
}
mkdir -p "${WORK}/jars-clean" "${WORK}/jars-llama" "${WORK}/jars-lib" "${WORK}/jars-addon" "${WORK}/jars-nowhisper"
jarof "${WORK}/jars-clean/app.jar" com/donfuldauer/kggml/samples/whisper/MainKt.class
jarof "${WORK}/jars-clean/kggml-whisper-jvm.jar" com/donfuldauer/kggml/whisper/WhisperEngine.class
jarof "${WORK}/jars-clean/kggml-runtime-jvm.jar" com/donfuldauer/kggml/runtime/KggmlRuntime.class kggml/native/linux-x86_64/libggml.so
expect pass "clean whisper-only desktop classpath passes" 'OK' jars "${WORK}/jars-clean"
cp "${WORK}"/jars-clean/*.jar "${WORK}/jars-llama/"
jarof "${WORK}/jars-llama/kggml-llama-jvm.jar" com/donfuldauer/kggml/llama/LlamaEngine.class
expect fail "llama classes on the desktop classpath fail" 'kggml-llama-jvm.jar contains Kggml llama' jars "${WORK}/jars-llama"
cp "${WORK}"/jars-clean/*.jar "${WORK}/jars-lib/"
jarof "${WORK}/jars-lib/natives.jar" kggml/native/linux-x86_64/libkggml_diffusion.so
expect fail "a diffusion native library in a jar fails" 'forbidden native library .*libkggml_diffusion.so' jars "${WORK}/jars-lib"
cp "${WORK}"/jars-clean/*.jar "${WORK}/jars-addon/"
jarof "${WORK}/jars-addon/gpu.jar" kggml/native/linux-x86_64/libggml-vulkan.so
expect fail "a GPU add-on library in a jar fails" 'forbidden native library .*libggml-vulkan.so' jars "${WORK}/jars-addon"
jarof "${WORK}/jars-nowhisper/app.jar" com/donfuldauer/kggml/samples/whisper/MainKt.class
expect fail "a classpath without the whisper engine fails" 'no jar contains com.donfuldauer.kggml.whisper' jars "${WORK}/jars-nowhisper"

echo "== usage"
expect fail "unknown file fails" 'not found' ios "${WORK}/does-not-exist"
rc=0; "${AUDIT}" bogus >/dev/null 2>&1 || rc=$?
if [[ ${rc} -eq 2 ]]; then ok "unknown mode exits 2"; else ko "unknown mode exits 2 (got ${rc})"; fi

echo
echo "audit-binary-symbols tests: ${passed} passed, ${failed} failed"
[[ ${failed} -eq 0 ]]
