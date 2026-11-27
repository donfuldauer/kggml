#!/usr/bin/env bash
# Audits dynamic ggml linkage of the host desktop build (Linux: readelf/nm, macOS: otool/nm).
# Expects ./kggml-runtime/native/scripts/build-host.sh to have produced build/native/lib.
# Shared by CI (.github/workflows/ci.yml) and ./scripts/ci-local.sh.
#
# Usage: scripts/ci/audit-host-linkage.sh [lib-dir]   (default: build/native/lib)
# A cross-compiled Linux tree (e.g. build/native-linux-aarch64/lib) can pass its binutils as READELF / NM
# (READELF=aarch64-linux-gnu-readelf NM=aarch64-linux-gnu-nm); the host's are used by default.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
lib="${1:-${REPO_ROOT}/build/native/lib}"

fail() { echo "FAIL: $*" >&2; exit 1; }
# Outputs are captured first so `grep -q` exiting early can never SIGPIPE a producer under pipefail.
contains() { grep -Eq -- "$2" <<<"$1"; }
READELF="${READELF:-readelf}"
NM="${NM:-nm}"

case "$(uname -s)" in
  Linux)
    rt_dyn="$("${READELF}" -d "${lib}/libkggml_runtime.so")"
    echo "${rt_dyn}"
    contains "${rt_dyn}" 'NEEDED.*\[libggml\.so' || fail "libkggml_runtime.so does not link libggml.so"
    contains "${rt_dyn}" 'NEEDED.*\[libggml-base\.so' || fail "libkggml_runtime.so does not link libggml-base.so"
    # kggml_runtime must import ggml, never define (embed) it.
    for so in libkggml_runtime.so libkggml_runtime_jni.so; do
      defs="$("${NM}" -D --defined-only "${lib}/${so}")"
      if contains "${defs}" ' (ggml|gguf)_'; then
        grep -E ' (ggml|gguf)_' <<<"${defs}" >&2
        fail "${so} embeds ggml symbols"
      fi
    done
    jni_dyn="$("${READELF}" -d "${lib}/libkggml_runtime_jni.so")"
    contains "${jni_dyn}" 'NEEDED.*\[libkggml_runtime\.so' || fail "libkggml_runtime_jni.so does not link libkggml_runtime.so"
    # The JNI bridge exports only Java_* entry points.
    stray="$("${NM}" -D --defined-only "${lib}/libkggml_runtime_jni.so" | awk '$2 ~ /[TDB]/ {print $3}' \
      | grep -v '^Java_com_donfuldauer_kggml_runtime_KggmlRuntimeJni_' || true)"
    [[ -z "${stray}" ]] || { echo "${stray}" >&2; fail "libkggml_runtime_jni.so exports non-JNI symbols"; }
    contains "$("${NM}" -D --defined-only "${lib}/libggml-base.so")" ' ggml_init$' \
      || fail "libggml-base.so does not export ggml_init"
    contains "$("${NM}" -D --defined-only "${lib}/libggml.so")" ' ggml_backend_reg_by_name$' \
      || fail "libggml.so does not export ggml_backend_reg_by_name"
    ;;
  Darwin)
    rt_libs="$(otool -L "${lib}/libkggml_runtime.dylib")"
    echo "${rt_libs}"
    contains "${rt_libs}" 'libggml\.' || fail "libkggml_runtime.dylib does not link libggml"
    contains "${rt_libs}" 'libggml-base\.' || fail "libkggml_runtime.dylib does not link libggml-base"
    for dylib in libkggml_runtime.dylib libkggml_runtime_jni.dylib; do
      defs="$(nm -gU "${lib}/${dylib}")"
      if contains "${defs}" ' _(ggml|gguf)_'; then
        grep -E ' _(ggml|gguf)_' <<<"${defs}" >&2
        fail "${dylib} embeds ggml symbols"
      fi
    done
    contains "$(otool -L "${lib}/libkggml_runtime_jni.dylib")" 'libkggml_runtime\.' \
      || fail "libkggml_runtime_jni.dylib does not link libkggml_runtime"
    ;;
  *) fail "unsupported host for linkage audit: $(uname -s)" ;;
esac
echo "host linkage audit: OK"
