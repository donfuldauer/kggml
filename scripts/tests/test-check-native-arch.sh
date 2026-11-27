#!/usr/bin/env bash
# Regression test for scripts/ci/check_native_arch.py with synthetic ELF / PE / Mach-O headers (issue #91).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/ci/python.sh"
kggml_require_python 3.9
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT

# fake <out> <elf|pe|macho> <x86_64|aarch64>: a file with just enough header for the checker.
fake() {
  kggml_python - "$@" <<'PY'
import struct, sys
out, fmt, arch = sys.argv[1:4]
if fmt == "elf":
    data = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9) + struct.pack("<HH", 3, {"x86_64": 62, "aarch64": 183}[arch])
elif fmt == "pe":
    data = bytearray(0x80); data[:2] = b"MZ"; data[0x3C:0x40] = struct.pack("<I", 0x40)
    data[0x40:0x44] = b"PE\0\0"; data[0x44:0x46] = struct.pack("<H", {"x86_64": 0x8664, "aarch64": 0xAA64}[arch])
    data = bytes(data)
else:
    data = struct.pack("<II", 0xFEEDFACF, {"x86_64": 0x01000007, "aarch64": 0x0100000C}[arch]) + bytes(24)
open(out, "wb").write(data)
PY
}
# stage <case> <platform> <fmt> <arch> <names...>: kggml/native/<platform>/ with an index and fake binaries.
stage() {
  local c="$1" p="$2" fmt="$3" arch="$4"; shift 4
  local d="$T/$c/kggml/native/$p"; mkdir -p "$d"; : > "$d/index.txt"
  for n in "$@"; do echo "$n" >> "$d/index.txt"; fake "$d/$n" "$fmt" "$arch"; done
}
fails=0
expect() { # name want args...
  local name="$1" want="$2"; shift 2
  set +e; kggml_python "${REPO_ROOT}/scripts/ci/check_native_arch.py" "$@" >"$T/out" 2>&1; local got=$?; set -e
  if [[ $got -eq $want ]]; then echo "ok   $name (exit $got)"; else echo "FAIL $name: want $want got $got"; cat "$T/out"; fails=$((fails+1)); fi
}

stage good linux-aarch64 elf aarch64 libggml-base.so.0 libkggml_runtime_jni.so
stage good windows-aarch64 pe aarch64 ggml-base.dll kggml_runtime_jni.dll
stage good macos-aarch64 macho aarch64 libggml-base.0.dylib libkggml_runtime_jni.dylib
stage good windows-x86_64 pe x86_64 ggml-base.dll kggml_runtime_jni.dll
expect all-staged 0 "$T/good"
expect one-platform 0 "$T/good" linux-aarch64
expect platform-dir 0 "$T/good/kggml/native/windows-aarch64"
stage wrongarch linux-aarch64 elf x86_64 libggml-base.so.0 libkggml_runtime_jni.so
expect wrong-arch 1 "$T/wrongarch"
stage wrongfmt windows-aarch64 elf aarch64 ggml-base.dll kggml_runtime_jni.dll
expect wrong-format 1 "$T/wrongfmt"
stage nojni linux-aarch64 elf aarch64 libggml-base.so.0
expect no-jni 1 "$T/nojni"
stage missing linux-aarch64 elf aarch64 libkggml_runtime_jni.so
echo "libggml.so.0" >> "$T/missing/kggml/native/linux-aarch64/index.txt"
expect missing-file 1 "$T/missing"
mkdir -p "$T/empty"
expect nothing-staged 1 "$T/empty"
expect unstaged-platform 1 "$T/good" linux-x86_64

[[ $fails -eq 0 ]] && echo "check-native-arch: all tests passed" || { echo "check-native-arch: $fails failure(s)"; exit 1; }
