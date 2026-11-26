#!/usr/bin/env python3
"""Checks that a staged JVM desktop runtime matches its platform directory (issue #91).

A cross-compiled runtime (linux-aarch64 on x86_64, windows-aarch64 on x64) cannot be loaded on the build host, so
nothing else notices a library built for the wrong architecture. This reads the binary headers directly (no
readelf / dumpbin / otool needed, so it runs on every host):

  * every file listed in <dir>/index.txt exists and is an ELF (linux), PE (windows) or Mach-O (macos) binary
    of the architecture the platform name says;
  * the index lists the JNI bridge (kggml_runtime_jni).

Usage: check_native_arch.py <staged-root-or-platform-dir> [<platform> ...]
  With a root (holding kggml/native/<platform>/), every listed platform is checked, or every staged one if none is
  given. With a platform directory itself, its name is the platform.
"""
import os
import struct
import sys

ELF_MACHINES = {"x86_64": 62, "aarch64": 183}
PE_MACHINES = {"x86_64": 0x8664, "aarch64": 0xAA64}
MACHO_CPUS = {"x86_64": 0x01000007, "aarch64": 0x0100000C}
MACHO_MAGICS = {0xFEEDFACF, 0xCFFAEDFE}
FAT_MAGIC = 0xCAFEBABE


def binary_arch(path):
    """Returns (format, arch or None) for an ELF / PE / Mach-O file, or (None, None) when it is none of them."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if head[:4] == b"\x7fELF":
        little = head[5] == 1
        machine = struct.unpack("<H" if little else ">H", head[18:20])[0]
        return "elf", next((a for a, m in ELF_MACHINES.items() if m == machine), f"e_machine={machine}")
    if head[:2] == b"MZ" and len(head) >= 0x40:
        pe = struct.unpack("<I", head[0x3C:0x40])[0]
        if head[pe:pe + 4] == b"PE\0\0":
            machine = struct.unpack("<H", head[pe + 4:pe + 6])[0]
            return "pe", next((a for a, m in PE_MACHINES.items() if m == machine), f"machine=0x{machine:04x}")
    if len(head) >= 8:
        magic_le = struct.unpack("<I", head[:4])[0]
        if magic_le in MACHO_MAGICS:
            cpu = struct.unpack("<I", head[4:8])[0]
            return "macho", next((a for a, c in MACHO_CPUS.items() if c == cpu), f"cputype=0x{cpu:x}")
        if struct.unpack(">I", head[:4])[0] == FAT_MAGIC:
            return "macho", "universal"
    return None, None


def check_platform(platform_dir, platform):
    errors = []
    os_part, _, arch = platform.partition("-")
    want_format = {"linux": "elf", "windows": "pe", "macos": "macho"}.get(os_part)
    if want_format is None or arch not in ELF_MACHINES:
        return [f"{platform}: unknown platform (expected <linux|windows|macos>-<x86_64|aarch64>)"]
    index = os.path.join(platform_dir, "index.txt")
    if not os.path.isfile(index):
        return [f"{platform}: missing index.txt"]
    with open(index, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not any("kggml_runtime_jni" in n for n in names):
        errors.append(f"{platform}: index.txt does not list the kggml_runtime_jni bridge")
    for name in names:
        path = os.path.join(platform_dir, name)
        if not os.path.isfile(path):
            errors.append(f"{platform}: {name} is listed but missing")
            continue
        fmt, got = binary_arch(path)
        if fmt != want_format:
            errors.append(f"{platform}: {name} is {fmt or 'not a binary'}, expected {want_format}")
        elif got != arch:
            errors.append(f"{platform}: {name} is built for {got}, expected {arch}")
        else:
            print(f"ok   {platform}/{name} ({fmt}, {got})")
    return errors


def main(argv):
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    root, platforms = argv[0], argv[1:]
    if os.path.isfile(os.path.join(root, "index.txt")):
        targets = [(root, os.path.basename(os.path.normpath(root)))]
    else:
        base = os.path.join(root, "kggml", "native")
        if not platforms:
            platforms = sorted(p for p in os.listdir(base) if os.path.isfile(os.path.join(base, p, "index.txt"))) \
                if os.path.isdir(base) else []
        if not platforms:
            print(f"FAIL: no kggml/native/<platform>/index.txt under {root}", file=sys.stderr)
            return 1
        targets = [(os.path.join(base, p), p) for p in platforms]
    errors = [e for d, p in targets for e in check_platform(d, p)]
    for e in errors:
        print(f"FAIL {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
