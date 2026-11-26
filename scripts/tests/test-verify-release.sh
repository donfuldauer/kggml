#!/usr/bin/env bash
# Tests for scripts/ci/verify_release.py (the Maven Central staging audit) against small synthetic staging repos.
#
# Builds a valid repository (a BOM plus one Kotlin Multiplatform library with a JVM and an iOS platform component),
# signed with a throwaway gpg key, and asserts the audit passes for it and FAILS for: a missing signature, a
# signature that does not verify, a wrong checksum, a missing javadoc jar, a POM without a license, a POM without
# scm, a dangling `available-at` target, a .module file hash mismatch, a BOM constraint at the wrong version, a
# staged component no root coordinate reaches, and a wrong expected version. Also checks --skip-targets.
#
# Usage: scripts/tests/test-verify-release.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT="${REPO_ROOT}/scripts/ci/verify_release.py"
WORK="$(mktemp -d)"
GNUPG="${WORK}/gnupg"
mkdir -m 700 "${GNUPG}"
trap 'gpgconf --homedir "${GNUPG}" --kill all >/dev/null 2>&1 || true; rm -rf "${WORK}"' EXIT

passed=0
failed=0
ok() { echo "  ok   $1"; passed=$((passed + 1)); }
ko() { echo "  FAIL $1" >&2; failed=$((failed + 1)); }

gpg --homedir "${GNUPG}" --batch --quiet --passphrase '' --pinentry-mode loopback \
  --quick-generate-key "verify-release test <test@example.invalid>" rsa2048 sign 1d

# make_repo <dir>: a valid signed staging repository for com.example:{x-bom,x-lib} at 2.0.0.
make_repo() {
  local dir="$1"
  python3 -I - "${dir}" <<'PY'
import hashlib, json, sys
from pathlib import Path

repo = Path(sys.argv[1])
G, V = "com.example", "2.0.0"

def comp(artifact):
    d = repo / "com" / "example" / artifact / V
    d.mkdir(parents=True, exist_ok=True)
    return d

def pom(artifact, packaging=None, extra=""):
    pk = f"<packaging>{packaging}</packaging>" if packaging else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>{G}</groupId><artifactId>{artifact}</artifactId><version>{V}</version>{pk}
  <name>{artifact}</name><description>Test {artifact}</description><url>https://example.invalid</url>
  <licenses><license><name>MIT License</name><url>https://opensource.org/licenses/MIT</url></license></licenses>
  <developers><developer><id>dev</id><name>Dev</name></developer></developers>
  <scm><url>https://example.invalid</url><connection>scm:git:https://example.invalid.git</connection></scm>
  {extra}
</project>
"""

def file_entry(d, name):
    data = (d / name).read_bytes()
    return {"name": name, "url": name, "size": len(data),
            "sha512": hashlib.sha512(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest(),
            "sha1": hashlib.sha1(data).hexdigest(), "md5": hashlib.md5(data).hexdigest()}

# BOM
d = comp("x-bom")
(d / f"x-bom-{V}.pom").write_text(pom("x-bom", "pom",
    f"<dependencyManagement><dependencies><dependency><groupId>{G}</groupId><artifactId>x-lib</artifactId>"
    f"<version>{V}</version></dependency></dependencies></dependencyManagement>"))
(d / f"x-bom-{V}.module").write_text(json.dumps({"formatVersion": "1.1",
    "component": {"group": G, "module": "x-bom", "version": V}, "variants": []}))

# Platform components
for plat in ("x-lib-jvm", "x-lib-iosarm64"):
    d = comp(plat)
    ext = "jar" if plat.endswith("jvm") else "klib"
    (d / f"{plat}-{V}.{ext}").write_bytes(f"binary of {plat}".encode())
    for c in ("sources", "javadoc"):
        (d / f"{plat}-{V}-{c}.jar").write_bytes(f"{c} of {plat}".encode())
    (d / f"{plat}-{V}.pom").write_text(pom(plat, None if ext == "jar" else "klib"))
    (d / f"{plat}-{V}.module").write_text(json.dumps({"formatVersion": "1.1",
        "component": {"url": f"../../x-lib/{V}/x-lib-{V}.module", "group": G, "module": "x-lib", "version": V},
        "variants": [{"name": "main", "files": [file_entry(d, f"{plat}-{V}.{ext}")]}]}))

# Root
d = comp("x-lib")
(d / f"x-lib-{V}.jar").write_bytes(b"metadata")
for c in ("sources", "javadoc"):
    (d / f"x-lib-{V}-{c}.jar").write_bytes(c.encode())
(d / f"x-lib-{V}.pom").write_text(pom("x-lib"))
(d / f"x-lib-{V}.module").write_text(json.dumps({"formatVersion": "1.1",
    "component": {"group": G, "module": "x-lib", "version": V},
    "variants": [
        {"name": "metadataApiElements", "files": [file_entry(d, f"x-lib-{V}.jar")]},
        {"name": "jvmApiElements-published", "available-at": {"url": f"../../x-lib-jvm/{V}/x-lib-jvm-{V}.module", "group": G, "module": "x-lib-jvm", "version": V}},
        {"name": "iosArm64ApiElements-published", "available-at": {"url": f"../../x-lib-iosarm64/{V}/x-lib-iosarm64-{V}.module", "group": G, "module": "x-lib-iosarm64", "version": V}},
    ]}))
PY
  finalize "${dir}"
}

# finalize <dir>: (re)write checksums and signatures for every artifact, as Gradle's maven-publish + signing do.
finalize() {
  local f
  while IFS= read -r -d '' f; do
    case "${f}" in *.asc|*.md5|*.sha1|*.sha256|*.sha512) continue ;; esac
    rm -f "${f}.asc"
    gpg --homedir "${GNUPG}" --batch --quiet --armor --detach-sign --output "${f}.asc" "${f}"
    python3 -I - "${f}" <<'PY'
import hashlib, sys
p = sys.argv[1]
data = open(p, "rb").read()
for a in ("md5", "sha1", "sha256", "sha512"):
    open(f"{p}.{a}", "w").write(hashlib.new(a, data).hexdigest())
PY
  done < <(find "$1" -type f -print0)
}

audit() { python3 -I "${AUDIT}" "$1" --version "${2:-2.0.0}" --group com.example --bom x-bom --libraries x-lib --gnupg-home "${GNUPG}" "${@:3}"; }
expect_pass() {
  local name="$1"; shift
  if audit "$@" >/dev/null 2>&1; then ok "${name}"; else ko "${name} (expected pass)"; audit "$@" || true; fi
}
expect_fail() {
  local name="$1"; shift
  if audit "$@" >/dev/null 2>&1; then ko "${name} (expected failure)"; else ok "${name}"; fi
}
lib() { echo "$1/com/example/$2/2.0.0"; }

make_repo "${WORK}/clean"
expect_pass "valid signed repository" "${WORK}/clean"
expect_fail "wrong expected version" "${WORK}/clean" 9.9.9

make_repo "${WORK}/no-asc"
rm "$(lib "${WORK}/no-asc" x-lib-jvm)/x-lib-jvm-2.0.0.jar.asc"
expect_fail "missing signature" "${WORK}/no-asc"

make_repo "${WORK}/bad-sig"
f="$(lib "${WORK}/bad-sig" x-lib-jvm)/x-lib-jvm-2.0.0-sources.jar"
cp "$(lib "${WORK}/bad-sig" x-lib-jvm)/x-lib-jvm-2.0.0-javadoc.jar.asc" "${f}.asc"
expect_fail "signature of another file" "${WORK}/bad-sig"

make_repo "${WORK}/bad-sum"
echo 0000 >"$(lib "${WORK}/bad-sum" x-lib)/x-lib-2.0.0.pom.sha1"
expect_fail "wrong checksum" "${WORK}/bad-sum"

make_repo "${WORK}/no-javadoc"
rm "$(lib "${WORK}/no-javadoc" x-lib-iosarm64)"/x-lib-iosarm64-2.0.0-javadoc.jar*
expect_fail "missing javadoc jar" "${WORK}/no-javadoc"

make_repo "${WORK}/no-license"
sed -i.bak 's|<licenses>.*</licenses>||' "$(lib "${WORK}/no-license" x-lib-jvm)/x-lib-jvm-2.0.0.pom"
rm -f "$(lib "${WORK}/no-license" x-lib-jvm)"/*.bak
finalize "${WORK}/no-license"
expect_fail "POM without license" "${WORK}/no-license"

make_repo "${WORK}/no-scm"
sed -i.bak 's|<scm>.*</scm>||' "$(lib "${WORK}/no-scm" x-bom)/x-bom-2.0.0.pom"
rm -f "$(lib "${WORK}/no-scm" x-bom)"/*.bak
finalize "${WORK}/no-scm"
expect_fail "POM without scm" "${WORK}/no-scm"

make_repo "${WORK}/dangling"
rm -rf "${WORK}/dangling/com/example/x-lib-iosarm64"
expect_fail "dangling available-at target" "${WORK}/dangling"
expect_pass "dangling target skipped with --skip-targets ios" "${WORK}/dangling" 2.0.0 --skip-targets ios

make_repo "${WORK}/module-hash"
printf 'changed' >"$(lib "${WORK}/module-hash" x-lib-jvm)/x-lib-jvm-2.0.0.jar"
finalize "${WORK}/module-hash"
expect_fail ".module file hash mismatch" "${WORK}/module-hash"

make_repo "${WORK}/bom"
sed -i.bak 's|<artifactId>x-lib</artifactId><version>2.0.0</version>|<artifactId>x-lib</artifactId><version>1.0.0</version>|' \
  "$(lib "${WORK}/bom" x-bom)/x-bom-2.0.0.pom"
rm -f "$(lib "${WORK}/bom" x-bom)"/*.bak
finalize "${WORK}/bom"
expect_fail "BOM constraint at the wrong version" "${WORK}/bom"

make_repo "${WORK}/stray"
cp -R "${WORK}/stray/com/example/x-lib-jvm" "${WORK}/stray/com/example/x-extra"
expect_fail "staged component not reachable from a root" "${WORK}/stray"

echo "${passed} passed, ${failed} failed"
[[ ${failed} -eq 0 ]]
