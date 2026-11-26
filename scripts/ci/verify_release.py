#!/usr/bin/env python3
"""Audits a staged Maven repository against Maven Central's publishing requirements.

Used by scripts/ci/stage-release.sh (release dry-run in CI and ./scripts/ci-local.sh, and the real release in
.github/workflows/release.yml) on build/kggml-staging-repo before it is bundled and uploaded.

For every published root coordinate (default: the 7 Kggml coordinates under com.donfuldauer) it checks:
  * the root component has a .pom and a Gradle .module file (formatVersion 1.1, matching component coordinates);
  * every platform component a root .module points at (`available-at`) exists, except targets excluded with
    --skip-targets (e.g. `ios` on a Linux host, where Kotlin/Native iOS targets are disabled);
  * every file a .module variant lists exists with the declared size, sha1, sha256 and sha512;
  * every component that is not `pom` packaging has a -sources.jar and a -javadoc.jar;
  * every POM has groupId/artifactId/version matching its path, name, description, url, a license (name + url),
    a developer (id + name) and scm (url + connection), and `com.donfuldauer` dependencies at the release version;
  * the BOM POM's <dependencyManagement> constrains exactly the library coordinates at the release version;
  * every artifact has .md5 and .sha1 checksums (Central requires them; .sha256/.sha512 too, as Gradle writes
    them) whose values match the file, and a .asc signature;
  * with --gnupg-home, every .asc verifies with `gpg` against the keys in that home.

Exit status 0 when everything passes, 1 with one line per problem otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_GROUP = "com.donfuldauer"
DEFAULT_LIBRARIES = [
    "kggml-core-api",
    "kggml-runtime",
    "kggml-runtime-vulkan",
    "kggml-whisper",
    "kggml-llama",
    "kggml-diffusion",
    "kggml-model-manager",
]
DEFAULT_BOM = "kggml-bom"
CHECKSUMS = ("md5", "sha1", "sha256", "sha512")
NS = {"m": "http://maven.apache.org/POM/4.0.0"}


class Audit:
    def __init__(self) -> None:
        self.problems: list[str] = []

    def fail(self, message: str) -> None:
        self.problems.append(message)


def is_derived(path: Path) -> bool:
    """Checksum and signature files, which are checked alongside the artifact they belong to."""
    name = path.name
    return name.endswith(".asc") or any(name.endswith("." + c) for c in CHECKSUMS)


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text(element: ET.Element | None, tag: str) -> str:
    if element is None:
        return ""
    child = element.find(f"m:{tag}", NS)
    return (child.text or "").strip() if child is not None and child.text else ""


def component_dir(repo: Path, group: str, artifact: str, version: str) -> Path:
    return repo.joinpath(*group.split("."), artifact, version)


def check_files(audit: Audit, comp: Path, gnupg_home: str | None) -> None:
    for f in sorted(p for p in comp.iterdir() if p.is_file() and not is_derived(p)):
        rel = f"{comp.parent.name}/{comp.name}/{f.name}"
        for algorithm in CHECKSUMS:
            sidecar = f.with_name(f"{f.name}.{algorithm}")
            if not sidecar.is_file():
                audit.fail(f"{rel}: missing .{algorithm} checksum")
                continue
            declared = sidecar.read_text().split()[0].strip().lower() if sidecar.read_text().split() else ""
            if declared != digest(f, algorithm):
                audit.fail(f"{rel}: .{algorithm} checksum does not match the file")
        signature = f.with_name(f.name + ".asc")
        if not signature.is_file():
            audit.fail(f"{rel}: missing .asc signature")
            continue
        if "BEGIN PGP SIGNATURE" not in signature.read_text(errors="replace"):
            audit.fail(f"{rel}.asc: not an ASCII-armored PGP signature")
            continue
        if gnupg_home is not None:
            result = subprocess.run(
                ["gpg", "--homedir", gnupg_home, "--batch", "--verify", str(signature), str(f)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                audit.fail(f"{rel}.asc: signature does not verify: {result.stderr.strip().splitlines()[-1:]}")


def check_pom(audit: Audit, pom: Path, group: str, artifact: str, version: str) -> ET.Element | None:
    rel = pom.name
    try:
        root = ET.parse(pom).getroot()
    except ET.ParseError as e:
        audit.fail(f"{rel}: not valid XML: {e}")
        return None
    for tag, expected in (("groupId", group), ("artifactId", artifact), ("version", version)):
        if text(root, tag) != expected:
            audit.fail(f"{rel}: <{tag}> is '{text(root, tag)}', expected '{expected}'")
    for tag in ("name", "description", "url"):
        if not text(root, tag):
            audit.fail(f"{rel}: missing <{tag}>")
    license_ = root.find("m:licenses/m:license", NS)
    if license_ is None or not text(license_, "name") or not text(license_, "url"):
        audit.fail(f"{rel}: missing <licenses><license> with <name> and <url>")
    developer = root.find("m:developers/m:developer", NS)
    if developer is None or not text(developer, "id") or not text(developer, "name"):
        audit.fail(f"{rel}: missing <developers><developer> with <id> and <name>")
    scm = root.find("m:scm", NS)
    if scm is None or not text(scm, "url") or not text(scm, "connection"):
        audit.fail(f"{rel}: missing <scm> with <url> and <connection>")
    for dep in root.findall("m:dependencies/m:dependency", NS):
        if text(dep, "groupId") == group and text(dep, "version") != version:
            audit.fail(f"{rel}: dependency {group}:{text(dep, 'artifactId')} is at '{text(dep, 'version')}', expected {version}")
    return root


def check_component(
    audit: Audit, repo: Path, group: str, artifact: str, version: str, gnupg_home: str | None, root_artifact: str | None = None
) -> dict | None:
    """Checks one component directory; returns its parsed .module (if any).

    A platform component of a Kotlin Multiplatform library (`root_artifact` set) declares the root component it
    belongs to (`component.url` pointing back at the root .module) instead of its own coordinates.
    """
    comp = component_dir(repo, group, artifact, version)
    if not comp.is_dir():
        audit.fail(f"{group}:{artifact}:{version}: component missing from the staging repository")
        return None
    base = f"{artifact}-{version}"
    pom = comp / f"{base}.pom"
    root = check_pom(audit, pom, group, artifact, version) if pom.is_file() else None
    if root is None and not pom.is_file():
        audit.fail(f"{artifact}: missing {pom.name}")
    packaging = text(root, "packaging") or "jar" if root is not None else "jar"
    if packaging != "pom":
        for classifier in ("sources", "javadoc"):
            if not (comp / f"{base}-{classifier}.jar").is_file():
                audit.fail(f"{artifact}: missing {base}-{classifier}.jar (required for '{packaging}' packaging)")
    check_files(audit, comp, gnupg_home)

    module_file = comp / f"{base}.module"
    if not module_file.is_file():
        audit.fail(f"{artifact}: missing Gradle module metadata {module_file.name}")
        return None
    try:
        module = json.loads(module_file.read_text())
    except json.JSONDecodeError as e:
        audit.fail(f"{module_file.name}: not valid JSON: {e}")
        return None
    if module.get("formatVersion") != "1.1":
        audit.fail(f"{module_file.name}: formatVersion is {module.get('formatVersion')!r}, expected '1.1'")
    component = module.get("component", {})
    owner = root_artifact or artifact
    if (component.get("group"), component.get("module"), component.get("version")) != (group, owner, version):
        audit.fail(f"{module_file.name}: component is {component}, expected {group}:{owner}:{version}")
    if root_artifact is not None:
        expected_url = f"../../{root_artifact}/{version}/{root_artifact}-{version}.module"
        if component.get("url") != expected_url:
            audit.fail(f"{module_file.name}: component url is {component.get('url')!r}, expected {expected_url!r}")
    for variant in module.get("variants", []):
        for entry in variant.get("files", []):
            f = comp / entry.get("url", "")
            if not f.is_file():
                audit.fail(f"{module_file.name}: variant {variant.get('name')} lists missing file {entry.get('url')}")
                continue
            if entry.get("size") != f.stat().st_size:
                audit.fail(f"{module_file.name}: {entry.get('url')} size {entry.get('size')} != {f.stat().st_size}")
            for algorithm in ("sha1", "sha256", "sha512"):
                if entry.get(algorithm) != digest(f, algorithm):
                    audit.fail(f"{module_file.name}: {entry.get('url')} {algorithm} does not match")
    return module


def check_bom(audit: Audit, repo: Path, group: str, bom: str, libraries: list[str], version: str) -> None:
    pom = component_dir(repo, group, bom, version) / f"{bom}-{version}.pom"
    if not pom.is_file():
        return  # reported by check_component
    root = ET.parse(pom).getroot()
    if text(root, "packaging") != "pom":
        audit.fail(f"{pom.name}: packaging is '{text(root, 'packaging')}', expected 'pom'")
    found = sorted(
        (text(d, "groupId"), text(d, "artifactId"), text(d, "version"))
        for d in root.findall("m:dependencyManagement/m:dependencies/m:dependency", NS)
    )
    expected = sorted((group, lib, version) for lib in libraries)
    if found != expected:
        audit.fail(f"{pom.name}: <dependencyManagement> constraints {found} != expected {expected}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path, help="staged Maven repository (e.g. build/kggml-staging-repo)")
    parser.add_argument("--version", required=True, help="expected release version (VERSION_NAME)")
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--bom", default=DEFAULT_BOM)
    parser.add_argument("--libraries", default=",".join(DEFAULT_LIBRARIES), help="comma-separated root library artifactIds")
    parser.add_argument("--skip-targets", default="", help="comma-separated target name fragments to skip (e.g. ios)")
    parser.add_argument("--gnupg-home", default=None, help="verify every .asc with gpg using this GNUPGHOME")
    args = parser.parse_args(argv)

    repo: Path = args.repo
    if not repo.is_dir():
        print(f"FAIL: staging repository {repo} does not exist", file=sys.stderr)
        return 1
    libraries = [x for x in args.libraries.split(",") if x]
    skip = [x.lower() for x in args.skip_targets.split(",") if x]
    audit = Audit()

    checked: set[str] = set()
    skipped: set[str] = set()
    for artifact in [args.bom, *libraries]:
        module = check_component(audit, repo, args.group, artifact, args.version, args.gnupg_home)
        checked.add(artifact)
        if module is None:
            continue
        for variant in module.get("variants", []):
            at = variant.get("available-at")
            if not at:
                continue
            target = at.get("module", "")
            if at.get("group") != args.group or at.get("version") != args.version:
                audit.fail(f"{artifact}.module: variant {variant.get('name')} points at {at}, expected {args.group}:*:{args.version}")
            if any(s in target.lower() for s in skip):
                skipped.add(target)
                continue
            if target in checked:
                continue
            checked.add(target)
            target_module = check_component(audit, repo, args.group, target, args.version, args.gnupg_home, root_artifact=artifact)
            if target_module is None:
                audit.fail(f"{artifact}.module: available-at {target} has no module metadata in the staging repository")
    check_bom(audit, repo, args.group, args.bom, libraries, args.version)

    # Anything staged that no root coordinate reaches would be uploaded without having been audited.
    group_dir = repo.joinpath(*args.group.split("."))
    staged = {p.name for p in group_dir.iterdir() if p.is_dir()} if group_dir.is_dir() else set()
    for stray in sorted(staged - checked - skipped):
        audit.fail(f"{args.group}:{stray}: staged but not reachable from any root coordinate")
    for other in sorted(p for p in repo.iterdir() if p.is_dir() and p.name != args.group.split(".")[0]):
        audit.fail(f"{other.name}/: staging repository contains a group outside {args.group}")

    if audit.problems:
        for p in audit.problems:
            print(f"FAIL: {p}", file=sys.stderr)
        print(f"Release audit: {len(audit.problems)} problem(s) in {repo}", file=sys.stderr)
        return 1
    signed = "verified with gpg" if args.gnupg_home else "present (not verified: no --gnupg-home)"
    note = f"; skipped targets: {', '.join(sorted(skipped))}" if skipped else ""
    print(
        f"Release audit OK: {len(checked)} components under {args.group} at {args.version} "
        f"({len([args.bom, *libraries])} root coordinates); checksums match; signatures {signed}{note}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
