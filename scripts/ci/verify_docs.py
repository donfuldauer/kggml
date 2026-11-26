#!/usr/bin/env python3
"""Documentation audit (issue #22). Run through scripts/ci/verify-docs.sh; see that file for what it checks.

Reads tracked files only (git ls-files), so build outputs and untracked scratch files are never audited.
KGGML_DOCS_AUDIT_ROOT overrides the repository root (used by scripts/tests/test-verify-docs.sh).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("KGGML_DOCS_AUDIT_ROOT") or Path(__file__).resolve().parents[2])

# Markdown files whose code blocks must be compiled snippets.
SNIPPET_DOCS = ["README.md"]
# Sources that hold the compiled snippet regions.
SNIPPET_SOURCE_DIR = "docs-snippets/"
# User-facing documents: checked for internal-only references and Maven coordinates.
PUBLIC_DOC_GLOBS = [
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "NOTICE",
    ".github/*.md",
    ".github/ISSUE_TEMPLATE/*.md",
    "kggml-*/README.md",
    "kggml-*/native/README.md",
    "samples/*.md",
    "samples/*/README.md",
    "docs-snippets/README.md",
    "integration-tests/README.md",
]
# Patterns that must never appear in user-facing documents.
INTERNAL_PATTERNS = [
    (re.compile(r"(?<![\w.])/(?:home|Users)/[A-Za-z0-9_.-]+"), "absolute local machine path"),
    (re.compile(r"\b[A-Za-z]:\\\\?Users\\"), "absolute local machine path"),
    (re.compile(r"claude\.ai/code/session_"), "link to a private agent session"),
    (re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b"), "unfinished placeholder"),
    (re.compile(r"lorem ipsum", re.IGNORECASE), "placeholder text"),
]
# Source sets whose public declarations must carry KDoc.
KDOC_SOURCES = re.compile(r"^kggml-[a-z-]+/src/(?:common|jvm|android|ios|jni|jvmCommon|native)Main/.*\.kt$")

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def tracked(*patterns: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", *patterns], check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------------------------
# Markdown parsing helpers


FENCE = re.compile(r"^(\s*)(```+|~~~+)(.*)$")


def split_markdown(text: str) -> tuple[list[tuple[int, str]], list[dict]]:
    """Returns (prose lines with their 1-based numbers, fenced code blocks)."""
    prose: list[tuple[int, str]] = []
    blocks: list[dict] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m:
            indent, marker, info = m.group(1), m.group(2), m.group(3).strip()
            start = i
            body: list[str] = []
            i += 1
            while i < len(lines) and not (
                lines[i].strip().startswith(marker[0] * len(marker)) and lines[i].strip().strip(marker[0]) == ""
            ):
                body.append(lines[i][len(indent):] if lines[i].startswith(indent) else lines[i].lstrip())
                i += 1
            blocks.append({"line": start + 1, "lang": info.split()[0] if info else "", "body": "\n".join(body)})
            i += 1
            continue
        prose.append((i + 1, lines[i]))
        i += 1
    return prose, blocks


def strip_code_spans(line: str) -> str:
    return re.sub(r"(`+)(?:(?!\1).)+?\1", lambda m: " " * len(m.group(0)), line)


def github_slug(heading: str) -> str:
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors_of(rel: str, cache: dict[str, set[str]]) -> set[str]:
    if rel not in cache:
        prose, _ = split_markdown(read(rel))
        seen: dict[str, int] = {}
        slugs: set[str] = set()
        for _, line in prose:
            m = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
            if not m:
                for a in re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", line):
                    slugs.add(a)
                continue
            slug = github_slug(m.group(1))
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            slugs.add(slug if n == 0 else f"{slug}-{n}")
        cache[rel] = slugs
    return cache[rel]


LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
HTML_LINK = re.compile(r"<(?:a|img)\s[^>]*(?:href|src)=\"([^\"]+)\"")
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")


def links_in(rel: str) -> list[tuple[int, str]]:
    prose, _ = split_markdown(read(rel))
    found: list[tuple[int, str]] = []
    for number, raw in prose:
        line = strip_code_spans(raw)
        m = REF_DEF.match(line)
        if m:
            found.append((number, m.group(1)))
            continue
        for pattern in (LINK, HTML_LINK, AUTOLINK):
            found.extend((number, target) for target in pattern.findall(line))
    return found


# --------------------------------------------------------------------------------------------------------------
# 1. Links


def check_links(online: bool) -> None:
    files = set(tracked())
    dirs = {str(Path(f).parent) for f in files}
    dirs |= {str(p) for f in files for p in Path(f).parents}
    anchor_cache: dict[str, set[str]] = {}
    external: dict[str, list[str]] = {}
    for rel in tracked("*.md"):
        for number, target in links_in(rel):
            where = f"{rel}:{number}"
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                if target.startswith(("http://", "https://")):
                    if re.search(r"\s|[<>]", target):
                        err(f"{where}: malformed URL '{target}'")
                    external.setdefault(target, []).append(where)
                elif not target.startswith("mailto:"):
                    err(f"{where}: unsupported link scheme '{target}'")
                continue
            path_part, _, anchor = target.partition("#")
            if path_part:
                resolved = os.path.normpath(
                    path_part.lstrip("/") if path_part.startswith("/") else str(Path(rel).parent / path_part)
                )
                if resolved.startswith(".."):
                    err(f"{where}: link '{target}' points outside the repository")
                    continue
                if resolved not in files and resolved not in dirs:
                    err(f"{where}: broken link '{target}' ({resolved} is not a tracked file or directory)")
                    continue
            else:
                resolved = rel
            if anchor and resolved.endswith(".md") and resolved in files:
                if anchor not in anchors_of(resolved, anchor_cache):
                    err(f"{where}: broken anchor '#{anchor}' in {resolved}")
    if online:
        check_external(external)


def check_external(urls: dict[str, list[str]]) -> None:
    for url, places in sorted(urls.items()):
        status = None
        problem = None
        for method in ("HEAD", "GET"):
            request = urllib.request.Request(url, method=method, headers={"User-Agent": "kggml-docs-link-check"})
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    status = response.status
                    problem = None
                    break
            except urllib.error.HTTPError as e:
                status, problem = e.code, f"HTTP {e.code}"
                if e.code not in (403, 405, 429, 501):
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                problem = f"unreachable ({e})"
        if problem and status not in (401, 403, 429):
            err(f"{places[0]}: external link {url} failed: {problem}")
        elif problem:
            print(f"docs audit: warning: {url} answered {problem} (not treated as broken)", file=sys.stderr)


# --------------------------------------------------------------------------------------------------------------
# 2. Internal-only references in user-facing documents


def public_docs() -> list[str]:
    return sorted(set(tracked(*PUBLIC_DOC_GLOBS)))


def check_internal_references() -> None:
    for rel in public_docs():
        for number, line in enumerate(read(rel).split("\n"), start=1):
            for pattern, what in INTERNAL_PATTERNS:
                if pattern.search(line):
                    err(f"{rel}:{number}: {what}: {line.strip()[:120]}")


# --------------------------------------------------------------------------------------------------------------
# 3. Maven coordinates


def published_artifacts() -> tuple[set[str], str]:
    bom = read("kggml-bom/build.gradle.kts")
    m = re.search(r"kggmlLibraryModules[^=]*=\s*listOf\((.*?)\)", bom, re.DOTALL)
    if not m:
        err("kggml-bom/build.gradle.kts: cannot find kggmlLibraryModules")
        return set(), ""
    modules = {name.lstrip(":") for name in re.findall(r"\"(:[\w-]+)\"", m.group(1))}
    modules.add("kggml-bom")
    version = re.search(r"^VERSION_NAME=(\S+)", read("gradle.properties"), re.MULTILINE)
    group = re.search(r"^GROUP=(\S+)", read("gradle.properties"), re.MULTILINE)
    if not version or not group or group.group(1) != "com.donfuldauer":
        err("gradle.properties: GROUP must be com.donfuldauer and VERSION_NAME must be set")
        return modules, ""
    return modules, version.group(1)


def check_coordinates() -> None:
    artifacts, version = published_artifacts()
    sources = public_docs() + tracked(SNIPPET_SOURCE_DIR)
    for rel in sources:
        for number, line in enumerate(read(rel).split("\n"), start=1):
            for artifact, ver in re.findall(r"com\.donfuldauer:([\w.-]+?)(?::([\w.+-]+))?(?![\w.:-])", line):
                if artifact not in artifacts:
                    err(f"{rel}:{number}: com.donfuldauer:{artifact} is not a published artifact {sorted(artifacts)}")
                if ver and version and ver != version:
                    err(f"{rel}:{number}: com.donfuldauer:{artifact}:{ver} does not match VERSION_NAME={version}")


# --------------------------------------------------------------------------------------------------------------
# 4. README code blocks are compiled snippets


MARKER = re.compile(r"^\s*<!--\s*snippet:\s*([\w-]+)\s*-->\s*$")


def snippet_regions() -> dict[str, list[tuple[str, str]]]:
    regions: dict[str, list[tuple[str, str]]] = {}
    for rel in tracked(SNIPPET_SOURCE_DIR):
        if not rel.endswith((".kt", ".kts")):
            continue
        lines = read(rel).split("\n")
        open_regions: dict[str, int] = {}
        for i, line in enumerate(lines):
            start = re.match(r"^\s*//\s*snippet:start\s+([\w-]+)\s*$", line)
            end = re.match(r"^\s*//\s*snippet:end\s+([\w-]+)\s*$", line)
            if start:
                if start.group(1) in open_regions:
                    err(f"{rel}:{i + 1}: snippet '{start.group(1)}' started twice")
                open_regions[start.group(1)] = i
            elif end:
                name = end.group(1)
                if name not in open_regions:
                    err(f"{rel}:{i + 1}: snippet:end '{name}' without snippet:start")
                    continue
                body = lines[open_regions.pop(name) + 1:i]
                body = [line for line in body if not re.match(r"^\s*//\s*snippet:(?:start|end)\s", line)]
                regions.setdefault(name, []).append((f"{rel}:{i + 1}", textwrap.dedent("\n".join(body)).strip("\n")))
        for name, i in open_regions.items():
            err(f"{rel}:{i + 1}: snippet '{name}' is never closed")
    return regions


def check_snippets() -> None:
    regions = snippet_regions()
    used: set[str] = set()
    for name, places in regions.items():
        if len(places) > 1:
            err(f"snippet '{name}' is defined more than once: {', '.join(p for p, _ in places)}")
    for rel in SNIPPET_DOCS:
        if not (ROOT / rel).is_file():
            continue
        text = read(rel)
        lines = text.split("\n")
        _, blocks = split_markdown(text)
        for block in blocks:
            if block["lang"] not in ("kotlin", "kt", "kts", "groovy", "java", "swift"):
                continue
            where = f"{rel}:{block['line']}"
            j = block["line"] - 2
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            marker = MARKER.match(lines[j]) if j >= 0 else None
            if not marker:
                err(f"{where}: ```{block['lang']} block has no <!-- snippet: <name> --> marker (README code must be compiled)")
                continue
            name = marker.group(1)
            used.add(name)
            if name not in regions:
                err(f"{where}: snippet '{name}' does not exist in {SNIPPET_SOURCE_DIR}")
                continue
            source, expected = regions[name][0]
            actual = textwrap.dedent(block["body"]).strip("\n")
            if actual != expected:
                diff = next(
                    (k for k, (a, b) in enumerate(zip(actual.split("\n"), expected.split("\n"))) if a != b),
                    min(len(actual.split("\n")), len(expected.split("\n"))),
                )
                err(f"{where}: block differs from snippet '{name}' ({source}) at block line {diff + 1}")
    for name, places in regions.items():
        if name not in used:
            err(f"{places[0][0]}: snippet '{name}' is not used by {', '.join(SNIPPET_DOCS)}")


# --------------------------------------------------------------------------------------------------------------
# 5. KDoc on public API


DECL = re.compile(
    r"^(?P<indent>\s*)(?:@[\w.]+(?:\([^)]*\))?\s+)*public\s+(?P<mods>(?:[a-z]+\s+)*?)"
    r"(?P<kind>class|interface|object|fun|val|var|typealias|constructor)\b(?P<rest>.*)$"
)
DATA_PARAM = re.compile(r"^(?P<indent>\s+)(?:@[\w.]+(?:\([^)]*\))?\s+)*(?:override\s+)?va[lr]\s+(?P<name>\w+)\s*:")
CLASS_START = re.compile(r"^\s*(?:@[\w.]+(?:\([^)]*\))?\s+)*(?:public\s+)?(?:[a-z]+\s+)*(?:class|interface|object)\s+(\w+)")


def kdoc_above(lines: list[str], i: int) -> str | None:
    j = i - 1
    while j >= 0 and re.match(r"^\s*@[\w.]+", lines[j]):
        j -= 1
    if j < 0 or not lines[j].strip().endswith("*/"):
        return None
    end = j
    while j >= 0 and "/**" not in lines[j]:
        if "/*" in lines[j] and "/**" not in lines[j]:
            return None
        j -= 1
    return "\n".join(lines[j:end + 1]) if j >= 0 else None


def enclosing_class_doc(lines: list[str], i: int, indent: int) -> str | None:
    for j in range(i - 1, -1, -1):
        line = lines[j]
        if not line.strip():
            continue
        current = len(line) - len(line.lstrip())
        if current < indent and CLASS_START.match(line):
            return kdoc_above(lines, j) or ""
        if current < indent and not line.lstrip().startswith(("@", "*", "/*", "//")) and line.rstrip().endswith("{"):
            return None
    return None


def documented_by_class(lines: list[str], i: int, name: str, indent: int) -> bool:
    doc = enclosing_class_doc(lines, i, indent)
    return bool(doc) and re.search(rf"@(?:property|param)\s+{re.escape(name)}\b", doc) is not None


def in_primary_constructor(lines: list[str], i: int) -> bool:
    depth = 0
    for j in range(i - 1, -1, -1):
        stripped = strip_code_spans(lines[j])
        depth += stripped.count(")") - stripped.count("(")
        if depth < 0:
            return CLASS_START.match(lines[j]) is not None or re.search(r"\bconstructor\s*\($", lines[j].strip()) is not None
        if lines[j].rstrip().endswith("{") and depth == 0:
            return False
    return False


def check_kdoc() -> None:
    for rel in tracked("kggml-*/src/*"):
        if not KDOC_SOURCES.match(rel):
            continue
        lines = read(rel).split("\n")
        in_comment = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if in_comment:
                in_comment = "*/" not in stripped
                continue
            if stripped.startswith("/*") and "*/" not in stripped:
                in_comment = True
                continue
            m = DECL.match(line)
            name = None
            if m:
                mods = m.group("mods").split()
                if "override" in mods or "actual" in mods:
                    continue
                if "companion" in mods and m.group("kind") == "object":
                    continue  # documented through its members
                name_match = re.match(r"\s*(?:<[^>]*>\s*)?(?:[\w.<>?, ]+\.)?`?(\w+)`?", m.group("rest"))
                name = name_match.group(1) if name_match else m.group("kind")
                indent = len(m.group("indent"))
            else:
                p = DATA_PARAM.match(line)
                if not p or "override" in line.split("va")[0] or not in_primary_constructor(lines, i):
                    continue
                name, indent = p.group("name"), len(p.group("indent"))
                if "public" not in line and not _public_data_class(lines, i):
                    continue
            if kdoc_above(lines, i) is not None:
                continue
            if m and m.group("kind") not in ("val", "var"):
                if m.group("kind") == "constructor" and documented_by_class(lines, i, "constructor", indent):
                    continue
                err(f"{rel}:{i + 1}: public {m.group('kind')} '{name}' has no KDoc")
                continue
            if documented_by_class(lines, i, name, indent):
                continue
            err(f"{rel}:{i + 1}: public property '{name}' has no KDoc (own /** */ or @property in its class KDoc)")


def _public_data_class(lines: list[str], i: int) -> bool:
    for j in range(i - 1, -1, -1):
        m = CLASS_START.match(lines[j])
        if m:
            decl = lines[j]
            return "data class" in decl and "private" not in decl and "internal" not in decl
    return False


# --------------------------------------------------------------------------------------------------------------


CHECKS = {
    "links": lambda args: check_links(args.online),
    "internal": lambda args: check_internal_references(),
    "coordinates": lambda args: check_coordinates(),
    "snippets": lambda args: check_snippets(),
    "kdoc": lambda args: check_kdoc(),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="also request every external http(s) link")
    parser.add_argument("--only", default=",".join(CHECKS), help=f"comma-separated subset of {','.join(CHECKS)}")
    args = parser.parse_args()
    selected = [c for c in args.only.split(",") if c]
    unknown = [c for c in selected if c not in CHECKS]
    if unknown:
        parser.error(f"unknown check(s): {unknown}")
    for check in selected:
        CHECKS[check](args)
    for message in errors:
        print(f"docs audit: {message}", file=sys.stderr)
    if errors:
        print(f"docs audit: FAILED ({len(errors)} problem(s))", file=sys.stderr)
        return 1
    print(f"docs audit: OK ({', '.join(selected)}{', online' if args.online else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
