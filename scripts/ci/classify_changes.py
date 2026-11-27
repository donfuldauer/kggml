#!/usr/bin/env python3
"""Regex-based change classifier for Kggml GitHub Actions workflows.

Maps every changed file in a pull request (by file extension, path regex, git change status, and binary
detection) to the exact CI platforms (`ci.yml`) and hygiene checks (`repo-hygiene.yml`) that actually need
to run. Unknown files fail safe by enabling all CI platforms plus license/docs audits.

Shared by `.github/workflows/ci.yml`, `.github/workflows/repo-hygiene.yml`, and
`scripts/tests/test-classify-changes.sh` (`./scripts/ci-local.sh`).

Usage:
  scripts/ci/classify_changes.py --base <ref> --head <ref> [--github-output <file>] [--summary <file>] [--json]
  scripts/ci/classify_changes.py --paths <path> [<path> ...] [--json]
  scripts/ci/classify_changes.py --all [--github-output <file>] [--summary <file>] [--json]
  scripts/ci/classify_changes.py --verify-repo [--repo <dir>]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PLATFORMS = ("linux", "macos", "windows")

SCRIPT_TEST_KEYS = (
    "test_wait_pr_checks",
    "test_verify_release",
    "test_check_native_arch",
    "test_audit_history",
    "test_audit_binary_symbols",
    "test_classify_changes",
)

ALL_OUTPUT_KEYS = (
    "ci_linux",
    "ci_macos",
    "ci_windows",
    "ci_any",
    "agent_parity",
    "workflow_audit",
    "license_audit",
    "docs_audit",
    "script_tests",
    *SCRIPT_TEST_KEYS,
)


@dataclasses.dataclass(frozen=True)
class ChangedFile:
    """A single file entry from a git diff or synthetic test input."""

    path: str
    status: str = "M"  # A, M, D, T, ...
    is_binary: bool = False


@dataclasses.dataclass(frozen=True)
class CiRule:
    """First-match classification rule mapping a path regex to its category and CI platforms."""

    category: str
    pattern: re.Pattern[str]
    platforms: frozenset[str]
    description: str


# --------------------------------------------------------------------------------------------------------------
# Layer 1: Primary content classification (ordered most-specific -> most-general; first match wins)
# --------------------------------------------------------------------------------------------------------------

CI_RULES: tuple[CiRule, ...] = (
    CiRule(
        category="docs-markdown",
        pattern=re.compile(r"(?i)\.md$"),
        platforms=frozenset(),
        description="Markdown documentation (checked by docs-audit, not compiled by ci.yml)",
    ),
    CiRule(
        category="docs-tree",
        pattern=re.compile(r"^docs/"),
        platforms=frozenset(),
        description="Engineering documentation tree under docs/",
    ),
    CiRule(
        category="agent-harness-config",
        pattern=re.compile(r"^\.(?:agents|antigravity|claude|codex|gemini)/"),
        platforms=frozenset(),
        description="AI agent and multi-harness instructions/configs",
    ),
    CiRule(
        category="license-notice",
        pattern=re.compile(r"^(?:LICENSE|NOTICE)$"),
        platforms=frozenset(),
        description="License and upstream attribution files (verified by license-audit/docs-audit)",
    ),
    CiRule(
        category="repo-metadata-config",
        pattern=re.compile(
            r"^(?:\.gitignore|\.coderabbit\.yaml|\.gitleaks\.toml|\.githooks/.*|"
            r"\.github/(?:ISSUE_TEMPLATE/.*|dependabot\.ya?ml|labeler\.ya?ml|CODEOWNERS|FUNDING\.ya?ml))$"
        ),
        platforms=frozenset(),
        description="Repository metadata, git hooks, templates, Dependabot, Labeler, CodeRabbit, Gitleaks",
    ),
    CiRule(
        category="workflow-ci",
        pattern=re.compile(r"^\.github/workflows/ci\.yml$"),
        platforms=frozenset(PLATFORMS),
        description="Main cross-platform CI workflow itself",
    ),
    CiRule(
        category="workflow-non-ci",
        pattern=re.compile(r"^\.github/workflows/.*\.ya?ml$"),
        platforms=frozenset(),
        description="Non-ci.yml GitHub Actions workflows (repo-hygiene, release, label, stale)",
    ),
    CiRule(
        category="scripts-hygiene-and-local",
        pattern=re.compile(
            r"^(?:scripts/tests/.*|"
            r"scripts/(?:ci-local|install-git-hooks|sync-agent-mirrors|verify-agent-parity|issue-status)\.sh|"
            r"scripts/ci/(?:verify-docs\.sh|verify_docs\.py|verify-licenses\.sh|verify-hosted-runners\.sh|"
            r"audit-history\.sh|wait-pr-checks\.sh|coverage\.sh|run-android-instrumented\.sh|"
            r"upload-central-bundle\.sh))$"
        ),
        platforms=frozenset(),
        description="Repo-hygiene, local-dev, release-upload, and script-test shell/Python scripts",
    ),
    CiRule(
        category="scripts-ci-classifier",
        pattern=re.compile(r"^scripts/ci/classify_changes\.py$"),
        platforms=frozenset({"linux"}),
        description="Change classifier script (smoke-tested in ci.yml Linux + full unit suite in repo-hygiene)",
    ),
    CiRule(
        category="scripts-ci-linux-only",
        pattern=re.compile(
            r"^(?:scripts/ci/setup/linux\.sh|"
            r"scripts/ci/audit-(?:android-linkage|gpu-addon)\.sh|"
            r"scripts/ci/verify-(?:whisper|llama|diffusion)-pin\.sh|"
            r"kggml-[^/]+/native/scripts/build-android\.sh|"
            r"kggml-runtime-vulkan/native/scripts/build-host\.sh)$"
        ),
        platforms=frozenset({"linux"}),
        description="Linux/Android-only CI setup, audit, pin, and native build scripts",
    ),
    CiRule(
        category="scripts-ci-macos-only",
        pattern=re.compile(
            r"^(?:scripts/ci/setup/macos\.sh|"
            r"scripts/ios/.*\.sh|"
            r"kggml-[^/]+/native/scripts/build-ios-xcframework\.sh)$"
        ),
        platforms=frozenset({"macos"}),
        description="macOS/iOS-only CI setup, XCFramework, and Xcode build scripts",
    ),
    CiRule(
        category="scripts-ci-windows-only",
        pattern=re.compile(
            r"^(?:scripts/ci/setup/windows\.ps1|"
            r"kggml-runtime/native/scripts/build-windows-arm64\.cmd|"
            r"gradlew\.bat|"
            r".*\.(?:ps1|cmd|bat))$"
        ),
        platforms=frozenset({"windows"}),
        description="Windows-only setup, batch/PowerShell, and ARM64 build scripts",
    ),
    CiRule(
        category="scripts-ci-linux-macos",
        pattern=re.compile(
            r"^(?:scripts/ci/verify-ggml-pin\.sh|"
            r"scripts/ci/audit-(?:host|whisper|llama|diffusion)-linkage\.sh|"
            r"scripts/ci/stage-release\.sh|"
            r"scripts/ci/verify_release\.py|"
            r"kggml-[^/]+/native/scripts/build-host\.sh)$"
        ),
        platforms=frozenset({"linux", "macos"}),
        description="Linux + macOS CI scripts (host builds, linkage audits, ggml pin, release dry-run)",
    ),
    CiRule(
        category="scripts-ci-linux-windows",
        pattern=re.compile(r"^(?:scripts/ci/check_native_arch\.py|scripts/ci/python\.sh)$"),
        platforms=frozenset({"linux", "windows"}),
        description="Linux + Windows cross-arch / Python helper scripts",
    ),
    CiRule(
        category="scripts-ci-all-platforms",
        pattern=re.compile(
            r"^(?:scripts/audit-binary-symbols\.sh|scripts/ci/audit-minimal-sample\.sh|scripts/ci/.*)$"
        ),
        platforms=frozenset(PLATFORMS),
        description="Cross-platform CI audit scripts and fail-safe for new scripts under scripts/ci/",
    ),
    CiRule(
        category="scripts-dev-other",
        pattern=re.compile(r"^scripts/[^/]+\.sh$"),
        platforms=frozenset(),
        description="Top-level developer/workflow helper scripts not invoked by ci.yml",
    ),
    CiRule(
        category="apple-ios-sources",
        pattern=re.compile(r"^(?:.+/)?src/(?:ios|apple|macos)(?:Main|Test)/"),
        platforms=frozenset({"macos"}),
        description="Apple (iOS / macOS) Kotlin Multiplatform sources and tests",
    ),
    CiRule(
        category="apple-cinterop-and-xcode",
        pattern=re.compile(
            r"(?:\.(?:def|swift|pbxproj|plist|xcscheme|xcworkspacedata|xib|storyboard|entitlements|metal|m|mm)$|"
            r"^samples/[^/]+/iosApp/)"
        ),
        platforms=frozenset({"macos"}),
        description="Kotlin/Native cinterop .def, Swift, Objective-C, Metal, and Xcode host app files",
    ),
    CiRule(
        category="android-sources-and-config",
        pattern=re.compile(
            r"(?:^(?:.+/)?src/android(?:Main|UnitTest|SharedTest|InstrumentedTest)/|"
            r"(?:^|/)AndroidManifest\.xml$|"
            r"(?:^|/)consumer-rules\.pro$|"
            r"(?:^|/)robolectric\.properties$)"
        ),
        platforms=frozenset({"linux"}),
        description="Android-only Kotlin sources, tests, manifests, ProGuard rules, and Robolectric config",
    ),
    CiRule(
        category="linux-android-native-only",
        pattern=re.compile(
            r"(?:^kggml-runtime-(?:vulkan|opencl)/native/|"
            r"^kggml-runtime/native/cmake/toolchains/aarch64-linux-gnu\.cmake$|"
            r"^kggml-[^/]+/native/jni/.*\.map$)"
        ),
        platforms=frozenset({"linux"}),
        description="Android/Linux Vulkan & OpenCL native builds, Linux ARM64 toolchain, ELF JNI version scripts",
    ),
    CiRule(
        category="windows-native-toolchain",
        pattern=re.compile(r"^kggml-runtime/native/cmake/toolchains/arm64-windows-llvm\.cmake$"),
        platforms=frozenset({"windows"}),
        description="Windows ARM64 LLVM cross-compile toolchain",
    ),
    CiRule(
        category="docs-snippets-sources",
        pattern=re.compile(r"^docs-snippets/src/"),
        platforms=frozenset({"linux"}),
        description="README snippet Kotlin sources (compiled by :docs-snippets in Linux ./gradlew check)",
    ),
    CiRule(
        category="kotlin-shared-and-jvm",
        pattern=re.compile(
            r"(?:\.(?:kt|java)$|"
            r"^(?:.+/)?src/(?:common|jvm|jvmCommon|jni|desktop)(?:Main|Test)/|"
            r"^integration-tests/src/)"
        ),
        platforms=frozenset(PLATFORMS),
        description="Shared KMP, JVM, JNI, Desktop, build-logic, and integration-test Kotlin/Java sources",
    ),
    CiRule(
        category="native-cpp-cmake-submodules",
        pattern=re.compile(
            r"(?:\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx|inl|cl|comp|glsl|s|S|asm|cmake)$|"
            r"\.cmake\.in$|"
            r"(?:^|/)CMakeLists\.txt$|"
            r"(?:^|/)CMakePresets\.json$|"
            r"^\.gitmodules$|"
            r"^kggml-[^/]+/native/)"
        ),
        platforms=frozenset(PLATFORMS),
        description="Cross-platform native C/C++/asm sources, headers, CMake, presets, and core submodules",
    ),
    CiRule(
        category="gradle-build-system",
        pattern=re.compile(r"(?:\.(?:kts|gradle|groovy|properties|xml)$|^gradle/|^build-logic/|^gradlew$)"),
        platforms=frozenset(PLATFORMS),
        description="Gradle build scripts, convention plugins, version catalog, wrapper, and properties",
    ),
)

# --------------------------------------------------------------------------------------------------------------
# Layer 2: Repo-hygiene audit rules (additive: a file may require multiple hygiene checks)
# --------------------------------------------------------------------------------------------------------------

RE_REPO_HYGIENE_WF = re.compile(r"^\.github/workflows/repo-hygiene\.yml$")

RE_AGENT_PARITY = re.compile(
    r"^(?:(?:AGENTS|CLAUDE|GEMINI)\.md|"
    r"\.(?:agents|antigravity|claude|codex|gemini)/.*|"
    r"scripts/(?:verify-agent-parity|sync-agent-mirrors)\.sh|"
    r"scripts/ci/python\.sh)$"
)

RE_WORKFLOW_AUDIT = re.compile(
    r"^(?:\.github/workflows/.*\.ya?ml|"
    r"scripts/ci/verify-hosted-runners\.sh|"
    r"scripts/tests/test-verify-hosted-runners\.sh)$"
)

# Matches the data/binary extension list audited by scripts/ci/verify-licenses.sh.
RE_DATA_BINARY_EXT = re.compile(
    r"(?i)\.(?:bin|gguf|ggml|safetensors|ckpt|pt|pth|onnx|tflite|mlmodel|mlpackage|"
    r"npy|npz|wav|pcm|raw|f32|mp3|flac|ogg|m4a|png|jpe?g|gif|webp|bmp|ico|icns|"
    r"zip|jar|aar|tar|gz|tgz|xz|7z)$"
)

RE_LICENSE_AUDIT = re.compile(
    r"(?:^(?:LICENSE|NOTICE|\.gitmodules)$|"
    r"^kggml-[^/]+/native/third_party/|"
    r"(?:^|/)CMakeLists\.txt$|"
    r"\.cmake$|"
    r"^scripts/ci/verify-licenses\.sh$|"
    r"^scripts/tests/test-verify-licenses\.sh$)"
)

RE_DOCS_AUDIT = re.compile(
    r"(?:(?i:\.md$)|"
    r"^(?:NOTICE|kggml-bom/build\.gradle\.kts|gradle\.properties)$|"
    r"^docs-snippets/|"
    r"^kggml-[a-z0-9-]+/src/[A-Za-z0-9]*Main/.*\.kt$|"
    r"^scripts/ci/verify[-_]docs\.(?:sh|py)$|"
    r"^scripts/tests/test-verify-docs\.sh$)"
)

SCRIPT_TEST_PATTERNS: dict[str, re.Pattern[str]] = {
    "test_wait_pr_checks": re.compile(r"^scripts/(?:ci/wait-pr-checks|tests/test-wait-pr-checks)\.sh$"),
    "test_verify_release": re.compile(r"^scripts/(?:ci/verify_release\.py|ci/python\.sh|tests/test-verify-release\.sh)$"),
    "test_check_native_arch": re.compile(
        r"^scripts/(?:ci/check_native_arch\.py|ci/python\.sh|tests/test-check-native-arch\.sh)$"
    ),
    "test_audit_history": re.compile(r"^scripts/(?:ci/audit-history|tests/test-audit-history)\.sh$"),
    "test_audit_binary_symbols": re.compile(
        r"^scripts/(?:audit-binary-symbols|tests/test-audit-binary-symbols)\.sh$"
    ),
    "test_classify_changes": re.compile(
        r"^(?:scripts/ci/classify_changes\.py|scripts/tests/test-classify-changes\.sh|"
        r"\.github/workflows/(?:ci|repo-hygiene)\.yml)$"
    ),
}


def classify_file(change: ChangedFile) -> tuple[str, frozenset[str]]:
    """Returns (category, ci_platforms) for a single changed file."""
    for rule in CI_RULES:
        if rule.pattern.search(change.path):
            return rule.category, rule.platforms
    return "unclassified-failsafe", frozenset(PLATFORMS)


def classify_changes(changes: Iterable[ChangedFile], force_all: bool = False) -> dict:
    """Classifies a sequence of ChangedFile objects into workflow outputs and per-file details."""
    outputs: dict[str, bool] = {key: force_all for key in ALL_OUTPUT_KEYS}
    details: list[dict] = []

    if force_all:
        return {"outputs": outputs, "files": details, "forced_all": True}

    for change in changes:
        category, platforms = classify_file(change)
        triggers: list[str] = []

        for platform in PLATFORMS:
            if platform in platforms:
                outputs[f"ci_{platform}"] = True
                triggers.append(f"ci:{platform}")

        is_hygiene_wf = bool(RE_REPO_HYGIENE_WF.search(change.path))
        is_unclassified = category == "unclassified-failsafe"

        if is_hygiene_wf or RE_AGENT_PARITY.search(change.path):
            outputs["agent_parity"] = True
            triggers.append("agent-parity")

        if is_hygiene_wf or RE_WORKFLOW_AUDIT.search(change.path):
            outputs["workflow_audit"] = True
            triggers.append("workflow-audit")

        if (
            is_hygiene_wf
            or is_unclassified
            or change.status == "D"
            or change.is_binary
            or RE_LICENSE_AUDIT.search(change.path)
            or RE_DATA_BINARY_EXT.search(change.path)
        ):
            outputs["license_audit"] = True
            triggers.append("license-audit")

        if is_hygiene_wf or is_unclassified or change.status == "D" or RE_DOCS_AUDIT.search(change.path):
            outputs["docs_audit"] = True
            triggers.append("docs-audit")

        for test_key, pattern in SCRIPT_TEST_PATTERNS.items():
            if is_hygiene_wf or pattern.search(change.path):
                outputs[test_key] = True
                triggers.append(test_key.replace("_", "-"))

        details.append(
            {
                "path": change.path,
                "status": change.status,
                "binary": change.is_binary,
                "category": category,
                "triggers": triggers,
            }
        )

    outputs["ci_any"] = outputs["ci_linux"] or outputs["ci_macos"] or outputs["ci_windows"]
    outputs["script_tests"] = any(outputs[k] for k in SCRIPT_TEST_KEYS)
    return {"outputs": outputs, "files": details, "forced_all": False}


# --------------------------------------------------------------------------------------------------------------
# Git diff reader
# --------------------------------------------------------------------------------------------------------------


def git_changed_files(repo: Path, base: str, head: str) -> list[ChangedFile]:
    """Reads changed files, status (A/M/D), and binary flags between base...head."""
    diff_range = f"{base}...{head}"
    status_proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-status", "--no-renames", "-z", diff_range],
        check=True,
        capture_output=True,
    )
    numstat_proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "--numstat", "--no-renames", "-z", diff_range],
        check=True,
        capture_output=True,
    )

    binary_paths: set[str] = set()
    for entry in numstat_proc.stdout.split(b"\0"):
        if not entry:
            continue
        parts = entry.decode("utf-8", errors="surrogateescape").split("\t", 2)
        if len(parts) == 3 and parts[0] == "-" and parts[1] == "-":
            binary_paths.add(parts[2])

    tokens = status_proc.stdout.split(b"\0")
    changes: list[ChangedFile] = []
    i = 0
    while i + 1 < len(tokens):
        raw_status = tokens[i].decode("utf-8", errors="surrogateescape")
        if not raw_status:
            i += 1
            continue
        path = tokens[i + 1].decode("utf-8", errors="surrogateescape")
        changes.append(ChangedFile(path=path, status=raw_status[0], is_binary=path in binary_paths))
        i += 2
    return changes


def parse_path_specs(specs: Iterable[str]) -> list[ChangedFile]:
    """Parses CLI path specs: `path`, `STATUS\\tpath`, or `STATUS\\tpath\\tbinary`."""
    changes: list[ChangedFile] = []
    for raw in specs:
        line = raw.rstrip("\r\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 1:
            changes.append(ChangedFile(path=parts[0]))
        elif len(parts) == 2:
            changes.append(ChangedFile(path=parts[1], status=parts[0]))
        else:
            changes.append(ChangedFile(path=parts[1], status=parts[0], is_binary=parts[2].lower() in ("1", "true", "binary")))
    return changes


# --------------------------------------------------------------------------------------------------------------
# Repository self-audit (--verify-repo): ensures zero unclassified tracked files & 100% parity with ci.yml paths
# --------------------------------------------------------------------------------------------------------------


def gha_glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translates a GitHub Actions `paths` glob pattern to an anchored Python regex."""
    i = 0
    n = len(pattern)
    out: list[str] = ["^"]
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                if i + 2 < n and pattern[i + 2] == "/":
                    # `**/` matches zero or more directories (including repo root).
                    out.append("(?:.*/)?")
                    i += 3
                else:
                    out.append(".*")
                    i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "[":
            j = i + 1
            if j < n and pattern[j] in ("!", "^"):
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(ch))
                i += 1
            else:
                inner = pattern[i + 1 : j]
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                out.append(f"[{inner}]")
                i = j + 1
        else:
            out.append(re.escape(ch))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def matches_gha_paths(path: str, patterns: list[str]) -> bool:
    """Evaluates an ordered GitHub Actions `paths:` list (including `!` negations) against one path."""
    matched = False
    for pat in patterns:
        if pat.startswith("!"):
            if gha_glob_to_regex(pat[1:]).match(path):
                matched = False
        else:
            if gha_glob_to_regex(pat).match(path):
                matched = True
    return matched


def extract_pull_request_paths(workflow_path: Path) -> list[str]:
    """Extracts the `on.pull_request.paths` list from a workflow YAML without needing PyYAML."""
    lines = workflow_path.read_text(encoding="utf-8").splitlines()
    in_on = False
    in_pr = False
    in_paths = False
    paths_indent = 0
    patterns: list[str] = []

    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent == 0:
            in_on = stripped == "on:"
            in_pr = False
            in_paths = False
            continue
        if not in_on:
            continue
        if indent == 2:
            in_pr = stripped == "pull_request:"
            in_paths = False
            continue
        if not in_pr:
            continue
        if indent == 4:
            in_paths = stripped == "paths:"
            paths_indent = indent
            continue
        if in_paths:
            if indent <= paths_indent:
                in_paths = False
                continue
            m = re.match(r"^\s*-\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))\s*(?:#.*)?$", raw)
            if m:
                patterns.append(m.group(1) or m.group(2) or m.group(3))
    return patterns


def verify_repo(repo: Path) -> int:
    """Audits every tracked file in the repository and workflow wiring for consistency."""
    errors: list[str] = []

    tracked = [
        line
        for line in subprocess.run(
            ["git", "-C", str(repo), "ls-files"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
        if line
    ]
    if not tracked:
        print("classify_changes: no tracked files found", file=sys.stderr)
        return 1

    ci_wf = repo / ".github/workflows/ci.yml"
    ci_paths = extract_pull_request_paths(ci_wf) if ci_wf.is_file() else []
    if not ci_paths:
        errors.append(f"{ci_wf}: missing on.pull_request.paths filter")

    unclassified: list[str] = []
    missing_from_ci_paths: list[str] = []
    extra_in_ci_paths: list[str] = []

    for rel in tracked:
        category, platforms = classify_file(ChangedFile(path=rel))
        if category == "unclassified-failsafe":
            unclassified.append(rel)
        needs_ci = bool(platforms)
        if ci_paths:
            matched_ci = matches_gha_paths(rel, ci_paths)
            if needs_ci and not matched_ci:
                missing_from_ci_paths.append(f"{rel} ({category} -> {sorted(platforms)})")
            elif not needs_ci and matched_ci:
                extra_in_ci_paths.append(f"{rel} ({category})")

    if unclassified:
        errors.append(f"{len(unclassified)} tracked file(s) fell into unclassified-failsafe: {unclassified[:10]}")
    if missing_from_ci_paths:
        errors.append(
            "tracked file(s) need ci.yml per classify_changes.py but are NOT matched by ci.yml on.pull_request.paths: "
            + ", ".join(missing_from_ci_paths[:10])
        )
    if extra_in_ci_paths:
        errors.append(
            "tracked file(s) do NOT need ci.yml per classify_changes.py but ARE matched by ci.yml on.pull_request.paths: "
            + ", ".join(extra_in_ci_paths[:10])
        )

    # Verify every `needs.classify.outputs.<key>` referenced in workflows is emitted by this classifier.
    known_keys = set(ALL_OUTPUT_KEYS)
    for wf_rel in (".github/workflows/ci.yml", ".github/workflows/repo-hygiene.yml"):
        wf_file = repo / wf_rel
        if not wf_file.is_file():
            continue
        text = wf_file.read_text(encoding="utf-8")
        for m in re.finditer(r"needs\.classify\.outputs\.([a-z0-9_]+)", text):
            key = m.group(1)
            if key not in known_keys:
                errors.append(f"{wf_rel}: references unknown classify output '{key}'")

    if errors:
        for msg in errors:
            print(f"classify_changes: FAIL: {msg}", file=sys.stderr)
        return 1

    print(
        f"classify_changes: OK ({len(tracked)} tracked files explicitly classified; "
        f"100% parity with {len(ci_paths)} ci.yml paths patterns)"
    )
    return 0


# --------------------------------------------------------------------------------------------------------------
# Output formatters (GitHub Actions outputs & Markdown job summary)
# --------------------------------------------------------------------------------------------------------------


def write_github_output(path: Path, outputs: dict[str, bool]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for key in ALL_OUTPUT_KEYS:
            fh.write(f"{key}={'true' if outputs[key] else 'false'}\n")


def write_summary(path: Path, report: dict) -> None:
    outputs = report["outputs"]
    lines = ["### Change Classification", ""]
    if report["forced_all"]:
        lines.append("Manual dispatch (`--all`): all CI platforms and repo-hygiene checks enabled.")
        lines.append("")
    else:
        lines.extend(
            [
                "| File | Status | Category | Triggered Checks |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in report["files"]:
            triggers = ", ".join(f"`{t}`" for t in item["triggers"]) or "*(secret-scan only)*"
            flag = f"{item['status']} (binary)" if item["binary"] else item["status"]
            lines.append(f"| `{item['path']}` | `{flag}` | `{item['category']}` | {triggers} |")
        if not report["files"]:
            lines.append("| *(no changed files)* | — | — | *(none)* |")
        lines.append("")

    lines.extend(
        [
            "**Selected jobs:** "
            + (
                ", ".join(f"`{k}`" for k in ALL_OUTPUT_KEYS if outputs[k] and k not in ("ci_any", "script_tests"))
                or "*(secret-scan only)*"
            ),
            "",
        ]
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]), help="repository root")
    parser.add_argument("--base", help="base git ref/SHA (used with --head)")
    parser.add_argument("--head", help="head git ref/SHA (used with --base)")
    parser.add_argument("--paths", nargs="*", help="explicit changed file paths (or STATUS\\tpath[\\tbinary])")
    parser.add_argument("--paths-stdin", action="store_true", help="read path specs from stdin (one per line)")
    parser.add_argument("--all", action="store_true", dest="force_all", help="enable every output (workflow_dispatch)")
    parser.add_argument("--github-output", help="append key=true|false lines to this file ($GITHUB_OUTPUT)")
    parser.add_argument("--summary", help="append Markdown classification table to this file ($GITHUB_STEP_SUMMARY)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="print JSON report to stdout")
    parser.add_argument("--verify-repo", action="store_true", help="audit all tracked files and ci.yml paths parity")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()

    if args.verify_repo:
        return verify_repo(repo)

    if args.force_all:
        report = classify_changes([], force_all=True)
    elif args.paths is not None:
        report = classify_changes(parse_path_specs(args.paths))
    elif args.paths_stdin:
        report = classify_changes(parse_path_specs(sys.stdin))
    elif args.base and args.head:
        report = classify_changes(git_changed_files(repo, args.base, args.head))
    else:
        parser.error("specify --verify-repo, --all, --base/--head, --paths, or --paths-stdin")

    if args.github_output:
        write_github_output(Path(args.github_output), report["outputs"])
    if args.summary:
        write_summary(Path(args.summary), report)

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        for key in ALL_OUTPUT_KEYS:
            print(f"{key}={'true' if report['outputs'][key] else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
