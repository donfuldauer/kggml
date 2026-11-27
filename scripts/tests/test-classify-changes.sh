#!/usr/bin/env bash
# Tests for scripts/ci/classify_changes.py:
#   1. Verifies every single tracked file in this repository is explicitly classified (zero unclassified-failsafe
#      hits) and matches .github/workflows/ci.yml's on.pull_request.paths filter with 100% parity.
#   2. Exercises every classification category, file extension regex, platform routing (linux / macos / windows),
#      repo-hygiene audit routing, script-test routing, deletion/rename handling, binary detection, unknown-file
#      fail-safe, --all mode, --github-output, --summary, real git diff --base/--head, and --verify-repo failures.
#
# Usage: scripts/tests/test-classify-changes.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLASSIFIER="${REPO_ROOT}/scripts/ci/classify_changes.py"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

python3 - "${REPO_ROOT}" "${CLASSIFIER}" "${WORK}" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
classifier = Path(sys.argv[2])
work = Path(sys.argv[3])

passed = 0
failed = 0


def ok(name: str) -> None:
    global passed
    print(f"  ok   {name}")
    passed += 1


def ko(name: str, detail: str = "") -> None:
    global failed
    msg = f"  FAIL {name}" + (f": {detail}" if detail else "")
    print(msg, file=sys.stderr)
    failed += 1


def run_classify(*args: str, stdin: str | None = None) -> dict:
    proc = subprocess.run(
        [sys.executable, str(classifier), *args, "--json"],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def expect_exact(name: str, path_specs: list[str], expected_true: set[str]) -> None:
    report = run_classify("--paths", *path_specs)
    actual_true = {k for k, v in report["outputs"].items() if v}
    expected = set(expected_true)
    if expected & {"ci_linux", "ci_macos", "ci_windows"}:
        expected.add("ci_any")
    if any(k.startswith("test_") for k in expected):
        expected.add("script_tests")
    if actual_true == expected:
        ok(name)
    else:
        ko(name, f"expected={sorted(expected)} actual={sorted(actual_true)}")


# ------------------------------------------------------------------------------
# 1. Whole-repo audit (--verify-repo on this repository)
# ------------------------------------------------------------------------------
verify = subprocess.run(
    [sys.executable, str(classifier), "--verify-repo", "--repo", str(repo_root)],
    capture_output=True,
    text=True,
)
if verify.returncode == 0:
    ok(f"this repository --verify-repo ({verify.stdout.strip()})")
else:
    ko("this repository --verify-repo", verify.stderr.strip())

# ------------------------------------------------------------------------------
# 2. Pure docs, workflow docs, agent mirrors, repo metadata, non-CI workflows
# ------------------------------------------------------------------------------
expect_exact(
    "workflow docs only (docs/RELEASING.md)",
    ["docs/RELEASING.md"],
    {"docs_audit"},
)
expect_exact(
    "public README.md only",
    ["README.md"],
    {"docs_audit"},
)
expect_exact(
    "module native README.md only",
    ["kggml-runtime/native/README.md"],
    {"docs_audit"},
)
expect_exact(
    "agent instructions + mirrors + helper script (PR #110 scenario)",
    [
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        ".gemini/GEMINI.md",
        ".agents/agents/orchestrator.md",
        ".agents/skills/work-on-issue/SKILL.md",
        ".codex/agents/orchestrator.toml",
        "CONTRIBUTING.md",
        ".github/ISSUE_TEMPLATE/engineering-task.md",
        "A\tscripts/issue-status.sh",
    ],
    {"agent_parity", "docs_audit"},
)
expect_exact(
    "single .gemini/GEMINI.md mirror triggers agent_parity + docs_audit",
    [".gemini/GEMINI.md"],
    {"agent_parity", "docs_audit"},
)
expect_exact(
    "single .codex/agents/*.toml triggers agent_parity only",
    [".codex/agents/orchestrator.toml"],
    {"agent_parity"},
)
expect_exact(
    "single .agents/skills/ SKILL.md triggers agent_parity + docs_audit",
    [".agents/skills/work-on-issue/SKILL.md"],
    {"agent_parity", "docs_audit"},
)
expect_exact(
    "dependabot.yml only (PR #108 scenario)",
    ["A\t.github/dependabot.yml"],
    set(),
)
expect_exact(
    "repo metadata (.coderabbit.yaml, .gitignore, .gitleaks.toml, .github/labeler.yml, .githooks)",
    [
        ".coderabbit.yaml",
        ".gitignore",
        ".gitleaks.toml",
        ".github/labeler.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".githooks/commit-msg",
    ],
    set(),
)
expect_exact(
    "non-ci workflow edit (stale.yml, label.yml, release.yml)",
    [".github/workflows/stale.yml", ".github/workflows/label.yml", ".github/workflows/release.yml"],
    {"workflow_audit"},
)
expect_exact(
    "repo-hygiene.yml edit runs all hygiene audits & script tests, not ci.yml",
    [".github/workflows/repo-hygiene.yml"],
    {
        "agent_parity",
        "workflow_audit",
        "license_audit",
        "docs_audit",
        "test_wait_pr_checks",
        "test_verify_release",
        "test_check_native_arch",
        "test_audit_history",
        "test_audit_binary_symbols",
        "test_classify_changes",
    },
)
expect_exact(
    "LICENSE only",
    ["LICENSE"],
    {"license_audit"},
)
expect_exact(
    "NOTICE only",
    ["NOTICE"],
    {"license_audit", "docs_audit"},
)
expect_exact(
    "local dev scripts only (ci-local.sh, install-git-hooks.sh, coverage.sh, run-android-instrumented.sh)",
    [
        "scripts/ci-local.sh",
        "scripts/install-git-hooks.sh",
        "scripts/ci/coverage.sh",
        "scripts/ci/run-android-instrumented.sh",
        "scripts/ci/upload-central-bundle.sh",
    ],
    set(),
)
expect_exact(
    "hygiene audit scripts trigger only their own audit",
    ["scripts/ci/verify_docs.py"],
    {"docs_audit"},
)
expect_exact(
    "verify-licenses.sh triggers only license_audit",
    ["scripts/ci/verify-licenses.sh"],
    {"license_audit"},
)
expect_exact(
    "verify-hosted-runners.sh triggers only workflow_audit",
    ["scripts/ci/verify-hosted-runners.sh"],
    {"workflow_audit"},
)
expect_exact(
    "wait-pr-checks.sh triggers only test_wait_pr_checks",
    ["scripts/ci/wait-pr-checks.sh"],
    {"test_wait_pr_checks"},
)
expect_exact(
    "audit-history.sh triggers only test_audit_history",
    ["scripts/ci/audit-history.sh"],
    {"test_audit_history"},
)

# ------------------------------------------------------------------------------
# 3. Platform-specific routing in ci.yml
# ------------------------------------------------------------------------------
expect_exact(
    "Apple/iOS Swift + Xcode app -> macOS only",
    [
        "samples/kggml-showcase/iosApp/iosApp/AppDelegate.swift",
        "samples/kggml-showcase/iosApp/iosApp.xcodeproj/project.pbxproj",
    ],
    {"ci_macos"},
)
expect_exact(
    "iOS library main Kotlin -> macOS + docs_audit (KDoc)",
    ["kggml-whisper/src/iosMain/kotlin/com/donfuldauer/kggml/whisper/IosWhisperBridge.kt"],
    {"ci_macos", "docs_audit"},
)
expect_exact(
    "iOS cinterop .def + xcframework build script + macos setup -> macOS only",
    [
        "kggml-runtime/src/nativeInterop/cinterop/kggmlRuntime.def",
        "kggml-runtime/native/scripts/build-ios-xcframework.sh",
        "scripts/ios/build-sample-apps.sh",
        "scripts/ci/setup/macos.sh",
    ],
    {"ci_macos"},
)
expect_exact(
    "Windows setup + arm64 cmd + gradlew.bat -> Windows only",
    [
        "scripts/ci/setup/windows.ps1",
        "kggml-runtime/native/scripts/build-windows-arm64.cmd",
        "gradlew.bat",
    ],
    {"ci_windows"},
)
expect_exact(
    "Windows arm64 toolchain cmake -> Windows + license_audit",
    ["kggml-runtime/native/cmake/toolchains/arm64-windows-llvm.cmake"],
    {"ci_windows", "license_audit"},
)
expect_exact(
    "Android sample + Android unit test + build-android.sh + linux setup -> Linux only",
    [
        "samples/kggml-showcase/src/androidMain/AndroidManifest.xml",
        "samples/kggml-showcase/src/androidUnitTest/resources/robolectric.properties",
        "kggml-whisper/consumer-rules.pro",
        "kggml-whisper/native/scripts/build-android.sh",
        "scripts/ci/setup/linux.sh",
        "scripts/ci/audit-android-linkage.sh",
        "scripts/ci/audit-gpu-addon.sh",
        "kggml-llama/native/jni/kggml_llama_jni.map",
    ],
    {"ci_linux"},
)
expect_exact(
    "Android library androidMain Kotlin -> Linux + docs_audit (KDoc)",
    ["kggml-runtime/src/androidMain/kotlin/com/donfuldauer/kggml/runtime/AndroidRuntimeLoader.kt"],
    {"ci_linux", "docs_audit"},
)
expect_exact(
    "Vulkan/OpenCL native CMake + submodules -> Linux + license_audit",
    [
        "kggml-runtime-vulkan/native/CMakeLists.txt",
        "kggml-runtime-vulkan/native/third_party/Vulkan-Headers",
        "kggml-runtime-opencl/native/third_party/OpenCL-Headers",
    ],
    {"ci_linux", "license_audit"},
)
expect_exact(
    "docs-snippets Kotlin source -> Linux + docs_audit",
    ["docs-snippets/src/commonMain/kotlin/com/donfuldauer/kggml/docs/WhisperSnippets.kt"],
    {"ci_linux", "docs_audit"},
)
expect_exact(
    "host build + linkage audit + release staging -> Linux + macOS",
    [
        "kggml-runtime/native/scripts/build-host.sh",
        "scripts/ci/audit-host-linkage.sh",
        "scripts/ci/stage-release.sh",
    ],
    {"ci_linux", "ci_macos"},
)
expect_exact(
    "verify_release.py -> Linux + macOS + test_verify_release",
    ["scripts/ci/verify_release.py"],
    {"ci_linux", "ci_macos", "test_verify_release"},
)
expect_exact(
    "check_native_arch.py -> Linux + Windows + test_check_native_arch",
    ["scripts/ci/check_native_arch.py"],
    {"ci_linux", "ci_windows", "test_check_native_arch"},
)
expect_exact(
    "python.sh -> Linux + Windows + agent_parity + test_verify_release + test_check_native_arch",
    ["scripts/ci/python.sh"],
    {"ci_linux", "ci_windows", "agent_parity", "test_verify_release", "test_check_native_arch"},
)
expect_exact(
    "classify_changes.py -> Linux + test_classify_changes",
    ["scripts/ci/classify_changes.py"],
    {"ci_linux", "test_classify_changes"},
)

# ------------------------------------------------------------------------------
# 4. Cross-platform shared code, C++, Gradle, submodules, ci.yml
# ------------------------------------------------------------------------------
expect_exact(
    "library commonMain Kotlin -> all 3 platforms + docs_audit (KDoc)",
    ["kggml-core-api/src/commonMain/kotlin/com/donfuldauer/kggml/core/BackendConfig.kt"],
    {"ci_linux", "ci_macos", "ci_windows", "docs_audit"},
)
expect_exact(
    "commonTest Kotlin only -> all 3 platforms, no docs_audit/license_audit",
    ["kggml-core-api/src/commonTest/kotlin/com/donfuldauer/kggml/core/BackendConfigTest.kt"],
    {"ci_linux", "ci_macos", "ci_windows"},
)
expect_exact(
    "native C++ source edit -> all 3 platforms, no hygiene audits",
    ["kggml-diffusion/native/src/kggml_diffusion.cpp"],
    {"ci_linux", "ci_macos", "ci_windows"},
)
expect_exact(
    "core CMakeLists.txt + ggml submodule -> all 3 platforms + license_audit",
    ["kggml-runtime/native/CMakeLists.txt", "kggml-runtime/native/third_party/ggml"],
    {"ci_linux", "ci_macos", "ci_windows", "license_audit"},
)
expect_exact(
    "kggml-bom/build.gradle.kts + gradle.properties -> all 3 platforms + docs_audit",
    ["kggml-bom/build.gradle.kts", "gradle.properties"],
    {"ci_linux", "ci_macos", "ci_windows", "docs_audit"},
)
expect_exact(
    "ci.yml edit -> all 3 platforms + workflow_audit + test_classify_changes",
    [".github/workflows/ci.yml"],
    {"ci_linux", "ci_macos", "ci_windows", "workflow_audit", "test_classify_changes"},
)
expect_exact(
    "scripts/audit-binary-symbols.sh -> all 3 platforms + test_audit_binary_symbols",
    ["scripts/audit-binary-symbols.sh"],
    {"ci_linux", "ci_macos", "ci_windows", "test_audit_binary_symbols"},
)

# ------------------------------------------------------------------------------
# 5. Deletions, binaries, unknown fail-safe, --all
# ------------------------------------------------------------------------------
expect_exact(
    "deleted file triggers docs_audit (links) and license_audit (NOTICE fixtures)",
    ["D\tkggml-whisper/native/tests/tiny_model.cpp"],
    {"ci_linux", "ci_macos", "ci_windows", "docs_audit", "license_audit"},
)
expect_exact(
    "added binary file triggers license_audit even with non-standard extension",
    ["A\tdocs/diagram.svgz\tbinary"],
    {"license_audit"},
)
expect_exact(
    "unknown/unclassified file fails safe (all 3 CI platforms + license_audit + docs_audit)",
    ["A\tsome-unknown-top-level.xyz"],
    {"ci_linux", "ci_macos", "ci_windows", "license_audit", "docs_audit"},
)

all_report = run_classify("--all")
if all(all_report["outputs"].values()):
    ok("--all enables every output key")
else:
    ko("--all enables every output key", str(all_report["outputs"]))

# ------------------------------------------------------------------------------
# 6. Real git diff (--base / --head) + --github-output + --summary
# ------------------------------------------------------------------------------
repo = work / "git-repo"
repo.mkdir()
git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)
git("init", "-q")
git("config", "user.name", "Test")
git("config", "user.email", "test@example.com")
(repo / "docs").mkdir()
(repo / "docs" / "a.md").write_text("# A\n", encoding="utf-8")
(repo / "samples" / "app" / "iosApp").mkdir(parents=True)
(repo / "samples" / "app" / "iosApp" / "App.swift").write_text("// swift\n", encoding="utf-8")
git("add", "-A")
git("commit", "-q", "-m", "base")
base_sha = git("rev-parse", "HEAD").stdout.strip()

git("checkout", "-q", "-b", "feat")
(repo / "samples" / "app" / "iosApp" / "App.swift").write_text("// updated\n", encoding="utf-8")
git("rm", "-q", "docs/a.md")
(repo / "docs").mkdir(exist_ok=True)
(repo / "docs" / "blob.dat").write_bytes(b"\x00\x01\x02\xff\xfe")
git("add", "-A")
git("commit", "-q", "-m", "pr change")
head_sha = git("rev-parse", "HEAD").stdout.strip()

gh_out = work / "github_output"
gh_sum = work / "github_summary.md"
diff_proc = subprocess.run(
    [
        sys.executable,
        str(classifier),
        "--repo",
        str(repo),
        "--base",
        base_sha,
        "--head",
        head_sha,
        "--github-output",
        str(gh_out),
        "--summary",
        str(gh_sum),
        "--json",
    ],
    check=True,
    capture_output=True,
    text=True,
)
diff_report = json.loads(diff_proc.stdout)
expected_diff = {"ci_macos", "ci_any", "docs_audit", "license_audit"}
actual_diff = {k for k, v in diff_report["outputs"].items() if v}
if actual_diff == expected_diff and "ci_macos=true" in gh_out.read_text() and "App.swift" in gh_sum.read_text():
    ok("real git diff (--base/--head) detects iOS edit + deletion + binary + outputs/summary")
else:
    ko("real git diff", f"expected={sorted(expected_diff)} actual={sorted(actual_diff)}")

# ------------------------------------------------------------------------------
# 7. Negative --verify-repo checks (unclassified file, ci.yml paths drift, output typo)
# ------------------------------------------------------------------------------
def make_verify_repo(name: str, extra_files: dict[str, str], ci_yml: str) -> Path:
    d = work / name
    (d / ".github" / "workflows").mkdir(parents=True)
    subprocess.run(["git", "-C", str(d), "init", "-q"], check=True)
    (d / ".github" / "workflows" / "ci.yml").write_text(ci_yml, encoding="utf-8")
    (d / "README.md").write_text("# Hi\n", encoding="utf-8")
    for rel, content in extra_files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
    return d


clean_ci = """on:
  pull_request:
    paths:
      - "**/*.kt"
      - ".github/workflows/ci.yml"
jobs:
  linux:
    if: ${{ needs.classify.outputs.ci_linux == 'true' }}
"""
r_unclassified = make_verify_repo("neg-unclassified", {"mystery.foo": "x"}, clean_ci)
if subprocess.run([sys.executable, str(classifier), "--verify-repo", "--repo", str(r_unclassified)], capture_output=True).returncode != 0:
    ok("--verify-repo rejects unclassified tracked file")
else:
    ko("--verify-repo rejects unclassified tracked file")

r_drift = make_verify_repo("neg-drift", {"kggml-core-api/src/commonMain/Foo.kt": "class Foo"}, """on:
  pull_request:
    paths:
      - ".github/workflows/ci.yml"
""")
if subprocess.run([sys.executable, str(classifier), "--verify-repo", "--repo", str(r_drift)], capture_output=True).returncode != 0:
    ok("--verify-repo rejects ci.yml paths missing a buildable file")
else:
    ko("--verify-repo rejects ci.yml paths missing a buildable file")

r_typo = make_verify_repo("neg-typo", {}, """on:
  pull_request:
    paths:
      - ".github/workflows/ci.yml"
jobs:
  linux:
    if: ${{ needs.classify.outputs.ci_linx == 'true' }}
""")
if subprocess.run([sys.executable, str(classifier), "--verify-repo", "--repo", str(r_typo)], capture_output=True).returncode != 0:
    ok("--verify-repo rejects unknown classify output reference in workflow")
else:
    ko("--verify-repo rejects unknown classify output reference in workflow")

print(f"\nclassify-changes tests: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
PY
