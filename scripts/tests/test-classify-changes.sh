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
# 3. Platform-specific routing & step flags in ci.yml
# ------------------------------------------------------------------------------
expect_exact(
    "Apple/iOS Swift + Xcode app -> macOS + samples_packaging (no native rebuild)",
    [
        "samples/kggml-showcase/iosApp/iosApp/AppDelegate.swift",
        "samples/kggml-showcase/iosApp/iosApp.xcodeproj/project.pbxproj",
    ],
    {"ci_macos", "ci_samples_packaging"},
)
expect_exact(
    "iOS library main Kotlin -> macOS + docs_audit (no native rebuild, no samples packaging)",
    ["kggml-whisper/src/iosMain/kotlin/com/donfuldauer/kggml/whisper/IosWhisperBridge.kt"],
    {"ci_macos", "docs_audit"},
)
expect_exact(
    "iOS cinterop .def only -> macOS only (uses cached XCFrameworks)",
    ["kggml-runtime/src/nativeInterop/cinterop/kggmlRuntime.def"],
    {"ci_macos"},
)
expect_exact(
    "iOS xcframework build script + macos setup -> macOS + native + macos_jvm + samples_packaging",
    [
        "kggml-runtime/native/scripts/build-ios-xcframework.sh",
        "scripts/ci/setup/macos.sh",
    ],
    {"ci_macos", "ci_native", "ci_macos_jvm", "ci_samples_packaging"},
)
expect_exact(
    "scripts/ios/build-sample-apps.sh -> macOS + samples_packaging (no native rebuild)",
    ["scripts/ios/build-sample-apps.sh"],
    {"ci_macos", "ci_samples_packaging"},
)
expect_exact(
    "Windows setup + arm64 cmd -> Windows + native + samples_packaging",
    [
        "scripts/ci/setup/windows.ps1",
        "kggml-runtime/native/scripts/build-windows-arm64.cmd",
    ],
    {"ci_windows", "ci_native", "ci_samples_packaging"},
)
expect_exact(
    "gradlew.bat only -> Windows only (no native rebuild)",
    ["gradlew.bat"],
    {"ci_windows"},
)
expect_exact(
    "Windows arm64 toolchain cmake -> Windows + native + samples_packaging + license_audit",
    ["kggml-runtime/native/cmake/toolchains/arm64-windows-llvm.cmake"],
    {"ci_windows", "ci_native", "ci_samples_packaging", "license_audit"},
)
expect_exact(
    "Android sample + manifest + ProGuard -> Linux + samples_packaging (no native rebuild)",
    [
        "samples/kggml-showcase/src/androidMain/AndroidManifest.xml",
        "samples/kggml-showcase/src/androidUnitTest/resources/robolectric.properties",
        "kggml-whisper/consumer-rules.pro",
    ],
    {"ci_linux", "ci_samples_packaging"},
)
expect_exact(
    "Android/Linux native build & audit scripts -> Linux + native + samples_packaging",
    [
        "kggml-whisper/native/scripts/build-android.sh",
        "scripts/ci/setup/linux.sh",
        "scripts/ci/audit-android-linkage.sh",
        "scripts/ci/audit-gpu-addon.sh",
        "kggml-llama/native/jni/kggml_llama_jni.map",
    ],
    {"ci_linux", "ci_native", "ci_samples_packaging"},
)
expect_exact(
    "Android library androidMain Kotlin -> Linux + docs_audit (no native rebuild)",
    ["kggml-runtime/src/androidMain/kotlin/com/donfuldauer/kggml/runtime/AndroidRuntimeLoader.kt"],
    {"ci_linux", "docs_audit"},
)
expect_exact(
    "Vulkan/OpenCL native CMake + submodules -> Linux + native + samples_packaging + license_audit",
    [
        "kggml-runtime-vulkan/native/CMakeLists.txt",
        "kggml-runtime-vulkan/native/third_party/Vulkan-Headers",
        "kggml-runtime-opencl/native/third_party/OpenCL-Headers",
    ],
    {"ci_linux", "ci_native", "ci_samples_packaging", "license_audit"},
)
expect_exact(
    "docs-snippets Kotlin source -> Linux + docs_audit (no native rebuild)",
    ["docs-snippets/src/commonMain/kotlin/com/donfuldauer/kggml/docs/WhisperSnippets.kt"],
    {"ci_linux", "docs_audit"},
)
expect_exact(
    "host build + linkage audit -> Linux + macOS + native + macos_jvm + samples_packaging",
    [
        "kggml-runtime/native/scripts/build-host.sh",
        "scripts/ci/audit-host-linkage.sh",
    ],
    {"ci_linux", "ci_macos", "ci_native", "ci_macos_jvm", "ci_samples_packaging"},
)
expect_exact(
    "stage-release.sh only -> Linux + macOS + release_dry_run + samples_packaging (no native rebuild, no macos_jvm)",
    ["scripts/ci/stage-release.sh"],
    {"ci_linux", "ci_macos", "ci_release_dry_run", "ci_samples_packaging"},
)
expect_exact(
    "verify_release.py -> Linux + macOS + release_dry_run + samples_packaging + test_verify_release",
    ["scripts/ci/verify_release.py"],
    {"ci_linux", "ci_macos", "ci_release_dry_run", "ci_samples_packaging", "test_verify_release"},
)
expect_exact(
    "check_native_arch.py -> Linux + Windows + native + samples_packaging + test_check_native_arch",
    ["scripts/ci/check_native_arch.py"],
    {"ci_linux", "ci_windows", "ci_native", "ci_samples_packaging", "test_check_native_arch"},
)
expect_exact(
    "python.sh -> Linux + Windows + agent_parity + test_verify_release + test_check_native_arch",
    ["scripts/ci/python.sh"],
    {"ci_linux", "ci_windows", "agent_parity", "test_verify_release", "test_check_native_arch"},
)
expect_exact(
    "classify_changes.py -> Linux + test_classify_changes (no native rebuild)",
    ["scripts/ci/classify_changes.py"],
    {"ci_linux", "test_classify_changes"},
)

# ------------------------------------------------------------------------------
# 4. Common Kotlin vs JVM/JNI/Desktop, native C++, Gradle, submodules, ci.yml
# ------------------------------------------------------------------------------
expect_exact(
    "library commonMain Kotlin -> Linux + macOS + docs_audit (drops Windows, skips native/release/packaging)",
    ["kggml-core-api/src/commonMain/kotlin/com/donfuldauer/kggml/core/BackendConfig.kt"],
    {"ci_linux", "ci_macos", "docs_audit"},
)
expect_exact(
    "commonTest Kotlin only -> Linux + macOS only (drops Windows, skips native/release/packaging)",
    ["kggml-core-api/src/commonTest/kotlin/com/donfuldauer/kggml/core/BackendConfigTest.kt"],
    {"ci_linux", "ci_macos"},
)
expect_exact(
    "sample commonMain Kotlin -> Linux + macOS + samples_packaging (drops Windows, skips native/release)",
    ["samples/kggml-showcase/src/commonMain/kotlin/com/donfuldauer/kggml/samples/showcase/ShowcaseApp.kt"],
    {"ci_linux", "ci_macos", "ci_samples_packaging"},
)
expect_exact(
    "library jvmMain / jniMain Kotlin -> all 3 platforms + macos_jvm + docs_audit (skips native/release/packaging)",
    [
        "kggml-runtime/src/jvmMain/kotlin/com/donfuldauer/kggml/runtime/JvmNativeLibraryLoader.kt",
        "kggml-runtime/src/jniMain/kotlin/com/donfuldauer/kggml/runtime/JniRuntimeBridge.kt",
    ],
    {"ci_linux", "ci_macos", "ci_windows", "ci_macos_jvm", "docs_audit"},
)
expect_exact(
    "integration-tests Kotlin -> all 3 platforms + macos_jvm (skips native/release/packaging)",
    ["integration-tests/src/jvmTest/kotlin/com/donfuldauer/kggml/integration/EngineCoexistenceTest.kt"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_macos_jvm"},
)
expect_exact(
    "sample desktopMain Kotlin -> all 3 platforms + macos_jvm + samples_packaging (skips native/release)",
    ["samples/kggml-showcase/src/desktopMain/kotlin/com/donfuldauer/kggml/samples/showcase/Main.kt"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_macos_jvm", "ci_samples_packaging"},
)
expect_exact(
    "native C++ source edit -> all 3 platforms + native + macos_jvm + samples_packaging (skips release_dry_run)",
    ["kggml-diffusion/native/src/kggml_diffusion.cpp"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_native", "ci_macos_jvm", "ci_samples_packaging"},
)
expect_exact(
    "core CMakeLists.txt + ggml submodule -> all 3 platforms + native + macos_jvm + samples_packaging + license_audit",
    ["kggml-runtime/native/CMakeLists.txt", "kggml-runtime/native/third_party/ggml"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_native", "ci_macos_jvm", "ci_samples_packaging", "license_audit"},
)
expect_exact(
    "build-logic Kotlin convention plugin -> all 3 platforms + macos_jvm + release_dry_run + samples_packaging (skips native)",
    ["build-logic/convention/src/main/kotlin/com/donfuldauer/kggml/buildlogic/KggmlPublishingPlugin.kt"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_macos_jvm", "ci_release_dry_run", "ci_samples_packaging"},
)
expect_exact(
    "kggml-bom/build.gradle.kts + gradle.properties -> all 3 platforms + macos_jvm + release_dry_run + samples_packaging + docs_audit",
    ["kggml-bom/build.gradle.kts", "gradle.properties"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_macos_jvm", "ci_release_dry_run", "ci_samples_packaging", "docs_audit"},
)
expect_exact(
    "sample build.gradle.kts only -> all 3 platforms + macos_jvm + samples_packaging (skips release_dry_run & native)",
    ["samples/kggml-showcase/build.gradle.kts"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_macos_jvm", "ci_samples_packaging"},
)
expect_exact(
    "ci.yml edit -> all 3 platforms + all CI step flags + workflow_audit + test_classify_changes",
    [".github/workflows/ci.yml"],
    {
        "ci_linux",
        "ci_macos",
        "ci_windows",
        "ci_native",
        "ci_macos_jvm",
        "ci_release_dry_run",
        "ci_samples_packaging",
        "workflow_audit",
        "test_classify_changes",
    },
)
expect_exact(
    "scripts/audit-binary-symbols.sh -> all 3 platforms + macos_jvm + samples_packaging + test_audit_binary_symbols",
    ["scripts/audit-binary-symbols.sh"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_macos_jvm", "ci_samples_packaging", "test_audit_binary_symbols"},
)

# ------------------------------------------------------------------------------
# 5. Deletions, binaries, unknown fail-safe, --all, native_cache_key
# ------------------------------------------------------------------------------
expect_exact(
    "deleted native C++ file triggers native + docs_audit (links) + license_audit (NOTICE fixtures)",
    ["D\tkggml-whisper/native/tests/tiny_model.cpp"],
    {"ci_linux", "ci_macos", "ci_windows", "ci_native", "ci_macos_jvm", "ci_samples_packaging", "docs_audit", "license_audit"},
)
expect_exact(
    "added binary file triggers license_audit even with non-standard extension",
    ["A\tdocs/diagram.svgz\tbinary"],
    {"license_audit"},
)
expect_exact(
    "unknown/unclassified file fails safe (all 3 CI platforms + all step flags + license_audit + docs_audit)",
    ["A\tsome-unknown-top-level.xyz"],
    {
        "ci_linux",
        "ci_macos",
        "ci_windows",
        "ci_native",
        "ci_macos_jvm",
        "ci_release_dry_run",
        "ci_samples_packaging",
        "license_audit",
        "docs_audit",
    },
)

all_report = run_classify("--all")
if all(all_report["outputs"].values()) and len(all_report["native_cache_key"]) == 16 and all(
    c in "0123456789abcdef" for c in all_report["native_cache_key"]
):
    ok(f"--all enables every output key and emits 16-char native_cache_key ({all_report['native_cache_key']})")
else:
    ko("--all enables every output key and native_cache_key", str(all_report))

# ------------------------------------------------------------------------------
# 6. Real git diff (--base / --head) + --github-output + --summary + native_cache_key stability
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
(repo / "kggml-runtime" / "native").mkdir(parents=True)
(repo / "kggml-runtime" / "native" / "CMakeLists.txt").write_text("# cmake v1\n", encoding="utf-8")
(repo / "kggml-runtime" / "native" / "README.md").write_text("# readme v1\n", encoding="utf-8")
git("add", "-A")
git("commit", "-q", "-m", "base")
base_sha = git("rev-parse", "HEAD").stdout.strip()
key_v1 = run_classify("--repo", str(repo), "--all")["native_cache_key"]

git("checkout", "-q", "-b", "feat")
(repo / "samples" / "app" / "iosApp" / "App.swift").write_text("// updated\n", encoding="utf-8")
(repo / "kggml-runtime" / "native" / "README.md").write_text("# readme v2\n", encoding="utf-8")
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
expected_diff = {"ci_macos", "ci_any", "ci_samples_packaging", "docs_audit", "license_audit"}
actual_diff = {k for k, v in diff_report["outputs"].items() if v}
key_v2 = diff_report["native_cache_key"]
gh_out_text = gh_out.read_text()
gh_sum_text = gh_sum.read_text()
if (
    actual_diff == expected_diff
    and key_v2 == key_v1
    and f"native_cache_key={key_v2}" in gh_out_text
    and "ci_macos=true" in gh_out_text
    and "App.swift" in gh_sum_text
    and key_v2 in gh_sum_text
):
    ok("real git diff (--base/--head) detects iOS edit + deletion + binary; native README edit preserves native_cache_key")
else:
    ko(
        "real git diff",
        f"expected={sorted(expected_diff)} actual={sorted(actual_diff)} key_v1={key_v1} key_v2={key_v2}",
    )

# Modifying a real native CMake file MUST change native_cache_key.
(repo / "kggml-runtime" / "native" / "CMakeLists.txt").write_text("# cmake v2\n", encoding="utf-8")
git("add", "-A")
git("commit", "-q", "-m", "native change")
key_v3 = run_classify("--repo", str(repo), "--all")["native_cache_key"]
if key_v3 != key_v1 and len(key_v3) == 16:
    ok(f"native CMakeLists.txt change rotates native_cache_key ({key_v1} -> {key_v3})")
else:
    ko("native CMakeLists.txt change rotates native_cache_key", f"key_v1={key_v1} key_v3={key_v3}")

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

# Unknown top-level files must still trigger ci.yml at the workflow level so classify can fail safe.
sys.path.insert(0, str(classifier.parent))
import classify_changes as cc

ci_paths = cc.extract_pull_request_paths(repo_root / ".github" / "workflows" / "ci.yml")
if cc.matches_gha_paths("some-unknown-top-level.xyz", ci_paths) and not cc.matches_gha_paths("docs/RELEASING.md", ci_paths):
    ok("ci.yml on.pull_request.paths lets unknown paths reach classify while skipping docs")
else:
    ko("ci.yml on.pull_request.paths unknown-path fallback")

# Non-ASCII filenames must not be quoted by git ls-files in --verify-repo.
r_unicode = make_verify_repo(
    "unicode-md",
    {"docs/café.md": "# Café\n"},
    """on:
  pull_request:
    paths:
      - "**"
      - "!**/*.md"
      - "!docs/**"
""",
)
if subprocess.run([sys.executable, str(classifier), "--verify-repo", "--repo", str(r_unicode)], capture_output=True).returncode == 0:
    ok("--verify-repo handles non-ASCII tracked filenames via git ls-files -z")
else:
    ko("--verify-repo handles non-ASCII tracked filenames via git ls-files -z")

print(f"\nclassify-changes tests: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
PY
