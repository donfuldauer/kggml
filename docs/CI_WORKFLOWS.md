# GitHub Actions Workflow Triggers & Content Classification

Every GitHub Actions workflow in `.github/workflows/` runs only when the changed content actually requires it.
The canonical rules live in [`scripts/ci/classify_changes.py`](../scripts/ci/classify_changes.py), which uses file-extension
and path regular expressions (plus git deletion/rename and binary detection) and is verified against every tracked file in the
repository by [`scripts/tests/test-classify-changes.sh`](../scripts/tests/test-classify-changes.sh) (`--verify-repo`).

---

## 1. Workflow Overview

| Workflow | Trigger | When It Runs |
| --- | --- | --- |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | `pull_request` (`paths:` filter) + `workflow_dispatch` | Heavy cross-platform native/JVM/Android/iOS/Windows builds, CTest, Gradle tests, symbol audits, and release dry-run. Skipped completely (0 runners) when no build/code/native/CI files change. When triggered, its `classify` job gates `linux`, `macos`, and `windows` independently. Draft PRs run nothing. |
| [`.github/workflows/repo-hygiene.yml`](../.github/workflows/repo-hygiene.yml) | `pull_request` + `workflow_dispatch` | Cheap Ubuntu repository audits. Its `classify` job gates `agent-parity`, `workflow-audit`, `license-audit`, `docs-audit`, and `script-tests` (and individual script tests inside `script-tests`). `secret-scan` runs in parallel on every ready PR because any commit, file, or commit message can introduce secrets, private keys, oversized blobs, or developer-machine paths. Draft PRs run nothing. |
| [`.github/workflows/label.yml`](../.github/workflows/label.yml) | `pull_request_target` | Applies module, CI, and documentation labels using [`.github/labeler.yml`](../.github/labeler.yml). |
| [`.github/workflows/release.yml`](../.github/workflows/release.yml) | `push` (`tags: ["v*"]`) + `workflow_dispatch` | Builds, signs, audits, and publishes the Maven Central bundle and draft GitHub release. Never runs on pull requests. |
| [`.github/workflows/stale.yml`](../.github/workflows/stale.yml) | `schedule` (daily cron) | Marks inactive issues and PRs as stale. Never triggered by file changes. |

Manual `workflow_dispatch` runs of `ci.yml` and `repo-hygiene.yml` pass `--all` to `classify_changes.py`, enabling every job.

---

## 2. Primary Content Classification (`ci.yml` Triggers & Platform Routing)

Rules are evaluated in order from most specific to most general in `scripts/ci/classify_changes.py`.
`.github/workflows/ci.yml`'s `on.pull_request.paths:` filter mirrors the union of every category that requires at least one
platform, and `--verify-repo` enforces 100% parity across all tracked files.

| Category | Extension / Path Regex | `ci.yml` Runners |
| --- | --- | --- |
| `docs-markdown` | `(?i)\.md$` | None |
| `docs-tree` | `^docs/` | None |
| `agent-harness-config` | `^\.(?:agents\|antigravity\|claude\|codex\|gemini)/` | None |
| `license-notice` | `^(?:LICENSE\|NOTICE)$` | None |
| `repo-metadata-config` | `.gitignore`, `.coderabbit.yaml`, `.gitleaks.toml`, `.githooks/**`, `.github/ISSUE_TEMPLATE/**`, `.github/dependabot.y(a)ml`, `.github/labeler.y(a)ml`, `.github/CODEOWNERS`, `.github/FUNDING.y(a)ml` | None |
| `workflow-ci` | `^\.github/workflows/ci\.yml$` | `linux`, `macos`, `windows` |
| `workflow-non-ci` | `^\.github/workflows/.*\.ya?ml$` (all other workflows) | None |
| `scripts-hygiene-and-local` | `scripts/tests/**`, `scripts/{ci-local,install-git-hooks,sync-agent-mirrors,verify-agent-parity,issue-status}.sh`, `scripts/ci/{verify-docs.sh,verify_docs.py,verify-licenses.sh,verify-hosted-runners.sh,audit-history.sh,wait-pr-checks.sh,coverage.sh,run-android-instrumented.sh,upload-central-bundle.sh}` | None |
| `scripts-ci-classifier` | `^scripts/ci/classify_changes\.py$` | `linux` |
| `scripts-ci-linux-only` | `scripts/ci/setup/linux.sh`, `scripts/ci/audit-{android-linkage,gpu-addon}.sh`, `scripts/ci/verify-{whisper,llama,diffusion}-pin.sh`, `kggml-*/native/scripts/build-android.sh`, `kggml-runtime-vulkan/native/scripts/build-host.sh` | `linux` |
| `scripts-ci-macos-only` | `scripts/ci/setup/macos.sh`, `scripts/ios/*.sh`, `kggml-*/native/scripts/build-ios-xcframework.sh` | `macos` |
| `scripts-ci-windows-only` | `scripts/ci/setup/windows.ps1`, `kggml-runtime/native/scripts/build-windows-arm64.cmd`, `gradlew.bat`, `.*\.(ps1\|cmd\|bat)$` | `windows` |
| `scripts-ci-linux-macos` | `scripts/ci/verify-ggml-pin.sh`, `scripts/ci/audit-{host,whisper,llama,diffusion}-linkage.sh`, `scripts/ci/stage-release.sh`, `scripts/ci/verify_release.py`, `kggml-*/native/scripts/build-host.sh` | `linux`, `macos` |
| `scripts-ci-linux-windows` | `scripts/ci/check_native_arch.py`, `scripts/ci/python.sh` | `linux`, `windows` |
| `scripts-ci-all-platforms` | `scripts/audit-binary-symbols.sh`, `scripts/ci/audit-minimal-sample.sh`, any new `scripts/ci/*` (fail-safe) | `linux`, `macos`, `windows` |
| `scripts-dev-other` | `^scripts/[^/]+\.sh$` (other top-level helper scripts) | None |
| `apple-ios-sources` | `^(.+/)?src/(ios\|apple\|macos)(Main\|Test)/` | `macos` |
| `apple-cinterop-and-xcode` | `\.(def\|swift\|pbxproj\|plist\|xcscheme\|xcworkspacedata\|xib\|storyboard\|entitlements\|metal\|m\|mm)$` or `^samples/[^/]+/iosApp/` | `macos` |
| `android-sources-and-config` | `^(.+/)?src/android(Main\|UnitTest\|HostTest\|SharedTest\|InstrumentedTest\|DeviceTest)/`, `^samples/[^/]+/androidApp/`, `AndroidManifest.xml`, `consumer-rules.pro`, `robolectric.properties` | `linux` |
| `linux-android-native-only` | `kggml-runtime-{vulkan,opencl}/native/**`, `aarch64-linux-gnu.cmake`, `kggml-*/native/jni/*.map` | `linux` |
| `windows-native-toolchain` | `kggml-runtime/native/cmake/toolchains/arm64-windows-llvm.cmake` | `windows` |
| `docs-snippets-sources` | `^docs-snippets/src/` (compiled in Linux `./gradlew check`) | `linux` |
| `kotlin-shared-and-jvm` | `\.(kt\|java)$`, `src/(common\|jvm\|jvmCommon\|jni\|desktop)(Main\|Test)/`, `integration-tests/src/` | `linux`, `macos`, `windows` |
| `native-cpp-cmake-submodules` | `\.(c\|cc\|cpp\|cxx\|h\|hh\|hpp\|hxx\|inl\|cl\|comp\|glsl\|s\|S\|asm\|cmake)$`, `*.cmake.in`, `CMakeLists.txt`, `CMakePresets.json`, `.gitmodules`, `kggml-*/native/**` | `linux`, `macos`, `windows` |
| `gradle-build-system` | `\.(kts\|gradle\|groovy\|properties\|xml)$`, `gradle/**`, `build-logic/**`, `gradlew` | `linux`, `macos`, `windows` |
| `unclassified-failsafe` | Any path matching none of the rules above | `linux`, `macos`, `windows` |

---

## 3. Repo-Hygiene Audit Classification (`repo-hygiene.yml`)

Hygiene rules are additive: a single file can trigger multiple audits (for example, `kggml-runtime/src/commonMain/.../*.kt`
triggers all three `ci.yml` platforms plus `docs-audit` for public KDoc; `NOTICE` triggers `license-audit` and `docs-audit`).

| Job | Triggered By |
| --- | --- |
| `agent-parity` | `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.agents/**`, `.antigravity/**`, `.claude/**`, `.codex/**`, `.gemini/**`, `scripts/verify-agent-parity.sh`, `scripts/sync-agent-mirrors.sh`, `scripts/ci/python.sh`, `.github/workflows/repo-hygiene.yml` |
| `workflow-audit` | `.github/workflows/*.y(a)ml`, `scripts/ci/verify-hosted-runners.sh`, `scripts/tests/test-verify-hosted-runners.sh` |
| `license-audit` | `LICENSE`, `NOTICE`, `.gitmodules`, `kggml-*/native/third_party/**`, `CMakeLists.txt`, `*.cmake`, binary/model/audio/image/archive extensions, any git-detected binary file, any deleted/renamed file (guards `NOTICE` fixture paths), `scripts/ci/verify-licenses.sh`, `scripts/tests/test-verify-licenses.sh`, `.github/workflows/repo-hygiene.yml`, or any unclassified file |
| `docs-audit` | `*.md`, `NOTICE`, `docs-snippets/**`, `kggml-bom/build.gradle.kts`, `gradle.properties`, library main Kotlin sources (`kggml-*/src/*Main/**/*.kt`, for KDoc), any deleted/renamed file (guards Markdown relative links), `scripts/ci/verify-docs.sh`, `scripts/ci/verify_docs.py`, `scripts/tests/test-verify-docs.sh`, `.github/workflows/repo-hygiene.yml`, or any unclassified file |
| `script-tests` | Only the unit test(s) under `scripts/tests/` whose subject script or test file changed (`wait-pr-checks`, `verify_release`, `check_native_arch`, `audit-history`, `audit-binary-symbols`, `classify_changes`), plus `test-classify-changes.sh` on `.github/workflows/ci.yml` edits and all tests on `.github/workflows/repo-hygiene.yml` edits |
| `secret-scan` | Every ready PR (runs in parallel with `classify`) and `workflow_dispatch` |

---

## 4. Verifying Locally

```bash
# Inspect what workflows and jobs your branch changes would run
python3 scripts/ci/classify_changes.py --base origin/main --head HEAD --json

# Audit every tracked file and verify 100% parity with ci.yml paths
python3 scripts/ci/classify_changes.py --verify-repo

# Run the full classifier test suite
./scripts/tests/test-classify-changes.sh
```
