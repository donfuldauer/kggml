# GitHub Actions Workflow Triggers & Content Classification

Every GitHub Actions workflow in `.github/workflows/` runs only when the changed content actually requires it.
The canonical rules live in [`scripts/ci/classify_changes.py`](../scripts/ci/classify_changes.py), which uses file-extension
and path regular expressions (plus git deletion/rename and binary detection) and is verified against every tracked file in the
repository by [`scripts/tests/test-classify-changes.sh`](../scripts/tests/test-classify-changes.sh) (`--verify-repo`).

---

## 1. Workflow Overview

| Workflow | Trigger | When It Runs |
| --- | --- | --- |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | `pull_request` (`paths:` filter) + `schedule` (nightly 07:00 UTC) + `workflow_dispatch` | Cross-platform native/JVM/Android/iOS/Windows builds, CTest, Gradle tests, symbol audits, and release dry-run. Skipped completely (0 runners) when no build/code/native/CI files change. When triggered on a PR, its `classify` job gates `linux`, `macos`, and `windows` **and** heavy step groups (`ci_native`, `ci_macos_jvm`, `ci_release_dry_run`, `ci_samples_packaging`) independently, restoring prebuilt native binaries/XCFrameworks via `actions/cache@v6` when native files did not change. Draft PRs run nothing. |
| [`.github/workflows/repo-hygiene.yml`](../.github/workflows/repo-hygiene.yml) | `pull_request` + `workflow_dispatch` | Cheap Ubuntu repository audits. Its `classify` job gates `agent-parity`, `workflow-audit`, `license-audit`, `docs-audit`, and `script-tests` (and individual script tests inside `script-tests`). `secret-scan` runs in parallel on every ready PR because any commit, file, or commit message can introduce secrets, private keys, oversized blobs, or developer-machine paths. Draft PRs run nothing. |
| [`.github/workflows/label.yml`](../.github/workflows/label.yml) | `pull_request_target` | Applies module, CI, and documentation labels using [`.github/labeler.yml`](../.github/labeler.yml). |
| [`.github/workflows/release.yml`](../.github/workflows/release.yml) | `push` (`tags: ["v*"]`) + `workflow_dispatch` | Builds, signs, audits, and publishes the Maven Central bundle and draft GitHub release. Never runs on pull requests. |
| [`.github/workflows/stale.yml`](../.github/workflows/stale.yml) | `schedule` (daily cron) | Marks inactive issues and PRs as stale. Never triggered by file changes. |

Full-matrix runs (`--all`, enabling every platform and step) run automatically on:
- Nightly `schedule` (`0 7 * * *` UTC on `main`), which keeps Windows, full native CTest, Android NDK + GPU add-ons, release dry-runs, sample packaging, and `main`-branch native caches warm for PRs.
- Manual `workflow_dispatch` of `ci.yml` or `repo-hygiene.yml`.
- Any PR carrying the `ci:full` label.

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
| `gradle-build-system` | `\.(kts\|gradle\|groovy\|properties\|xml)$`, `gradle/**`, `build-logic/**` (incl. convention plugin `.kt`), `gradlew` | `linux`, `macos`, `windows` |
| `kotlin-jvm-jni-desktop` | `src/(jvm\|jvmCommon\|jni\|desktop)(Main\|Test)/`, `integration-tests/src/`, `\.java$` | `linux`, `macos`, `windows` |
| `kotlin-common` | `src/common(Main\|Test)/`, other `\.kt$` (Linux JVM/Android + macOS K/N & iOS; Windows deferred to nightly) | `linux`, `macos` |
| `native-cpp-cmake-submodules` | `\.(c\|cc\|cpp\|cxx\|h\|hh\|hpp\|hxx\|inl\|cl\|comp\|glsl\|s\|S\|asm\|cmake)$`, `*.cmake.in`, `CMakeLists.txt`, `CMakePresets.json`, `.gitmodules`, `kggml-*/native/**` | `linux`, `macos`, `windows` |
| `unclassified-failsafe` | Any path matching none of the rules above | `linux`, `macos`, `windows` |

---

## 3. Step-Level Gating & Prebuilt Native Caching (`ci.yml`)

Within each selected OS job, `classify_changes.py` emits four step flags plus a deterministic `native_cache_key` (16-char SHA-256 of `git ls-tree -r HEAD` across `.gitmodules` and `kggml-{runtime,whisper,llama,diffusion}/native`, excluding `*.md`):

| Output | Controls | Triggered By |
| --- | --- | --- |
| `ci_native` | Full native C/C++ rebuilds, upstream pin checks, CTest, dynamic linkage audits, `linux-aarch64` QEMU cross-compile, full Android NDK + Vulkan/OpenCL add-ons, desktop Vulkan Mesa lavapipe test, `windows-arm64` cross-compile, and XCFramework rebuilds | `native-cpp-cmake-submodules`, `linux-android-native-only`, `windows-native-toolchain`, `.metal`/`.m`/`.mm`, `kggml-*/native/scripts/**`, `scripts/ios/build-xcframeworks.sh`, `scripts/ci/setup/*`, pin/linkage/GPU/arch scripts, `ci.yml`, unclassified |
| `ci_macos_jvm` | macOS host dylibs + desktop `jvmTest` / `desktopTest` / desktop minimal audit on `macos` (skipped on pure `commonMain` or iOS-only PRs, where `macos` focuses on `macosArm64Test`, `iosSimulatorArm64Test`, and `linkDebugFrameworkIos*`) | `ci_native` (when `macos` runs), `kotlin-jvm-jni-desktop`, `gradle-build-system`, `scripts-ci-all-platforms`, host/linkage scripts in `scripts-ci-linux-macos`, `ci.yml`, unclassified |
| `ci_release_dry_run` | Signed Maven Central staging dry-run (`scripts/ci/stage-release.sh --dry-run` + `verify_release.py`) on Linux and macOS | Root/library/BOM Gradle files (`build.gradle.kts`, `settings.gradle.kts`, `gradle.properties`, `gradlew`, `gradle/**`, `build-logic/**`, `kggml-bom/**`, `kggml-*/build.gradle.kts`), `scripts/ci/stage-release.sh`, `scripts/ci/verify_release.py`, `ci.yml`, unclassified |
| `ci_samples_packaging` | Sample Android debug/release APKs, desktop jar, `whisper-only-minimal` APK/desktop/iOS binary symbol audits, and both Xcode sample apps (`scripts/ios/build-sample-apps.sh all`) | `ci_native`, `ci_release_dry_run`, any buildable file under `samples/**`, `AndroidManifest.xml`, `consumer-rules.pro`, `scripts/audit-binary-symbols.sh`, `scripts/ci/audit-minimal-sample.sh`, `scripts/ios/build-sample-apps.sh`, `ci.yml`, unclassified |
| `native_cache_key` | Exact `actions/cache@v6` key suffix for prebuilt Linux/macOS/Windows host libs, Android minimal jniLibs (`runtime` + `whisper`), and Apple XCFrameworks (`build/native-ios`) | Always emitted; on a cache miss when `ci_native == false`, the runner automatically performs a minimal native build (skipping CTest and unrelated cross-compiles) so `-Pkggml.requireNative=true` still passes and warms the cache |

---

## 4. Repo-Hygiene Audit Classification (`repo-hygiene.yml`)

Hygiene rules are additive: a single file can trigger multiple audits (for example, `kggml-runtime/src/commonMain/.../*.kt`
triggers Linux + macOS in `ci.yml` plus `docs-audit` for public KDoc; `NOTICE` triggers `license-audit` and `docs-audit`).

| Job | Triggered By |
| --- | --- |
| `agent-parity` | `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.agents/**`, `.antigravity/**`, `.claude/**`, `.codex/**`, `.gemini/**`, `scripts/verify-agent-parity.sh`, `scripts/sync-agent-mirrors.sh`, `scripts/ci/python.sh`, `.github/workflows/repo-hygiene.yml` |
| `workflow-audit` | `.github/workflows/*.y(a)ml`, `scripts/ci/verify-hosted-runners.sh`, `scripts/tests/test-verify-hosted-runners.sh` |
| `license-audit` | `LICENSE`, `NOTICE`, `.gitmodules`, `kggml-*/native/third_party/**`, `CMakeLists.txt`, `*.cmake`, binary/model/audio/image/archive extensions, any git-detected binary file, any deleted/renamed file (guards `NOTICE` fixture paths), `scripts/ci/verify-licenses.sh`, `scripts/tests/test-verify-licenses.sh`, `.github/workflows/repo-hygiene.yml`, or any unclassified file |
| `docs-audit` | `*.md`, `NOTICE`, `docs-snippets/**`, `kggml-bom/build.gradle.kts`, `gradle.properties`, library main Kotlin sources (`kggml-*/src/*Main/**/*.kt`, for KDoc), any deleted/renamed file (guards Markdown relative links), `scripts/ci/verify-docs.sh`, `scripts/ci/verify_docs.py`, `scripts/tests/test-verify-docs.sh`, `.github/workflows/repo-hygiene.yml`, or any unclassified file |
| `script-tests` | Only the unit test(s) under `scripts/tests/` whose subject script or test file changed (`wait-pr-checks`, `verify_release`, `check_native_arch`, `audit-history`, `audit-binary-symbols`, `classify_changes`), plus `test-classify-changes.sh` on `.github/workflows/ci.yml` edits and all tests on `.github/workflows/repo-hygiene.yml` edits |
| `secret-scan` | Every ready PR (runs in parallel with `classify`) and `workflow_dispatch` |

---

## 5. Verifying Locally

```bash
# Inspect what workflows, platforms, step flags, and native cache key your branch changes would run
python3 scripts/ci/classify_changes.py --base origin/main --head HEAD --json

# Audit every tracked file and verify 100% parity with ci.yml paths
python3 scripts/ci/classify_changes.py --verify-repo

# Run the full classifier test suite
./scripts/tests/test-classify-changes.sh
```
