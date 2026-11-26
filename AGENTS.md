# Kggml Engineering & Agent Guidelines

This is the canonical repository instruction file for **Kggml** (`com.donfuldauer.kggml`), a modular on-device AI SDK for Kotlin Multiplatform wrapping `ggml` (`whisper.cpp`, `llama.cpp`, `libmtmd`, and `stable-diffusion.cpp`) across **Android**, **iOS**, and **JVM Desktop**.

> **Multi-Harness Sync:** `AGENTS.md`, `.agents/agents/`, and `.agents/skills/` are the canonical sources of truth. After editing any of them, run `./scripts/sync-agent-mirrors.sh` and verify parity with `./scripts/verify-agent-parity.sh`. CI enforces strict parity across `CLAUDE.md`, `GEMINI.md`, `.gemini/`, `.claude/`, `.antigravity/`, and `.codex/`.

---

## 1. Module Topology & Package Rules

Maven Group: `com.donfuldauer` | BOM Coordinate: `com.donfuldauer:kggml-bom:1.0.0`

| Module | Root Package | Kind | Responsibility |
| --- | --- | --- | --- |
| `:kggml-bom` | — | `java-platform` | Version constraints across all KMP root coordinates |
| `:kggml-core-api` | `com.donfuldauer.kggml.core` | Pure Kotlin | `InferenceEngine` (`AutoCloseable`), `ComputeBackend`, `BackendConfig`, `TranscriptionEvent`, `TokenEvent`, `InferenceException` hierarchy, `MemoryPressureObserver`. **Zero native code.** |
| `:kggml-runtime` | `com.donfuldauer.kggml.runtime` | KMP Native | Shared dynamic `ggml` runtime (`ggml-base`, `ggml-cpu`, `ggml-metal` w/ embedded `.metallib`, `ggml-vulkan` on desktop with `KGGML_ENABLE_VULKAN`). Ships `GgmlRuntime.xcframework` (iOS), `libggml.so` (Android), and desktop native libs (`macosArm64`, `linuxX64`, `windowsX64`). Registers Android GPU add-ons found in the app (`kggml_runtime_load_backend`). |
| `:kggml-runtime-vulkan` | `com.donfuldauer.kggml.runtime.vulkan` | KMP (Android native) | Optional Android GPU add-on: ggml's Vulkan backend as a `GGML_BACKEND_DL` module (`libggml-vulkan.so`, API 28+) linking `:kggml-runtime`'s `libggml-base.so`; pinned Khronos `Vulkan-Headers`. No-op on JVM / iOS. Never a dependency of `whisper-only-minimal` (its audit forbids it). |
| `:kggml-whisper` | `com.donfuldauer.kggml.whisper` | KMP Native | `whisper.cpp` dynamically linked against `:kggml-runtime`, native PCM resampler/downmixer, ring-buffer VAD gate, `Flow<TranscriptionEvent>` streaming. |
| `:kggml-llama` | `com.donfuldauer.kggml.llama` | KMP Native | `llama.cpp` + `libmtmd` (VLM) dynamically linked against `:kggml-runtime`, `Flow<TokenEvent>`, KV-cache snapshots, GBNF/JSON constraints, embeddings. |
| `:kggml-diffusion` | `com.donfuldauer.kggml.diffusion` | KMP Native | `stable-diffusion.cpp` dynamically linked against `:kggml-runtime`. |
| `:kggml-model-manager` | `com.donfuldauer.kggml.models` | Pure Kotlin (Optional) | Resumable chunked downloads, SHA-256 verification, GGUF/Whisper header RAM estimator, OS low-memory cache eviction hooks. |
| `:samples:kggml-showcase` | `com.donfuldauer.kggml.samples.showcase` | Compose Multiplatform | Full multi-engine showcase (Android, iOS, JVM Desktop) consuming SDK via `kggml-bom`. |
| `:samples:whisper-only-minimal` | `com.donfuldauer.kggml.samples.whisper` | Android + iOS | Minimal sample depending **only** on `kggml-bom` + `kggml-whisper` for CI binary symbol leakage audits. |
| `:integration-tests` | `com.donfuldauer.kggml.integration` | JVM tests (unpublished) | Cross-engine integration tests: whisper + llama (+ diffusion) loaded into one process on the single shared `:kggml-runtime`, inferring concurrently. |
| `:docs-snippets` | `com.donfuldauer.kggml.docs` | JVM (unpublished) | Every Kotlin/Gradle block of the root `README.md`, as `// snippet:start/end <name>` regions compiled against the published `com.donfuldauer` coordinates (build-local verification repo, BOM, no `project()` deps). |

---

## 2. Non-Negotiable Architecture & Safety Rules

1. **Strict Binary Modularity & Single-Symbol `ggml` Linkage:**
   - All `ggml-*` targets live exclusively in `:kggml-runtime`.
   - Engine modules (`:kggml-whisper`, `:kggml-llama`, `:kggml-diffusion`) must link **dynamically** against `:kggml-runtime` and never statically embed duplicate `ggml` symbols.
   - `:samples:whisper-only-minimal` artifacts must contain **zero** `llama` or `stable-diffusion` symbols.
2. **Explicit Runtime Compute Backend Control:**
   - Every engine accepts `BackendConfig(backend, gpuLayers, cpuThreads, useFlashAttention, useMmap)` at construction time.
   - On iOS for Whisper, `ComputeBackend.CPU_ONLY` must set `whisper_context_params.use_gpu = false` before `whisper_init_from_file_with_params()` so execution routes strictly through Apple Accelerate / ARM NEON (preventing background Metal `MTLCommandBufferErrorDomain` `accessRevoked` aborts).
   - Build iOS `whisper.cpp` with `WHISPER_COREML=1` behind a capability flag with automatic Metal/CPU fallback.
3. **No Global Singletons & Serialized Native Dispatch:**
   - `WhisperEngine`, `LlamaEngine`, and `DiffusionEngine` are instantiable `AutoCloseable` classes supporting multiple concurrent instances.
   - Because `ggml` contexts are not thread-safe for concurrent mutation, each engine instance owns a dedicated single-threaded `CoroutineDispatcher` serializing `init`, `infer`, and `close`.
   - Every inference call registers a native abort callback (`abort_callback`, `llama_set_abort_callback`, etc.) tied to the active `CoroutineContext.job`.
4. **Zero-Copy Streaming & FFI Boundary Safety:**
   - Zero temporary WAV files in Whisper streaming and zero temporary image files in VLM (`libmtmd`) inference.
   - Use `FloatArray.usePinned { it.addressOf(0) }` on Kotlin/Native and `GetPrimitiveArrayCritical` (with guaranteed release on all paths) or direct `ByteBuffer` on JNI.
   - Wrap all C++ FFI entry points in `extern "C"` `try/catch` returning structured error codes mapped to typed Kotlin `InferenceException` subclasses—never let C++ exceptions cross JNI or Kotlin/Native boundaries.
   - Enforce a >= 1,000 ms floor in the Whisper ring buffer by zero-padding final flush slices below 1,000 ms before calling `whisper_full_with_state`.
5. **Strict On-Device Focus:** No server/LAN inference modules.
6. **Secret & Attribution Hygiene:**
   - Never commit secrets, signing keys, local machine paths, or large/unlicensed model binaries.
   - Record every pinned upstream C/C++ commit hash and every bundled CC0/MIT test fixture in `NOTICE`. `scripts/ci/verify-licenses.sh` (every PR, `repo-hygiene.yml`) fails on an unrecorded or stale pin, a tracked binary/model/audio/image file missing from `NOTICE`, or a fixture that is not MIT/CC0.
   - `scripts/ci/audit-history.sh` (every PR, `repo-hygiene.yml` `secret-scan`; `ci-local.sh` step `history`) audits the full history of every branch (every PR head on a manual dispatch): gitleaks, blobs over 1 MiB, credential/key/`local.properties` files, private-key blocks and developer-machine paths (files and commit messages), and binaries missing from `NOTICE`. A finding can only be removed by an owner-run `git filter-repo` rewrite (`docs/OSS_LAUNCH.md`); report it instead of rewriting history.
7. **Verified Public Documentation:**
   - Every public declaration in a library module's main source sets has KDoc (its own `/** */` or an `@property`/`@param` line in its class KDoc).
   - Every Kotlin/Gradle block in the root `README.md` is preceded by `<!-- snippet: <name> -->` and is identical to that region in `docs-snippets/`; change both together. `scripts/ci/verify-docs.sh` (every PR, `repo-hygiene.yml`) enforces this plus Markdown links/anchors, published Maven coordinates/BOM version, and no local paths, session links or TODOs in user-facing docs.

---

## 3. Build, Test & Verification Commands

**Before every push, run `./scripts/ci-local.sh`.** It runs the same checks as the Linux PR CI jobs (agent parity, license/NOTICE/fixture audit, gitleaks, ggml pin, host native build + CTest, dynamic-linkage audit, `:kggml-runtime:jvmTest`, Android cross-compile + audit). Native steps run only when native paths changed versus `origin/main`; tools it cannot find (gitleaks, Android NDK) are reported as `SKIPPED`. Paste its summary block into the PR's Verification section. Fix every `FAIL` locally before pushing rather than using CI to find failures.

Also run whichever of these apply to the modules your change touches:

```bash
# Agent & harness parity check (always run if .agents/ or AGENTS.md changed)
./scripts/verify-agent-parity.sh

# Docs audit + README snippet compile (always run if README.md, docs-snippets/ or public API changed)
./scripts/ci/verify-docs.sh
./gradlew :docs-snippets:build

# Pure Kotlin & JVM/Android tests (once Gradle is initialized in Issue #1)
./gradlew check
./gradlew :kggml-core-api:allTests
./gradlew jvmTest androidUnitTest

# iOS simulator tests (macOS runners / hosts)
./gradlew iosSimulatorArm64Test

# Native C++ CTest suites (when native code is added)
ctest --test-dir build/native --output-on-failure

# Android instrumented tests on a connected device / emulator (every library + both samples, real native libraries
# bundled and required; KGGML_AVD=<avd> boots a headless emulator first). ci-local step: --only instrumented.
# CI runs it in the macOS job on the owner's Mac (local-mac) with the AVD named by the ANDROID_AVD repo variable.
./scripts/ci/run-android-instrumented.sh

# Merged Kover coverage of every JVM / Robolectric test -> build/reports/kover/html (ci-local step: --only coverage)
./scripts/ci/coverage.sh -Pkggml.requireNative=true

# iOS apps of both samples (samples/*/iosApp/iosApp.xcodeproj, macOS): build the xcframeworks once, then open the
# project in Xcode, or build both apps unsigned for the simulator and devices (ci-local step: --only ios-apps; CI:
# macOS job)
./scripts/ios/build-xcframeworks.sh
./scripts/ios/build-sample-apps.sh all
```

### Mandatory Testing & Coverage Policy (Section 9.4)
- **Pure Kotlin Logic (`:kggml-core-api`, ring buffer, VAD, GGUF/Whisper header parser, `BackendConfig`):** Thorough `commonTest` unit tests with deterministic fakes and high statement/branch coverage.
- **Native C++ & Bridge Layers (`:kggml-runtime`, `:kggml-whisper`, `:kggml-llama`, `:kggml-diffusion`):** CTest/GoogleTest suites (resampling, short-buffer padding, abort callbacks, exception handling) + platform integration tests (`androidUnitTest`, `iosSimulatorArm64Test`, `jvmTest`) using tiny CC0/MIT synthetic fixtures and PCM arrays.
- **Modularity & UI:** CI binary symbol audit on `:samples:whisper-only-minimal` (`nm`/`otool` on iOS, `apkanalyzer`/`readelf` on Android) and Compose UI tests on `:samples:kggml-showcase`.
- **Every platform:** put tests in `commonTest` unless they need a platform API. Library `commonTest` and the shared native contracts (`Native*EngineContract`) run on the JVM, the iOS simulator and Android devices (`androidInstrumentedTest` subclasses); the showcase's Compose UI tests (`UiTestBase`) run on desktop, the iOS simulator, Robolectric and Android devices. Real-engine tests skip without native libraries and fail with `-Pkggml.requireNative=true`.

### Deferred Manual & Physical Verification
Agent sessions never wait on a person to test anything by hand. Manual and physical-device checks (real iPhone/Android hardware, backgrounding on a device, thermal/battery behaviour, listening to audio, eyeballing UI on a real screen) are reserved for a single owner acceptance pass **after every phase is complete**.
- Automate everything that can be automated (simulator/emulator tests, `jvmTest`, CTest, binary audits, scripted sample runs) and gate the PR on those.
- For whatever still needs a person, write the exact procedure (in the module or sample `README.md`) and add one line to `docs/MANUAL_VERIFICATION.md` (`- [ ] [#<issue>] <check> — <procedure link>`) in the same PR.
- A pending manual check is **never** a merge blocker, a reason to leave a PR open, a `phase-reviewer` blocking finding, or a reason to escalate to the user. Mark the PR's manual-verification checkbox as deferred, merge when the automated gates pass, and continue to the next issue.
- Issue acceptance criteria that call for physical or manual verification are satisfied for merge by the automated evidence plus the `docs/MANUAL_VERIFICATION.md` entry.

### CI Minutes Policy (private repository)
Included Actions minutes are limited, and hosted Windows and macOS runners are billed at 2× and 10× Linux minutes, so CI is split by cost and nothing is billed twice:
- **Every ready PR (Linux, 1×):** `repo-hygiene.yml` (parity, gitleaks) and the Ubuntu job of `ci.yml` (ggml/whisper pins, Linux host build + CTest + linkage audits, `./gradlew check` incl. `jvmTest`/`androidUnitTest`, Android cross-compile + audits, sample builds + `whisper-only-minimal` symbol audit). Draft PRs and PRs that only touch Markdown, `docs/` or agent-instruction mirrors skip `ci.yml`, and a new push cancels the previous run. `runtime-native.yml` duplicates the native part of the Ubuntu job and is manual-dispatch only.
- **Release dry-run on every PR:** the Ubuntu job runs `scripts/ci/stage-release.sh --dry-run --skip-targets ios` (every publication signed with a throwaway key, then the `verify_release.py` audit of POMs, `.module` files, BOM constraints, sources/javadoc jars, checksums and signatures); the `ci:macos` job runs it with the iOS components too. `release.yml` itself runs only on a `v<VERSION_NAME>` tag or a manual dispatch (see `docs/RELEASING.md`); agents never push release tags.
- **No re-runs on `main`:** pushes to `main` (i.e. merges) run no workflow; PRs merge with a merge commit after their head passed.
- **Windows on demand only:** on hosted runners, the Windows job in `ci.yml` runs only on PRs labelled `ci:windows` or a manual dispatch (while the owner's Windows runner is on it runs on every ready PR instead; see "Runner order"). Add `ci:windows` to **phase-final PRs** and to PRs that change Windows-specific code (MSVC flags, `windows-x64` presets, `_WIN32` `#ifdef`s, DLL loading).
- **macOS on demand only:** the macOS job in `ci.yml` runs only when a PR has the `ci:macos` label or on a manual dispatch (on the owner's Mac while it is on, else hosted; see "Runner order"); `runtime-native-macos.yml` runs only on manual dispatch or weekly (Mondays) if `main` changed runtime paths that week. Add `ci:macos` to **phase-final PRs** and to PRs that change macOS/iOS-specific native code (Metal, CoreML, Accelerate, `cinterop`, xcframework packaging, Apple-only `#ifdef`s). Do not add either label to other PRs.
- New CI jobs must put shared logic in `scripts/ci/` and call it from both the workflow and `./scripts/ci-local.sh`, so every Linux CI check can be reproduced locally. Any new macOS/iOS job goes in the label-gated macOS workflow, not the per-PR one.
- Batch fixes into one push where possible: every push to an open PR starts the Linux jobs again. Open PRs as drafts while iterating and mark them ready once `./scripts/ci-local.sh` passes.

#### Runner order
While the repository is private, the owner's own machines run jobs at no billed minutes (#62). Each runner sets its repository variable to `on` when it starts and `off` when it stops, and a job uses it only while that variable is `on` and the PR comes from this repository (fork PRs never reach a private runner):
1. **Linux (`local-linux`, `LINUX_RUNNER`)**: the Ubuntu job of `ci.yml` and every `repo-hygiene.yml` job.
2. **Mac (`local-mac`, `MAC_RUNNER`)**: the `ci:macos` job; and, when no Linux runner is on, `Mac (full local gate)` (`ci-local.sh --full --strict` on macOS arm64) replaces the Ubuntu job. That does not build linux-x64: for PRs that change Linux-specific code (`__linux__` `#ifdef`s, `linux-x64` presets, ELF packaging) with only the Mac on, also run `gh workflow run ci.yml --ref <branch> -f runner=hosted`.
3. **Windows (`local-win`, `WIN_RUNNER`)**: the Windows job on **every** ready PR, not only `ci:windows` ones.
4. **GitHub-hosted runners** for everything else, exactly as in "CI Minutes Policy" above; `runner: hosted` on a manual dispatch forces them.
5. **Local gate** when hosted jobs are refused (next sections).

If a self-hosted job is still queued after 10 minutes while that machine is not busy with another job, its flag is stale (the machine slept or lost power without switching it off): run `gh variable set <MAC|WIN|LINUX>_RUNNER --body off`, cancel the run, then close and reopen the PR to start it on hosted runners. A self-hosted job that fails with build or test output is a real failure: fix it, do not fall back.

`scripts/ci/verify-hosted-runners.sh` (every PR) allows only these three label sets, requires the same-repository guard on every job that can reach one, and **fails while the repository is public**: every runner and runner variable must be removed before the visibility flip (#63, `docs/OSS_LAUNCH.md`).

#### Waiting for CI
PR activity subscriptions (Claude Code `subscribe_pr_activity`, and equivalents) wake the agent only for **failed** checks, comments, reviews and the merge/close. A check that finishes green sends **no** event, so silence never means "still running". Timers that live only in the session (Claude Code `CronCreate` / `send_later`) can disappear before they fire. Neither is a wake-up you can rely on for a green result, so:
- **Never end a turn while a check on the PR head is pending.** Right after every push (or label change that starts a job), start `scripts/ci/wait-pr-checks.sh <PR>`. It polls every 60 s and exits as soon as anything is decided: `0` all green, `1` a check failed or was cancelled (fail-fast), `2` the queued 20 min / running 60 min deadline was hit (infrastructure failure), `3` checks cannot be queried. In Claude Code run it with `run_in_background: true`: the harness re-invokes the agent when the command exits, which is a tracked, guaranteed wake-up. In other harnesses run it in the foreground or with their tracked-background equivalent.
- **Act on the exit code at once:**
  - `0`: go to the merge step.
  - `1`: fix the failure (code failure), or apply "When a runner fails or hangs" (no logs or billing refusal = infrastructure failure). Start the waiter again for the remaining checks if any are still pending.
  - `2`: apply "When a runner fails or hangs".
  - `3` (no authenticated `gh`, as in some cloud sessions): poll the head's check runs another way, e.g. GitHub MCP `pull_request_read` method `get_check_runs`, every 1–2 minutes. Pace the polls with a tracked background timer (Claude Code: Bash `sleep 90` with `run_in_background: true`; foreground `sleep` is blocked), applying the same deadlines.
- **Subscriptions are an extra, never the wait mechanism.** Still subscribe to every PR you open, so review comments and failures arrive early, and handle those events when they come. But keep the waiter or poll timer running until every check is terminal.
- **No long or unverified backstops.** Do not schedule 30-minute "safety-net" check-ins. If a turn must end with checks still pending (e.g. the user interrupts), schedule one check-in at most 10 minutes out. Then confirm it exists (Claude Code: `CronList` shows the job ID) and say when it fires; an unconfirmed timer counts as no timer. Prefer a tracked background command over any scheduled check-in.

#### When a runner fails or hangs
CI is a gate the agent checks, never something it waits on open-endedly. Every `ci.yml` job and slow step has a `timeout-minutes`, and iOS simulator test tasks are capped in Gradle (`-Pkggml.iosTestTimeoutMinutes`), so every check ends. On top of that:
- **Bounded waiting.** Wait for CI as described in "Waiting for CI" above, never open-endedly. A check still queued 20 minutes after the push, or running 60 minutes after it started, counts as an infrastructure failure: cancel the run and continue as below.
- **Infrastructure failure** = the job never started, was refused for billing / spending limit, lost its runner, was cancelled by anything other than the agent's own push, or timed out / failed in a setup step (checkout, toolchain setup, simulator boot) before any build or test output. Do not re-run, re-push or wait. Run `./scripts/ci-local.sh --full`, paste its summary into the PR's Verification section under `CI unavailable (<runner> failed: <reason>): local gate`, add `- [ ] [#<issue>] CI re-run on <macOS|Windows> — <PR link>` to `docs/MANUAL_VERIFICATION.md`, and merge on the local gate. This is not a reason to escalate.
- **Code failure** = a compile error, a failing or timed-out test, or a failed audit, with the log showing which. Fix it like any other failure. If it occurs only on a platform the agent cannot run locally (macOS/iOS or Windows) and is still failing after two fix pushes, stop iterating: open a follow-up issue (`bug`, the module label, the same phase label, `status:ready`) with the log excerpt and the failing test/step name, link it from the PR body and from `docs/MANUAL_VERIFICATION.md`, and merge on the remaining green gates. Phase-final PRs instead go to `phase-reviewer`, which decides whether the follow-up blocks the phase.

#### When the Actions quota is exhausted
GitHub stops starting hosted jobs once the included minutes (and any spending limit) are used up: checks stay queued or fail at once with a billing / spending-limit annotation and no logs. That is not a code failure and must not block work:
- Do not re-run, re-push or wait for CI. Run `./scripts/ci-local.sh --full` (every Linux check regardless of what changed, plus `./gradlew check` and the sample builds) and paste its summary into the PR's Verification section under the heading `CI unavailable (Actions quota exhausted): local gate`.
- Any step reported `SKIPPED` for a missing tool (NDK, SDK, gitleaks) is listed in the PR body. Labelled macOS/Windows checks that cannot run are added to `docs/MANUAL_VERIFICATION.md` as `- [ ] [#<issue>] CI re-run on macOS/Windows once Actions minutes reset — <PR link>`.
- With a green local gate the PR is mergeable under "Autonomous merge" below; the quota is not a reason to escalate. Phase-final PRs still need `phase-reviewer` approval.
- The owner can unblock hosted CI by raising the Actions budget (Settings → Billing → Budgets), starting one of the self-hosted runners ("Runner order"), or waiting for the monthly reset. A job a running self-hosted runner picks up is not affected by the quota.

---

## 4. GitHub Workflow, Labels & Multi-Agent Roles

### Issue Labels
- **Phase:** `phase-1`, `phase-2`, `phase-3`, `phase-4`, `phase-oss`
- **Module:** `mod:bom`, `mod:core-api`, `mod:runtime`, `mod:whisper`, `mod:llama`, `mod:diffusion`, `mod:model-manager`, `mod:samples`, `infra:ci`
- **Status Lifecycle:** `status:ready`, `status:blocked`, `status:in-progress`, `status:in-review`
- **CI (PRs only):** `ci:macos` — runs the 10×-billed macOS workflow; see Section 3 "CI Minutes Policy"

### Branch & PR Conventions
- Branch name: `feat/issue-<N>-<slug>` cut from `main`.
- **Commit attribution (all harnesses):** Agent-made commits must credit the human contributor the agent works for, as well as the agent. Run `./scripts/install-git-hooks.sh` once per clone (Claude Code does this automatically via the `SessionStart` hook in `.claude/settings.json`; `./scripts/ci-local.sh` also runs it). It enables `.githooks/commit-msg` and records the contributor in `git config kggml.coauthor` (from `$KGGML_COAUTHOR`, an existing `kggml.coauthor`, a non-agent `user.name`/`user.email`, or the GitHub account `gh` is signed in to, via its no-reply address). The hook then appends `Co-authored-by: <contributor>` to every commit authored by an agent identity (e.g. `noreply@anthropic.com`, `[bot]` accounts). Keep the agent's own identity as author (or its `Co-authored-by:` trailer); never rewrite authorship to hide the agent, and do not bypass the hook with `--no-verify`.
- PR body must include `Closes #<N>`, summary of changes, verification output, and checklist completion.
- Merge strategy: merge to `main` with a **merge commit** (`gh pr merge <PR> --merge`; squash merges are disabled on the repository, and merge commits preserve each PR's individual commits) after CI passes (including the `ci:macos` run on phase-final or Apple-specific PRs, plus `phase-reviewer` approval on phase-final issues), and promote newly unblocked downstream issues from `status:blocked` to `status:ready`. The repository deletes head branches automatically on merge, so never delete the branch yourself (no `--delete-branch`, no `git push origin --delete`).
- **Autonomous merge:** A `"work on next issue"` / `"work on issue #X"` request authorizes the agent (in any harness) to perform that merge itself without asking for confirmation. Escalate to the user only when a gate fails and cannot be fixed within the issue's scope (red CI that is neither an exhausted Actions quota nor a runner failure / platform-only failure handled by "When a runner fails or hangs", unresolvable conflict, missing `phase-reviewer` approval, or unresolved requested changes). Never wait on or escalate for manual/physical testing: defer it to `docs/MANUAL_VERIFICATION.md` (Section 3, "Deferred Manual & Physical Verification").

### Specialized Agents (`.agents/agents/`)
1. **`issue-triager`** — Verifies `Blocked by` prerequisites, inspects codebase state, identifies edge cases, and produces a technical blueprint.
2. **`orchestrator`** — Claims issues, manages branch/labels, delegates to `implementer` and `tester`, enforces quality gates, handles PR creation/merge and downstream unblocking.
3. **`implementer`** — Writes KMP, C++/CMake, JNI, `cinterop`, and Compose Multiplatform production code.
4. **`tester`** — Writes unit, CTest, memory-safety, binary-audit, integration, and Compose UI tests; validates coverage.
5. **`phase-reviewer`** — Invoked **exclusively** on the final issue of each Phase/Milestone (never on routine mid-phase PRs) to audit cumulative phase changes against Section 5 Exit Criteria before unblocking the next phase.

### Autonomous Skill
- Use `.agents/skills/work-on-issue/SKILL.md` for `"work on next issue"` or `"work on issue #X"` workflows using standard `gh`, `git`, `./gradlew`, and `ctest` commands.
