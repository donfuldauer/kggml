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
| `:kggml-runtime` | `com.donfuldauer.kggml.runtime` | KMP Native | Shared dynamic `ggml` runtime (`ggml-base`, `ggml-cpu`, `ggml-metal` w/ embedded `.metallib`, `ggml-vulkan`). Ships `GgmlRuntime.xcframework` (iOS), `libggml.so` (Android), and desktop native libs (`macosArm64`, `linuxX64`, `windowsX64`). |
| `:kggml-whisper` | `com.donfuldauer.kggml.whisper` | KMP Native | `whisper.cpp` dynamically linked against `:kggml-runtime`, native PCM resampler/downmixer, ring-buffer VAD gate, `Flow<TranscriptionEvent>` streaming. |
| `:kggml-llama` | `com.donfuldauer.kggml.llama` | KMP Native | `llama.cpp` + `libmtmd` (VLM) dynamically linked against `:kggml-runtime`, `Flow<TokenEvent>`, KV-cache snapshots, GBNF/JSON constraints, embeddings. |
| `:kggml-diffusion` | `com.donfuldauer.kggml.diffusion` | KMP Native | `stable-diffusion.cpp` dynamically linked against `:kggml-runtime`. |
| `:kggml-model-manager` | `com.donfuldauer.kggml.models` | Pure Kotlin (Optional) | Resumable chunked downloads, SHA-256 verification, GGUF/Whisper header RAM estimator, OS low-memory cache eviction hooks. |
| `:samples:kggml-showcase` | `com.donfuldauer.kggml.samples.showcase` | Compose Multiplatform | Full multi-engine showcase (Android, iOS, JVM Desktop) consuming SDK via `kggml-bom`. |
| `:samples:whisper-only-minimal` | `com.donfuldauer.kggml.samples.whisper` | Android + iOS | Minimal sample depending **only** on `kggml-bom` + `kggml-whisper` for CI binary symbol leakage audits. |

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
   - Record every pinned upstream C/C++ commit hash and every bundled CC0/MIT test fixture in `NOTICE`.

---

## 3. Build, Test & Verification Commands

**Before every push, run `./scripts/ci-local.sh`.** It runs the same checks as the Linux PR CI jobs (agent parity, gitleaks, ggml pin, host native build + CTest, dynamic-linkage audit, `:kggml-runtime:jvmTest`, Android cross-compile + audit). Native steps run only when native paths changed versus `origin/main`; tools it cannot find (gitleaks, Android NDK) are reported as `SKIPPED`. Paste its summary block into the PR's Verification section. Fix every `FAIL` locally before pushing rather than using CI to find failures.

Also run whichever of these apply to the modules your change touches:

```bash
# Agent & harness parity check (always run if .agents/ or AGENTS.md changed)
./scripts/verify-agent-parity.sh

# Pure Kotlin & JVM/Android tests (once Gradle is initialized in Issue #1)
./gradlew check
./gradlew :kggml-core-api:allTests
./gradlew jvmTest androidUnitTest

# iOS simulator tests (macOS runners / hosts)
./gradlew iosSimulatorArm64Test

# Native C++ CTest suites (when native code is added)
ctest --test-dir build/native --output-on-failure
```

### Mandatory Testing & Coverage Policy (Section 9.4)
- **Pure Kotlin Logic (`:kggml-core-api`, ring buffer, VAD, GGUF/Whisper header parser, `BackendConfig`):** Thorough `commonTest` unit tests with deterministic fakes and high statement/branch coverage.
- **Native C++ & Bridge Layers (`:kggml-runtime`, `:kggml-whisper`, `:kggml-llama`, `:kggml-diffusion`):** CTest/GoogleTest suites (resampling, short-buffer padding, abort callbacks, exception handling) + platform integration tests (`androidUnitTest`, `iosSimulatorArm64Test`, `jvmTest`) using tiny CC0/MIT synthetic fixtures and PCM arrays.
- **Modularity & UI:** CI binary symbol audit on `:samples:whisper-only-minimal` (`nm`/`otool` on iOS, `apkanalyzer`/`readelf` on Android) and Compose UI tests on `:samples:kggml-showcase`.

### CI Minutes Policy (private repository)
Hosted macOS runners are billed at 10× Linux minutes, so CI is split by cost:
- **Every PR (Linux, 1×):** `repo-hygiene.yml` (parity, gitleaks) and `runtime-native.yml` (Linux host build + CTest + linkage audit + `jvmTest`, Android cross-compile + audit). Draft PRs skip the native jobs, and a new push cancels the previous run.
- **macOS on demand only:** `runtime-native-macos.yml` runs only when a PR has the `ci:macos` label, when dispatched manually, or weekly (Mondays) if `main` changed runtime paths that week. Add `ci:macos` to **phase-final PRs** and to PRs that change macOS/iOS-specific native code (Metal, CoreML, Accelerate, `cinterop`, xcframework packaging, Apple-only `#ifdef`s). Do not add it to other PRs.
- New CI jobs must put shared logic in `scripts/ci/` and call it from both the workflow and `./scripts/ci-local.sh`, so every Linux CI check can be reproduced locally. Any new macOS/iOS job goes in the label-gated macOS workflow, not the per-PR one.
- Batch fixes into one push where possible: every push to an open PR starts the Linux jobs again.

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
- Merge strategy: merge to `main` with a **merge commit** (`gh pr merge <PR> --merge --delete-branch`; squash merges are disabled on the repository, and merge commits preserve each PR's individual commits) after CI passes (including the `ci:macos` run on phase-final or Apple-specific PRs, plus `phase-reviewer` approval on phase-final issues), delete branch, and promote newly unblocked downstream issues from `status:blocked` to `status:ready`.
- **Autonomous merge:** A `"work on next issue"` / `"work on issue #X"` request authorizes the agent (in any harness) to perform that merge itself without asking for confirmation. Escalate to the user only when a gate fails and cannot be fixed within the issue's scope (red CI, unresolvable conflict, missing `phase-reviewer` approval, or unresolved requested changes).

### Specialized Agents (`.agents/agents/`)
1. **`issue-triager`** — Verifies `Blocked by` prerequisites, inspects codebase state, identifies edge cases, and produces a technical blueprint.
2. **`orchestrator`** — Claims issues, manages branch/labels, delegates to `implementer` and `tester`, enforces quality gates, handles PR creation/merge and downstream unblocking.
3. **`implementer`** — Writes KMP, C++/CMake, JNI, `cinterop`, and Compose Multiplatform production code.
4. **`tester`** — Writes unit, CTest, memory-safety, binary-audit, integration, and Compose UI tests; validates coverage.
5. **`phase-reviewer`** — Invoked **exclusively** on the final issue of each Phase/Milestone (never on routine mid-phase PRs) to audit cumulative phase changes against Section 5 Exit Criteria before unblocking the next phase.

### Autonomous Skill
- Use `.agents/skills/work-on-issue/SKILL.md` for `"work on next issue"` or `"work on issue #X"` workflows using standard `gh`, `git`, `./gradlew`, and `ctest` commands.
