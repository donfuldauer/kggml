# Contributing to Kggml

Thank you for your interest in contributing to Kggml (`com.donfuldauer.kggml`).

## Workflow

1. **Pick a ready issue:** Work from open GitHub issues labeled `status:ready` with all `Blocked by` prerequisites closed.
2. **Branch naming:** Create feature branches from `main` using `feat/issue-<N>-<slug>`.
3. **Architecture & coding rules:** Follow [AGENTS.md](AGENTS.md), including:
   - Root package `com.donfuldauer.kggml` (`.core`, `.whisper`, `.llama`, `.diffusion`, `.models`).
   - No global engine singletons; engines must be instantiable `AutoCloseable` classes backed by a dedicated serial single-thread `CoroutineDispatcher`.
   - Single-symbol `ggml` linkage exclusively via `:kggml-runtime`.
   - Zero-copy JNI / Kotlin/Native buffer crossing and `extern "C"` `try/catch` boundaries on all C++ entry points.
4. **Tests are mandatory:** Every PR must ship automated tests covering pure Kotlin (`commonTest`), C++ (`ctest`), platform integration (`androidUnitTest`, `iosSimulatorArm64Test`, `jvmTest`), and UI/binary checks as applicable.
5. **Agentic harness parity:** If you edit `AGENTS.md`, `.agents/agents/`, or `.agents/skills/`, run `./scripts/sync-agent-mirrors.sh` and verify with `./scripts/verify-agent-parity.sh`.
6. **Run CI locally first:** Run `./scripts/ci-local.sh` before pushing; it runs the same checks as the Linux PR CI jobs. macOS and Windows CI are label-gated (`ci:macos`, `ci:windows`) to save billed minutes; if the Actions quota is exhausted, `./scripts/ci-local.sh --full` is the merge gate. See `AGENTS.md` Section 3, "CI Minutes Policy".
7. **Pull requests:** Link `Closes #<N>` in the PR description and ensure all CI checks (and `phase-reviewer` audit on phase-final issues) pass prior to merging (merge commit; squash merges are disabled).

## Licensing & Attribution

- All contributions are licensed under the MIT License ([LICENSE](LICENSE)).
- Any upstream C/C++ dependency additions/updates or bundled test fixtures must be recorded in [NOTICE](NOTICE) with license compatibility verified.
- Never commit secrets, signing keys, local machine paths, or unapproved model weights.
