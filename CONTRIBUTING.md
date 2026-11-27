# Contributing to Kggml

Thanks for your interest in Kggml. Bug reports, documentation fixes and code changes are all welcome. This guide
explains how to set up the project, the rules the code follows, and how pull requests are checked.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Please report security problems
privately, as described in [SECURITY.md](SECURITY.md), not in public issues.

## Ways to contribute

- **Report a bug** with the [bug report form](https://github.com/donfuldauer/kggml/issues/new?template=bug-report.md).
  Include the platform, the engine, the `BackendConfig` and the model you used.
- **Suggest a feature** with the
  [feature request form](https://github.com/donfuldauer/kggml/issues/new?template=feature-request.md).
- **Send a pull request.** For anything bigger than a small fix, open an issue first so the design can be agreed
  before you write the code. Issues labelled `status:ready` are ready to be worked on.

## Setting up

Prerequisites:

- JDK 17
- CMake 3.24+ and a C++17 compiler (Clang, GCC or MSVC)
- Python 3.9+ (3.11+ for the agent-mirror scripts) and Bash, for the scripts in `scripts/`
- Optional: the Android SDK and NDK for Android builds; macOS with Xcode for iOS builds

```bash
git clone --recurse-submodules https://github.com/donfuldauer/kggml.git
cd kggml
./scripts/install-git-hooks.sh   # enables the commit-msg hook in .githooks/
./scripts/ci-local.sh            # builds and tests what your change touches
```

The [README](README.md#building-from-source) and each module's `native/README.md` describe the native builds for
every platform.

## Project layout

| Directory | Contents |
| --- | --- |
| `kggml-core-api/` | Public API types shared by every engine. Pure Kotlin, no native code. |
| `kggml-runtime/` | The one shared ggml build (CPU, plus Metal on Apple platforms) and its JNI / cinterop bindings |
| `kggml-runtime-vulkan/`, `kggml-runtime-opencl/` | Optional GPU add-ons: ggml's Vulkan (Android, desktop Linux) and OpenCL (Android) backends as modules the runtime loads |
| `kggml-whisper/`, `kggml-llama/`, `kggml-diffusion/` | One engine each: the upstream library as a git submodule, a C wrapper in `native/`, Kotlin bindings in `src/` |
| `kggml-model-manager/` | Optional pure-Kotlin downloads, model inspection and caching |
| `kggml-bom/` | The BOM that aligns all published versions |
| `docs-snippets/` | The README code examples, compiled against the published artifacts |
| `integration-tests/` | Tests that load several engines into one process |
| `samples/` | The showcase app and the Whisper-only sample |
| `scripts/` | Local CI (`ci-local.sh`) and the checks it shares with GitHub Actions (`scripts/ci/`) |

## Rules for code

[AGENTS.md](AGENTS.md) is the full architecture reference. The rules that matter most:

- **Packages.** Everything lives under `com.donfuldauer.kggml` (`.core`, `.runtime`, `.whisper`, `.llama`,
  `.diffusion`, `.models`). Library modules use Kotlin explicit API mode.
- **One ggml.** Only `kggml-runtime` builds ggml. Engine libraries link to it dynamically and must not contain
  `ggml_*` symbols of their own. CI checks this.
- **No singletons.** Engines are `AutoCloseable` classes. Each instance serializes `init`, inference and `close`
  on its own single-threaded dispatcher, and every inference call registers a native abort callback tied to the
  calling coroutine's job.
- **Safe native boundaries.** Every C/C++ entry point is `extern "C"`, catches all exceptions and returns an error
  code; Kotlin maps the code to an `InferenceException` subclass. Pass arrays without copying
  (`usePinned` on Kotlin/Native, `GetPrimitiveArrayCritical` with a release on every path, or direct buffers on
  JNI). Never write temporary audio or image files.
- **On-device only.** No server or network inference.
- **Public API needs KDoc.** Every public class, function and property needs a `/** */` comment, or an `@property`
  / `@param` line in its class's KDoc.

## Tests

Every change ships with tests:

| Code | Tests |
| --- | --- |
| Pure Kotlin | `commonTest` with deterministic fakes (`./gradlew :<module>:allTests`) |
| C/C++ | CTest / GoogleTest suites in `native/tests`, run by each `build-host.sh` |
| Bindings | `jvmTest`, `androidUnitTest`, `iosSimulatorArm64Test` and (core-api, runtime) `macosArm64Test` against the real native libraries |
| Samples and UI | Compose UI tests and the binary symbol audit of `samples/whisper-only-minimal` |

Test fixtures must be small, synthetic or MIT / CC0 licensed, and recorded in [NOTICE](NOTICE). Never commit model
weights. Checks that need a real device or a person (for example listening to audio) are written down in the
module's README and listed in [`docs/MANUAL_VERIFICATION.md`](docs/MANUAL_VERIFICATION.md).

## Documentation

- Every Kotlin block in `README.md` is a region of a file in `docs-snippets/`, marked with
  `// snippet:start <name>` and `// snippet:end <name>`, and the README marks the block with
  `<!-- snippet: <name> -->`. Change both together; `./gradlew :docs-snippets:build` compiles the snippets against
  the published artifacts.
- `./scripts/ci/verify-docs.sh` checks Markdown links and anchors, the README snippets, the Maven coordinates and the
  KDoc coverage of the public API. Add `--online` to also check external links.

## Pull requests

1. Branch from `main`. Maintainers use `feat/issue-<N>-<slug>`.
2. Keep each pull request to one issue, and write `Closes #<N>` in its description.
3. Run `./scripts/ci-local.sh` and fix every `FAIL` before you push. It runs the same checks as the Linux CI job,
   and only rebuilds native code when native paths changed. Paste its summary into the pull request.
4. If you change `AGENTS.md`, `.agents/agents/` or `.agents/skills/`, run `./scripts/sync-agent-mirrors.sh` and
   `./scripts/verify-agent-parity.sh`.
5. Fill in the pull request template. A maintainer reviews the change and merges it with a merge commit once CI
   passes. Every ready pull request runs the Ubuntu, macOS and Windows CI jobs.

Commits made by AI coding agents keep the agent as author and credit the person it works for with a
`Co-authored-by:` trailer, which the hook from `./scripts/install-git-hooks.sh` adds.

## Licensing

Kggml is MIT licensed ([LICENSE](LICENSE)). By contributing you agree that your contribution is licensed under the
same terms. New or updated upstream C/C++ dependencies and new test fixtures must be recorded in [NOTICE](NOTICE),
with a license compatible with MIT; `scripts/ci/verify-licenses.sh` enforces this. Never commit secrets, signing
keys, local machine paths or model weights.
