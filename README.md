# Kggml

**Modular On-Device AI SDK for Kotlin Multiplatform (`com.donfuldauer.kggml`)**

Kggml wraps the native `ggml` ecosystem (`whisper.cpp`, `llama.cpp`, `libmtmd`, and `stable-diffusion.cpp`) for **Android**, **iOS**, and **JVM Desktop** with three core architectural guarantees:

1. **Strict Binary Modularity via KMP BOM:** Import `com.donfuldauer:kggml-bom:1.0.0` and pull only the inference engines you use (for example, `:kggml-whisper` alone) with zero unused LLM or diffusion native binaries linked.
2. **Explicit Runtime Compute Backend Control:** Per-instance `BackendConfig` supporting `AUTO`, `CPU_ONLY`, `GPU_METAL`, `GPU_VULKAN`, and `NPU_COREML`—including `CPU_ONLY` (`use_gpu = false`) on iOS for Whisper to prevent background Metal `accessRevoked` aborts.
3. **Zero-Copy Streaming Pipelines:** Live PCM `FloatArray`/`ShortArray` audio streaming with native resampling, ring-buffer VAD gating, and incremental `Flow<TranscriptionEvent>` / `Flow<TokenEvent>` emissions without temporary WAV files.

## Module Topology

| Module | Artifact Coordinate | Description |
| --- | --- | --- |
| `:kggml-bom` | `com.donfuldauer:kggml-bom` | Gradle `java-platform` BOM managing all module versions |
| `:kggml-core-api` | `com.donfuldauer:kggml-core-api` | Pure Kotlin interfaces (`InferenceEngine`, `BackendConfig`, events, errors) |
| `:kggml-runtime` | `com.donfuldauer:kggml-runtime` | Shared dynamic `ggml` native runtime (`ggml-base`, `cpu`, `metal`, `vulkan`) |
| `:kggml-whisper` | `com.donfuldauer:kggml-whisper` | `whisper.cpp` engine with native resampler, VAD ring buffer, and streaming `Flow` |
| `:kggml-llama` | `com.donfuldauer:kggml-llama` | `llama.cpp` + `libmtmd` (VLM) engine with KV-cache snapshots, GBNF/JSON, and embeddings |
| `:kggml-diffusion` | `com.donfuldauer:kggml-diffusion` | `stable-diffusion.cpp` engine for on-device image generation |
| `:kggml-model-manager` | `com.donfuldauer:kggml-model-manager` | Optional pure-Kotlin resumable downloads, SHA-256 verification, GGUF/Whisper header RAM estimation, and low-memory eviction hooks |
| `:samples:kggml-showcase` | *(sample)* | Full Compose Multiplatform showcase app (Android, iOS, JVM Desktop) |
| `:samples:whisper-only-minimal` | *(sample)* | Minimal Whisper-only app used in CI binary-symbol leakage audits |

## Development Status

Kggml is currently under active development across five tracked GitHub Milestones:

- **Phase 1:** Build System, KMP BOM, Shared `ggml` Runtime, & Sample Scaffolding
- **Phase 2:** `:kggml-whisper` Core & Runtime Backend Toggle
- **Phase 3:** `:kggml-whisper` Live Streaming Pipeline
- **Phase 4:** `:kggml-llama`, `:kggml-diffusion`, `:kggml-model-manager`, & Full Showcase
- **Phase OSS:** Public Launch, Licensing Audit, & Maven Central Publishing

See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for architecture rules, testing requirements, and multi-agent development workflows.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
