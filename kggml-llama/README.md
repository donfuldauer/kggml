# `:kggml-llama`

On-device text generation with [llama.cpp](https://github.com/ggml-org/llama.cpp) for Android, iOS and JVM Desktop.
`llama.cpp` is linked dynamically against the shared ggml in `:kggml-runtime`; see [`native/README.md`](native/README.md)
for the native layer, its C ABI and the build scripts.

```kotlin
val engine = LlamaEngine(
    modelPath = "/path/to/model.gguf",
    backendConfig = BackendConfig(backend = ComputeBackend.AUTO, gpuLayers = BackendConfig.ALL_GPU_LAYERS),
    contextConfig = LlamaContextConfig(contextLength = 4096),
)

engine.generate("Write a haiku about the sea.", GenerationOptions(maxTokens = 64)).collect { event ->
    when (event) {
        is TokenEvent.Token -> print(event.text)
        is TokenEvent.Completed -> println("\n[${event.stopReason}]")
        is TokenEvent.Error -> throw event.exception
    }
}
engine.close()
```

## Features

| Feature | API | Notes |
| --- | --- | --- |
| Streaming generation | `generate(prompt, options): Flow<TokenEvent>` | Cold flow; tokens stream as they are sampled, multi-byte UTF-8 characters split across tokens are held back until complete. Stops on end of generation, `maxTokens`, `stopSequences` or a full context (`StopReason`). `generateText()` collects it into a `GenerationResult`. |
| Cancellation | cancel the collector, `abortActiveGeneration()`, `close()` | Wired to `llama_set_abort_callback`: a running `llama_decode` stops within one graph step. Collector cancellation throws `CancellationException`; `abortActiveGeneration()` / `close()` end the flow with a terminal `TokenEvent.Error(InferenceCancelledException)`. |
| Sampling | `GenerationOptions` | temperature (`<= 0` = greedy), top-k, top-p, min-p, repetition penalty, seed. |
| Constrained decoding | `GenerationOptions.grammar` | `LlamaGrammar.gbnf(text)`, `LlamaGrammar.jsonSchema(schema)` (JSON Schema → GBNF in common Kotlin), `LlamaGrammar.json()`. |
| Prompt cache | `GenerationOptions.reusePromptCache` | The engine keeps one conversation in its KV cache; a new prompt only evaluates the part after the longest common token prefix. |
| KV-cache sessions | `saveSession(): ByteArray`, `restoreSession(bytes)`, `clearSession()` | Token history + `llama_state_seq_*` data in a versioned, CRC-32-checked container with a model fingerprint. Restore on any engine with the same model. Only restore sessions from trusted storage. |
| Embeddings | `embed(text, normalize = true): FloatArray` | Separate lazily created embedding context (pooling: `LlamaContextConfig.embeddingPooling`), never disturbs the generation session. |
| Tokenizer | `tokenize()`, `detokenize()` | |
| Vision-language models | `LlamaEngineOptions(projectorPath)`, `generate(prompt, images, options)` | libmtmd projector (`mmproj-*.gguf`) loaded next to the model on the same backend. `LlamaImage.Encoded` (PNG / JPEG / BMP / GIF / ... bytes) or `LlamaImage.Rgb` (packed RGB888) — in memory only, pinned zero-copy for preprocessing, never written to a temporary file. One `LlamaEngine.MEDIA_MARKER` per image in the prompt (prepended when absent). |

## Images (VLM)

```kotlin
val vlm = LlamaEngine(
    modelPath = "/path/to/SmolVLM-500M-Instruct-Q8_0.gguf",
    options = LlamaEngineOptions(projectorPath = "/path/to/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"),
)
check(vlm.supportsVision)
val photo: ByteArray = readJpegFromCameraOrGallery() // stays in memory
val prompt = "<|im_start|>User:${LlamaEngine.MEDIA_MARKER}What is in this picture?<end_of_utterance>\nAssistant:"
println(vlm.generateText(prompt, listOf(LlamaImage.Encoded(photo)), GenerationOptions(maxTokens = 96)).text)
```

The projector uses the engine's backend (`CPU_ONLY` creates no GPU backend for the image encoder either) and its
`cpuThreads` / `useFlashAttention`; `imageMinTokens` / `imageMaxTokens` bound dynamic-resolution projectors. An image
prompt starts a new session (image embeddings are not token ids, so there is no prefix to reuse); sessions with images
save and restore like text ones. Cancellation aborts the text decodes and is checked between image encodes. The
preprocessed images are freed as soon as they are in the KV cache, and the projector is freed by `close()` before
the model; invalid images, a marker / image count mismatch or a prompt that does not fit the context end the flow
with `InvalidInputException`.

## Backends

`BackendConfig` is applied when the model is loaded: `CPU_ONLY` hands llama.cpp an empty device list (no GPU backend
is ever created — safe in the iOS background), `GPU_METAL` / `GPU_OPENCL` / `GPU_VULKAN` offload `gpuLayers` layers
to `LlamaEngineOptions.gpuDevice`, `AUTO` prefers Metal, then OpenCL, then Vulkan, then CPU. Vulkan and OpenCL are only
available with the `:kggml-runtime-vulkan` / `:kggml-runtime-opencl` add-ons. llama.cpp has no Core ML path:
`NPU_COREML` falls back (or fails with `BackendUnavailableException` when `allowBackendFallback = false`).
`cpuThreads`, `useFlashAttention` and `useMmap` map to the llama.cpp context / model parameters.

## Threading

Every engine owns one dedicated thread (`SerialExecutor`); model loading, every inference call and the native
release in `close()` run on it, one at a time. Engines are ordinary objects: any number can coexist and run
concurrently. `abortActiveGeneration()` and `close()` are safe from any thread.

## Tests

- `commonTest`: engine lifecycle, generation, cancellation, sessions, embeddings and image prompts (projector
  loading, marker handling, chunk release, abort, invalid images) against a deterministic fake of the C ABI;
  JSON-schema → GBNF conversion; UTF-8 streaming; session container; option validation.
- `jvmTest` / `iosSimulatorArm64Test`: `NativeLlamaEngineContract` against the real library, driven by
  `TinyLlamaModel` (a synthetic 100 KB GGUF generated at test time, byte-identical to the native CTest fixture) and
  `TinyMmproj` (a synthetic 64 KB LLaVA-style vision projector for it, likewise byte-identical): real libmtmd
  in-memory RGB / BMP decoding, encoding, generation, cancellation and error codes. `jvmTest` also checks that no
  temporary file is written and that projector / chunk objects return to their live-object baseline.
  `jvmTest` needs the host build (`native/scripts/build-host.sh`); `-Pkggml.requireNative=true` turns a missing
  library into a failure instead of a skip.
- `androidUnitTest`: the Android binding degrades into a typed `NativeFailureException` on the host JVM.

## Manual verification (real models, physical devices)

Automated tests use a synthetic model, so output quality and GPU offload with real weights are checked by hand
during the final acceptance pass (`docs/MANUAL_VERIFICATION.md`):

1. Download a small instruct GGUF (e.g. a 0.5–1.5 B Q4_K_M model) to the device / desktop.
2. Load it with `BackendConfig.DEFAULT` and with `BackendConfig.CPU`; check `activeBackend` / `usesGpu` (Metal on
   iPhone / Apple Silicon, OpenCL on an Adreno GPU or Vulkan on a supported Android GPU with the matching add-on,
   CPU otherwise).
3. Stream a chat-formatted prompt with `GenerationOptions(maxTokens = 128)`: text must be coherent and arrive token by
   token; cancel the collecting coroutine mid-stream and confirm generation stops within ~100 ms.
4. Generate with `LlamaGrammar.jsonSchema(...)` for a small object schema; the output must parse and match the schema.
5. `saveSession()` after a first answer, `restoreSession()` on a fresh engine and ask a follow-up: it must answer in
   context without re-evaluating the earlier prompt (visibly faster first token).
6. `embed()` two paraphrases and one unrelated sentence with an embedding GGUF: the paraphrases' cosine similarity
   must be clearly higher.
7. VLM (#15): load a small vision model and its projector (e.g. SmolVLM-500M-Instruct Q8_0 + `mmproj`) with
   `BackendConfig.DEFAULT` and with `BackendConfig.CPU`. Describe a JPEG taken with the camera (`LlamaImage.Encoded`)
   and an `Rgb` bitmap converted from a platform image: the answers must describe the pictures. Abort a description
   mid-stream (stops promptly after the image encode) and confirm the app's cache / temp directories gain no files.
