# `:kggml-diffusion`

On-device text-to-image generation with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) for
Android, iOS and JVM Desktop. stable-diffusion.cpp is linked dynamically against the shared ggml in `:kggml-runtime`;
see [`native/README.md`](native/README.md) for the native layer, its C ABI and the build scripts.

```kotlin
val engine = DiffusionEngine(
    modelPath = "/path/to/sd-v1-5-q8_0.gguf",
    backendConfig = BackendConfig(backend = ComputeBackend.AUTO),
)

val params = DiffusionGenerationParams(
    prompt = "a lighthouse on a cliff at sunset, oil painting",
    negativePrompt = "blurry, low quality",
    width = 512,
    height = 512,
    sampleSteps = 20,
    cfgScale = 7f,
    seed = 42,
)

engine.generate(params).collect { event ->
    when (event) {
        is DiffusionEvent.Progress -> println("${event.phase} ${event.step}/${event.totalSteps}")
        is DiffusionEvent.Completed -> show(event.image) // RGB8: width * height * 3 bytes
        is DiffusionEvent.Error -> throw event.exception
    }
}

// Or, without the event stream:
val image: DiffusionImage = engine.generateImage(params) { progress -> updateBar(progress.step, progress.totalSteps) }
engine.close()
```

## API

| Type | Purpose |
| --- | --- |
| `DiffusionEngine(modelPath, backendConfig, options)` / `DiffusionEngine(DiffusionModelFiles, ...)` | Suspend factories: load the weights on the new engine's own thread. `DiffusionModelFiles` takes either a full checkpoint (`modelPath`) or a standalone diffusion model (`diffusionModelPath`) plus optional VAE / TAESD / CLIP-L / CLIP-G / T5-XXL / LLM text-encoder files. |
| `generate(params): Flow<DiffusionEvent>` | Cold flow, one generation per collection: `Progress(phase, step, totalSteps, secondsPerStep)` (`step = 0` at the start, then one per denoising step; `VAE_TILES` progress follows with `vaeTiling = true`), then `Completed(image)` or a terminal `Error(InferenceException)`. Progress is buffered, so a slow collector never stalls the sampler. |
| `generateImage(params, onProgress)` | Collects `generate` and returns the `DiffusionImage`, throwing the typed `InferenceException` of an `Error`. |
| `DiffusionGenerationParams` | `prompt` (non-blank), `negativePrompt`, `width` / `height` (positive multiples of 8, `<= 16384`), `sampleSteps` (`1..10000`), `cfgScale` (finite, `>= 0`; `1` disables CFG), `distilledGuidance` (Flux-style models), `seed` (`< 0` = random), `sampler` / `scheduler` (`MODEL_DEFAULT` = stable-diffusion.cpp's per-model choice), `clipSkip`, `vaeTiling`. Validated on construction (`IllegalArgumentException`) before any native call. |
| `DiffusionImage` | `width`, `height`, `channels` (3) and packed, row-major RGB8 `pixels` without row padding. The bytes are copied out of the native buffer, which is freed (`kggml_diffusion_image_free`) before `generate` returns — on success, failure and cancellation alike. |
| `abortActiveGeneration()`, `close()`, `isGenerating`, `activeBackend`, `usesGpu`, `modelInfo`, `DiffusionEngine.nativeInfo()` | Control and introspection. `close()` is idempotent and safe from any thread. |

Errors are the typed `InferenceException`s of `:kggml-core-api`: `ModelLoadException` (missing / unreadable / not a
diffusion model, naming the file), `BackendUnavailableException`, `OutOfMemoryException`, `InvalidInputException`,
`NativeFailureException`, and `InferenceCancelledException` for `abortActiveGeneration()` / `close()`.

## Backends

`BackendConfig` is applied when the model is loaded:

| `BackendConfig` | stable-diffusion.cpp |
| --- | --- |
| `CPU_ONLY` | Every module on the ggml CPU device; no GPU backend is ever created (safe in the iOS background). |
| `GPU_METAL` / `GPU_VULKAN` | The `DiffusionEngineOptions.gpuDevice`-th device of that kind. `BACKEND_UNAVAILABLE` from native falls back along `ComputeBackend.fallback` (to CPU) unless `allowBackendFallback = false`. |
| `NPU_COREML` | Not supported by stable-diffusion.cpp: falls back to Metal / CPU, or fails with `BackendUnavailableException`. |
| `AUTO` | Metal, then Vulkan, then CPU. |
| `gpuLayers = 0` with a GPU backend | `offload_params_to_cpu`: weights stay in host memory and stream to the GPU per graph (stable-diffusion.cpp has no per-layer offload; any other value keeps the weights on the GPU). |
| `cpuThreads`, `useFlashAttention`, `useMmap` | `n_threads` (`AUTO_THREADS` = performance cores), flash attention in every module, mmap of the weight files. |

`DiffusionEngineOptions.keepTextEncoderOnCpu` / `keepVaeOnCpu` save GPU memory; `weightType` converts the weights at
load time (`sd_type_t`).

## Cancellation

Cancellation is cooperative at **denoising-step granularity**. Cancelling the collecting coroutine (or calling
`abortActiveGeneration()` / `close()`) latches the native abort flag of the engine's context
(`kggml_diffusion_request_abort`); stable-diffusion.cpp checks it at every progress point, so the step that is
computing when the abort arrives finishes first and the generation stops before the next one (with `vaeTiling`, also
between VAE tiles). stable-diffusion.cpp has no abort hook inside a graph computation at the pinned commit, so a single
step (and the non-tiled VAE decode) is never interrupted midway; on a phone a step of a 512×512 SD 1.5 model takes
roughly 1–5 s. A collector cancellation surfaces as `CancellationException`; `abortActiveGeneration()` / `close()` end
the flow with `DiffusionEvent.Error(InferenceCancelledException)`. No image buffer is produced or leaked for an
aborted run, and the engine stays usable: the latch is cleared at the start of the next generation.

## Threading

Every engine owns one dedicated thread (`SerialExecutor`); model loading, every generation and the native release in
`close()` run on it, one at a time. Engines are ordinary objects: any number can coexist and generate concurrently
(stable-diffusion.cpp's process-wide progress hook is dispatched per generating thread by the native layer).
Progress callbacks are delivered from native code on the engine thread and forwarded into the flow without blocking.

## Platform bindings

- **Android / JVM Desktop**: `kggml_diffusion_jni` (`native/jni/kggml_diffusion_jni.cpp`). Every entry point is
  `extern "C"` behind a `try` / `catch` returning `kggml_status`; prompts and paths cross as UTF-8 `byte[]`; progress
  calls `DiffusionProgressListener.onProgress(int, int, float)` by name (kept by `consumer-rules.pro`), and a Java
  exception there is cleared and turned into an abort. The JVM loader looks in `kggml.diffusion.native.path` /
  `KGGML_DIFFUSION_NATIVE_PATH`, then the runtime's `kggml.native.path`, then `java.library.path`. Android AARs bundle
  `libkggml_diffusion(+_jni).so` with `-Pkggml.bundleNative=true`.
- **iOS**: cinterop (`src/nativeInterop/cinterop/kggmlDiffusion.def`) against `KggmlDiffusion.xcframework`; the
  progress listener reaches the C callback through a `StableRef`, and the image is copied and freed in a `finally`.

## Tests

- `commonTest`: engine lifecycle and backend mapping / fallback, parameter forwarding (prompt, negative prompt, size,
  steps, CFG, guidance, seed, sampler, scheduler, clip skip, VAE tiling), progress / cancellation state machines
  (collector cancellation, `abortActiveGeneration`, close during and queued behind a generation, stale abort latches,
  native image buffers always released), error mapping and validation, against a deterministic fake of the C ABI.
- `jvmTest` / `iosSimulatorArm64Test`: `NativeDiffusionEngineContract` against the real library: pinned native
  version, missing / garbage / empty / directory model files → `ModelLoadException`, Core ML without fallback. No
  diffusion model small enough to commit exists, so real image generation is a manual check (below).
- `jvmTest` additionally drives a full `DiffusionEngine` through the real JNI bridge with the native test hooks of the
  host build (a model-less stub context + the wrapper's simulated sampler, which runs the real progress dispatch,
  latched abort, image hand-over and `kggml_diffusion_image_free` path): progress and RGB8 pixels across JNI,
  mid-generation cancellation and `abortActiveGeneration` at the native step, close during a generation, concurrent
  engines, a throwing progress listener, invalid JNI arguments, and the native live-context count returning to its
  baseline. It needs the host build (`native/scripts/build-host.sh`); `-Pkggml.requireNative=true` turns a missing
  library into a failure instead of a skip.
- `androidUnitTest`: the Android binding degrades into a typed `NativeFailureException` on the host JVM, and the Kotlin
  JNI declarations, the C++ entry points and the progress callback signature cannot drift apart.

## Manual verification (real model, physical devices)

Automated tests cannot generate a real image, so this is checked by hand during the final acceptance pass
(`docs/MANUAL_VERIFICATION.md`):

1. Download a small SD 1.5-class checkpoint (e.g. a q8_0 SD 1.5 GGUF, ~1.6 GB) to the desktop and to a physical
   Android device and iPhone (app sandbox).
2. Create `DiffusionEngine(modelPath, BackendConfig.DEFAULT)` and `DiffusionEngine(modelPath, BackendConfig.CPU)`;
   check `activeBackend` / `usesGpu` (Metal on iPhone / Apple Silicon, Vulkan on a supported Android GPU or desktop,
   CPU otherwise).
3. `generate(DiffusionGenerationParams(prompt = "a lighthouse on a cliff at sunset", sampleSteps = 20, seed = 42))`:
   `Progress` steps `0..20` arrive in order, then a 512×512 `Completed` image that shows the prompt (convert the RGB8
   pixels to a platform bitmap or write a PPM: `P6 512 512 255\n` + `pixels`).
4. Repeat with the same seed: the image is identical. Change `negativePrompt`, `cfgScale` and `seed`: the image changes
   accordingly.
5. Cancel the collecting coroutine after ~5 steps: no further `Progress` arrives after the step in flight (within one
   step's time), no image is produced, and a following generation on the same engine succeeds. Repeat with
   `abortActiveGeneration()` (flow ends with `InferenceCancelledException`) and with `close()` mid-generation.
6. On iPhone, with `BackendConfig.CPU`, background the app during a generation: it completes without a Metal
   `accessRevoked` abort.
7. Run 10 generations in a row and watch the process memory (Android Studio profiler / Xcode memory gauge / `top`): it
   returns to the same level after each image (no leaked native image buffers), and drops after `close()`.
