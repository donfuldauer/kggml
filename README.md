# Kggml

Kggml is a Kotlin Multiplatform SDK for running AI models locally on **Android**, **iOS**, **macOS** and **JVM
desktop** (Linux, Windows, macOS). It wraps three C/C++ inference libraries from the
[ggml](https://github.com/ggml-org/ggml) ecosystem behind one Kotlin API:

| Task | Upstream library | Kggml module |
| --- | --- | --- |
| Speech-to-text, one-shot and live streaming | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | `kggml-whisper` |
| Text generation, structured output, embeddings, image understanding | [llama.cpp](https://github.com/ggml-org/llama.cpp) (with `libmtmd` for images) | `kggml-llama` |
| Text-to-image generation | [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | `kggml-diffusion` |

Models run on the device's CPU or GPU (Metal, Vulkan, OpenCL) and, for Whisper on iOS, the Neural Engine through
Core ML. Nothing is sent to a server. Model files are not bundled: your app downloads them (the optional
`kggml-model-manager` module does this) or ships them itself.

> **Status: 0.1.0, pre-1.0.** The API may still change between minor versions. What has to happen before a stable
> 1.0.0 is tracked in issue #104.

## Contents

- [Project structure](#project-structure)
- [Platform support](#platform-support)
- [Installation](#installation)
- [Usage](#usage): [Whisper](#speech-to-text-kggml-whisper), [Llama](#text-generation-kggml-llama),
  [Diffusion](#image-generation-kggml-diffusion), [Model manager](#downloads-and-memory-kggml-model-manager)
- [Compute backends](#compute-backends)
- [Native libraries](#native-libraries)
- [iOS background audio](#ios-background-audio)
- [Memory and quantization](#memory-and-quantization)
- [Threading, cancellation and errors](#threading-cancellation-and-errors)
- [Samples](#samples)
- [Building from source](#building-from-source)
- [Contributing, security and license](#contributing-security-and-license)

## Project structure

Kggml is split into small Gradle modules so an app includes only the native code it uses.

```text
                         your app
                            │  com.donfuldauer:kggml-bom (one version for everything)
   ┌──────────────┬─────────┴─────┬──────────────────┐
   ▼              ▼               ▼                  ▼
kggml-whisper  kggml-llama     kggml-diffusion    kggml-model-manager
whisper.cpp    llama.cpp       stable-diffusion   (pure Kotlin)
               + libmtmd       .cpp
   │              │               │                  │
   └──────────────┴───────┬───────┘                  │
                          ▼                          │
                    kggml-runtime  ◄── optional GPU add-ons:
        the one shared copy of ggml     kggml-runtime-vulkan
        (CPU everywhere, Metal on       kggml-runtime-opencl
         Apple platforms)                            │
                          │                          │
                          ▼                          ▼
                    kggml-core-api  (pure Kotlin API types)
```

| Module | What it contains |
| --- | --- |
| `kggml-core-api` | The types every module shares: `InferenceEngine`, `BackendConfig`, `ComputeBackend`, event types, the `InferenceException` hierarchy and memory-pressure observers. Pure Kotlin. |
| `kggml-runtime` | The shared ggml native runtime (CPU backends, plus Metal on iOS and macOS), its loaders for each platform, and `KggmlRuntime` for querying versions, available backends and CPU threads. |
| `kggml-runtime-vulkan` | Optional GPU add-on: ggml's Vulkan backend for Android and desktop Linux x86_64. The runtime finds and registers it at load; no code is needed. |
| `kggml-runtime-opencl` | Optional Android GPU add-on: ggml's OpenCL backend (kernels tuned for Qualcomm Adreno), used through the device vendor's `libOpenCL.so`. |
| `kggml-whisper` | `WhisperEngine`: whisper.cpp transcription, live streaming with voice activity detection, Core ML encoder support. |
| `kggml-llama` | `LlamaEngine`: llama.cpp generation, sampling controls, grammars and JSON Schema output, KV-cache sessions, embeddings, vision-language models. |
| `kggml-diffusion` | `DiffusionEngine`: stable-diffusion.cpp image generation with progress and cancellation. |
| `kggml-model-manager` | Optional. Resumable, SHA-256-checked downloads, model header inspection, peak-RAM estimates, disk and memory caches. Pure Kotlin. |
| `kggml-bom` | Bill of materials that aligns the versions of all the modules above. |

How the pieces fit together:

- **One copy of ggml.** Each engine module contains only its upstream library and a thin C wrapper, linked
  dynamically against `kggml-runtime`. An app that uses only `kggml-whisper` contains no llama.cpp or
  stable-diffusion.cpp code, and an app that uses all three engines loads ggml once. CI checks the linkage with
  `readelf` / `nm` / `otool`.
- **Native bridges.** Android and the JVM call native code through JNI; iOS and macOS use Kotlin/Native cinterop.
  Every native entry point catches C++ exceptions and returns an error code that Kotlin turns into an
  `InferenceException`.
- **Pinned upstreams.** ggml, whisper.cpp, llama.cpp and stable-diffusion.cpp are git submodules under each module's
  `native/third_party/`, pinned to the commits listed in [`NOTICE`](NOTICE). CI checks that each submodule matches
  its pin.

The rest of the repository:

| Path | Contents |
| --- | --- |
| `kggml-*/native/` | CMake projects, presets and build scripts for each module's native code, with CTest suites |
| `samples/` | The [showcase app](samples/kggml-showcase/README.md) and the [Whisper-only sample](samples/whisper-only-minimal/README.md) |
| `integration-tests/` | Tests that load several engines into one process against the single shared runtime (not published) |
| `docs-snippets/` | Compiles every Kotlin snippet in this README through the BOM, the way an app would (not published) |
| `build-logic/` | Gradle convention plugins (Kotlin Multiplatform setup, publishing, signing) |
| `scripts/` | Local and CI scripts: builds, linkage and license audits, release staging |
| `docs/` | [Releasing](docs/RELEASING.md), [manual verification](docs/MANUAL_VERIFICATION.md) and design notes |

## Platform support

| Platform | Kotlin targets | Native ABIs | Compute backends |
| --- | --- | --- | --- |
| Android 8.0+ (API 26) | `androidTarget` | `arm64-v8a`, `x86_64` | CPU (ARM NEON). GPU on Android 9+ with the optional add-ons: Vulkan (`kggml-runtime-vulkan`, any Vulkan 1.1 driver) and OpenCL (`kggml-runtime-opencl`, devices whose vendor ships OpenCL, e.g. Adreno). |
| iOS | `iosArm64`, `iosSimulatorArm64` | device and Apple-silicon simulator | CPU (Accelerate / NEON), Metal, Core ML (Whisper encoder) |
| macOS 13.3+ (Kotlin/Native) | `macosArm64` (`kggml-core-api` and `kggml-runtime` so far; the engines follow in issue #93) | Apple silicon | CPU (Accelerate / NEON), Metal. Link the `macos-arm64` slice of `GgmlRuntime.xcframework`. |
| JVM desktop (JDK 17+) | `jvm` | macOS arm64, Linux x86_64 and arm64, Windows x86_64 and arm64 | CPU everywhere (arm64 builds need ARMv8.2 with dotprod: Raspberry Pi 5, Graviton 2+, Snapdragon X; not Raspberry Pi 4). Metal on macOS. Vulkan on Linux x86_64 with the optional `kggml-runtime-vulkan` add-on; Windows Vulkan is issue #92. |

What each engine can use:

| Engine | CPU | Metal | Vulkan | OpenCL | Core ML |
| --- | --- | --- | --- | --- | --- |
| `kggml-whisper` | yes | yes | yes | yes | Encoder only, with a `<model>-encoder.mlmodelc` next to the model |
| `kggml-llama` | yes | yes | yes | yes | No (falls back to Metal, then CPU) |
| `kggml-diffusion` | yes | yes | yes | yes | No (falls back to Metal, then CPU) |

## Installation

Kggml is published under the `com.donfuldauer` group. Import the BOM once, then add only the modules you need,
without versions. In a Kotlin Multiplatform module:

<!-- snippet: gradle-dependencies -->
```kotlin
sourceSets {
    commonMain.dependencies {
        // One version for every Kggml module.
        implementation(project.dependencies.platform("com.donfuldauer:kggml-bom:0.1.0"))

        // Pick only the engines you ship; unused engines add no native code to your app.
        implementation("com.donfuldauer:kggml-whisper")        // speech-to-text
        implementation("com.donfuldauer:kggml-llama")          // text generation, embeddings, vision
        implementation("com.donfuldauer:kggml-diffusion")      // text-to-image
        implementation("com.donfuldauer:kggml-model-manager")  // optional: downloads, RAM estimates
        implementation("com.donfuldauer:kggml-runtime-vulkan") // optional: GPU via Vulkan (Android, desktop Linux)
        implementation("com.donfuldauer:kggml-runtime-opencl") // optional: Android GPU (OpenCL, Adreno)
    }
}
```

The same coordinates work in an Android-only or JVM-only module (`dependencies { implementation(platform(...)) }`);
Gradle picks the right platform variant. Every engine depends on `kggml-core-api` and `kggml-runtime`, so you never
declare those yourself. On iOS, also embed the xcframeworks described under [Native libraries](#native-libraries).

## Usage

Engines are created by suspend factory functions that load the model on the engine's own thread. Always `close()`
an engine (or use `use { }`) to free its native memory.

### Speech-to-text (`kggml-whisper`)

Use whisper.cpp `ggml` models (`ggml-base.en.bin`, `ggml-small-q5_1.bin`, ...). One-shot transcription:

<!-- snippet: whisper-transcribe -->
```kotlin
// `pcm`: mono float samples in [-1, 1]. Any rate / channel count works; 16 kHz mono is passed zero-copy.
WhisperEngine(modelPath, BackendConfig.DEFAULT).use { engine ->
    val result = engine.transcribe(pcm, sampleRate = 16_000, options = TranscriptionOptions(language = "en"))
    println("${result.text} (${result.audioDurationMs} ms on ${result.backend})")
    result.segments.forEach { println("[${it.startMs}–${it.endMs} ms] ${it.text}") }
}
```

Live transcription from a microphone. Kggml downmixes and resamples to 16 kHz in native code. An energy-based voice
activity detector splits the audio into speech turns, so silence is never decoded. While a turn is in progress you
get `PartialHypothesis` events that may still change; when it ends you get `CommittedSegment`s that will not. If the
decoder falls behind, the input flow is suspended rather than dropping audio.

<!-- snippet: whisper-stream -->
```kotlin
// `microphone`: 16-bit PCM chunks straight from the platform recorder (AudioRecord, AVAudioEngine, javax.sound).
engine.transcribeStream(microphone, sampleRate = 48_000, channels = 1, options = StreamingOptions.DEFAULT)
    .collect { event ->
        when (event) {
            is TranscriptionEvent.PartialHypothesis -> showDraft(event.text)  // may still change
            is TranscriptionEvent.CommittedSegment -> appendLine(event.text)  // final
            is TranscriptionEvent.Completed -> println("${event.segmentCount} segments")
            is TranscriptionEvent.Error -> throw event.exception
        }
    }
```

`TranscriptionOptions` sets the language (`"auto"` to detect), translation to English, an initial prompt and the
sampling temperature. `StreamingOptions` tunes the voice activity detector (`VadConfig`), the partial-result interval
and the buffer size.

### Text generation (`kggml-llama`)

Use any llama.cpp GGUF model:

<!-- snippet: llama-generate -->
```kotlin
val engine = LlamaEngine(
    modelPath = modelPath,  // any llama.cpp GGUF, e.g. a 1–3 B Q4_K_M instruct model
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

// Structured output: decoding is constrained by a grammar generated from a JSON Schema.
val schema = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"""
val json = engine.generateText(
    "Which city is the Eiffel Tower in? Answer in JSON.",
    GenerationOptions(maxTokens = 32, grammar = LlamaGrammar.jsonSchema(schema)),
).text

val vector: FloatArray = engine.embed("on-device inference")  // L2-normalized by default
engine.close()
```

The engine passes the prompt to the model as given, so format it with the chat template your model expects.
`kggml-llama` also provides:

- sampling controls in `GenerationOptions` (temperature, top-k, top-p, min-p, repetition penalty, seed, stop
  sequences);
- grammars: `LlamaGrammar.gbnf(...)`, `LlamaGrammar.jsonSchema(...)` and `LlamaGrammar.json()`;
- prompt-prefix reuse, and `saveSession()` / `restoreSession()` to persist the KV cache of a conversation;
- vision-language models: pass `LlamaEngineOptions(projectorPath = "mmproj-....gguf")` and call
  `generate(prompt, images, options)` with in-memory `LlamaImage.Encoded` (PNG, JPEG, ...) or `LlamaImage.Rgb` images.

See [`kggml-llama/README.md`](kggml-llama/README.md) for details.

### Image generation (`kggml-diffusion`)

Use a stable-diffusion.cpp model (SD 1.x / 2.x / XL checkpoints, or Flux / SD 3.5 with their component files via
`DiffusionModelFiles`):

<!-- snippet: diffusion-generate -->
```kotlin
DiffusionEngine(modelPath, BackendConfig.DEFAULT).use { engine ->  // e.g. an SD 1.5 q8_0 GGUF
    val image: DiffusionImage = engine.generateImage(
        DiffusionGenerationParams(
            prompt = "a lighthouse on a cliff at sunset, oil painting",
            negativePrompt = "blurry, low quality",
            width = 512,
            height = 512,
            sampleSteps = 20,
            seed = 42,
        ),
    ) { progress -> println("${progress.phase} ${progress.step}/${progress.totalSteps}") }
    // Packed RGB8, row-major: image.width * image.height * 3 bytes.
    check(image.pixels.size == image.width * image.height * image.channels)
}
```

`generate(params)` returns a `Flow<DiffusionEvent>` with `Progress`, `Completed` and `Error` events if you prefer a
stream. Cancellation takes effect between denoising steps. See
[`kggml-diffusion/README.md`](kggml-diffusion/README.md).

### Downloads and memory (`kggml-model-manager`)

<!-- snippet: model-manager -->
```kotlin
val cache = ModelCache(directory = "$appFilesDir/models", maxBytes = 8L shl 30)  // 8 GiB disk budget
ModelDownloader().download(
    cache.downloadRequest(url = modelUrl, fileName = "model-Q4_K_M.gguf", expectedSha256 = sha256),
).collect { event ->
    when (event) {
        is DownloadEvent.Started -> println("resuming at ${event.resumedFromBytes} bytes")
        is DownloadEvent.Progress -> println("${((event.fraction ?: 0.0) * 100).toInt()} %")
        is DownloadEvent.Verifying -> println("verifying SHA-256")
        is DownloadEvent.Completed -> println("ready: ${event.path}")
    }
}

// Read the header only (no weights) to decide whether the model fits before loading it.
val path = cache.pathFor("model-Q4_K_M.gguf")
val info = ModelInspector.inspectFile(path)
println("${info.parameterCount} params, ${info.quantization}, ~${info.estimatedPeakRam.totalBytes shr 20} MiB peak")

// Close idle engines when the OS reports memory pressure.
// Android: AndroidMemoryPressureObserver(context); iOS: IosMemoryPressureObserver(); desktop: manual.
val observer = ManualMemoryPressureObserver()
val resident = ResidentModelCache(budgetBytes = 3L shl 30)
resident.attachTo(observer)
resident.getOrPut(path, info.estimatedPeakRam.totalBytes) { LlamaEngine(modelPath = path) }.use { lease ->
    println(lease.resource.generateText("Hello").text)
}
```

Downloads go to `<file>.part` in HTTP `Range` chunks. They resume after a cancellation, network failure or app
restart, and the size and SHA-256 are checked before the file is renamed into place. See
[`kggml-model-manager/README.md`](kggml-model-manager/README.md).

## Compute backends

Every engine takes a `BackendConfig` when it is created and reports the backend it actually got.
`BackendConfig(backend, gpuLayers, cpuThreads, useFlashAttention, useMmap)` is applied when the engine loads its
model:

| Field | Default | Meaning |
| --- | --- | --- |
| `backend` | `AUTO` | A `ComputeBackend`: `AUTO`, `CPU_ONLY`, `GPU_METAL`, `GPU_VULKAN`, `GPU_OPENCL` or `NPU_COREML`. `AUTO` picks the first available of Core ML, Metal, OpenCL, Vulkan, CPU. `CPU_ONLY` never creates a GPU backend. |
| `gpuLayers` | `ALL_GPU_LAYERS` (`-1`) | Layers to offload to the GPU (llama.cpp). Must be `0` or `-1` with `CPU_ONLY`. |
| `cpuThreads` | `AUTO_THREADS` (`-1`) | CPU threads. `AUTO_THREADS` uses the number of processors, capped at 8. |
| `useFlashAttention` | `false` | Flash-attention kernels where the backend supports them. |
| `useMmap` | `true` | Memory-map the weights instead of reading them into memory. |

`BackendConfig.CPU` and `BackendConfig.DEFAULT` are ready-made configurations. If the requested backend is not
available, the engine falls back along `ComputeBackend.fallback` (Core ML → Metal → CPU; Vulkan, OpenCL → CPU). Set
`allowBackendFallback = false` in the engine options to get a `BackendUnavailableException` instead. After loading,
`engine.activeBackend` tells you what was chosen, and `KggmlRuntime.availableBackends()` lists what this device and
build support.

### Built-in backends and GPU add-ons

A backend is built into `kggml-runtime` when nearly every app wants it and it costs little. It is a separate add-on
module when it is large, depends on the device's drivers, or many apps would rather leave it out.

- **Metal (iOS, macOS)** is built in: every supported Apple device has it, the driver is Apple's, and its shaders add
  about 0.6 MB (compiled on the device). `ComputeBackend.CPU_ONLY` never creates it.
- **Vulkan on Android** is the add-on `kggml-runtime-vulkan`: ggml's Vulkan backend carries tens of MB of precompiled
  shaders per ABI, and phone GPU drivers vary. Add the dependency and the runtime registers it when it loads. On
  devices without a usable Vulkan 1.1 driver (or Android 8.x) the engines keep running on the CPU. Emulators expose
  only software Vulkan, which ggml ignores, so there `AUTO` stays on the CPU. Imagination PowerVR GPUs (Pixel 10 and
  later) are excluded as well: their Vulkan driver computes ggml's shaders incorrectly and can hang a model load, so
  those phones use the CPU unless the `KGGML_VULKAN_ALLOW_POWERVR=1` environment variable is set.
- **OpenCL on Android** is the add-on `kggml-runtime-opencl` (`ComputeBackend.GPU_OPENCL`): ggml's OpenCL backend with
  the matrix-multiplication kernels Qualcomm tuned for Adreno, used through the device vendor's `libOpenCL.so`
  (declared optional in the add-on's manifest). It loads only where the vendor ships OpenCL (in practice Adreno
  phones; not Pixel phones or emulators), and `AUTO` prefers it over Vulkan there.
- **Vulkan on desktop Linux** is the same `kggml-runtime-vulkan` add-on: its jar carries the module for
  `linux-x86_64`, which the runtime registers at load. It needs the user's Vulkan loader and driver (any Mesa, NVIDIA
  or AMD install has them); without them the engines stay on the CPU. Windows is not packaged yet (issue #92). A
  runtime you build yourself can link Vulkan in instead (`KGGML_ENABLE_VULKAN=ON`); the add-on is then skipped.

## Native libraries

| Platform | How the native code is shipped |
| --- | --- |
| Android | The AARs carry their `.so` files for `arm64-v8a` and `x86_64`: `libggml*.so` and `libkggml_runtime*.so` in `kggml-runtime`, `libkggml_whisper*.so` in `kggml-whisper`, `libggml-vulkan.so` / `libggml-opencl.so` in the GPU add-ons, and so on. Nothing to configure. |
| JVM desktop | The `kggml-runtime` jar carries the ggml runtime for `linux-x86_64`, `linux-aarch64`, `windows-x86_64`, `windows-aarch64` and `macos-aarch64` under `/kggml/native/<os>-<arch>/` and extracts it on first use; `-Dkggml.native.path=<dir>` (or `KGGML_NATIVE_PATH`) points it at your own build instead. Each engine loads its JNI library from `-Dkggml.whisper.native.path` / `kggml.llama.native.path` / `kggml.diffusion.native.path` (or the matching `KGGML_*_NATIVE_PATH` variable), then from the `kggml.native.path` directory, then from `java.library.path`. The engines' `build-host.sh` scripts build these libraries. |
| iOS and macOS | The Kotlin frameworks link against `GgmlRuntime.xcframework` plus one xcframework per engine (`KggmlWhisper`, `KggmlLlama`, `KggmlDiffusion`). Embed the ones you use in your Xcode app target. Each release attaches them to its GitHub release with a `SHA256SUMS` file; the `build-ios-xcframework.sh` scripts build them from source. |

## iOS background audio

iOS does not let apps in the background use the GPU. If a background app submits Metal work, the command buffer
fails with `MTLCommandBufferErrorDomain` / `accessRevoked` and the app crashes. For recording or transcribing in the
background, keep Whisper off the GPU:

<!-- snippet: whisper-ios-background -->
```kotlin
// Background-safe from the start: whisper.cpp is loaded with use_gpu = false (Accelerate / NEON only).
val backgroundEngine = WhisperEngine(modelPath, BackendConfig.CPU)
check(!backgroundEngine.usesGpu)

// Or use Metal / Core ML in the foreground and drop to the CPU while the app is in the background.
val engine = WhisperEngine(modelPath, BackendConfig(backend = ComputeBackend.NPU_COREML))
val fallback = WhisperBackgroundFallback(engine, appScope)
// Call these from your lifecycle callbacks. On iOS, `engine.observeAppLifecycle(appScope)` wires them to the
// UIApplication notifications for you.
fallback.onEnterBackground()  // aborts in-flight GPU work, reloads with ComputeBackend.CPU_ONLY
fallback.onEnterForeground()  // restores the previous BackendConfig
```

With `ComputeBackend.CPU_ONLY`, `whisper_context_params.use_gpu` is `false` before the model loads, so whisper.cpp
runs on Apple Accelerate and ARM NEON only. Smaller models (`tiny`, `base`) keep CPU transcription fast. Also enable
the `audio` background mode in your app's `Info.plist` and keep an `AVAudioSession` active, or iOS suspends the app
anyway.

## Memory and quantization

Peak RAM is roughly the size of the model file plus the KV cache and compute buffers. These are typical figures
for planning. The Whisper figures come from whisper.cpp's memory table and the Stable Diffusion figure from
stable-diffusion.cpp's, both at the pinned commits. The LLM figures are computed for Llama 3.2 1B / 3B, Qwen2.5 3B / 7B,
Mistral 7B and Llama 3.1 8B. Use `ModelInspector.inspectFile(path).estimatedPeakRam` from `kggml-model-manager` to
estimate a specific Whisper or LLM file before loading it. Its estimate is deliberately conservative: it counts logits
for a full batch and, without flash attention, the whole attention-score buffer, so for LLMs it reads a few hundred MB
above these figures. For diffusion models it counts only the weights plus a fixed overhead.

| Model | File size | Approximate peak RAM |
| --- | --- | --- |
| Whisper `tiny` / `base` (F16) | 75 MB / 142 MB | 0.3 GB / 0.4 GB |
| Whisper `small` (F16, or Q5_1) | 466 MB (190 MB) | 0.9 GB (0.5 GB) |
| Whisper `medium` / `large-v3` (Q5_0) | 515 MB / 1.1 GB | 1.0 GB / 1.8 GB |
| LLM, 1 B parameters, Q4_K_M, 4 K context | 0.8 GB | 1.2 GB |
| LLM, 3 B parameters, Q4_K_M, 4 K context | 1.9–2.0 GB | 2.4–2.8 GB |
| LLM, 7–8 B parameters, Q4_K_M, 4 K context | 4.4–4.9 GB | 5.2–5.8 GB |
| Stable Diffusion 1.5, Q8_0, 512×512 | 1.6 GB | 2.1 GB (1.6 GB with flash attention) |

Guidelines:

- **Quantization.** For LLMs, `Q4_K_M` is the usual trade-off between size and quality on phones; `Q5_K_M` and
  `Q8_0` are larger and closer to full quality. Whisper models work well at `Q5_1` / `Q5_0`. Kggml runs any
  quantization the pinned upstream libraries support.
- **Phones.** Plan for about half of the device's RAM. On a 6–8 GB phone that means a Whisper `base` / `small` model
  or a 1–3 B LLM; 7–8 B models need 12 GB or more.
- **Context length.** The KV cache grows with `LlamaContextConfig.contextLength`. `MODEL_CONTEXT_LENGTH` uses the
  model's training length, which can be very large; set an explicit length on mobile.
- **Memory pressure.** `AndroidMemoryPressureObserver` (`onTrimMemory`) and `IosMemoryPressureObserver` (memory
  warnings), together with `ResidentModelCache`, close idle engines when the OS runs low on memory.

## Threading, cancellation and errors

- Engines are ordinary `AutoCloseable` objects, not singletons; any number can be open at once. Each engine owns one
  thread, and loading, inference and `close()` run on it one at a time because ggml contexts are not thread-safe.
  Different engines run in parallel.
- Cancelling the coroutine that collects a flow (or calls `transcribe` / `generateText`) sets the native abort flag.
  Whisper and llama stop within one ggml graph step; diffusion stops after the denoising step in progress. The
  engine stays usable.
- `abortActiveTranscription()` / `abortActiveGeneration()` and `close()` can be called from any thread. They end the
  running call with `InferenceCancelledException`.
- Audio and image buffers are passed to native code without copies where the format allows it; nothing is written to
  temporary files.
- Native failures surface as subclasses of `InferenceException`: `ModelLoadException`, `InvalidInputException`,
  `BackendUnavailableException`, `OutOfMemoryException`, `InferenceCancelledException` and
  `NativeFailureException`. C++ exceptions never cross into Kotlin.

## Samples

- The [showcase app](samples/kggml-showcase/README.md) is a Compose Multiplatform app for Android, iOS and desktop
  that uses every module: a backend inspector, a model manager with downloads, live Whisper dictation, LLM chat with
  structured output and vision, and an image generation studio.
- The [Whisper-only sample](samples/whisper-only-minimal/README.md) is the smallest integration. CI uses it to check
  that an app depending on one engine contains no code from the others.

## Building from source

You need JDK 17, CMake 3.24+ and a C++17 compiler. Android builds also need the Android SDK and NDK; iOS and macOS
builds need macOS with Xcode.

```bash
git clone --recurse-submodules https://github.com/donfuldauer/kggml.git
cd kggml

# Native libraries for this computer (runtime and engines), with their CTest suites
kggml-runtime/native/scripts/build-host.sh
kggml-whisper/native/scripts/build-host.sh
kggml-llama/native/scripts/build-host.sh
kggml-diffusion/native/scripts/build-host.sh

# Kotlin tests on the JVM, against the native libraries built above
./gradlew jvmTest -Pkggml.requireNative=true

# Every Linux CI check (what pull requests must pass)
./scripts/ci-local.sh
```

Each `native/README.md` ([runtime](kggml-runtime/native/README.md), [whisper](kggml-whisper/native/README.md),
[llama](kggml-llama/native/README.md), [diffusion](kggml-diffusion/native/README.md)) covers the Android
(`build-android.sh`), iOS (`build-ios-xcframework.sh`) and cross-compiled builds. The GPU add-ons are described in
[`kggml-runtime-vulkan/README.md`](kggml-runtime-vulkan/README.md) and
[`kggml-runtime-opencl/README.md`](kggml-runtime-opencl/README.md). [`docs/RELEASING.md`](docs/RELEASING.md) explains
how releases are built, signed and published to Maven Central.

## Contributing, security and license

- [CONTRIBUTING.md](CONTRIBUTING.md) explains how to set up the project, the coding rules and how pull requests are
  checked.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
- Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

Kggml is released under the [MIT License](LICENSE). The bundled upstream libraries (ggml, whisper.cpp, llama.cpp
with libmtmd, stable-diffusion.cpp) are MIT-licensed. Some of the code they vendor uses other permissive licenses:
stb_image (MIT or public domain), miniaudio (MIT-0 or public domain), miniz (MIT, with parts under the Unlicense) and
darts-clone (BSD-3-Clause). See [NOTICE](NOTICE) for every third-party component, its license and its pinned
commit.
