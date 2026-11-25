# Kggml Showcase

Compose Multiplatform app (Android, iOS, JVM Desktop) that consumes the SDK only through `kggml-bom`. Screens:

| Screen | Phase | What it shows |
| --- | --- | --- |
| Hardware Backend Inspector | 1 | Compute backends, thread counts and version metadata from `KggmlRuntime`. |
| **Model Manager** | 4 | Curated catalog, resumable downloads, SHA-256 status, GGUF / Whisper header metadata with a peak-RAM estimate, and manual / OS low-memory eviction, with `:kggml-model-manager`. |
| **Live Whisper Dictation & Audio Studio** | 3 | Live microphone dictation and PCM file transcription with `:kggml-whisper`. |
| **LLM Chat, Structured Output & Vision** | 4 | Multi-turn streaming chat, thread / GPU-layer controls, KV-cache snapshots, JSON-schema extraction with live validation, embeddings, and in-memory VLM image analysis with `:kggml-llama` (llama.cpp + libmtmd). |
| **Diffusion Image Studio** | 4 | Text-to-image with prompt / negative prompt, steps, CFG, seed, live progress, cancel and PNG export with `:kggml-diffusion`. |

All screens run on Android, iOS and JVM Desktop. Their state lives in plain-Kotlin controllers created once in
`ShowcaseApp` (so dictation, downloads and generations keep running while another screen is shown); every engine is
behind a small interface (`Transcriber`, `ChatModel`, `ImageGenerator`, `ModelStore`) whose production adapter wraps
the SDK and whose test fake is deterministic. Native model loading and file I/O run on the I/O dispatcher, engines are
closed when the app is disposed, and cancelling a coroutine job aborts native work through the engine's abort
callback.

## Model Manager

- **Catalog** (`ModelCatalog.kt`): Whisper tiny.en, Qwen2.5 0.5B Instruct Q4_K_M, SmolVLM 256M Instruct + its
  `mmproj` projector, and Stable Diffusion v1.5, all public files on Hugging Face. Sizes are approximate and for display
  only. No SHA-256 is published in the catalog yet, so those downloads are labelled **Unverified (no published
  SHA-256)**: the transfer is still size-checked, the digest is computed and shown, and an entry with a `sha256` is
  verified and rejected on mismatch. No model is ever committed to the repository.
- **Downloads** use `ModelDownloader` (HTTP `Range` chunks into `<file>.part`). The row shows bytes, percent and a
  progress bar; **Pause** cancels the collection and keeps the `.part` file, **Resume** continues from it. Phases:
  Downloading → Verifying SHA-256 → Installed (or Failed with the reason; a SHA-256 mismatch deletes the partial file).
  **Delete partial downloads** removes abandoned `.part` files.
- **Verification status:** *SHA-256 verified*, *Unverified (no published SHA-256)*, *Not checked this session* (files
  found on disk at start-up) or *SHA-256 mismatch*. **Verify SHA-256** re-hashes an installed file.
- **Metadata:** installed models are inspected with `ModelInspector` (header only, weights are never loaded): format,
  architecture, name, parameter count, quantization, context length, and the **estimated peak RAM** with its weights /
  KV cache / compute / overhead breakdown. Files that are not GGUF / Whisper ggml (e.g. `.safetensors`) say so.
  **Inspect a model file** does the same for any path.
- **Use in Chat / Whisper / Diffusion** hands the installed path to the matching screen (a vision LLM also gets its
  installed projector).
- **Loaded engines & low-memory eviction:** the Chat and Diffusion engines live in a `ResidentModelCache` (budget:
  half the physical RAM) subscribed to the platform's memory-pressure observer — `AndroidMemoryPressureObserver`
  (`onTrimMemory`), `IosMemoryPressureObserver` (memory warnings), or a manual observer on desktop. `MODERATE` trims
  idle engines to half the budget, `CRITICAL` closes every idle engine; an engine in the middle of a generation is
  never closed and an evicted one is reloaded on next use (the Chat screen says so). **Trim to half budget** and
  **Evict all idle engines** evict directly; **Simulate OS low-memory warning** emits `CRITICAL` through the platform
  observer, exactly the path a real OS signal takes. The eviction log shows the source, level and evicted engines.
- **Storage:** Android app files dir (`files/models`), iOS Application Support (`kggml/models`), desktop
  `~/.kggml/models` (override with `-Pkggml.models.dir=…` on `run`, `-Dkggml.models.dir` or `KGGML_MODELS_DIR`).

## LLM Chat, Structured Output & Vision

- **Model controls:** GGUF model path, optional vision projector (`mmproj`), prompt template (ChatML, Gemma,
  User / Assistant for SmolVLM), system prompt, compute backend, **CPU threads** (Auto or 1…), **GPU layers** (All,
  or a count; always 0 on CPU) and max tokens. **Load model / Reload with these settings** applies them right away;
  otherwise a changed setting reloads the engine on next use (the old one is closed). The card shows the backend in
  use, threads, GPU layers, context, vision support and whether the engine is resident or was evicted.
- **Chat:** every turn re-renders the whole conversation with the template and streams `Flow<TokenEvent>` into an
  italic "streaming" bubble; the engine reuses the KV cache of the common prefix, so only the new turn is evaluated.
  **Cancel** aborts the decode natively (the partial reply is kept). **Clear** empties the conversation and the KV
  cache.
- **KV-cache snapshots:** **Save KV snapshot** stores `LlamaEngine.saveSession()` bytes in memory with the
  conversation; **Restore** loads them back with `restoreSession()` and the conversation it belonged to.
- **Structured output:** a JSON schema (editable, checked live) is compiled to a GBNF grammar with
  `LlamaGrammar.jsonSchema`, and a greedy constrained generation extracts JSON from the input text. The output (also
  editable) is validated live with a small JSON parser: *Incomplete* while it is streaming, *Invalid JSON*, *Schema
  violations* (`type`, `required`, `properties`, `items`, `enum`) or *Valid JSON matching the schema*.
- **Embeddings:** cosine similarity of `LlamaEngine.embed()` vectors of two texts.
- **Vision (libmtmd):** **Photo library** / **Camera** (Android `GetContent` / `TakePicturePreview`; iOS
  `UIImagePickerController`) or, on every platform, a file path read into memory. The image is passed as
  `LlamaImage.Encoded` bytes with `LlamaEngine.MEDIA_MARKER` in the prompt; **no temporary image file is written**.
  A model without a projector reports a clear error.

## Diffusion Image Studio

- Model path (`.gguf`, `.safetensors`, `.ckpt`), compute backend, prompt and negative prompt, **sampling steps**
  slider (1–50), **CFG scale** slider (1–20 in 0.5 steps), size (256 / 384 / 512), **seed** field with **Randomize**
  and a "new random seed for every image" switch (the chosen seed is always shown, so an image can be reproduced).
- **Generate** streams `DiffusionEvent.Progress` into a progress bar (`Denoising step n / N`, seconds per step) and
  renders the final RGB image (the SDK reports step progress, not intermediate latents). **Cancel** stops the
  generation at the next step boundary; the engine stays loaded.
- **Export PNG** encodes the image with a small pure-Kotlin PNG encoder (`PngEncoder`: stored deflate, CRC-32,
  Adler-32) and writes it to the path in the field (default: `~/Pictures` on desktop, the app's external
  `Pictures` dir on Android, Documents on iOS).

## Live Whisper Dictation & Audio Studio

- **Live dictation:** microphone PCM is captured in memory (Android `AudioRecord` 48 kHz mono float; iOS
  `AVAudioEngine` input tap at the hardware format, usually 48 kHz; Desktop `javax.sound.sampled` 44.1 kHz 16-bit,
  mono, or stereo when the input only supports stereo). Chunks are streamed as a `Flow<FloatArray>` straight into
  `WhisperEngine.transcribeStream`, which resamples and downmixes them natively. **No WAV or other temporary file is
  ever written.**
- **Transcript:** the unstable `PartialHypothesis` of the turn in progress is shown in italics with its lag behind the
  microphone (captured audio minus the hypothesis end; target < 400 ms). Each `CommittedSegment` is appended as a
  locked, timestamped line (`m:ss.s → m:ss.s`). **Stop** ends capture and commits the last turn; **Cancel**
  abandons the stream.
- **Waveform visualizer:** a scrolling bar chart of the RMS level of the most recent chunks.
- **Backend switcher:** `Auto`, `CPU`, `Metal`, `Vulkan`, `Core ML`. Selecting a backend while a model is loaded and
  idle reloads it right away (`WhisperEngine.reinitialize`); the line below shows the backend actually in use and
  `use_gpu`. Unavailable backends fall back (Core ML → Metal → CPU, Vulkan → CPU).
- **iOS background stress toggle:** by default the studio installs `WhisperEngine.observeAppLifecycle`, which moves
  the engine to `CPU_ONLY` while the app is in the background. With **Background stress test** on, that fallback is
  removed so the selected backend keeps running in the background: `CPU` keeps transcribing, while `Metal` runs into
  iOS revoking GPU access (the stream ends with an in-band error, or iOS aborts the process).
- **File transcription:** a WAV file (16-bit PCM or 32-bit float, any rate / channel count) or raw 16 kHz mono
  16-bit PCM is read into memory and transcribed in one pass.

## Running

```bash
# Native libraries for the desktop app (once): host runtime + engines
./kggml-runtime/native/scripts/build-host.sh
./kggml-whisper/native/scripts/build-host.sh
./kggml-llama/native/scripts/build-host.sh
./kggml-diffusion/native/scripts/build-host.sh

# Desktop, pre-filling the Whisper model path (or set KGGML_WHISPER_MODEL) and choosing the Model Manager directory
# (default ~/.kggml/models, or set KGGML_MODELS_DIR)
./gradlew :samples:kggml-showcase:run -Pkggml.whisper.model=/path/to/ggml-tiny.en.bin -Pkggml.models.dir=/path/to/models

# Android (grant the microphone permission when asked; push a model with adb and type its path)
./gradlew :samples:kggml-showcase:installDebug -Pkggml.bundleNative=true
```

The desktop app passes `-Dkggml.{native,whisper.native,llama.native,diffusion.native}.path` for every host build
that exists. Android needs the `INTERNET` permission (declared) for downloads; the camera uses the system camera app
through `TakePicturePreview`, so no `CAMERA` permission is declared.

iOS: embed `KggmlShowcase.framework` together with `GgmlRuntime.xcframework`, `KggmlWhisper.xcframework`,
`KggmlLlama.xcframework` and `KggmlDiffusion.xcframework` in an Xcode host app whose `Info.plist` contains
`NSMicrophoneUsageDescription`, `NSCameraUsageDescription` (VLM camera) and, for background dictation,
`UIBackgroundModes` = `audio`. A `ggml-tiny.en.bin` bundled as an app resource is pre-filled as the Whisper model path.

## Automated tests

```bash
./gradlew :samples:kggml-showcase:desktopTest        # controller state tests + Compose UI tests
./gradlew :samples:kggml-showcase:testDebugUnitTest  # controller state tests on Android
```

All tests use deterministic fakes: no native libraries, no network, no model files.

- **commonTest** (desktop and Android unit tests):
  - `WhisperStudioControllerTest` (`WhisperFakes.kt`): start / stop / cancel, backend switching and reload, partial
    vs. committed rendering, file transcription, model-load / microphone / in-band stream errors, background stress.
  - `ModelManagerControllerTest` (`ModelFakes.kt`, an in-memory `ModelStore` with scripted `DownloadEvent`s): scan of
    installed / partial files, progress → verify → install → inspect, pause and resume from the `.part` file,
    SHA-256 mismatch, network failure, re-verification, delete / clear partials, "use", path inspection, manual and
    simulated-OS eviction (leased engines survive).
  - `LlmChatControllerTest` (`LlmFakes.kt`): multi-turn streaming and prefix-preserving prompts, cancel mid-stream,
    in-band and load errors, thread / GPU-layer / backend reloads, snapshot save / restore, clear, eviction and
    reload, JSON-schema extraction (grammar, greedy, live validation, schema violations, invalid schema), embedding
    similarity, VLM bytes from a path / gallery / camera, missing projector.
  - `DiffusionStudioControllerTest` (`DiffusionFakes.kt`): parameter clamping, progress then image, seed handling,
    cancel mid-generation, in-band / load / parameter errors, backend reload, eviction, PNG export and its failure.
  - `JsonValidationTest`, `PromptTemplateTest`, `PngEncoderTest` (decodes the PNG and checks every CRC / Adler-32),
    `FormatTest`.
- **desktopTest** (Compose UI, `runComposeUiTest` with `UiTestSupport.kt`'s `FakeShowcase`):
  `ModelManagerUiTest`, `LlmChatUiTest`, `DiffusionStudioUiTest`, `WhisperStudioUiTest`, `ShowcaseAppUiTest` (every
  screen reachable, work continues across navigation); plus `DiskModelStoreDesktopTest` (real file system, synthetic
  GGUF header) and the exported PNG decoded by `javax.imageio`.

## Manual verification (deferred to the final acceptance pass)

These need a person and real hardware, so they are listed in [`docs/MANUAL_VERIFICATION.md`](../../docs/MANUAL_VERIFICATION.md).

### Model Manager: real downloads on devices

1. Install the showcase on a physical Android phone (`./gradlew :samples:kggml-showcase:installDebug
   -Pkggml.bundleNative=true`) and on an iPhone (host app as above); open **Models**.
2. **Download** *Qwen2.5 0.5B Instruct Q4_K_M*. Expected: bytes / percent / progress bar advance; the row ends in
   *Installed* with *Unverified (no published SHA-256)*, a SHA-256 prefix, and metadata (`GGUF v3 · qwen2`,
   ~0.49 B parameters, `Q4_K_M`, a peak-RAM estimate).
3. Start *SmolVLM 256M*, press **Pause** at ~30%, then **Resume**: the percentage continues from where it stopped (not
   from 0). Repeat with airplane mode turned on mid-download (the row shows the error and stays resumable), then off
   and **Resume**. Kill the app mid-download, relaunch: the row shows *Paused (resumable)* with the `.part` size.
4. **Verify SHA-256** on an installed model and compare the digest with the "SHA256" shown on the file's Hugging Face
   page.
5. **Use in Chat** on the LLM: the Chat screen opens with the path filled in.

### Model Manager: OS low-memory eviction on devices

1. On Android, load a model in **Chat** (send one message), go to **Models**: one engine is resident. Run
   `adb shell am send-trim-memory com.donfuldauer.kggml.samples.showcase RUNNING_CRITICAL`. Expected: the eviction
   log shows `OS memory-pressure signal: CRITICAL → evicted llm:…`, the engine list is empty, the Chat card says
   *Evicted from memory*, and the app's RSS (`adb shell dumpsys meminfo …`) drops by roughly the model size. Sending a
   new chat message reloads the model and works.
2. Repeat with `RUNNING_MODERATE` with two engines loaded (Chat + Diffusion): only the least recently used one is
   closed if both exceed half the budget.
3. On iOS: Simulator → Device → Simulate Memory Warning (and on a device under memory pressure): same result.
4. Start a long generation and send the signal while it runs: the generation completes (leased engines are never
   evicted) and the engine is evicted only on a later signal.

### LLM Chat, structured output and vision on hardware

1. Desktop (macOS Metal, Linux Vulkan / CPU), Android and iPhone with *Qwen2.5 0.5B Instruct* (ChatML): hold a
   3-turn conversation. Expected: coherent streamed replies; the second turn starts streaming faster than the first
   (prefix reuse). **Cancel** mid-reply stops within ~1 token.
2. Set **CPU threads** to 2 and **GPU layers** to 0, **Reload**: the card shows the new values and generation still
   works (slower); set **CPU** backend: card shows `CPU (GPU off)`.
3. **Save KV snapshot**, send two more turns, **Restore**: the conversation returns to the snapshot and the next reply
   is consistent with it.
4. **Structured output** with the default schema: output is valid JSON `{"name": "Maria", "age": 34, "city": "Lisbon"}`
   (or equivalent) and the validation line reads *Valid JSON matching the schema*.
5. **Embeddings**: the default pair scores clearly higher than "The cat sat on the mat." vs "Quarterly revenue
   grew 4%.".
6. **Vision** with *SmolVLM 256M* + its projector (User / Assistant template): pick a photo from the library and take
   one with the camera; **Analyze image** describes each correctly. Confirm no image file appeared in the app's cache /
   temp directories (Android `adb shell run-as … ls cache files`, iOS app container `tmp/`).

### Diffusion Image Studio on hardware

1. Desktop (Metal / Vulkan / CPU), Android and iPhone with *Stable Diffusion v1.5* (or an SD 1.5 GGUF): 20 steps,
   CFG 7, seed 42, 512×512. Expected: the progress bar advances `0 … 20`, a coherent image of the prompt appears.
2. Generate again with seed 42: the same image. **Randomize** gives a different one, and the shown seed reproduces it.
3. Start a 50-step generation and **Cancel** at ~step 10: it stops within one step; a new generation works.
4. **Export PNG** and open the file in an image viewer: identical to the on-screen image.

### Live dictation latency and multi-minute continuity

1. Run the showcase on a desktop (or Android device) with `ggml-base.en.bin` or `ggml-tiny.en.bin` and the `Auto`
   backend.
2. Open **Whisper**, press **Start dictation** and read a text aloud continuously for at least **three minutes**,
   with occasional pauses.
3. Expected: the waveform moves with your voice; italic partial text appears while you speak and its reported lag
   stays mostly **below 400 ms** on a GPU backend; each pause locks a punctuated, timestamped committed line; the
   timestamps keep increasing across the whole run (including past 30 s windows) and wording stays consistent
   across pauses (prompt carry-over).
4. Press **Stop**: the last sentence is committed and the status reads `Completed: N segment(s), … of audio`.
5. Confirm nothing was written to disk, e.g. on desktop compare `find "$TMPDIR" /tmp -newer <marker> -name '*.wav'`
   before and after the run (expected: no new files).

### iOS background stress: `CPU_ONLY` vs. Metal

1. Build the iOS host app as above (with the `audio` background mode) and install it on a **physical iPhone**.
2. Select **CPU**, turn **Background stress test** on, start dictation, speak, and lock the phone / switch apps
   for 30 s while still speaking. Return: committed lines covering the background period are present and no error
   is shown.
3. Select **Metal**, keep the toggle on and repeat. Expected: the stream fails (in-band error card, e.g. a Metal /
   cancelled error) or iOS terminates the app — demonstrating why the fallback exists.
4. Turn the toggle off (fallback installed), keep **Metal** and repeat: the app is never terminated. If a GPU decode
   was running at the moment of backgrounding it is aborted and the stream ends with an `InferenceCancelledException`
   error card; otherwise dictation continues on the CPU in the background. On return the engine is back on Metal
   and a new dictation works normally.
