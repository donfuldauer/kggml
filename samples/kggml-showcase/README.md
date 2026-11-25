# Kggml Showcase

Compose Multiplatform app (Android, iOS, JVM Desktop) that consumes the SDK only through `kggml-bom`. Screens:

| Screen | Status | What it shows |
| --- | --- | --- |
| Hardware Backend Inspector | Phase 1 | Compute backends, thread counts and version metadata from `KggmlRuntime`. |
| **Live Whisper Dictation & Audio Studio** | Phase 3 | Live microphone dictation and PCM file transcription with `:kggml-whisper`. |
| Model Manager, LLM Chat / VLM, Diffusion Studio | Phase 4 placeholders | — |

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

### Running

```bash
# Native libraries for the desktop app (once): host runtime + whisper
./kggml-runtime/native/scripts/build-host.sh
./kggml-whisper/native/scripts/build-host.sh

# Desktop, pre-filling the model path (or set KGGML_WHISPER_MODEL)
./gradlew :samples:kggml-showcase:run -Pkggml.whisper.model=/path/to/ggml-tiny.en.bin

# Android (grant the microphone permission when asked; push a model with adb and type its path)
./gradlew :samples:kggml-showcase:installDebug -Pkggml.bundleNative=true
```

iOS: embed `KggmlShowcase.framework` together with `GgmlRuntime.xcframework` and `KggmlWhisper.xcframework` in an
Xcode host app whose `Info.plist` contains `NSMicrophoneUsageDescription` and, for background dictation,
`UIBackgroundModes` = `audio`. A `ggml-tiny.en.bin` bundled as an app resource is pre-filled as the model path.

### Automated tests

```bash
./gradlew :samples:kggml-showcase:desktopTest        # controller state tests + Compose UI tests
./gradlew :samples:kggml-showcase:testDebugUnitTest  # controller state tests on Android
```

`WhisperStudioControllerTest` (commonTest) and `WhisperStudioUiTest` (desktopTest) drive the studio with a
deterministic microphone and transcriber (`WhisperFakes.kt`): start / stop / cancel, backend switching and reload,
partial vs. committed rendering, file transcription, model-load / microphone / in-band stream errors, and the
background stress toggle.

### Manual verification (deferred to the final acceptance pass)

These need a person and real hardware, so they are listed in [`docs/MANUAL_VERIFICATION.md`](../../docs/MANUAL_VERIFICATION.md).

#### Live dictation latency and multi-minute continuity

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

#### iOS background stress: `CPU_ONLY` vs. Metal

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
