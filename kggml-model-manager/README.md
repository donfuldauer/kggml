# `:kggml-model-manager`

Optional, pure-Kotlin helpers for getting models onto a device and deciding whether they fit, for Android, iOS and
JVM Desktop. The module has no native C/C++ code and does not depend on any engine module. It depends only on
`:kggml-core-api` (for `MemoryPressureObserver`) and `kotlinx-coroutines`.

| Feature | API | Notes |
| --- | --- | --- |
| Header inspection | `ModelInspector.inspect(source)` / `inspectFile(path)` | Reads the magic number and parses GGUF (v1–v3) or legacy whisper.cpp `ggml` headers. Tensor data is skipped by position, so weights are never loaded. It reports the parameter count, quantization (`Q4_K_M`, `F16`, …), context length and a `PeakRamEstimate`. |
| RAM estimation | `estimateLlmPeakRam(header, LlmRamEstimateOptions)`, `estimateWhisperPeakRam(header)` | Weights, KV cache (llama.cpp / whisper.cpp formulas), compute buffers and overhead. The compute and overhead parts are heuristics, meant for planning a load. |
| Resumable downloads | `ModelDownloader.download(DownloadRequest): Flow<DownloadEvent>` | Downloads in HTTP `Range` chunks (8 MiB by default) into `<dest>.part`. Network errors and 408/429/5xx are retried with backoff. It resumes after interruption or cancellation, and restarts if the server ignores `Range`. Size and SHA-256 are checked before an atomic rename to `<dest>`; on a mismatch the partial file is deleted. |
| SHA-256 | `Sha256` | Incremental and pure Kotlin. `Sha256.hashFileHex(fs, path)`. |
| Disk cache | `ModelCache(directory, maxBytes)` | Evicts least-recently-used files once the disk budget is exceeded. `pin(fileName)` protects files an engine has loaded. `clearPartialDownloads()`. |
| Low-memory eviction | `ResidentModelCache(budgetBytes).attachTo(observer)` | Holds loaded engines (any `AutoCloseable`). `MODERATE` trims idle models to half the budget; `CRITICAL` closes every idle model. Leased (in-use) models are never closed. |
| Platform I/O | `defaultModelFileSystem()`, `defaultHttpTransport()` | Android and JVM use `java.io`/`java.nio` and `HttpURLConnection`. iOS uses POSIX `stdio` and `NSURLSession`. Both interfaces are small, so you can plug in your own HTTP stack. |

## Download, verify and inspect

```kotlin
val cache = ModelCache(directory = "$appFilesDir/models", maxBytes = 8L shl 30)
val downloader = ModelDownloader()

downloader.download(
    cache.downloadRequest(
        url = "https://huggingface.co/.../resolve/main/model-Q4_K_M.gguf",
        fileName = "model-Q4_K_M.gguf",
        expectedSha256 = "…64 hex digits…",
    ),
).collect { event ->
    when (event) {
        is DownloadEvent.Started -> println("resuming at ${event.resumedFromBytes}")
        is DownloadEvent.Progress -> updateProgressBar(event.fraction)
        is DownloadEvent.Verifying -> println("verifying SHA-256…")
        is DownloadEvent.Completed -> println("ready: ${event.path}")
    }
}

val info = ModelInspector.inspectFile(cache.pathFor("model-Q4_K_M.gguf"))
println("${info.parameterCount} params, ${info.quantization}, ~${info.estimatedPeakRam.totalBytes shr 20} MiB")
```

If collection is cancelled or the network fails for good, the `.part` file is kept and the next `download()` of the
same request resumes from it. Failures are thrown from the flow:
- `ChecksumMismatchException` and `DownloadSizeMismatchException`: the partial file is deleted.
- `DownloadHttpException`: for example 404.
- `DownloadProtocolException`: the server broke the Range protocol.
- `DownloadTransportException`: the retries were used up.

## Evicting loaded models on low memory

```kotlin
val observer = AndroidMemoryPressureObserver(context)   // iOS: IosMemoryPressureObserver(); desktop: ManualMemoryPressureObserver()
val resident = ResidentModelCache(budgetBytes = 3L shl 30)
resident.attachTo(observer)

val info = ModelInspector.inspectFile(path)
resident.getOrPut(path, info.estimatedPeakRam.totalBytes) { LlamaEngine(modelPath = path) }.use { lease ->
    lease.resource.generateText("Hello")
}
// While nothing holds a lease, an onTrimMemory / memory warning closes the engine.
```

`IosMemoryPressureObserver` lives in `:kggml-core-api` (`iosMain`) next to `AndroidMemoryPressureObserver`. It
reports every `UIApplicationDidReceiveMemoryWarningNotification` as `CRITICAL`.

## Tests

- `commonTest` runs on JVM, Android unit tests and the iOS simulator. It covers:
  - synthetic GGUF v1/v2/v3 and Whisper header byte arrays: valid files, every truncation point, corrupt magic and
    version, unknown types, implausible counts;
  - SHA-256 test vectors;
  - downloader scenarios against an in-memory scripted HTTP server and file system: resume, dropped connections,
    retry and backoff limits, Range ignored, unknown size, 416, SHA-256 and size mismatches, cancellation;
  - disk-cache LRU and pinning;
  - simulated `MemoryPressureLevel` eviction.
- `jvmTest` runs the real `HttpURLConnection` transport and `java.nio` file system against a local Range-capable
  `com.sun.net.httpserver` server, including dropped connections, cancellation followed by resume, and servers
  without Range support.

## Manual verification

These checks are deferred to the owner's final acceptance pass (`docs/MANUAL_VERIFICATION.md`).

1. **Real download on devices.** On a physical Android phone and iPhone, start a download of a real
   ~500 MB GGUF with `expectedSha256` set.
   - Toggle airplane mode mid-download, then turn it off and collect again: the download resumes from the `.part`
     size rather than from 0.
   - Kill the app mid-download and relaunch: the download resumes again.
   - Then try a wrong SHA: it must fail and leave no `.part` file.
2. **Low-memory eviction.**
   - Android: run `adb shell am send-trim-memory <package> RUNNING_CRITICAL` with an idle engine in
     `ResidentModelCache`; the engine must be closed and its RSS released.
   - iOS: use Simulator → Debug → Simulate Memory Warning, and check the same on a device under memory pressure.
3. **RAM estimate sanity.** For `ggml-tiny.bin`, `ggml-base.en-q5_1.bin` and one 1–3B Q4_K_M GGUF, compare
   `estimatedPeakRam.totalBytes` with the peak RSS after the first transcription or generation. The estimate should
   be within about ±30%.
