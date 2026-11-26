package com.donfuldauer.kggml.docs

import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.llama.LlamaEngine
import com.donfuldauer.kggml.models.DownloadEvent
import com.donfuldauer.kggml.models.ModelCache
import com.donfuldauer.kggml.models.ModelDownloader
import com.donfuldauer.kggml.models.ModelInspector
import com.donfuldauer.kggml.models.ResidentModelCache

// README.md snippets for :kggml-model-manager. Compiled by :docs-snippets; see build.gradle.kts.

internal suspend fun modelManager(appFilesDir: String, modelUrl: String, sha256: String) {
    // snippet:start model-manager
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
    // snippet:end model-manager
}
