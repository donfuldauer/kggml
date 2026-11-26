package com.donfuldauer.kggml.samples.whisper

import java.io.File

internal actual fun listFileNames(directory: String): List<String> =
    File(directory).listFiles()?.filter { it.isFile }?.map { it.name }.orEmpty()

/**
 * The desktop model folder: `-Dkggml.models.dir`, else `KGGML_MODELS_DIR`, else `~/.kggml/models`, the folder the
 * showcase downloads into, so its whisper models show up here too.
 */
fun desktopModelFolder(): String =
    System.getProperty("kggml.models.dir")?.takeIf { it.isNotBlank() }
        ?: System.getenv("KGGML_MODELS_DIR")?.takeIf { it.isNotBlank() }
        ?: File(System.getProperty("user.home"), ".kggml/models").path
