package com.donfuldauer.kggml.samples.whisper

import java.io.File

internal actual fun listFileNames(directory: String): List<String> =
    File(directory).listFiles()?.filter { it.isFile }?.map { it.name }.orEmpty()
