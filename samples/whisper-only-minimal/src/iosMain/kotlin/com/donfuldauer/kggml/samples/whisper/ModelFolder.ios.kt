@file:OptIn(ExperimentalForeignApi::class)

package com.donfuldauer.kggml.samples.whisper

import kotlinx.cinterop.BooleanVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.value
import platform.Foundation.NSFileManager

internal actual fun listFileNames(directory: String): List<String> {
    val manager = NSFileManager.defaultManager
    val entries = manager.contentsOfDirectoryAtPath(directory, null) ?: return emptyList()
    return entries.filterIsInstance<String>().filter { name ->
        memScoped {
            val isDirectory = alloc<BooleanVar>()
            manager.fileExistsAtPath(directory.trimEnd('/') + "/" + name, isDirectory.ptr) && !isDirectory.value
        }
    }
}
