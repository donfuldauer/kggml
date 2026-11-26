package com.donfuldauer.kggml.samples.whisper

import java.nio.file.Files

actual object NativeTestEnvironment {
    /** -Pkggml.requireNative=true (desktopTest passes it as `kggml.test.requireNative`): missing libraries fail. */
    actual val requireNative: Boolean = System.getProperty("kggml.test.requireNative").toBoolean()

    /** AUTO resolves to an available backend (Metal on macOS; the CPU on Linux / Windows builds without Vulkan). */
    actual val gpuBackendsUsable: Boolean = true

    actual fun writeTempFile(name: String, bytes: ByteArray): String {
        val dir = Files.createTempDirectory("kggml-minimal").toFile()
        return dir.resolve(name).apply { writeBytes(bytes) }.absolutePath
    }

    actual fun skip(reason: String): Nothing = throw org.junit.AssumptionViolatedException(reason)
}
