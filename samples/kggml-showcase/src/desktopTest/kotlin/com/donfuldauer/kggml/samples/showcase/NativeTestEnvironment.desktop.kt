package com.donfuldauer.kggml.samples.showcase

import java.nio.file.Files

actual object NativeTestEnvironment {
    actual val requireNative: Boolean = System.getProperty("kggml.test.requireNative").toBoolean()

    actual fun newTempDirectory(): String =
        Files.createTempDirectory("kggml-showcase-e2e").toFile().apply { deleteOnExit() }.absolutePath

    actual fun skip(reason: String): Nothing = throw org.junit.AssumptionViolatedException(reason)
}
