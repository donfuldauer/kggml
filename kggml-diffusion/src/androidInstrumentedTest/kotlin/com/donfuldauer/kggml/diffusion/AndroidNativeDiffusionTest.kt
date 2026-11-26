package com.donfuldauer.kggml.diffusion

import androidx.test.platform.app.InstrumentationRegistry
import java.io.File

/**
 * [NativeDiffusionEngineContract] on an Android device or emulator through the JNI bridge, against the real native libraries
 * packaged into the test APK (`connectedDebugAndroidTest -Pkggml.bundleNative=true`).
 */
class AndroidNativeDiffusionTest : NativeDiffusionEngineContract() {
    private val cacheDir: File get() = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir

    override fun writeTempFile(prefix: String, bytes: ByteArray): String =
        File.createTempFile(prefix, ".gguf", cacheDir).apply { writeBytes(bytes) }.absolutePath

    override fun deleteFile(path: String) {
        File(path).delete()
    }

    override fun directoryPath(): String = cacheDir.absolutePath
}
