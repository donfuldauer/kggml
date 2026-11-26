package com.donfuldauer.kggml.llama

import androidx.test.platform.app.InstrumentationRegistry
import java.io.File

/**
 * [NativeLlamaEngineContract] on an Android device or emulator through the JNI bridge, against the real native libraries
 * packaged into the test APK (`connectedDebugAndroidTest -Pkggml.bundleNative=true`).
 */
class AndroidNativeLlamaTest : NativeLlamaEngineContract() {
    private val cacheDir: File get() = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir

    override fun writeTempFile(prefix: String, bytes: ByteArray): String =
        File.createTempFile(prefix, ".gguf", cacheDir).apply { writeBytes(bytes) }.absolutePath

    override fun deleteFile(path: String) {
        File(path).delete()
    }

    override fun directoryPath(): String = cacheDir.absolutePath
}
