package com.donfuldauer.kggml.whisper

import androidx.test.platform.app.InstrumentationRegistry
import java.io.File

/**
 * [NativeWhisperEngineContract] on an Android device or emulator through the JNI bridge, against the real native libraries
 * packaged into the test APK (`connectedDebugAndroidTest -Pkggml.bundleNative=true`).
 */
class AndroidNativeWhisperTest : NativeWhisperEngineContract() {
    private val cacheDir: File get() = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir

    override fun writeTempFile(prefix: String, bytes: ByteArray): String =
        File.createTempFile(prefix, ".bin", cacheDir).apply { writeBytes(bytes) }.absolutePath

    override fun deleteFile(path: String) {
        File(path).delete()
    }

    /** The app's cache directory (also `java.io.tmpdir` on Android): where a temporary audio file would land. */
    override fun temporaryFileNames(): Set<String> =
        (cacheDir.list().orEmpty().toList() + File(System.getProperty("java.io.tmpdir") ?: ".").list().orEmpty()).toSet()
}
