package com.donfuldauer.kggml.samples.showcase

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import com.donfuldauer.kggml.core.AndroidMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureObserver
import com.donfuldauer.kggml.samples.showcase.llm.ImageSource
import com.donfuldauer.kggml.samples.showcase.llm.NoImageSource
import java.io.File

/** The application context, set by [MainActivity] before the UI is composed. */
@SuppressLint("StaticFieldLeak") // Only the application context is kept, which lives as long as the process.
object ShowcaseAndroid {
    @Volatile
    private var context: Context? = null

    fun init(context: Context) {
        this.context = context.applicationContext
    }

    val appContext: Context
        get() = context ?: error("ShowcaseAndroid.init(context) must be called by the activity first")
}

actual fun defaultModelDirectory(): String = File(ShowcaseAndroid.appContext.filesDir, "models").absolutePath

actual fun defaultExportDirectory(): String =
    (ShowcaseAndroid.appContext.getExternalFilesDir("Pictures") ?: File(ShowcaseAndroid.appContext.filesDir, "exports")).absolutePath

actual fun createMemoryPressureObserver(): MemoryPressureObserver = AndroidMemoryPressureObserver(ShowcaseAndroid.appContext)

actual fun defaultResidentBudgetBytes(): Long {
    val manager = ShowcaseAndroid.appContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val info = ActivityManager.MemoryInfo().also(manager::getMemoryInfo)
    return (info.totalMem / 2).coerceAtLeast(512L shl 20)
}

/** The activity-bound picker is passed in by [MainActivity] (`LlmChatEnvironment.imageSource`). */
actual fun platformImageSource(): ImageSource = NoImageSource

actual fun readLocalFile(path: String): ByteArray = File(path).readBytes()

actual fun writeLocalFile(path: String, bytes: ByteArray) {
    val file = File(path)
    file.absoluteFile.parentFile?.mkdirs()
    file.writeBytes(bytes)
}

actual fun pixelsToImageBitmap(width: Int, height: Int, channels: Int, pixels: ByteArray): ImageBitmap {
    val rgba = toRgba(width, height, channels, pixels)
    val argb = IntArray(width * height) { i ->
        val o = i * 4
        ((rgba[o + 3].toInt() and 0xFF) shl 24) or ((rgba[o].toInt() and 0xFF) shl 16) or
            ((rgba[o + 1].toInt() and 0xFF) shl 8) or (rgba[o + 2].toInt() and 0xFF)
    }
    return Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888).asImageBitmap()
}
