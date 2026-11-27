package com.donfuldauer.kggml.samples.showcase

import android.content.Context
import android.content.ContextWrapper
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.donfuldauer.kggml.core.AndroidMemoryPressureObserver
import com.donfuldauer.kggml.samples.showcase.llm.NoImageSource
import com.donfuldauer.kggml.samples.showcase.whisper.AndroidAudioCapture
import com.donfuldauer.kggml.samples.showcase.whisper.defaultWhisperModelPath
import com.donfuldauer.kggml.samples.showcase.whisper.platformAudioCapture
import com.donfuldauer.kggml.samples.showcase.whisper.readAudioFile
import com.donfuldauer.kggml.samples.showcase.whisper.supportsBackgroundStress
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertSame

/**
 * The Android platform layer of the showcase (Robolectric locally, also on a device): app-private directories,
 * local file access, pixel conversion for display, and the platform defaults of every screen.
 */
@RunWith(AndroidJUnit4::class)
class PlatformAndroidTest {
    private val app: Context = ApplicationProvider.getApplicationContext()
    private val scratch = File(app.cacheDir, "platform-android-test")

    @Before
    fun initShowcaseAndroid() {
        ShowcaseAndroid.init(app)
    }

    @After
    fun cleanUp() {
        ShowcaseAndroid.init(app)
        scratch.deleteRecursively()
    }

    @Test
    fun modelsLiveInTheAppFilesDirectory() {
        assertEquals(File(app.filesDir, "models").absolutePath, defaultModelDirectory())
    }

    @Test
    fun exportsGoToExternalPicturesOrFallBackToAppFiles() {
        assertEquals(assertNotNull(app.getExternalFilesDir("Pictures")).absolutePath, defaultExportDirectory())

        // Without external storage (unmounted / removed): app-private files/exports.
        val noExternalStorage = object : ContextWrapper(app) {
            override fun getApplicationContext(): Context = this
            override fun getExternalFilesDir(type: String?): File? = null
        }
        ShowcaseAndroid.init(noExternalStorage)
        assertEquals(File(app.filesDir, "exports").absolutePath, defaultExportDirectory())
    }

    @Test
    fun initKeepsOnlyTheApplicationContext() {
        val activityLike = object : ContextWrapper(app) {}
        ShowcaseAndroid.init(activityLike)
        assertSame(app, ShowcaseAndroid.appContext, "an activity context must not be retained")
    }

    @Test
    fun appContextBeforeInitFailsWithGuidance() {
        val field = ShowcaseAndroid::class.java.getDeclaredField("context").apply { isAccessible = true }
        field.set(null, null)
        val e = assertFailsWith<IllegalStateException> { defaultModelDirectory() }
        assertEquals("ShowcaseAndroid.init(context) must be called by the activity first", e.message)
    }

    @Test
    fun writingCreatesParentDirectoriesAndReadingReturnsTheSameBytes() {
        val path = File(scratch, "a/b/image.png").absolutePath
        val bytes = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0, -1)
        writeLocalFile(path, bytes)
        assertContentEquals(bytes, readLocalFile(path))
        assertContentEquals(bytes, readAudioFile(path))
    }

    @Test
    fun pixelsBecomeArgbBitmaps() {
        val rgb = pixelsToImageBitmap(2, 1, 3, byteArrayOf(10, 20, 30, 200.toByte(), 100, 50)).asAndroidBitmap()
        assertEquals(2, rgb.width)
        assertEquals(1, rgb.height)
        assertEquals(0xFF0A141E.toInt(), rgb.getPixel(0, 0))
        assertEquals(0xFFC86432.toInt(), rgb.getPixel(1, 0))

        val gray = pixelsToImageBitmap(1, 2, 1, byteArrayOf(0x80.toByte(), 0)).asAndroidBitmap()
        assertEquals(0xFF808080.toInt(), gray.getPixel(0, 0))
        assertEquals(0xFF000000.toInt(), gray.getPixel(0, 1))

        val rgba = pixelsToImageBitmap(1, 1, 4, byteArrayOf(-1, 0, 0, 0x80.toByte())).asAndroidBitmap()
        assertEquals(0x80FF0000.toInt(), rgba.getPixel(0, 0))
    }

    @Test
    fun platformServicesOfTheScreens() {
        assertIs<AndroidMemoryPressureObserver>(createMemoryPressureObserver()).close()
        // The activity-bound picker comes from MainActivity; the platform default has none.
        assertSame(NoImageSource, platformImageSource())
        assertIs<AndroidAudioCapture>(platformAudioCapture())
        assertEquals("", defaultWhisperModelPath())
        assertFalse(supportsBackgroundStress)
    }
}
