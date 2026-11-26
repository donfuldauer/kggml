package com.donfuldauer.kggml.samples.showcase.llm

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.Shadows.shadowOf
import org.robolectric.android.controller.ActivityController
import org.robolectric.annotation.GraphicsMode
import java.io.ByteArrayInputStream
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * [AndroidImageSource] against the real activity-result machinery of a Robolectric `ComponentActivity`: the picker
 * and camera intents it launches are answered through `ShadowActivity.receiveResult`, exactly as the system would.
 */
@RunWith(AndroidJUnit4::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE) // real JPEG encoding / decoding of the camera bitmap
class AndroidImageSourceTest {
    private lateinit var controller: ActivityController<ComponentActivity>
    private lateinit var source: AndroidImageSource
    private val activity: ComponentActivity get() = controller.get()

    @Before
    fun createActivity() {
        controller = Robolectric.buildActivity(ComponentActivity::class.java).create()
        // Like MainActivity: the launchers are registered in onCreate, before the activity is started.
        source = AndroidImageSource(activity)
        controller.start().resume()
    }

    @After
    fun destroyActivity() {
        controller.pause().stop().destroy()
    }

    /** Runs [block] on this thread up to its first suspension (the launched picker), like a click handler would. */
    private fun <T> launch(block: suspend () -> T): Deferred<T> = CoroutineScope(Dispatchers.Unconfined).async { block() }

    private fun nextRequest(): Intent = assertNotNull(shadowOf(activity).nextStartedActivityForResult, "no activity was launched").intent

    @Test
    fun galleryPickLaunchesGetContentForImagesAndReadsTheChosenUri() {
        val uri = Uri.parse("content://media/external/images/media/7")
        val png = FakeImageSource.PNG_MAGIC + ByteArray(24) { it.toByte() }
        shadowOf(activity.contentResolver).registerInputStream(uri, ByteArrayInputStream(png))
        assertTrue(source.canPickFromGallery)

        val picked = launch { source.pickFromGallery() }
        val request = nextRequest()
        assertEquals(Intent.ACTION_GET_CONTENT, request.action)
        assertEquals("image/*", request.type)
        assertFalse(picked.isCompleted, "waits for the picker")

        shadowOf(activity).receiveResult(request, Activity.RESULT_OK, Intent().setData(uri))
        assertContentEquals(png, runBlocking { picked.await() })
    }

    @Test
    fun cancelledGalleryPickReturnsNull() {
        val picked = launch { source.pickFromGallery() }
        shadowOf(activity).receiveResult(nextRequest(), Activity.RESULT_CANCELED, null)
        assertNull(runBlocking { picked.await() })
    }

    @Test
    fun aNewPickAbandonsThePendingOne() {
        val uri = Uri.parse("content://media/external/images/media/8")
        val jpeg = FakeImageSource.JPEG_MAGIC + ByteArray(8)
        shadowOf(activity.contentResolver).registerInputStream(uri, ByteArrayInputStream(jpeg))
        val first = launch { source.pickFromGallery() }
        val second = launch { source.pickFromGallery() }
        assertTrue(first.isCancelled, "the first request is cancelled, not left hanging")

        shadowOf(activity).receiveResult(nextRequest(), Activity.RESULT_OK, Intent().setData(uri))
        assertContentEquals(jpeg, runBlocking { second.await() })
    }

    @Test
    fun cameraCaptureLaunchesTakePicturePreviewAndReturnsAJpeg() {
        val photo = Bitmap.createBitmap(4, 3, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.RED) }
        val captured = launch { source.capture() }
        val request = nextRequest()
        assertEquals(MediaStore.ACTION_IMAGE_CAPTURE, request.action)

        shadowOf(activity).receiveResult(request, Activity.RESULT_OK, Intent().putExtra("data", photo))
        val jpeg = assertNotNull(runBlocking { captured.await() })
        assertEquals("JPEG", imageFormatOf(jpeg))
        val decoded = assertNotNull(BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size))
        assertEquals(4, decoded.width)
        assertEquals(3, decoded.height)
        val center = decoded.getPixel(2, 1)
        assertTrue(Color.red(center) > 200 && Color.green(center) < 60 && Color.blue(center) < 60, "still red after JPEG: ${Integer.toHexString(center)}")
    }

    @Test
    fun cancelledCaptureReturnsNull() {
        val captured = launch { source.capture() }
        shadowOf(activity).receiveResult(nextRequest(), Activity.RESULT_CANCELED, null)
        assertNull(runBlocking { captured.await() })
    }

    @Test
    fun captureIsOfferedOnlyWithACamera() {
        val packageManager = shadowOf(activity.packageManager)
        packageManager.setSystemFeature("android.hardware.camera.any", false)
        assertFalse(source.canCapture)
        packageManager.setSystemFeature("android.hardware.camera.any", true)
        assertTrue(source.canCapture)
    }
}
