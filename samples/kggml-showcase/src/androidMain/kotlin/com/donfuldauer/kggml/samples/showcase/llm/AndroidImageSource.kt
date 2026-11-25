package com.donfuldauer.kggml.samples.showcase.llm

import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.io.ByteArrayOutputStream
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Photo library (`GetContent`, read through the content resolver) and camera (`TakePicturePreview`, an in-memory
 * bitmap compressed to JPEG) for the VLM panel. Nothing is written to disk. Must be created in the activity's
 * `onCreate` (activity-result launchers are registered at construction).
 */
class AndroidImageSource(private val activity: ComponentActivity) : ImageSource {
    private var pending: CompletableDeferred<Any?>? = null

    private val gallery = activity.registerForActivityResult(ActivityResultContracts.GetContent()) { uri -> pending?.complete(uri) }
    private val camera = activity.registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap -> pending?.complete(bitmap) }

    override val canPickFromGallery: Boolean = true
    override val canCapture: Boolean get() = activity.packageManager.hasSystemFeature("android.hardware.camera.any")

    override suspend fun pickFromGallery(): ByteArray? {
        val uri = await { gallery.launch("image/*") } as Uri? ?: return null
        return withContext(Dispatchers.IO) {
            activity.contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: error("Cannot open $uri")
        }
    }

    override suspend fun capture(): ByteArray? {
        val bitmap = await { camera.launch(null) } as Bitmap? ?: return null
        return withContext(Dispatchers.Default) {
            ByteArrayOutputStream().use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
                out.toByteArray()
            }
        }
    }

    private suspend fun await(launch: () -> Unit): Any? {
        pending?.cancel()
        val result = CompletableDeferred<Any?>()
        pending = result
        launch()
        return try {
            result.await()
        } finally {
            if (pending === result) pending = null
        }
    }
}
