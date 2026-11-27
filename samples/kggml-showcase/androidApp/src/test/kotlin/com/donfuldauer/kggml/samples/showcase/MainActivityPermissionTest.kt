package com.donfuldauer.kggml.samples.showcase

import android.Manifest
import android.app.Application
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Shadows.shadowOf
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/** [MainActivity]'s microphone permission request, as recorded by Robolectric's `ShadowActivity`. */
@RunWith(AndroidJUnit4::class)
class MainActivityPermissionTest {
    private val app: Application = ApplicationProvider.getApplicationContext()

    @Test
    fun microphonePermissionIsRequestedWhenNotGranted() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val request = assertNotNull(shadowOf(activity).lastRequestedPermission)
                assertContentEquals(arrayOf(Manifest.permission.RECORD_AUDIO), request.requestedPermissions)
                assertEquals(1, request.requestCode)
            }
        }
    }

    @Test
    fun noPermissionRequestWhenTheMicrophoneIsAlreadyGranted() {
        shadowOf(app).grantPermissions(Manifest.permission.RECORD_AUDIO)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity -> assertNull(shadowOf(activity).lastRequestedPermission) }
        }
    }
}
