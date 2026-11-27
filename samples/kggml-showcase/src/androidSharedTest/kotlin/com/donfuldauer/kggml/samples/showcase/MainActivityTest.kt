package com.donfuldauer.kggml.samples.showcase

import android.Manifest
import android.content.Context
import android.os.Build
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.donfuldauer.kggml.samples.showcase.models.ModelManagerTags
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.test.assertSame

/**
 * Launches the real [MainActivity] (Robolectric locally, also on a device): it initialises the platform layer and
 * shows the navigation shell over the production environments, and the shell switches destinations.
 */
@RunWith(AndroidJUnit4::class)
class MainActivityTest {
    @get:Rule
    val compose = createEmptyComposeRule()

    private val app: Context = ApplicationProvider.getApplicationContext()

    @Before
    fun grantMicrophoneOnDevice() {
        // On a device the runtime-permission dialog would cover the activity. Robolectric shows no dialog (the
        // request is only recorded, see MainActivityPermissionTest) and has no UiAutomation to grant with.
        if (Build.FINGERPRINT != "robolectric") {
            InstrumentationRegistry.getInstrumentation().uiAutomation
                .grantRuntimePermission(app.packageName, Manifest.permission.RECORD_AUDIO)
        }
    }

    @Test
    fun launchShowsTheNavigationShellAndSwitchesDestinations() {
        ActivityScenario.launch(MainActivity::class.java).use {
            ShowcaseDestination.entries.forEach { compose.onNodeWithTag(it.navTag).assertIsDisplayed() }
            compose.onNodeWithTag(ShowcaseDestination.BACKEND_INSPECTOR.navTag).assertIsSelected()
            compose.onNodeWithTag(ShowcaseDestination.BACKEND_INSPECTOR.screenTag).assertIsDisplayed()

            compose.onNodeWithTag(ShowcaseDestination.MODEL_MANAGER.navTag).performClick()
            compose.onNodeWithTag(ShowcaseDestination.MODEL_MANAGER.navTag).assertIsSelected()
            compose.onNodeWithTag(ShowcaseDestination.MODEL_MANAGER.screenTag).assertIsDisplayed()
            compose.onNodeWithTag(ShowcaseDestination.BACKEND_INSPECTOR.screenTag).assertDoesNotExist()
            // The production Model Manager stores models in the app's files directory.
            compose.onNodeWithTag(ModelManagerTags.DIRECTORY)
                .assertTextContains("Storage: ${File(app.filesDir, "models").absolutePath}", substring = true)

            compose.onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.navTag).performClick()
            compose.onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.screenTag).assertIsDisplayed()
        }
    }

    @Test
    fun launchInitialisesThePlatformLayerWithTheApplicationContext() {
        // Forget any context a previous test set, so only MainActivity.onCreate can provide it.
        ShowcaseAndroid::class.java.getDeclaredField("context").apply { isAccessible = true }.set(null, null)
        ActivityScenario.launch(MainActivity::class.java).use {
            assertSame(app, ShowcaseAndroid.appContext)
        }
    }
}
