package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.hasAnyDescendant
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.runComposeUiTest
import com.donfuldauer.kggml.core.NativeFailureException
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.Dispatchers
import kotlin.test.Test
import kotlin.test.assertEquals

/** Compose UI smoke tests of the navigation shell and the Hardware Backend Inspector view state. */
@OptIn(ExperimentalTestApi::class)
class ShowcaseAppUiTest {
    @Test
    fun navigationShellRendersEveryDestination() = runComposeUiTest {
        setContent { ShowcaseApp(inspector = { fakeSnapshot() }, ioDispatcher = Dispatchers.Unconfined) }
        ShowcaseDestination.entries.forEach { onNodeWithTag(it.navTag).assertIsDisplayed() }
        onNodeWithTag(ShowcaseDestination.BACKEND_INSPECTOR.navTag).assertIsSelected()
        onNodeWithTag(ShowcaseDestination.BACKEND_INSPECTOR.screenTag).assertIsDisplayed()
    }

    @Test
    fun navigatingToPlaceholderScreensShowsTheirPhase() = runComposeUiTest {
        setContent { ShowcaseApp(inspector = { fakeSnapshot() }, ioDispatcher = Dispatchers.Unconfined) }
        ShowcaseDestination.entries.filter { it.isPlaceholder }.forEach { destination ->
            onNodeWithTag(destination.navTag).performClick()
            onNodeWithTag(destination.navTag).assertIsSelected()
            onNodeWithTag(destination.screenTag).assertIsDisplayed()
            onNodeWithText(destination.title).assertIsDisplayed()
            onNodeWithText("Coming in Phase ${destination.plannedPhase}.").assertIsDisplayed()
        }
        onNodeWithTag(ShowcaseDestination.BACKEND_INSPECTOR.navTag).performClick()
        onNodeWithTag(ShowcaseDestination.BACKEND_INSPECTOR.screenTag).assertIsDisplayed()
    }

    @Test
    fun inspectorShowsCpuMetalAndVulkanBackends() = runComposeUiTest {
        setContent { ShowcaseApp(inspector = { fakeSnapshot() }, ioDispatcher = Dispatchers.Unconfined) }
        waitUntil { onAllNodes(hasText("Hardware threads: 10")).fetchSemanticsNodes().isNotEmpty() }
        onNodeWithTag("backend-row-cpu_only").assertIsDisplayed()
        onNodeWithTag("backend-row-gpu_metal").assertIsDisplayed()
        onNodeWithTag("backend-row-gpu_vulkan").assertExists()
        onNodeWithText("CPU").assertExists()
        onNodeWithText("Metal").assertExists()
        onNodeWithText("Vulkan").assertExists()
        onNode(hasTestTag("backend-row-gpu_metal") and hasAnyDescendant(hasText("Available"))).assertExists()
        onNode(hasTestTag("backend-row-gpu_vulkan") and hasAnyDescendant(hasText("Not compiled in this build")))
            .assertExists()
        onNodeWithText("Recommended threads (P-cores): 4").assertExists()
        onNodeWithTag("inspector-refresh").assertIsEnabled()
    }

    @Test
    fun inspectorShowsErrorStateWhenRuntimeCannotLoad() = runComposeUiTest {
        setContent {
            ShowcaseApp(
                inspector = { throw NativeFailureException(message = "Failed to load the Kggml native runtime") },
                ioDispatcher = Dispatchers.Unconfined,
            )
        }
        waitUntil { onAllNodes(hasText("Native runtime unavailable")).fetchSemanticsNodes().isNotEmpty() }
        onNodeWithTag("inspector-error").assertIsDisplayed()
        onNodeWithText("Failed to load the Kggml native runtime").assertExists()
    }

    @Test
    fun refreshQueriesTheRuntimeAgain() = runComposeUiTest {
        val calls = AtomicInteger()
        setContent {
            ShowcaseApp(inspector = { calls.incrementAndGet(); fakeSnapshot() }, ioDispatcher = Dispatchers.Unconfined)
        }
        waitUntil { calls.get() == 1 }
        waitForIdle()
        onNodeWithTag("inspector-refresh").performClick()
        waitUntil { calls.get() == 2 }
        waitForIdle()
        assertEquals(2, calls.get())
        onNodeWithTag("backend-row-cpu_only").assertExists()
    }

    @Test
    fun loadingStateIsShownWhileRuntimeQueryIsPending() = runComposeUiTest {
        setContent { BackendInspectorContent(state = BackendInspectorState.Loading, onRefresh = {}) }
        onNodeWithTag("inspector-loading").assertExists()
    }
}
