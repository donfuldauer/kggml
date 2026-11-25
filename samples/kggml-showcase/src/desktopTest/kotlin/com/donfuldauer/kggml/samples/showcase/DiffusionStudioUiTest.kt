package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.assertTextEquals
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.test.runComposeUiTest
import com.donfuldauer.kggml.core.OutOfMemoryException
import com.donfuldauer.kggml.samples.showcase.diffusion.DiffusionStudioTags
import com.donfuldauer.kggml.samples.showcase.diffusion.FakeImageGeneratorFactory
import com.donfuldauer.kggml.samples.showcase.diffusion.PngEncoderTest
import com.donfuldauer.kggml.samples.showcase.diffusion.fakeImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import kotlinx.coroutines.channels.Channel
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Compose UI tests of the Diffusion Image Studio with a deterministic generator (`DiffusionFakes.kt`): parameter
 * controls, live step progress, the rendered result, mid-generation cancel, PNG export and errors.
 */
@OptIn(ExperimentalTestApi::class)
class DiffusionStudioUiTest {
    private fun androidx.compose.ui.test.ComposeUiTest.setSlider(tag: String, value: Float) {
        onNodeWithTag(tag).performScrollTo().performSemanticsAction(SemanticsActions.SetProgress) { it(value) }
    }

    @Test
    fun generationShowsProgressThenTheImage() = runComposeUiTest {
        val gate = Channel<Unit>(Channel.UNLIMITED)
        val fakes = FakeShowcase(generatorFactory = FakeImageGeneratorFactory { it.stepGate = gate })
        showFakeShowcase(fakes, ShowcaseDestination.DIFFUSION_STUDIO)
        setSlider(DiffusionStudioTags.STEPS, 4f)
        onNodeWithTag(DiffusionStudioTags.STEPS_LABEL).assertTextEquals("Sampling steps: 4")
        setSlider(DiffusionStudioTags.CFG, 5.2f)
        onNodeWithTag(DiffusionStudioTags.CFG_LABEL).assertTextEquals("CFG scale: 5.0")
        click(DiffusionStudioTags.size(256))
        typeInto(DiffusionStudioTags.PROMPT, "a red fox")
        typeInto(DiffusionStudioTags.SEED, "1234")

        click(DiffusionStudioTags.GENERATE)
        waitForText(DiffusionStudioTags.PROGRESS_LABEL, "Denoising step 0 / 4")
        onNodeWithTag(DiffusionStudioTags.GENERATE).assertIsNotEnabled()
        onNodeWithTag(DiffusionStudioTags.CANCEL).assertIsEnabled()
        gate.trySend(Unit)
        gate.trySend(Unit)
        waitForText(DiffusionStudioTags.PROGRESS_LABEL, "Denoising step 2 / 4 · 0.50 s/step")
        repeat(2) { gate.trySend(Unit) }
        waitForTag(DiffusionStudioTags.IMAGE)
        onNodeWithTag(DiffusionStudioTags.IMAGE_INFO).assertTextEquals("256×256 · seed 1234 · 4 steps · CFG 5.0")
        onNodeWithTag(DiffusionStudioTags.STATUS).assertTextEquals("Generated 256x256 image (seed 1234)")
        onNodeWithTag(DiffusionStudioTags.ACTIVITY).assertTextEquals("Idle")
        val request = fakes.generatorFactory.opened.single().requests.single()
        assertEquals("a red fox", request.prompt)
        assertEquals(5f, request.cfgScale)
        assertEquals(1234L, request.seed)
    }

    @Test
    fun cancelStopsTheGeneration() = runComposeUiTest {
        val gate = Channel<Unit>(Channel.UNLIMITED)
        val fakes = FakeShowcase(generatorFactory = FakeImageGeneratorFactory { it.stepGate = gate })
        showFakeShowcase(fakes, ShowcaseDestination.DIFFUSION_STUDIO)
        setSlider(DiffusionStudioTags.STEPS, 10f)
        click(DiffusionStudioTags.GENERATE)
        gate.trySend(Unit)
        waitForText(DiffusionStudioTags.PROGRESS_LABEL, "Denoising step 1 / 10")
        click(DiffusionStudioTags.CANCEL)
        waitForText(DiffusionStudioTags.STATUS, "Cancelled at denoising step 1 / 10")
        onNodeWithTag(DiffusionStudioTags.IMAGE).assertDoesNotExist()
        onNodeWithTag(DiffusionStudioTags.GENERATE).assertIsEnabled()
    }

    @Test
    fun exportWritesADecodablePng() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.DIFFUSION_STUDIO)
        setSlider(DiffusionStudioTags.STEPS, 1f)
        click(DiffusionStudioTags.size(256))
        click(DiffusionStudioTags.GENERATE)
        waitForTag(DiffusionStudioTags.EXPORT_PATH)
        onNodeWithTag(DiffusionStudioTags.EXPORT_PATH).assertTextContains("/exports/kggml-diffusion-42.png")
        typeInto(DiffusionStudioTags.EXPORT_PATH, "/exports/fox.png")
        click(DiffusionStudioTags.EXPORT)
        waitForText(DiffusionStudioTags.STATUS, "to /exports/fox.png")
        val png = fakes.written.getValue("/exports/fox.png")
        // A real PNG decoder (javax.imageio) reads the pure-Kotlin encoder's output back pixel for pixel.
        val decoded = assertNotNull(ImageIO.read(ByteArrayInputStream(png)))
        assertEquals(256, decoded.width)
        val expected = fakeImage(256, 256, 42)
        val rgb = decoded.getRGB(3, 2)
        val offset = expected.offsetOf(3, 2)
        assertEquals(expected.pixels[offset].toInt() and 0xFF, (rgb shr 16) and 0xFF)
        assertEquals(expected.pixels[offset + 1].toInt() and 0xFF, (rgb shr 8) and 0xFF)
        assertEquals(expected.pixels[offset + 2].toInt() and 0xFF, rgb and 0xFF)
        assertTrue(PngEncoderTest.decodeStoredPng(png).pixels.contentEquals(expected.pixels))
    }

    @Test
    fun randomSeedAndErrors() = runComposeUiTest {
        val fakes = FakeShowcase(generatorFactory = FakeImageGeneratorFactory { it.failAtStep = 1; it.failure = OutOfMemoryException("VRAM") })
        showFakeShowcase(fakes, ShowcaseDestination.DIFFUSION_STUDIO)
        setSlider(DiffusionStudioTags.STEPS, 2f)
        click(DiffusionStudioTags.RANDOM_SEED)
        onNodeWithTag(DiffusionStudioTags.RANDOM_SEED).assertIsOn()
        onNodeWithTag(DiffusionStudioTags.SEED).assertIsNotEnabled()
        click(DiffusionStudioTags.GENERATE)
        waitForTag(DiffusionStudioTags.ERROR)
        onNodeWithTag(DiffusionStudioTags.ERROR).assertTextContains("OutOfMemoryException: VRAM", substring = true)
        val seed = fakes.generatorFactory.opened.single().requests.single().seed
        onNodeWithTag(DiffusionStudioTags.SEED).assertTextContains(seed.toString())

        typeInto(DiffusionStudioTags.PROMPT, " ")
        onNodeWithTag(DiffusionStudioTags.GENERATE).assertIsNotEnabled()
    }
}
