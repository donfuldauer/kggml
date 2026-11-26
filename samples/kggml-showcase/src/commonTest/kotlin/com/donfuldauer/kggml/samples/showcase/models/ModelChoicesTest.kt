package com.donfuldauer.kggml.samples.showcase.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Which downloaded files each feature screen's model picker offers ([choicesFor] / [ModelSlot.accepts]). */
class ModelChoicesTest {
    private fun model(id: String, kind: ModelKind, fileName: String, projectorId: String? = null) = CatalogModel(
        id = id,
        displayName = "Model $id",
        kind = kind,
        url = "https://example.invalid/$fileName",
        fileName = fileName,
        approxSizeBytes = 1,
        license = "MIT",
        projectorId = projectorId,
    )

    private val whisper = model("whisper", ModelKind.WHISPER, "ggml-tiny.bin")
    private val llm = model("llm", ModelKind.LLM, "llm.gguf")
    private val vlm = model("vlm", ModelKind.VISION_LLM, "vlm.gguf", projectorId = "proj")
    private val projector = model("proj", ModelKind.VISION_PROJECTOR, "mmproj-vlm.gguf")
    private val sd = model("sd", ModelKind.DIFFUSION, "sd.safetensors")

    private fun installed(model: CatalogModel) =
        CatalogRow(model, phase = DownloadPhase.INSTALLED, installedPath = "/m/${model.fileName}")

    private fun state(rows: List<CatalogRow>, others: List<String> = emptyList()) = ModelManagerState(
        directory = "/m",
        rows = rows,
        otherFiles = others.map { StoredFile(it, "/m/$it", 10) },
    )

    private fun List<ModelChoice>.ids() = map { it.id }

    @Test
    fun onlyInstalledCatalogModelsOfTheSlotsKindsAreOffered() {
        val s = state(
            listOf(
                installed(whisper),
                CatalogRow(llm, phase = DownloadPhase.PARTIAL),
                CatalogRow(sd, phase = DownloadPhase.DOWNLOADING),
                installed(vlm),
            ),
        )
        assertEquals(listOf("whisper"), s.choicesFor(ModelSlot.WHISPER).ids())
        // A paused LLM download is not offered; the installed vision LLM is.
        assertEquals(listOf("vlm"), s.choicesFor(ModelSlot.LLM).ids())
        assertEquals(emptyList(), s.choicesFor(ModelSlot.DIFFUSION))
        assertEquals(emptyList(), s.choicesFor(ModelSlot.VISION_PROJECTOR))
    }

    @Test
    fun choiceCarriesTheInstalledPathAndLabel() {
        val choice = state(listOf(installed(sd))).choicesFor(ModelSlot.DIFFUSION).single()
        assertEquals(ModelChoice("sd", "Model sd", "sd.safetensors", "/m/sd.safetensors", null), choice)
    }

    @Test
    fun visionLlmBringsItsInstalledProjector() {
        val withProjector = state(listOf(installed(vlm), installed(projector))).choicesFor(ModelSlot.LLM).single()
        assertEquals("/m/mmproj-vlm.gguf", withProjector.projectorPath)
        assertEquals("vlm.gguf · with its vision projector", withProjector.detail)

        val withoutProjector = state(listOf(installed(vlm), CatalogRow(projector))).choicesFor(ModelSlot.LLM).single()
        assertEquals(null, withoutProjector.projectorPath)
        assertEquals("vlm.gguf · vision projector not downloaded", withoutProjector.detail)

        // A text-only LLM has no projector note.
        assertEquals("llm.gguf", state(listOf(installed(llm))).choicesFor(ModelSlot.LLM).single().detail)
    }

    @Test
    fun otherFilesInTheModelFolderAreMatchedByExtensionAfterTheCatalogModels() {
        val s = state(
            rows = listOf(installed(llm), installed(projector)),
            others = listOf("zeta.gguf", "ggml-base.en.bin", "mmproj-extra.GGUF", "alpha.gguf", "notes.txt", "sd-xl.ckpt", "v2.safetensors"),
        )
        // Catalog first, then other files sorted by name; projectors (mmproj…) are kept apart from LLMs.
        assertEquals(listOf("llm", "alpha.gguf", "zeta.gguf"), s.choicesFor(ModelSlot.LLM).ids())
        assertEquals(listOf("proj", "mmproj-extra.GGUF"), s.choicesFor(ModelSlot.VISION_PROJECTOR).ids())
        assertEquals(listOf("ggml-base.en.bin"), s.choicesFor(ModelSlot.WHISPER).ids())
        assertEquals(listOf("sd-xl.ckpt", "v2.safetensors"), s.choicesFor(ModelSlot.DIFFUSION).ids())

        val other = s.choicesFor(ModelSlot.WHISPER).single()
        assertEquals(ModelChoice("ggml-base.en.bin", "ggml-base.en.bin", "Other file in the model folder", "/m/ggml-base.en.bin"), other)
    }

    @Test
    fun slotAcceptsFilesByExtensionCaseInsensitively() {
        assertTrue(ModelSlot.WHISPER.accepts("GGML-SMALL.BIN"))
        assertFalse(ModelSlot.WHISPER.accepts("model.gguf"))
        assertFalse(ModelSlot.WHISPER.accepts("bin"), "no extension")
        assertTrue(ModelSlot.LLM.accepts("Qwen.GGUF"))
        assertFalse(ModelSlot.LLM.accepts("MMPROJ-model.gguf"))
        assertTrue(ModelSlot.VISION_PROJECTOR.accepts("model-mmproj-f16.gguf"))
        assertFalse(ModelSlot.VISION_PROJECTOR.accepts("model.gguf"))
        assertTrue(ModelSlot.DIFFUSION.accepts("x.CKPT"))
        assertFalse(ModelSlot.DIFFUSION.accepts("x.gguf"), "a stray GGUF is offered to the LLM screen, not diffusion")
    }
}
