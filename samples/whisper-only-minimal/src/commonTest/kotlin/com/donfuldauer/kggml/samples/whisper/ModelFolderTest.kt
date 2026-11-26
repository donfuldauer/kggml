package com.donfuldauer.kggml.samples.whisper

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** The sample's model picker logic: which files in the model folder it offers and what it starts with. */
class ModelFolderTest {
    private val dir = "/data/files"

    @Test
    fun onlyBinFilesAreOfferedSortedByNameIgnoringCase() {
        val models = whisperModelsIn(
            dir,
            listOf("jfk.wav", "ggml-tiny.en.bin", ".hidden.bin", "Ggml-Base.BIN", "notes.txt", ".bin", "model.bin.part", "a.gguf"),
        )
        assertEquals(
            listOf(FolderModel("Ggml-Base.BIN", "/data/files/Ggml-Base.BIN"), FolderModel("ggml-tiny.en.bin", "/data/files/ggml-tiny.en.bin")),
            models,
        )
    }

    @Test
    fun pathsJoinWithOneSeparatorWhateverTheDirectoryEndsWith() {
        assertEquals("/d/m.bin", whisperModelsIn("/d/", listOf("m.bin")).single().path)
        assertEquals("/d/m.bin", whisperModelsIn("/d", listOf("m.bin")).single().path)
    }

    @Test
    fun initialPathPrefersTheRequestedThenTheDefaultThenTheFirstFolderModel() {
        val default = "$dir/ggml-tiny.en.bin"
        val tiny = FolderModel("ggml-tiny.en.bin", default)
        val base = FolderModel("ggml-base.en.bin", "$dir/ggml-base.en.bin")
        assertEquals("/sdcard/x.bin", initialModelPath("/sdcard/x.bin", default, listOf(base)))
        assertEquals(default, initialModelPath(null, default, listOf(base, tiny)), "the default model is in the folder")
        assertEquals(default, initialModelPath(null, default, emptyList()), "nothing in the folder: show where to put it")
        assertEquals(base.path, initialModelPath(null, default, listOf(base)), "default missing: use what is there")
        assertEquals(base.path, initialModelPath("  ", default, listOf(base)), "a blank request counts as none")
    }

    @Test
    fun chosenIndexMatchesTheExactPathOrIsNullForACustomPath() {
        val models = whisperModelsIn(dir, listOf("a.bin", "b.bin"))
        assertEquals(1, chosenModelIndex(models, "/data/files/b.bin"))
        assertEquals(0, chosenModelIndex(models, "  /data/files/a.bin "))
        assertNull(chosenModelIndex(models, "/elsewhere/a.bin"))
        assertNull(chosenModelIndex(models, ""))
        assertNull(chosenModelIndex(emptyList(), "/data/files/a.bin"))
    }

    @Test
    fun hintNamesTheFolder() {
        assertEquals("No whisper models (.bin) in /data/files yet: copy one there, or enter a path below.", noModelsHint(dir))
    }

    @Test
    fun scanListsTheModelFilesOfARealDirectory() {
        val model = NativeTestEnvironment.writeTempFile("ggml-tiny.en.bin", byteArrayOf(1, 2, 3))
        val folder = model.substringBeforeLast('/')
        assertEquals(listOf(FolderModel("ggml-tiny.en.bin", model)), scanWhisperModels(folder))
        assertEquals(listOf(FolderModel("ggml-tiny.en.bin", model)), scanWhisperModels("$folder/"))

        val audioOnly = NativeTestEnvironment.writeTempFile("jfk.wav", byteArrayOf(1)).substringBeforeLast('/')
        assertEquals(emptyList(), scanWhisperModels(audioOnly))
        assertEquals(emptyList(), scanWhisperModels("$folder/does-not-exist"))
    }
}
