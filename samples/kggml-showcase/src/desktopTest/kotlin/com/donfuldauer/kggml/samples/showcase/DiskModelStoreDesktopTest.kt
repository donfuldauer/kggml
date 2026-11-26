package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.models.ChecksumMismatchException
import com.donfuldauer.kggml.models.DownloadEvent
import com.donfuldauer.kggml.models.HttpResponse
import com.donfuldauer.kggml.models.HttpTransport
import com.donfuldauer.kggml.models.ModelDownloader
import com.donfuldauer.kggml.models.ModelFormatException
import com.donfuldauer.kggml.models.defaultModelFileSystem
import com.donfuldauer.kggml.samples.showcase.models.CatalogModel
import com.donfuldauer.kggml.samples.showcase.models.DiskModelStore
import com.donfuldauer.kggml.samples.showcase.models.ModelKind
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** [DiskModelStore] over the real `java.io` file system of `:kggml-model-manager` in a temporary directory. */
class DiskModelStoreDesktopTest {
    private val dir: File = Files.createTempDirectory("kggml-store").toFile()

    @AfterTest
    fun cleanUp() {
        dir.deleteRecursively()
    }

    @Test
    fun scanningDoesNotCreateTheDirectory() {
        val missing = File(dir, "not-yet")
        val store = DiskModelStore(missing.absolutePath)
        assertTrue(store.files().isEmpty())
        assertEquals(0, store.partialBytes("x.gguf"))
        assertFalse(missing.exists())
    }

    @Test
    fun listsInspectsHashesAndDeletes() {
        File(dir, "tiny.gguf").writeBytes(ggufHeader())
        File(dir, "big.gguf.part").writeBytes(ByteArray(123))
        File(dir, "abc.txt").writeText("abc")
        val store = DiskModelStore(dir.absolutePath)

        assertEquals(listOf("abc.txt", "tiny.gguf"), store.files().map { it.fileName })
        assertEquals(123, store.partialBytes("big.gguf"))
        assertEquals(File(dir, "tiny.gguf").absolutePath, store.pathFor("tiny.gguf"))

        val metadata = store.inspect(store.pathFor("tiny.gguf"))
        assertEquals("GGUF v3", metadata.format)
        assertEquals("llama", metadata.architecture)
        assertEquals("tiny-test", metadata.name)
        assertTrue(metadata.estimatedPeakRam.totalBytes > 0)
        assertFailsWith<ModelFormatException> { store.inspect(store.pathFor("abc.txt")) }

        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", store.sha256("abc.txt"))
        assertEquals(listOf("big.gguf.part"), store.clearPartialDownloads())
        assertTrue(store.delete("abc.txt"))
        assertFalse(store.delete("abc.txt"))
        assertEquals(listOf("tiny.gguf"), store.files().map { it.fileName })
    }

    @Test
    fun directoryWithATrailingSeparatorFindsPartials() {
        File(dir, "big.gguf.part").writeBytes(ByteArray(42))
        val store = DiskModelStore(dir.absolutePath + File.separator)
        assertEquals(42, store.partialBytes("big.gguf"))
        assertEquals(0, store.partialBytes("other.gguf"))
    }

    @Test
    fun inspectsWhisperHeaders() {
        File(dir, "ggml-tiny.en.bin").writeBytes(whisperHeader(vocabularySize = 51864, audioLayers = 4))
        File(dir, "ggml-tiny.bin").writeBytes(whisperHeader(vocabularySize = 51865, audioLayers = 4))
        File(dir, "ggml-odd.bin").writeBytes(whisperHeader(vocabularySize = 51865, audioLayers = 7))
        val store = DiskModelStore(dir.absolutePath)

        val english = store.inspect(store.pathFor("ggml-tiny.en.bin"))
        assertEquals("Whisper ggml", english.format)
        assertEquals("whisper", english.architecture)
        assertEquals("whisper-tiny.en", english.name, "n_vocab < 51865 is an English-only model")
        assertEquals("F16", english.quantization)
        assertEquals(448, english.contextLength, "the text context")
        assertEquals(16, english.parameterCount)
        assertTrue(english.estimatedPeakRam.totalBytes > 0)

        assertEquals("whisper-tiny", store.inspect(store.pathFor("ggml-tiny.bin")).name, "multilingual")
        assertEquals(null, store.inspect(store.pathFor("ggml-odd.bin")).name, "unknown encoder depth: no size name")
    }

    @Test
    fun downloadsTheCatalogUrlIntoTheDirectoryAndVerifiesTheDigest() = runBlocking {
        val body = "abc".encodeToByteArray()
        val requested = mutableListOf<String>()
        val transport = object : HttpTransport {
            override suspend fun <T> get(url: String, headers: Map<String, String>, handle: suspend (HttpResponse) -> T): T {
                requested += url
                var sent = false
                return handle(object : HttpResponse {
                    override val statusCode = 200
                    override fun header(name: String) = if (name.equals("Content-Length", ignoreCase = true)) "${body.size}" else null
                    override suspend fun read(buffer: ByteArray): Int {
                        if (sent) return -1
                        sent = true
                        body.copyInto(buffer)
                        return body.size
                    }
                })
            }
        }
        val fileSystem = defaultModelFileSystem()
        val store = DiskModelStore(dir.absolutePath, fileSystem, ModelDownloader(transport = transport, fileSystem = fileSystem))
        val abcSha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        val model = CatalogModel("abc", "ABC", ModelKind.LLM, "https://example.invalid/abc.gguf", "abc.gguf", 3, "MIT", sha256 = abcSha)

        val completed = store.download(model).toList().last() as DownloadEvent.Completed
        assertEquals(listOf("https://example.invalid/abc.gguf"), requested)
        assertEquals(File(dir, "abc.gguf").absolutePath, completed.path)
        assertEquals(abcSha, completed.sha256)
        assertEquals("abc", File(dir, "abc.gguf").readText())

        // A digest that does not match the catalog's rejects the file.
        val wrong = model.copy(id = "bad", fileName = "bad.gguf", sha256 = "0".repeat(64))
        assertFailsWith<ChecksumMismatchException> { store.download(wrong).toList() }
        assertFalse(File(dir, "bad.gguf").exists())
    }

    /** A whisper.cpp ggml header (tiny-like hyper-parameters) with one 16-element F32 tensor. */
    private fun whisperHeader(vocabularySize: Int, audioLayers: Int): ByteArray {
        val out = ByteArrayOutputStream()
        fun i32(v: Int) = out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array())
        i32(0x67676d6c)
        // n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, n_text_ctx, n_text_state, n_text_head,
        // n_text_layer, n_mels, ftype (1 = F16)
        listOf(vocabularySize, 1500, 384, 6, audioLayers, 448, 384, 6, 4, 80, 1).forEach(::i32)
        i32(80); i32(201); out.write(ByteArray(80 * 201 * 4)) // mel filters
        i32(1); i32(1); out.write("a".encodeToByteArray()) // vocabulary
        val name = "decoder.ln.bias".encodeToByteArray()
        i32(1); i32(name.size); i32(0) // dims, name length, type F32
        i32(16)
        out.write(name)
        out.write(ByteArray(16 * 4))
        return out.toByteArray()
    }

    /** A GGUF v3 header with no tensors and two metadata strings. */
    private fun ggufHeader(): ByteArray {
        val out = ByteArrayOutputStream()
        fun u32(v: Int) = out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array())
        fun u64(v: Long) = out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array())
        fun str(s: String) {
            val bytes = s.encodeToByteArray()
            u64(bytes.size.toLong())
            out.write(bytes)
        }
        out.write("GGUF".encodeToByteArray())
        u32(3)
        u64(0) // tensors
        u64(2) // metadata entries
        str("general.architecture"); u32(8); str("llama")
        str("general.name"); u32(8); str("tiny-test")
        return out.toByteArray()
    }
}
