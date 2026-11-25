package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.models.ModelFormatException
import com.donfuldauer.kggml.samples.showcase.models.DiskModelStore
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
