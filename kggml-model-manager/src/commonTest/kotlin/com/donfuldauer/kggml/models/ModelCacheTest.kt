package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ModelCacheTest {
    private val fs = FakeFileSystem()
    private var now = 1_000L
    private fun cache(max: Long = Long.MAX_VALUE) = ModelCache("/c", fs, max) { now }

    private fun put(name: String, size: Int, modified: Long) {
        fs.files["/c/$name"] = ByteArray(size)
        fs.modified["/c/$name"] = modified
    }

    @Test
    fun listsCompleteFilesLeastRecentlyUsedFirst() {
        put("b.gguf", 10, 20)
        put("a.gguf", 5, 10)
        put("c.gguf.part", 99, 0)
        val c = cache()
        assertEquals(listOf("a.gguf", "b.gguf"), c.entries().map { it.fileName })
        assertEquals(15L, c.totalBytes())
        assertTrue(c.contains("a.gguf"))
        assertFalse(c.contains("c.gguf"))
        c.touch("a.gguf")
        assertEquals(listOf("b.gguf", "a.gguf"), c.entries().map { it.fileName })
        assertEquals(1_000L, c.entries().last().lastAccessMillis)
    }

    @Test
    fun trimEvictsLruButNeverPinned() {
        put("a", 40, 1)
        put("b", 40, 2)
        put("c", 40, 3)
        val c = cache(max = 100)
        val pin = c.pin("a")
        now = 0 // pinning set a's access to the old clock value; make b the LRU unpinned
        assertEquals(listOf("b"), c.trimToSize())
        assertEquals(listOf("a", "c"), c.entries().map { it.fileName }.sorted())
        assertTrue(c.entries().first { it.fileName == "a" }.pinned)
        assertEquals(listOf("c"), c.trimToSize(0))
        assertFalse(c.remove("a"), "pinned files are not removed")
        pin.close()
        pin.close() // idempotent
        assertTrue(c.remove("a"))
        assertTrue(c.entries().isEmpty())
    }

    @Test
    fun pinsNestAndTouchEnforcesBudget() {
        put("a", 60, 1)
        put("b", 60, 2)
        val c = cache(max = 100)
        val p1 = c.pin("b")
        val p2 = c.pin("b")
        p1.close()
        c.touch("b")
        assertEquals(listOf("b"), c.entries().map { it.fileName }, "a evicted, b still pinned by p2")
        p2.close()
        assertEquals(listOf("b"), c.trimToSize(0))
    }

    @Test
    fun pathsAndRequests() {
        val c = cache()
        assertEquals("/c/m.gguf", c.pathFor("m.gguf"))
        assertTrue("/c" in fs.directories)
        assertEquals(DownloadRequest("u", "/c/m.gguf", null, 5), c.downloadRequest("u", "m.gguf", expectedSizeBytes = 5))
        listOf("", "../x", "a/b", "..", ".", "a\\b").forEach { name ->
            assertFailsWith<IllegalArgumentException>(name) { c.pathFor(name) }
        }
        assertFailsWith<IllegalArgumentException> { ModelCache("/c", fs, -1) }
        assertFailsWith<IllegalArgumentException> { c.trimToSize(-1) }
    }

    @Test
    fun clearsPartialDownloads() {
        put("a.part", 1, 0)
        put("b", 1, 0)
        assertEquals(listOf("a.part"), cache().clearPartialDownloads())
        assertEquals(listOf("/c/b"), fs.files.keys.toList())
    }

    @Test
    fun filesThatCannotBeDeletedAreSkippedByTrim() {
        put("a", 40, 1)
        put("b", 40, 2)
        put("c", 40, 3)
        val stuck = object : ModelFileSystem by fs {
            override fun delete(path: String) = path != "/c/a" && fs.delete(path)
        }
        val c = ModelCache("/c", stuck, maxBytes = 100) { now }
        assertEquals(listOf("b"), c.trimToSize(), "a (LRU) cannot be deleted, so b goes and 80 bytes remain")
        assertEquals(listOf("a", "c"), c.entries().map { it.fileName })
        assertEquals(listOf("c"), c.trimToSize(0))
        assertEquals(40L, c.totalBytes())
        assertFalse(c.remove("a"))
    }

    @Test
    fun filesThatVanishAfterListingAreIgnored() {
        put("a", 10, 1)
        put("gone", 10, 2)
        val racy = object : ModelFileSystem by fs {
            override fun size(path: String) = if (path == "/c/gone") -1L else fs.size(path)
        }
        val c = ModelCache("/c", racy) { now }
        assertEquals(listOf("a"), c.entries().map { it.fileName })
        assertEquals(10L, c.totalBytes())
        assertFalse(c.contains("gone"))
    }
}
