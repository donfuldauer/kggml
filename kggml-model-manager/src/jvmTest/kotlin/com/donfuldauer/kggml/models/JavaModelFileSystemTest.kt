package com.donfuldauer.kggml.models

import java.io.File
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** [JavaModelFileSystem] on a real temporary directory, following the [ModelFileSystem] contract. */
class JavaModelFileSystemTest {
    private val fs = JavaModelFileSystem
    private lateinit var dir: File

    @BeforeTest
    fun setUp() {
        dir = Files.createTempDirectory("kggml-fs").toFile()
    }

    @AfterTest
    fun tearDown() {
        dir.deleteRecursively()
    }

    private fun path(name: String) = File(dir, name).path

    @Test
    fun defaultFileSystemIsJavaModelFileSystem() {
        assertSame(JavaModelFileSystem, defaultModelFileSystem())
    }

    @Test
    fun createDirectoriesIsIdempotentAndCreatesParents() {
        val nested = path("a/b/c")
        fs.createDirectories(nested)
        assertTrue(File(nested).isDirectory)
        fs.createDirectories(nested) // already exists: no error
        assertTrue(fs.exists(nested))
        assertEquals(-1L, fs.size(nested), "a directory is not a regular file")
        assertFalse(fs.delete(nested), "delete only removes files")
        assertTrue(File(nested).isDirectory)
    }

    @Test
    fun listReturnsSortedRegularFileNamesOnly() {
        File(dir, "b.bin").writeBytes(ByteArray(2))
        File(dir, "a.bin").writeBytes(ByteArray(1))
        File(dir, "sub").mkdir()
        File(dir, "sub/inner.bin").writeBytes(ByteArray(1))
        assertEquals(listOf("a.bin", "b.bin"), fs.list(dir.path))
        assertEquals(emptyList(), fs.list(path("a.bin")), "a file is not a directory")
        assertEquals(2L, fs.size(path("b.bin")))
    }

    @Test
    fun appendAndTruncate() {
        val p = path("m.bin")
        fs.openAppend(p).use { it.write(byteArrayOf(1, 2, 3)) }
        fs.openAppend(p).use { it.write(byteArrayOf(9, 4, 5, 9), offset = 1, length = 2) }
        assertContentEquals(byteArrayOf(1, 2, 3, 4, 5), File(p).readBytes())
        fs.openAppend(p, truncate = true).use { it.write(byteArrayOf(7)); it.flush() }
        assertContentEquals(byteArrayOf(7), File(p).readBytes())
    }

    @Test
    fun writeFailuresAreFileSystemExceptions() {
        val sink = fs.openAppend(path("m.bin"))
        sink.close()
        val e = assertFailsWith<ModelFileSystemException> { sink.write(byteArrayOf(1)) }
        assertTrue(e.message!!.startsWith("write failed"), e.message)
        val dirAsFile = assertFailsWith<ModelFileSystemException> { fs.openAppend(dir.path) }
        assertTrue(dirAsFile.message!!.startsWith("cannot open ${dir.path} for writing"), dirAsFile.message)
    }

    @Test
    fun randomAccessReadsFollowTheByteSourceContract() {
        val data = ByteArray(10) { (it + 1).toByte() }
        val p = path("r.bin")
        File(p).writeBytes(data)
        val source = fs.openRead(p)
        assertEquals(10L, source.size)
        val buffer = ByteArray(8)
        assertEquals(4, source.read(6, buffer, 2, 8 - 2), "short read at the end of the file")
        assertContentEquals(byteArrayOf(0, 0, 7, 8, 9, 10, 0, 0), buffer)
        assertEquals(0, source.read(3, buffer, 0, 0), "zero-length read")
        assertEquals(-1, source.read(10, buffer, 0, 1), "at the end")
        assertEquals(-1, source.read(50, buffer, 0, 1), "past the end")
        source.close()
        val e = assertFailsWith<ModelFileSystemException> { source.read(0, buffer, 0, 1) }
        assertTrue(e.message!!.startsWith("read failed"), e.message)
    }

    @Test
    fun atomicMoveReplacesTheTarget() {
        val from = path("new.bin")
        val to = path("old.bin")
        File(from).writeBytes(byteArrayOf(1, 2))
        File(to).writeBytes(byteArrayOf(9, 9, 9))
        fs.atomicMove(from, to)
        assertFalse(fs.exists(from))
        assertContentEquals(byteArrayOf(1, 2), File(to).readBytes())
    }
}
