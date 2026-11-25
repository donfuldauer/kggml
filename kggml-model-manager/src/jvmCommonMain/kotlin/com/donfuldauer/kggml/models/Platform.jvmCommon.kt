package com.donfuldauer.kggml.models

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

public actual fun defaultModelFileSystem(): ModelFileSystem = JavaModelFileSystem

public actual fun defaultHttpTransport(): HttpTransport = UrlConnectionHttpTransport()

public actual fun modelIoDispatcher(): CoroutineDispatcher = Dispatchers.IO

internal actual fun currentTimeMillis(): Long = System.currentTimeMillis()

/** [ModelFileSystem] backed by `java.io` / `java.nio` (Android API 26+ and JVM desktop). */
public object JavaModelFileSystem : ModelFileSystem {
    override fun exists(path: String): Boolean = File(path).exists()

    override fun size(path: String): Long = File(path).let { if (it.isFile) it.length() else -1L }

    override fun lastModifiedMillis(path: String): Long = File(path).lastModified()

    override fun createDirectories(path: String) {
        val dir = File(path)
        if (!dir.isDirectory && !dir.mkdirs() && !dir.isDirectory) throw ModelFileSystemException("cannot create directory $path")
    }

    override fun list(directory: String): List<String> =
        File(directory).listFiles()?.filter { it.isFile }?.map { it.name }?.sorted() ?: emptyList()

    override fun delete(path: String): Boolean = File(path).let { it.isFile && it.delete() }

    override fun atomicMove(source: String, target: String) {
        val from = File(source).toPath()
        val to = File(target).toPath()
        try {
            try {
                Files.move(from, to, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(from, to, StandardCopyOption.REPLACE_EXISTING)
            }
        } catch (e: IOException) {
            throw ModelFileSystemException("cannot move $source to $target: ${e.message}", e)
        }
    }

    override fun openRead(path: String): ModelByteSource = try {
        RandomAccessFileSource(RandomAccessFile(path, "r"))
    } catch (e: IOException) {
        throw ModelFileSystemException("cannot open $path: ${e.message}", e)
    }

    override fun openAppend(path: String, truncate: Boolean): FileSink = try {
        StreamFileSink(FileOutputStream(path, !truncate))
    } catch (e: IOException) {
        throw ModelFileSystemException("cannot open $path for writing: ${e.message}", e)
    }

    private class RandomAccessFileSource(private val file: RandomAccessFile) : ModelByteSource {
        override val size: Long = file.length()

        override fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
            if (position >= size) return -1
            if (length == 0) return 0
            return try {
                file.seek(position)
                file.read(buffer, offset, length)
            } catch (e: IOException) {
                throw ModelFileSystemException("read failed: ${e.message}", e)
            }
        }

        override fun close() = file.close()
    }

    private class StreamFileSink(private val out: FileOutputStream) : FileSink {
        override fun write(buffer: ByteArray, offset: Int, length: Int) = try {
            out.write(buffer, offset, length)
        } catch (e: IOException) {
            throw ModelFileSystemException("write failed: ${e.message}", e)
        }

        override fun flush() = try {
            out.flush()
        } catch (e: IOException) {
            throw ModelFileSystemException("flush failed: ${e.message}", e)
        }

        override fun close() = out.close()
    }
}
