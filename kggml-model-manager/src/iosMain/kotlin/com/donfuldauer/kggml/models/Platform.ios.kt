package com.donfuldauer.kggml.models

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.IO
import platform.Foundation.NSDate
import platform.Foundation.NSFileManager
import platform.Foundation.NSFileModificationDate
import platform.Foundation.NSFileType
import platform.Foundation.NSFileTypeRegular
import platform.Foundation.timeIntervalSince1970
import platform.posix.FILE
import platform.posix.F_OK
import platform.posix.SEEK_SET
import platform.posix.access
import platform.posix.errno
import platform.posix.fclose
import platform.posix.fflush
import platform.posix.fopen
import platform.posix.fread
import platform.posix.fseeko
import platform.posix.fwrite
import platform.posix.rename
import platform.posix.stat
import platform.posix.strerror
import platform.posix.unlink
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.toKString

public actual fun defaultModelFileSystem(): ModelFileSystem = PosixModelFileSystem

public actual fun defaultHttpTransport(): HttpTransport = NSURLSessionHttpTransport()

public actual fun modelIoDispatcher(): CoroutineDispatcher = Dispatchers.IO

internal actual fun currentTimeMillis(): Long = (NSDate().timeIntervalSince1970 * 1000.0).toLong()

/** [ModelFileSystem] on POSIX `stdio` plus `NSFileManager` for directories (iOS). */
@OptIn(ExperimentalForeignApi::class)
public object PosixModelFileSystem : ModelFileSystem {
    override fun exists(path: String): Boolean = access(path, F_OK) == 0

    override fun size(path: String): Long = memScoped {
        val st = alloc<stat>()
        if (stat(path, st.ptr) != 0) return@memScoped -1L
        if ((st.st_mode.toInt() and 0xF000) != 0x8000) return@memScoped -1L // S_ISREG
        st.st_size
    }

    override fun lastModifiedMillis(path: String): Long {
        val attrs = NSFileManager.defaultManager.attributesOfItemAtPath(path, null) ?: return 0L
        val date = attrs[NSFileModificationDate] as? NSDate ?: return 0L
        return (date.timeIntervalSince1970 * 1000.0).toLong()
    }

    override fun createDirectories(path: String) {
        if (!NSFileManager.defaultManager.createDirectoryAtPath(path, true, null, null)) {
            throw ModelFileSystemException("cannot create directory $path")
        }
    }

    override fun list(directory: String): List<String> {
        val fm = NSFileManager.defaultManager
        val names = fm.contentsOfDirectoryAtPath(directory, null) ?: return emptyList()
        return names.filterIsInstance<String>().filter { name ->
            val attrs = fm.attributesOfItemAtPath(joinPath(directory, name), null)
            attrs?.get(NSFileType) == NSFileTypeRegular
        }.sorted()
    }

    override fun delete(path: String): Boolean = size(path) >= 0 && unlink(path) == 0

    override fun atomicMove(source: String, target: String) {
        if (rename(source, target) != 0) throw ModelFileSystemException("cannot move $source to $target: ${lastError()}")
    }

    override fun openRead(path: String): ModelByteSource {
        val size = size(path)
        val file = fopen(path, "rb") ?: throw ModelFileSystemException("cannot open $path: ${lastError()}")
        return PosixSource(file, size)
    }

    override fun openAppend(path: String, truncate: Boolean): FileSink {
        val file = fopen(path, if (truncate) "wb" else "ab")
            ?: throw ModelFileSystemException("cannot open $path for writing: ${lastError()}")
        return PosixSink(file)
    }

    private fun lastError(): String = strerror(errno)?.toKString() ?: "errno $errno"

    private class PosixSource(private var file: CPointer<FILE>?, override val size: Long) : ModelByteSource {
        override fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
            val f = file ?: throw ModelFileSystemException("source is closed")
            if (position >= size) return -1
            if (length == 0) return 0
            if (fseeko(f, position, SEEK_SET) != 0) throw ModelFileSystemException("seek failed: ${lastError()}")
            val n = buffer.usePinned { fread(it.addressOf(offset), 1.convert(), length.convert(), f) }.toInt()
            return if (n == 0) -1 else n
        }

        override fun close() {
            file?.let { fclose(it) }
            file = null
        }
    }

    private class PosixSink(private var file: CPointer<FILE>?) : FileSink {
        override fun write(buffer: ByteArray, offset: Int, length: Int) {
            val f = file ?: throw ModelFileSystemException("sink is closed")
            if (length == 0) return
            val n = buffer.usePinned { fwrite(it.addressOf(offset), 1.convert(), length.convert(), f) }.toInt()
            if (n != length) throw ModelFileSystemException("write failed: ${lastError()}")
        }

        override fun flush() {
            file?.let { if (fflush(it) != 0) throw ModelFileSystemException("flush failed: ${lastError()}") }
        }

        override fun close() {
            file?.let { fclose(it) }
            file = null
        }
    }
}
