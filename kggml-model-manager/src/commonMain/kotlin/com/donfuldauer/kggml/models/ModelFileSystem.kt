package com.donfuldauer.kggml.models

/**
 * Minimal blocking file-system API used by the model manager. Paths are platform paths (`/` separated on every
 * supported platform). Call it from an I/O dispatcher.
 *
 * [defaultModelFileSystem] returns the platform implementation (`java.io`/`java.nio` on Android and JVM, POSIX on
 * iOS); tests use an in-memory fake.
 */
public interface ModelFileSystem {
    /** Whether a file or directory exists at [path]. */
    public fun exists(path: String): Boolean

    /** Size of the regular file at [path] in bytes, or `-1` if it does not exist. */
    public fun size(path: String): Long

    /** Last modification time in epoch milliseconds, or `0` if unknown. */
    public fun lastModifiedMillis(path: String): Long

    /** Creates [path] and any missing parent directories. */
    public fun createDirectories(path: String)

    /** Names (not paths) of the regular files directly inside [directory]; empty if it does not exist. */
    public fun list(directory: String): List<String>

    /** Deletes the file at [path]. Returns `false` if it did not exist. */
    public fun delete(path: String): Boolean

    /**
     * Moves [source] to [target], replacing [target]. Atomic when both are on the same volume, so readers never see a
     * partially written [target].
     */
    public fun atomicMove(source: String, target: String)

    /** Opens [path] for random-access reading. */
    public fun openRead(path: String): ModelByteSource

    /** Opens [path] for appending, creating it if needed; with [truncate] it is emptied first. */
    public fun openAppend(path: String, truncate: Boolean = false): FileSink
}

/** Append-only output file. */
public interface FileSink : AutoCloseable {
    /** Appends [length] bytes of [buffer] from [offset]. */
    public fun write(buffer: ByteArray, offset: Int = 0, length: Int = buffer.size - offset)

    /** Flushes buffered bytes to the OS. */
    public fun flush()
}

/** The platform [ModelFileSystem]. */
public expect fun defaultModelFileSystem(): ModelFileSystem

internal fun parentOf(path: String): String? {
    val i = path.trimEnd('/').lastIndexOf('/')
    return when {
        i < 0 -> null
        i == 0 -> "/"
        else -> path.substring(0, i)
    }
}

internal fun joinPath(directory: String, name: String): String =
    if (directory.endsWith("/")) directory + name else "$directory/$name"
