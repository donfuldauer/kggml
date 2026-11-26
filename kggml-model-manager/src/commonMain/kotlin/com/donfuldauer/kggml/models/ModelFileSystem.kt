package com.donfuldauer.kggml.models

/**
 * Minimal blocking file-system API used by the model manager. Paths are platform paths: `/` separated on every
 * supported platform, and `\` separated too on Windows. Call it from an I/O dispatcher.
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

/** `/` everywhere; `\` too, because Windows JVM paths (`java.io.File.getPath()`) use it. */
private fun isSeparator(c: Char): Boolean = c == '/' || c == '\\'

/**
 * The parent directory of [path], or `null` when it has none. Handles `/` and `\` separators and keeps a root as a
 * root: `/a` -> `/`, `C:\a` -> `C:\`.
 */
internal fun parentOf(path: String): String? {
    val trimmed = path.trimEnd('/', '\\')
    val i = trimmed.indexOfLast(::isSeparator)
    return when {
        i < 0 -> null
        i == 0 -> trimmed.substring(0, 1)
        i == 2 && trimmed[1] == ':' -> trimmed.substring(0, 3)
        else -> trimmed.substring(0, i)
    }
}

/**
 * [directory] + [name] with exactly one separator between them, in the directory's own style: `\` for a Windows path
 * that uses only `\` (so the result equals `java.io.File(directory, name).path`), else `/`.
 */
internal fun joinPath(directory: String, name: String): String = when {
    directory.isNotEmpty() && isSeparator(directory.last()) -> directory + name
    '\\' in directory && '/' !in directory -> "$directory\\$name"
    else -> "$directory/$name"
}
