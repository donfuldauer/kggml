package com.donfuldauer.kggml.models

/**
 * One model file in a [ModelCache].
 *
 * @property fileName the file name inside [ModelCache.directory].
 * @property path the full path of the file.
 * @property sizeBytes the file size in bytes.
 * @property lastAccessMillis last [ModelCache.touch] (or the file's modification time if never touched).
 * @property pinned whether the file is pinned (in use) and therefore never evicted.
 */
public data class ModelCacheEntry(
    val fileName: String,
    val path: String,
    val sizeBytes: Long,
    val lastAccessMillis: Long,
    val pinned: Boolean,
)

/**
 * On-disk model cache: a directory of downloaded model files with a size budget and least-recently-used eviction.
 *
 * Files are addressed by plain file names (no directories). In-progress downloads (`*.part`) are ignored by the size
 * accounting. Pinned files (models currently loaded by an engine) are never evicted.
 *
 * @param directory cache directory; created on first use.
 * @param maxBytes disk budget enforced by [trimToSize] and after [touch].
 * @param clock epoch-millisecond clock (injectable for tests).
 */
public class ModelCache(
    public val directory: String,
    private val fileSystem: ModelFileSystem = defaultModelFileSystem(),
    public val maxBytes: Long = Long.MAX_VALUE,
    private val clock: () -> Long = { currentTimeMillis() },
) {
    private val lock = SpinLock()
    private val lastAccess = HashMap<String, Long>()
    private val pins = HashMap<String, Int>()

    init {
        require(maxBytes >= 0) { "maxBytes must be >= 0" }
    }

    /** Path for [fileName] inside the cache (the file may not exist yet; use it as a download destination). */
    public fun pathFor(fileName: String): String {
        require(fileName.isNotBlank() && '/' !in fileName && '\\' !in fileName && fileName != "." && fileName != "..") {
            "invalid cache file name '$fileName'"
        }
        fileSystem.createDirectories(directory)
        return joinPath(directory, fileName)
    }

    /** A download request whose destination is [fileName] in this cache. */
    public fun downloadRequest(
        url: String,
        fileName: String,
        expectedSha256: String? = null,
        expectedSizeBytes: Long? = null,
    ): DownloadRequest = DownloadRequest(url, pathFor(fileName), expectedSha256, expectedSizeBytes)

    /** Whether a complete [fileName] is cached. */
    public fun contains(fileName: String): Boolean = fileSystem.size(joinPath(directory, fileName)) >= 0

    /** Complete cached files, least recently used first. */
    public fun entries(): List<ModelCacheEntry> {
        val names = fileSystem.list(directory).filterNot { it.endsWith(DownloadRequest.PARTIAL_SUFFIX) }
        return lock.withLock {
            names.mapNotNull { name ->
                val path = joinPath(directory, name)
                val size = fileSystem.size(path)
                if (size < 0) return@mapNotNull null
                val access = lastAccess[name] ?: fileSystem.lastModifiedMillis(path)
                ModelCacheEntry(name, path, size, access, (pins[name] ?: 0) > 0)
            }
        }.sortedWith(compareBy<ModelCacheEntry> { it.lastAccessMillis }.thenBy { it.fileName })
    }

    /** Total bytes of complete cached files. */
    public fun totalBytes(): Long = entries().sumOf { it.sizeBytes }

    /** Marks [fileName] as used now, then enforces [maxBytes]. */
    public fun touch(fileName: String) {
        lock.withLock { lastAccess[fileName] = clock() }
        trimToSize()
    }

    /**
     * Pins [fileName] so it is never evicted while the returned handle is open (for example while an engine has it
     * loaded). Pins nest; closing a handle twice is a no-op.
     */
    public fun pin(fileName: String): AutoCloseable {
        lock.withLock {
            pins[fileName] = (pins[fileName] ?: 0) + 1
            lastAccess[fileName] = clock()
        }
        var closed = false
        return AutoCloseable {
            lock.withLock {
                if (!closed) {
                    closed = true
                    val n = (pins[fileName] ?: 1) - 1
                    if (n <= 0) pins.remove(fileName) else pins[fileName] = n
                }
            }
        }
    }

    /**
     * Deletes least-recently-used, unpinned files until the cache holds at most [targetBytes].
     *
     * @return the evicted file names, in eviction order.
     */
    public fun trimToSize(targetBytes: Long = maxBytes): List<String> {
        require(targetBytes >= 0) { "targetBytes must be >= 0" }
        val all = entries()
        var total = all.sumOf { it.sizeBytes }
        val evicted = ArrayList<String>()
        for (entry in all) {
            if (total <= targetBytes) break
            if (entry.pinned || isPinned(entry.fileName)) continue
            if (fileSystem.delete(entry.path)) {
                total -= entry.sizeBytes
                evicted += entry.fileName
                lock.withLock { lastAccess.remove(entry.fileName) }
            }
        }
        return evicted
    }

    /** Deletes [fileName] unless it is pinned. Returns whether it was deleted. */
    public fun remove(fileName: String): Boolean {
        if (isPinned(fileName)) return false
        lock.withLock { lastAccess.remove(fileName) }
        return fileSystem.delete(joinPath(directory, fileName))
    }

    /** Deletes abandoned `*.part` files (for example after the user cancels a download for good). */
    public fun clearPartialDownloads(): List<String> =
        fileSystem.list(directory).filter { it.endsWith(DownloadRequest.PARTIAL_SUFFIX) }
            .filter { fileSystem.delete(joinPath(directory, it)) }

    private fun isPinned(fileName: String): Boolean = lock.withLock { (pins[fileName] ?: 0) > 0 }
}

/** Epoch milliseconds. */
internal expect fun currentTimeMillis(): Long
