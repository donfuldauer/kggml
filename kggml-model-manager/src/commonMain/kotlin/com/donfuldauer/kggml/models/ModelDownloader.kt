package com.donfuldauer.kggml.models

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.last
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

/**
 * A model to download.
 *
 * @property url HTTP(S) URL.
 * @property destination final file path. Bytes are written to `"$destination.part"` and atomically moved here only
 *   after the size and SHA-256 checks pass, so [destination] is never partially written.
 * @property expectedSha256 lowercase or uppercase hex SHA-256; when set, a mismatch deletes the partial file and fails
 *   with [ChecksumMismatchException], and an existing [destination] is reused only if it matches.
 * @property expectedSizeBytes when set, a different size (announced by the server or written) fails with
 *   [DownloadSizeMismatchException].
 * @property headers extra request headers (for example `Authorization`).
 */
public data class DownloadRequest(
    val url: String,
    val destination: String,
    val expectedSha256: String? = null,
    val expectedSizeBytes: Long? = null,
    val headers: Map<String, String> = emptyMap(),
) {
    init {
        require(url.isNotBlank()) { "url must not be blank" }
        require(destination.isNotBlank()) { "destination must not be blank" }
        require(expectedSha256 == null || SHA256_HEX.matches(expectedSha256)) { "expectedSha256 must be 64 hex digits" }
        require(expectedSizeBytes == null || expectedSizeBytes >= 0) { "expectedSizeBytes must be >= 0" }
        require(headers.keys.none { it.equals("Range", ignoreCase = true) }) { "the Range header is managed by the downloader" }
    }

    /** Path of the partial file used while downloading. */
    val partialPath: String get() = "$destination$PARTIAL_SUFFIX"

    public companion object {
        /** Suffix of in-progress download files. */
        public const val PARTIAL_SUFFIX: String = ".part"
        private val SHA256_HEX = Regex("^[0-9a-fA-F]{64}$")
    }
}

/** Events emitted by [ModelDownloader.download]. */
public sealed interface DownloadEvent {
    /** The download started; [resumedFromBytes] bytes were already present in the partial file. */
    public data class Started(val resumedFromBytes: Long, val totalBytes: Long?) : DownloadEvent

    /** [downloadedBytes] of [totalBytes] (`null` if the server did not say) are on disk. */
    public data class Progress(val downloadedBytes: Long, val totalBytes: Long?) : DownloadEvent {
        /** Completed fraction in `0.0..1.0`, or `null` if [totalBytes] is unknown. */
        val fraction: Double? get() = totalBytes?.takeIf { it > 0 }?.let { downloadedBytes.toDouble() / it }
    }

    /** All bytes are on disk; the SHA-256 is being computed. */
    public data class Verifying(val sizeBytes: Long) : DownloadEvent

    /**
     * The verified file is at [path].
     *
     * @property path the final destination ([DownloadRequest.destination]).
     * @property sizeBytes the file size in bytes.
     * @property sha256 the lowercase hex SHA-256 of the file.
     * @property resumedFromBytes bytes that were already in the partial file when this download started.
     * @property alreadyPresent the destination already existed (and matched [DownloadRequest.expectedSha256]), so
     *   nothing was downloaded.
     */
    public data class Completed(
        val path: String,
        val sizeBytes: Long,
        val sha256: String,
        val resumedFromBytes: Long,
        val alreadyPresent: Boolean,
    ) : DownloadEvent
}

/**
 * Resumable, chunked model downloader using HTTP Range requests.
 *
 * - Data is requested in [chunkSizeBytes] ranges and appended to `<destination>.part`. An interrupted download
 *   (network error, cancellation, app kill) resumes from the partial file's size.
 * - Network errors and HTTP 408/429/5xx are retried up to [maxRetries] times in a row with exponential backoff
 *   starting at [initialRetryDelay]; the counter resets after every chunk that makes progress.
 * - Servers that ignore `Range` (HTTP 200) restart the download from byte 0.
 * - The finished file's size and SHA-256 are verified before it is atomically moved to the destination; on a
 *   mismatch the partial file is deleted so the next attempt starts clean.
 *
 * Collection is cancellable at chunk-read granularity; the partial file is kept for a later resume.
 */
public class ModelDownloader(
    private val transport: HttpTransport = defaultHttpTransport(),
    private val fileSystem: ModelFileSystem = defaultModelFileSystem(),
    private val chunkSizeBytes: Long = DEFAULT_CHUNK_SIZE,
    private val maxRetries: Int = 5,
    private val initialRetryDelay: Duration = 500.milliseconds,
    private val ioDispatcher: CoroutineDispatcher = modelIoDispatcher(),
    private val readBufferSize: Int = 64 * 1024,
) {
    init {
        require(chunkSizeBytes > 0) { "chunkSizeBytes must be > 0" }
        require(maxRetries >= 0) { "maxRetries must be >= 0" }
        require(readBufferSize > 0) { "readBufferSize must be > 0" }
    }

    /** Downloads [request], emitting [DownloadEvent]s and ending with [DownloadEvent.Completed]. */
    public fun download(request: DownloadRequest): Flow<DownloadEvent> = flow { run(request) }.flowOn(ioDispatcher)

    /** Downloads [request] and returns the [DownloadEvent.Completed] event. */
    public suspend fun downloadTo(request: DownloadRequest): DownloadEvent.Completed =
        download(request).filterIsInstance<DownloadEvent.Completed>().last()

    private suspend fun FlowCollector<DownloadEvent>.run(request: DownloadRequest) {
        val expectedSha = request.expectedSha256?.lowercase()
        val dest = request.destination
        val part = request.partialPath

        if (fileSystem.exists(dest)) {
            val size = fileSystem.size(dest)
            val sizeOk = request.expectedSizeBytes == null || request.expectedSizeBytes == size
            if (sizeOk) {
                emit(DownloadEvent.Verifying(size))
                val sha = Sha256.hashFileHex(fileSystem, dest)
                if (expectedSha == null || sha == expectedSha) {
                    emit(DownloadEvent.Completed(dest, size, sha, resumedFromBytes = 0, alreadyPresent = true))
                    return
                }
            }
        }
        parentOf(dest)?.let { fileSystem.createDirectories(it) }

        var position = fileSystem.size(part).coerceAtLeast(0)
        var total: Long? = request.expectedSizeBytes
        if (total != null && position > total) {
            fileSystem.delete(part)
            position = 0
        }
        val resumedFrom = position
        emit(DownloadEvent.Started(resumedFrom, total))

        val buffer = ByteArray(readBufferSize)
        var failures = 0
        var done = total != null && position == total
        while (!done) {
            currentCoroutineContext().ensureActive()
            val end = (position + chunkSizeBytes - 1).let { if (total != null) minOf(it, total - 1) else it }
            val headers = request.headers + ("Range" to "bytes=$position-$end")
            val outcome = try {
                transport.get(request.url, headers) { response ->
                    handleResponse(request, response, position, end, total, buffer)
                }
            } catch (e: DownloadTransportException) {
                failures++
                if (failures > maxRetries) throw e
                delay(initialRetryDelay * (1 shl minOf(failures - 1, 16)))
                position = fileSystem.size(part).coerceAtLeast(0)
                continue
            }
            if (outcome.total != null) total = outcome.total
            if (outcome.newPosition > position || outcome.restarted) failures = 0
            position = outcome.newPosition
            done = outcome.finished || (total != null && position >= total)
        }

        if (!fileSystem.exists(part)) fileSystem.openAppend(part).close() // empty remote file
        val size = fileSystem.size(part).coerceAtLeast(0)
        val expectedSize = request.expectedSizeBytes ?: total
        if (expectedSize != null && size != expectedSize) {
            fileSystem.delete(part)
            throw DownloadSizeMismatchException(expectedSize, size)
        }
        emit(DownloadEvent.Verifying(size))
        val sha = Sha256.hashFileHex(fileSystem, part)
        if (expectedSha != null && sha != expectedSha) {
            fileSystem.delete(part)
            throw ChecksumMismatchException(expectedSha, sha)
        }
        fileSystem.atomicMove(part, dest)
        emit(DownloadEvent.Completed(dest, size, sha, resumedFrom, alreadyPresent = false))
    }

    private class ChunkOutcome(val newPosition: Long, val total: Long?, val finished: Boolean, val restarted: Boolean)

    private suspend fun FlowCollector<DownloadEvent>.handleResponse(
        request: DownloadRequest,
        response: HttpResponse,
        position: Long,
        end: Long,
        knownTotal: Long?,
        buffer: ByteArray,
    ): ChunkOutcome {
        val status = response.statusCode
        when {
            status == 206 -> {
                val range = ContentRange.parse(response.header("Content-Range"))
                    ?: throw DownloadProtocolException("206 response without a valid Content-Range")
                if (range.first != position) {
                    throw DownloadProtocolException("server returned range starting at ${range.first}, requested $position")
                }
                val total = range.total ?: knownTotal
                checkAnnouncedSize(request, total)
                val expected = range.last!! - range.first + 1
                val written = copyBody(response, request.partialPath, truncate = false, position, total, buffer, expected)
                if (written == 0L) throw DownloadTransportException("empty 206 body at offset $position")
                val newPosition = position + written
                // Unknown total: a short chunk means the end of the file.
                val finished = total == null && newPosition <= end && written < end - position + 1
                return ChunkOutcome(newPosition, total, finished, restarted = false)
            }
            status == 200 -> {
                // Range ignored: the body is the whole file.
                val total = response.header("Content-Length")?.toLongOrNull()?.takeIf { it >= 0 }
                checkAnnouncedSize(request, total)
                val written = copyBody(response, request.partialPath, truncate = true, 0, total, buffer, Long.MAX_VALUE)
                return ChunkOutcome(written, total ?: written, finished = true, restarted = position > 0)
            }
            status == 416 -> {
                val range = ContentRange.parse(response.header("Content-Range"))
                val total = range?.total ?: knownTotal
                if (total != null && position == total) return ChunkOutcome(position, total, finished = true, restarted = false)
                // The partial file is longer than the remote file (or the remote changed): start over.
                fileSystem.delete(request.partialPath)
                if (position == 0L) throw DownloadHttpException(status, request.url)
                return ChunkOutcome(0, total, finished = false, restarted = true)
            }
            status == 408 || status == 429 || status in 500..599 ->
                throw DownloadTransportException("HTTP $status for ${request.url}")
            else -> throw DownloadHttpException(status, request.url)
        }
    }

    private fun checkAnnouncedSize(request: DownloadRequest, total: Long?) {
        val expected = request.expectedSizeBytes ?: return
        if (total != null && total != expected) {
            fileSystem.delete(request.partialPath)
            throw DownloadSizeMismatchException(expected, total)
        }
    }

    private suspend fun FlowCollector<DownloadEvent>.copyBody(
        response: HttpResponse,
        path: String,
        truncate: Boolean,
        startPosition: Long,
        total: Long?,
        buffer: ByteArray,
        limit: Long,
    ): Long {
        var written = 0L
        fileSystem.openAppend(path, truncate).use { sink ->
            try {
                while (written < limit) {
                    currentCoroutineContext().ensureActive()
                    val n = response.read(buffer)
                    if (n < 0) break
                    if (n == 0) continue
                    val take = minOf(n.toLong(), limit - written).toInt()
                    sink.write(buffer, 0, take)
                    written += take
                    emit(DownloadEvent.Progress(startPosition + written, total))
                }
            } finally {
                // Keep what was written so the next attempt resumes after it.
                sink.flush()
            }
        }
        return written
    }

    public companion object {
        /** Default Range chunk size (8 MiB). */
        public const val DEFAULT_CHUNK_SIZE: Long = 8L * 1024 * 1024
    }
}

/** The dispatcher blocking model-manager I/O runs on (`Dispatchers.IO`). */
public expect fun modelIoDispatcher(): CoroutineDispatcher
