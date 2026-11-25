package com.donfuldauer.kggml.models

/**
 * Minimal HTTP GET client used by [ModelDownloader]. Kept small so apps can plug in their own stack (OkHttp, Ktor,
 * a corporate proxy) and tests can use a fake server.
 *
 * Implementations must call `handle` in the caller's coroutine context (no `withContext` around it), because the
 * downloader emits progress from inside it.
 */
public interface HttpTransport {
    /**
     * Performs a GET of [url] with [headers] (for example `Range`) and passes the response to [handle]. The response
     * is only valid inside [handle]; it is closed when [handle] returns.
     *
     * @throws DownloadTransportException on network errors (retried by the downloader).
     */
    public suspend fun <T> get(url: String, headers: Map<String, String>, handle: suspend (HttpResponse) -> T): T
}

/** A response passed to the [HttpTransport.get] handler. */
public interface HttpResponse {
    /** HTTP status code. */
    public val statusCode: Int

    /** Value of response header [name] (case-insensitive), or `null`. */
    public fun header(name: String): String?

    /**
     * Reads up to `buffer.size` body bytes. Returns the count read, or `-1` at the end of the body.
     *
     * @throws DownloadTransportException on network errors.
     */
    public suspend fun read(buffer: ByteArray): Int
}

/** The platform [HttpTransport] (`HttpURLConnection` on Android and JVM, `NSURLSession` on iOS). */
public expect fun defaultHttpTransport(): HttpTransport

/** A parsed `Content-Range: bytes <first>-<last>/<total|*>` header. */
internal data class ContentRange(val first: Long?, val last: Long?, val total: Long?) {
    companion object {
        private val PATTERN = Regex("""^\s*bytes\s+(?:(\d+)-(\d+)|\*)/(\d+|\*)\s*$""", RegexOption.IGNORE_CASE)

        fun parse(value: String?): ContentRange? {
            val m = PATTERN.matchEntire(value ?: return null) ?: return null
            val first = m.groupValues[1].takeIf { it.isNotEmpty() }?.toLongOrNull()
            val last = m.groupValues[2].takeIf { it.isNotEmpty() }?.toLongOrNull()
            val total = m.groupValues[3].takeIf { it != "*" }?.toLongOrNull()
            if (first != null && last != null && last < first) return null
            return ContentRange(first, last, total)
        }
    }
}
