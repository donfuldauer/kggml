package com.donfuldauer.kggml.models

/** In-memory [ModelFileSystem]. Directories are implicit except those created explicitly. */
class FakeFileSystem : ModelFileSystem {
    val files = LinkedHashMap<String, ByteArray>()
    val directories = HashSet<String>()
    val modified = HashMap<String, Long>()
    var failMoves = false

    override fun exists(path: String) = path in files || path in directories
    override fun size(path: String) = files[path]?.size?.toLong() ?: -1L
    override fun lastModifiedMillis(path: String) = modified[path] ?: 0L
    override fun createDirectories(path: String) { directories += path }
    override fun list(directory: String): List<String> {
        val prefix = if (directory.endsWith("/")) directory else "$directory/"
        return files.keys.filter { it.startsWith(prefix) && '/' !in it.removePrefix(prefix) }.map { it.removePrefix(prefix) }.sorted()
    }
    override fun delete(path: String) = files.remove(path) != null
    override fun atomicMove(source: String, target: String) {
        if (failMoves) throw ModelFileSystemException("move failed")
        files[target] = files.remove(source) ?: throw ModelFileSystemException("no $source")
    }
    override fun openRead(path: String): ModelByteSource {
        val bytes = files[path] ?: throw ModelFileSystemException("no $path")
        return ByteArrayModelSource(bytes)
    }
    override fun openAppend(path: String, truncate: Boolean): FileSink {
        if (truncate || path !in files) files[path] = ByteArray(0)
        return object : FileSink {
            override fun write(buffer: ByteArray, offset: Int, length: Int) {
                files[path] = files.getValue(path) + buffer.copyOfRange(offset, offset + length)
            }
            override fun flush() = Unit
            override fun close() = Unit
        }
    }
}

/**
 * Scripted in-memory HTTP server for [ModelDownloader] tests. Serves [content] at any URL with Range support and
 * lets tests inject failures per request.
 */
class FakeHttpServer(var content: ByteArray) : HttpTransport {
    /** What the server does for one request. */
    sealed interface Behavior {
        /** Normal Range-aware response. */
        data object Normal : Behavior
        /** Serves the requested range but drops the connection after [bytes] body bytes. */
        data class DropAfter(val bytes: Int) : Behavior
        /** Fails before any response. */
        data object ConnectFail : Behavior
        /** Ignores Range and returns 200 with the full body. */
        data object IgnoreRange : Behavior
        /** Responds with [status] and no body. */
        data class Status(val status: Int) : Behavior
        /** 206 with a Content-Range starting at a wrong offset. */
        data object WrongOffset : Behavior
        /** 206 without a total (`bytes a-b/ *`). */
        data object UnknownTotal : Behavior
        /** 206 without a Content-Range header. */
        data object NoContentRange : Behavior
    }

    val requests = ArrayList<Map<String, String>>()
    val script = ArrayDeque<Behavior>()
    var readSize = 7

    override suspend fun <T> get(url: String, headers: Map<String, String>, handle: suspend (HttpResponse) -> T): T {
        requests += headers
        val behavior = script.removeFirstOrNull() ?: Behavior.Normal
        if (behavior == Behavior.ConnectFail) throw DownloadTransportException("connection refused")
        if (behavior is Behavior.Status) return handle(response(behavior.status, ByteArray(0), emptyMap()))
        if (behavior == Behavior.IgnoreRange) {
            return handle(response(200, content, mapOf("content-length" to content.size.toString())))
        }
        val range = headers["Range"]?.let { Regex("""bytes=(\d+)-(\d+)""").matchEntire(it) }
        if (range == null) return handle(response(200, content, mapOf("content-length" to content.size.toString())))
        val first = range.groupValues[1].toLong()
        val last = minOf(range.groupValues[2].toLong(), content.size - 1L)
        if (first >= content.size) {
            return handle(response(416, ByteArray(0), mapOf("content-range" to "bytes */${content.size}")))
        }
        val body = content.copyOfRange(first.toInt(), last.toInt() + 1)
        val headerFirst = if (behavior == Behavior.WrongOffset) first + 1 else first
        val total = if (behavior == Behavior.UnknownTotal) "*" else content.size.toString()
        val respHeaders = if (behavior == Behavior.NoContentRange) emptyMap() else mapOf("content-range" to "bytes $headerFirst-$last/$total")
        val drop = (behavior as? Behavior.DropAfter)?.bytes
        return handle(response(206, body, respHeaders, drop))
    }

    private fun response(status: Int, body: ByteArray, headers: Map<String, String>, dropAfter: Int? = null) = object : HttpResponse {
        var pos = 0
        override val statusCode = status
        override fun header(name: String) = headers[name.lowercase()]
        override suspend fun read(buffer: ByteArray): Int {
            if (dropAfter != null && pos >= dropAfter) throw DownloadTransportException("connection reset")
            if (pos >= body.size) return -1
            var n = minOf(buffer.size, readSize, body.size - pos)
            if (dropAfter != null) n = minOf(n, dropAfter - pos)
            body.copyInto(buffer, 0, pos, pos + n)
            pos += n
            return n
        }
    }
}
