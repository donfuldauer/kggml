package com.donfuldauer.kggml.models

import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import java.io.File
import java.net.InetSocketAddress
import java.nio.file.Files
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

/** Real `HttpURLConnection` + `java.nio` file system against a local Range-capable HTTP server. */
class JvmIntegrationTest {
    private val content = ByteArray(300_000) { (it * 31 + it / 7).toByte() }
    private val sha = Sha256.hashHex(content)
    private lateinit var server: HttpServer
    private lateinit var dir: File
    private val requests = AtomicInteger()
    private val dropNext = AtomicInteger() // number of upcoming requests to cut off mid-body
    @Volatile private var ignoreRange = false
    // Range requests starting at or after this offset wait for `gate`, so a test can stop a download part-way.
    @Volatile private var holdFrom = Int.MAX_VALUE
    private val gate = CountDownLatch(1)

    @BeforeTest
    fun start() {
        dir = Files.createTempDirectory("kggml-mm").toFile()
        server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/model.bin") { ex ->
            requests.incrementAndGet()
            val range = ex.requestHeaders.getFirst("Range")?.let { Regex("""bytes=(\d+)-(\d+)""").matchEntire(it) }
            if (range == null || ignoreRange) {
                ex.sendResponseHeaders(200, content.size.toLong())
                ex.responseBody.use { it.write(content) }
                return@createContext
            }
            val first = range.groupValues[1].toInt()
            if (first >= holdFrom) gate.await(30, TimeUnit.SECONDS)
            if (first >= content.size) {
                ex.responseHeaders.add("Content-Range", "bytes */${content.size}")
                ex.sendResponseHeaders(416, -1)
                ex.close()
                return@createContext
            }
            val last = minOf(range.groupValues[2].toInt(), content.size - 1)
            ex.responseHeaders.add("Content-Range", "bytes $first-$last/${content.size}")
            val len = last - first + 1
            ex.sendResponseHeaders(206, len.toLong())
            if (dropNext.getAndUpdate { if (it > 0) it - 1 else 0 } > 0) {
                ex.responseBody.write(content, first, len / 2)
                ex.responseBody.flush()
                ex.close() // connection closed before Content-Length bytes were sent
                return@createContext
            }
            ex.responseBody.use { it.write(content, first, len) }
        }
        server.createContext("/missing") { ex ->
            ex.sendResponseHeaders(404, -1)
            ex.close()
        }
        server.start()
    }

    @AfterTest
    fun stop() {
        gate.countDown()
        server.stop(0)
        dir.deleteRecursively()
    }

    private val url get() = "http://127.0.0.1:${server.address.port}/model.bin"
    private fun downloader() = ModelDownloader(chunkSizeBytes = 64 * 1024, initialRetryDelay = 10.milliseconds)

    @Test
    fun downloadsVerifiesAndPromotes() = runBlocking {
        val dest = File(dir, "nested/model.bin").path
        val completed = downloader().downloadTo(DownloadRequest(url, dest, sha, content.size.toLong()))
        assertEquals(sha, completed.sha256)
        assertContentEquals(content, File(dest).readBytes())
        assertFalse(File("$dest.part").exists())
        assertEquals(5, requests.get())
    }

    @Test
    fun droppedConnectionsResume() = runBlocking {
        dropNext.set(2)
        val dest = File(dir, "model.bin").path
        downloader().downloadTo(DownloadRequest(url, dest, sha))
        assertContentEquals(content, File(dest).readBytes())
        assertTrue(requests.get() > 5)
    }

    @Test
    fun cancelledDownloadResumesFromPartialFile() = runBlocking {
        val dest = File(dir, "model.bin").path
        val request = DownloadRequest(url, dest, sha)
        // download() runs on the I/O dispatcher ahead of this collector (flowOn buffers its events), so without a
        // hold it can finish the whole file before the collector cancels. The server holds every request from the
        // third 64 KiB chunk on, so the download can never get past 128 KiB. `first` returns only once the cancelled
        // download has finished, which needs the held request answered: release it shortly after cancelling.
        holdFrom = 2 * 64 * 1024
        downloader().download(request).filterIsInstance<DownloadEvent.Progress>().first {
            (it.downloadedBytes >= 100_000).also { reached ->
                if (reached) Thread { Thread.sleep(200); gate.countDown() }.apply { isDaemon = true }.start()
            }
        }
        holdFrom = Int.MAX_VALUE
        val partial = File("$dest.part").length()
        assertTrue(partial in 100_000 until content.size)
        val completed = downloader().downloadTo(request)
        assertEquals(partial, completed.resumedFromBytes)
        assertContentEquals(content, File(dest).readBytes())
    }

    @Test
    fun serverWithoutRangeSupport() = runBlocking {
        ignoreRange = true
        val dest = File(dir, "model.bin").path
        File("$dest.part").writeBytes(ByteArray(1234))
        downloader().downloadTo(DownloadRequest(url, dest, sha))
        assertContentEquals(content, File(dest).readBytes())
    }

    @Test
    fun mismatchAndHttpErrors() = runBlocking {
        val dest = File(dir, "model.bin").path
        assertFailsWith<ChecksumMismatchException> { downloader().downloadTo(DownloadRequest(url, dest, "f".repeat(64))) }
        assertFalse(File(dest).exists())
        assertFalse(File("$dest.part").exists())
        val missing = "http://127.0.0.1:${server.address.port}/missing"
        assertEquals(404, assertFailsWith<DownloadHttpException> { downloader().downloadTo(DownloadRequest(missing, dest)) }.statusCode)
        val refused = ModelDownloader(maxRetries = 1, initialRetryDelay = 1.milliseconds)
        assertFailsWith<DownloadTransportException> { refused.downloadTo(DownloadRequest("http://127.0.0.1:1/x", dest)) }
    }

    @Test
    fun fileSystemAndInspectorOnRealFiles() {
        val fs = JavaModelFileSystem
        val path = File(dir, "w.bin").path
        assertEquals(-1L, fs.size(path))
        fs.openAppend(path).use { it.write(tinyWhisperGgml()); it.flush() }
        assertIs<ModelHeaderInfo.Whisper>(ModelInspector.inspectFile(path))
        fs.openAppend(path, truncate = true).use { it.write(tinyLlamaGguf(withData = true)) }
        val info = assertIs<ModelHeaderInfo.Gguf>(ModelInspector.inspectFile(path))
        assertEquals("Q4_K_M", info.quantization)
        assertEquals(listOf("w.bin"), fs.list(dir.path))
        assertTrue(fs.lastModifiedMillis(path) > 0)
        fs.atomicMove(path, File(dir, "x.bin").path)
        assertFalse(fs.exists(path))
        assertTrue(fs.delete(File(dir, "x.bin").path))
        assertFalse(fs.delete(File(dir, "x.bin").path))
        assertTrue(fs.list(File(dir, "nope").path).isEmpty())
        assertFailsWith<ModelFileSystemException> { fs.openRead(File(dir, "nope").path) }
        assertFailsWith<ModelFileSystemException> { fs.atomicMove(File(dir, "nope").path, path) }
        File(dir, "f").writeText("x")
        assertFailsWith<ModelFileSystemException> { fs.createDirectories(File(dir, "f/sub").path) }
    }

    @Test
    fun diskCacheOnRealFiles() = runBlocking {
        val cache = ModelCache(File(dir, "cache").path, maxBytes = content.size.toLong())
        downloader().downloadTo(cache.downloadRequest(url, "a.bin", sha))
        downloader().downloadTo(cache.downloadRequest(url, "b.bin", sha))
        cache.touch("b.bin")
        assertEquals(listOf("b.bin"), cache.entries().map { it.fileName })
    }
}
