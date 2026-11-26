package com.donfuldauer.kggml.models

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.runBlocking
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.util.concurrent.ConcurrentHashMap
import kotlin.concurrent.thread
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** [UrlConnectionHttpTransport] against local servers on 127.0.0.1 (no external network). */
class UrlConnectionHttpTransportTest {
    private val body = ByteArray(1000) { (it * 13 + 5).toByte() }
    private lateinit var server: HttpServer
    private val rawServers = ArrayList<ServerSocket>()

    /** Request headers of the last request, keyed by lowercase name. */
    private val lastRequest = ConcurrentHashMap<String, String>()

    private val transport = UrlConnectionHttpTransport(connectTimeoutMillis = 5_000, readTimeoutMillis = 5_000)

    @BeforeTest
    fun start() {
        server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/file") { ex ->
            record(ex)
            val range = ex.requestHeaders.getFirst("Range")?.let { Regex("""bytes=(\d+)-(\d+)""").matchEntire(it) }
            ex.responseHeaders.add("X-Custom", "v1")
            if (range == null) {
                ex.sendResponseHeaders(200, body.size.toLong())
                ex.responseBody.use { it.write(body) }
            } else {
                val first = range.groupValues[1].toInt()
                val last = minOf(range.groupValues[2].toInt(), body.size - 1)
                ex.responseHeaders.add("Content-Range", "bytes $first-$last/${body.size}")
                ex.sendResponseHeaders(206, (last - first + 1).toLong())
                ex.responseBody.use { it.write(body, first, last - first + 1) }
            }
        }
        server.createContext("/denied") { ex ->
            record(ex)
            val text = "access denied".encodeToByteArray()
            ex.sendResponseHeaders(403, text.size.toLong())
            ex.responseBody.use { it.write(text) }
        }
        server.createContext("/missing") { ex ->
            record(ex)
            ex.sendResponseHeaders(404, -1)
            ex.close()
        }
        server.createContext("/unavailable") { ex ->
            ex.responseHeaders.add("Retry-After", "7")
            ex.sendResponseHeaders(503, -1)
            ex.close()
        }
        server.createContext("/moved") { ex ->
            ex.responseHeaders.add("Location", "/file")
            ex.sendResponseHeaders(302, -1)
            ex.close()
        }
        server.start()
    }

    @AfterTest
    fun stop() {
        server.stop(0)
        rawServers.forEach { it.close() }
    }

    private fun record(ex: HttpExchange) {
        lastRequest.clear()
        ex.requestHeaders.forEach { (k, v) -> lastRequest[k.lowercase()] = v.joinToString(",") }
    }

    private fun url(path: String) = "http://127.0.0.1:${server.address.port}$path"

    private suspend fun HttpResponse.readAll(bufferSize: Int = 64): ByteArray {
        val out = ByteArrayOutputStream()
        val buffer = ByteArray(bufferSize)
        while (true) {
            val n = read(buffer)
            if (n < 0) break
            out.write(buffer, 0, n)
        }
        return out.toByteArray()
    }

    /** A raw TCP server that reads each request's headers, writes [response] (if any) and closes the connection. */
    private fun rawServer(response: ByteArray?): Int {
        val socket = ServerSocket(0, 50, InetAddress.getLoopbackAddress())
        rawServers += socket
        thread(isDaemon = true) {
            while (true) {
                val client = try {
                    socket.accept()
                } catch (_: IOException) {
                    break
                }
                client.use {
                    val reader = it.getInputStream().bufferedReader(Charsets.ISO_8859_1)
                    while (true) {
                        val line = reader.readLine() ?: break
                        if (line.isEmpty()) break
                    }
                    if (response != null) {
                        it.getOutputStream().write(response)
                        it.getOutputStream().flush()
                    }
                }
            }
        }
        return socket.localPort
    }

    @Test
    fun fullGetReturnsStatusHeadersAndBody() = runBlocking {
        val (status, bytes) = transport.get(url("/file"), mapOf("Authorization" to "Bearer t")) { r ->
            assertEquals("1000", r.header("Content-Length"))
            assertEquals("1000", r.header("content-length"), "header names are case-insensitive")
            assertEquals("v1", r.header("X-CUSTOM"))
            assertNull(r.header("X-Not-Sent"))
            r.statusCode to r.readAll()
        }
        assertEquals(200, status)
        assertContentEquals(body, bytes)
        assertEquals("Bearer t", lastRequest["authorization"])
        // Byte offsets must refer to the stored bytes, so compression is refused.
        assertEquals("identity", lastRequest["accept-encoding"])
        assertNull(lastRequest["range"])
    }

    @Test
    fun rangeRequestReturnsPartialContent() = runBlocking {
        transport.get(url("/file"), mapOf("Range" to "bytes=100-299")) { r ->
            assertEquals(206, r.statusCode)
            assertEquals("bytes 100-299/1000", r.header("Content-Range"))
            assertContentEquals(body.copyOfRange(100, 300), r.readAll(bufferSize = 7))
        }
        assertEquals("bytes=100-299", lastRequest["range"])
    }

    @Test
    fun errorStatusesExposeTheErrorBody() = runBlocking {
        transport.get(url("/denied"), emptyMap()) { r ->
            assertEquals(403, r.statusCode)
            assertEquals("access denied", r.readAll().decodeToString())
        }
        transport.get(url("/missing"), emptyMap()) { r ->
            assertEquals(404, r.statusCode)
            assertEquals(-1, r.read(ByteArray(16)), "no error body reads as end of body")
        }
        transport.get(url("/unavailable"), emptyMap()) { r ->
            assertEquals(503, r.statusCode)
            assertEquals("7", r.header("Retry-After"))
        }
    }

    @Test
    fun redirectsAreFollowed() = runBlocking {
        val bytes = transport.get(url("/moved"), emptyMap()) { r ->
            assertEquals(200, r.statusCode)
            r.readAll()
        }
        assertContentEquals(body, bytes)
    }

    @Test
    fun handlerResultAndExceptionsPassThrough() = runBlocking {
        assertEquals("done", transport.get(url("/file"), emptyMap()) { "done" })
        val e = assertFailsWith<IllegalStateException> {
            transport.get(url("/file"), emptyMap()) { throw IllegalStateException("from handler") }
        }
        assertEquals("from handler", e.message)
    }

    @Test
    fun connectionRefusedIsATransportError() = runBlocking {
        val port = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).use { it.localPort }
        val target = "http://127.0.0.1:$port/x"
        val e = assertFailsWith<DownloadTransportException> { transport.get(target, emptyMap()) { it.statusCode } }
        assertTrue(e.message!!.startsWith("cannot connect to $target"), e.message)
        assertTrue(e.cause is IOException)
    }

    @Test
    fun malformedUrlsAreProtocolErrors() = runBlocking {
        // URISyntaxException: a space in the host.
        val spaced = assertFailsWith<DownloadProtocolException> { transport.get("http://bad host/x", emptyMap()) { 0 } }
        assertTrue(spaced.message!!.startsWith("invalid URL http://bad host/x"), spaced.message)
        // IllegalArgumentException: a relative URI cannot be turned into a URL.
        val relative = assertFailsWith<DownloadProtocolException> { transport.get("models/a.gguf", emptyMap()) { 0 } }
        assertTrue(relative.message!!.startsWith("invalid URL models/a.gguf"), relative.message)
    }

    @Test
    fun connectionClosedBeforeAnyResponseIsATransportError() = runBlocking {
        val target = "http://127.0.0.1:${rawServer(response = null)}/x"
        val e = assertFailsWith<DownloadTransportException> { transport.get(target, emptyMap()) { it.statusCode } }
        assertTrue(e.message!!.startsWith("no response from $target"), e.message)
    }

    @Test
    fun bodyCutOffMidStreamIsATransportError() = runBlocking {
        // Chunked body whose connection closes before the terminating zero-length chunk.
        val response = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n".encodeToByteArray()
        val target = "http://127.0.0.1:${rawServer(response)}/x"
        transport.get(target, emptyMap()) { r ->
            assertEquals(200, r.statusCode)
            val e = assertFailsWith<DownloadTransportException> { r.readAll() }
            assertTrue(e.message!!.startsWith("read from $target failed"), e.message)
        }
    }

    @Test
    fun defaultTransportIsUrlConnection() {
        assertTrue(defaultHttpTransport() is UrlConnectionHttpTransport)
    }
}
