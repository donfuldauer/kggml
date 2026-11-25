package com.donfuldauer.kggml.models

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

class ModelDownloaderTest {
    private val content = ByteArray(1000) { (it * 7 + 3).toByte() }
    private val sha = Sha256.hashHex(content)
    private val fs = FakeFileSystem()
    private val server = FakeHttpServer(content)
    private val dest = "/cache/model.gguf"
    private val part = "$dest.part"

    private fun TestScope.downloader(chunk: Long = 128, retries: Int = 3) = ModelDownloader(
        transport = server,
        fileSystem = fs,
        chunkSizeBytes = chunk,
        maxRetries = retries,
        initialRetryDelay = 100.milliseconds,
        ioDispatcher = StandardTestDispatcher(testScheduler),
        readBufferSize = 50,
    )

    private fun request(sha256: String? = sha, size: Long? = content.size.toLong()) = DownloadRequest("https://h/m", dest, sha256, size)

    @Test
    fun downloadsInRangeChunksAndPromotesAtomically() = runTest {
        val events = downloader().download(request()).toList()
        assertContentEquals(content, fs.files[dest])
        assertNull(fs.files[part])
        assertEquals("/cache", fs.directories.single())
        assertEquals(DownloadEvent.Started(0, 1000), events.first())
        val completed = assertIs<DownloadEvent.Completed>(events.last())
        assertEquals(DownloadEvent.Completed(dest, 1000, sha, 0, alreadyPresent = false), completed)
        assertEquals(DownloadEvent.Verifying(1000), events[events.size - 2])
        val progress = events.filterIsInstance<DownloadEvent.Progress>()
        assertEquals(progress.map { it.downloadedBytes }.sorted(), progress.map { it.downloadedBytes })
        assertEquals(1000L, progress.last().downloadedBytes)
        assertEquals(1.0, progress.last().fraction)
        assertEquals(8, server.requests.size)
        assertEquals("bytes=0-127", server.requests.first()["Range"])
        assertEquals("bytes=896-999", server.requests.last()["Range"])
    }

    @Test
    fun resumesFromExistingPartialFile() = runTest {
        fs.files[part] = content.copyOf(300)
        val events = downloader().download(request()).toList()
        assertEquals(DownloadEvent.Started(300, 1000), events.first())
        assertEquals("bytes=300-427", server.requests.first()["Range"])
        val completed = events.last() as DownloadEvent.Completed
        assertEquals(300L, completed.resumedFromBytes)
        assertContentEquals(content, fs.files[dest])
    }

    @Test
    fun interruptedChunksAreRetriedFromTheBytesOnDisk() = runTest {
        server.script += FakeHttpServer.Behavior.Normal
        server.script += FakeHttpServer.Behavior.DropAfter(60)
        server.script += FakeHttpServer.Behavior.ConnectFail
        server.script += FakeHttpServer.Behavior.Status(503)
        val start = testScheduler.currentTime
        downloader().downloadTo(request())
        assertContentEquals(content, fs.files[dest])
        assertEquals("bytes=128-255", server.requests[1]["Range"])
        assertEquals("bytes=188-315", server.requests[2]["Range"], "resumes after the 60 bytes kept on disk")
        assertEquals("bytes=188-315", server.requests[3]["Range"])
        // Backoff 100 + 200 + 400 ms; progress after the drop does not reset it because ConnectFail/503 made none.
        assertEquals(700L, testScheduler.currentTime - start)
    }

    @Test
    fun interruptedDownloadResumesInANewCollection() = runTest {
        server.script += FakeHttpServer.Behavior.Normal
        server.script += FakeHttpServer.Behavior.DropAfter(10)
        repeat(3) { server.script += FakeHttpServer.Behavior.ConnectFail }
        val d = downloader(retries = 2)
        assertFailsWith<DownloadTransportException> { d.downloadTo(request()) }
        assertEquals(138, fs.files[part]!!.size, "partial bytes are kept for resume")
        assertNull(fs.files[dest])
        val completed = d.downloadTo(request())
        assertEquals(138L, completed.resumedFromBytes)
        assertContentEquals(content, fs.files[dest])
    }

    @Test
    fun checksumMismatchDeletesPartialAndFails() = runTest {
        val wrong = "0".repeat(64)
        val e = assertFailsWith<ChecksumMismatchException> { downloader().downloadTo(request(sha256 = wrong)) }
        assertEquals(wrong, e.expectedSha256)
        assertEquals(sha, e.actualSha256)
        assertNull(fs.files[part])
        assertNull(fs.files[dest])
    }

    @Test
    fun corruptResumedPrefixIsCaughtBySha() = runTest {
        fs.files[part] = ByteArray(300) // zeros, not the real prefix
        assertFailsWith<ChecksumMismatchException> { downloader().downloadTo(request()) }
        assertNull(fs.files[part])
        // Next attempt starts clean and succeeds.
        downloader().downloadTo(request())
        assertContentEquals(content, fs.files[dest])
    }

    @Test
    fun uppercaseExpectedShaIsAccepted() = runTest {
        downloader().downloadTo(request(sha256 = sha.uppercase()))
        assertContentEquals(content, fs.files[dest])
    }

    @Test
    fun sizeMismatchesFail() = runTest {
        val e = assertFailsWith<DownloadSizeMismatchException> { downloader().downloadTo(request(size = 999)) }
        assertEquals(999L, e.expectedBytes)
        assertEquals(1000L, e.actualBytes)
        assertNull(fs.files[part])
        // The partial file is longer than the expected size → discarded and restarted.
        fs.files[part] = ByteArray(2000)
        downloader().downloadTo(request())
        assertContentEquals(content, fs.files[dest])
    }

    @Test
    fun serverIgnoringRangeRestartsFromZero() = runTest {
        fs.files[part] = content.copyOf(400)
        server.script += FakeHttpServer.Behavior.IgnoreRange
        val completed = downloader().downloadTo(request())
        assertEquals(1, server.requests.size)
        assertContentEquals(content, fs.files[dest])
        assertEquals(400L, completed.resumedFromBytes)
    }

    @Test
    fun unknownSizeDownloadsUntilShortChunk() = runTest {
        repeat(20) { server.script += FakeHttpServer.Behavior.UnknownTotal }
        val events = downloader().download(request(sha256 = null, size = null)).toList()
        assertEquals(DownloadEvent.Started(0, null), events.first())
        assertNull(events.filterIsInstance<DownloadEvent.Progress>().first().fraction)
        assertContentEquals(content, fs.files[dest])
        assertEquals(sha, (events.last() as DownloadEvent.Completed).sha256)
    }

    @Test
    fun exactMultipleOfChunkWithUnknownSizeEndsOn416() = runTest {
        server.content = content.copyOf(512)
        repeat(4) { server.script += FakeHttpServer.Behavior.UnknownTotal }
        downloader().downloadTo(request(sha256 = null, size = null))
        assertEquals(512, fs.files[dest]!!.size)
        assertEquals("bytes=512-639", server.requests.last()["Range"])
    }

    @Test
    fun partialLongerThanRemoteRestarts() = runTest {
        server.content = content.copyOf(100)
        fs.files[part] = ByteArray(300)
        downloader().downloadTo(request(sha256 = null, size = null))
        assertContentEquals(content.copyOf(100), fs.files[dest])
    }

    @Test
    fun permanentHttpErrorsAndProtocolViolationsFailFast() = runTest {
        server.script += FakeHttpServer.Behavior.Status(404)
        val e = assertFailsWith<DownloadHttpException> { downloader().downloadTo(request()) }
        assertEquals(404, e.statusCode)
        assertEquals(1, server.requests.size)

        server.script += FakeHttpServer.Behavior.WrongOffset
        assertFailsWith<DownloadProtocolException> { downloader().downloadTo(request()) }
        server.script += FakeHttpServer.Behavior.NoContentRange
        assertFailsWith<DownloadProtocolException> { downloader().downloadTo(request()) }

    }

    @Test
    fun emptyRemoteFileDownloadsAsEmptyFile() = runTest {
        server.content = ByteArray(0)
        val completed = downloader().downloadTo(request(sha256 = null, size = null))
        assertEquals(0L, completed.sizeBytes)
        assertEquals(Sha256.hashHex(ByteArray(0)), completed.sha256)
        assertContentEquals(ByteArray(0), fs.files[dest])
    }

    @Test
    fun announcedSizeDifferentFromExpectedFails() = runTest {
        server.content = content.copyOf(900)
        assertFailsWith<DownloadSizeMismatchException> { downloader().downloadTo(request(sha256 = null)) }
        server.script += FakeHttpServer.Behavior.IgnoreRange
        assertFailsWith<DownloadSizeMismatchException> { downloader().downloadTo(request(sha256 = null)) }
    }

    @Test
    fun retriesAreBounded() = runTest {
        repeat(10) { server.script += FakeHttpServer.Behavior.Status(500) }
        assertFailsWith<DownloadTransportException> { downloader(retries = 2).downloadTo(request()) }
        assertEquals(3, server.requests.size)
    }

    @Test
    fun existingVerifiedDestinationIsReused() = runTest {
        fs.files[dest] = content.copyOf()
        val events = downloader().download(request()).toList()
        assertEquals(DownloadEvent.Completed(dest, 1000, sha, 0, alreadyPresent = true), events.last())
        assertTrue(server.requests.isEmpty())
        // Without an expected SHA any existing file is accepted.
        assertTrue(downloader().downloadTo(request(sha256 = null)).alreadyPresent)
        // A stale destination with the wrong content is downloaded again.
        fs.files[dest] = ByteArray(1000)
        assertFalse(downloader().downloadTo(request()).alreadyPresent)
        assertContentEquals(content, fs.files[dest])
    }

    @Test
    fun cancellationKeepsPartialFile() = runTest {
        val d = downloader()
        val progress = d.download(request()).filterIsInstance<DownloadEvent.Progress>().first { it.downloadedBytes >= 300 }
        assertTrue(progress.downloadedBytes >= 300)
        val kept = fs.files[part]!!.size
        assertTrue(kept >= 300)
        assertNull(fs.files[dest])
        val completed = d.downloadTo(request())
        assertEquals(kept.toLong(), completed.resumedFromBytes)
        assertContentEquals(content, fs.files[dest])
    }

    @Test
    fun requestValidation() {
        assertFailsWith<IllegalArgumentException> { DownloadRequest("", dest) }
        assertFailsWith<IllegalArgumentException> { DownloadRequest("u", " ") }
        assertFailsWith<IllegalArgumentException> { DownloadRequest("u", dest, expectedSha256 = "abc") }
        assertFailsWith<IllegalArgumentException> { DownloadRequest("u", dest, expectedSizeBytes = -1) }
        assertFailsWith<IllegalArgumentException> { DownloadRequest("u", dest, headers = mapOf("range" to "x")) }
        assertFailsWith<IllegalArgumentException> { ModelDownloader(FakeHttpServer(ByteArray(0)), FakeFileSystem(), chunkSizeBytes = 0) }
        assertFailsWith<IllegalArgumentException> { ModelDownloader(FakeHttpServer(ByteArray(0)), FakeFileSystem(), maxRetries = -1) }
        assertEquals("$dest.part", DownloadRequest("u", dest).partialPath)
    }

    @Test
    fun extraHeadersAreSent() = runTest {
        downloader().downloadTo(DownloadRequest("https://h/m", dest, headers = mapOf("Authorization" to "Bearer t")))
        assertTrue(server.requests.all { it["Authorization"] == "Bearer t" })
    }

    @Test
    fun contentRangeParsing() {
        assertEquals(ContentRange(0, 9, 100), ContentRange.parse("bytes 0-9/100"))
        assertEquals(ContentRange(5, 9, null), ContentRange.parse("BYTES 5-9/*"))
        assertEquals(ContentRange(null, null, 100), ContentRange.parse("bytes */100"))
        assertNull(ContentRange.parse("bytes 9-5/100"))
        assertNull(ContentRange.parse("items 0-1/2"))
        assertNull(ContentRange.parse(null))
    }
}
