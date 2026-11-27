package com.donfuldauer.kggml.samples.showcase.models

import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureLevel
import com.donfuldauer.kggml.core.MemoryPressureListener
import com.donfuldauer.kggml.core.MemoryPressureObserver
import com.donfuldauer.kggml.models.ChecksumMismatchException
import com.donfuldauer.kggml.models.DownloadEvent
import com.donfuldauer.kggml.models.DownloadHttpException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class ModelManagerControllerTest {
    private class Harness(
        scope: TestScope,
        val store: FakeModelStore = FakeModelStore(),
        val observer: ManualMemoryPressureObserver = ManualMemoryPressureObserver(),
        budget: Long = 10_000,
    ) {
        val residents = ResidentModels(budget, observer)
        val used = mutableListOf<Pair<String, String>>()
        val controller = ModelManagerController(
            scope = CoroutineScope(StandardTestDispatcher(scope.testScheduler) + Job()),
            store = store,
            residents = residents,
            ioDispatcher = StandardTestDispatcher(scope.testScheduler),
            catalog = TEST_CATALOG,
            onUseModel = { model, path -> used += model.id to path },
        )
        val state get() = controller.state.value
        fun row(id: String) = state.row(id)
    }

    private val llmPath = "/data/models/llm.gguf"

    @Test
    fun initialScanFindsInstalledAndPartialFilesAndInspectsHeaders() = runTest {
        val store = FakeModelStore().apply {
            files["llm.gguf"] = 1000
            files["notes.txt"] = 5
            partials["ggml-tiny.bin"] = 200
            metadata[llmPath] = TEST_METADATA
        }
        val h = Harness(this, store)
        assertTrue(h.state.loading)
        advanceUntilIdle()
        assertFalse(h.state.loading)
        assertEquals(DownloadPhase.INSTALLED, h.row("llm").phase)
        assertEquals(Verification.NOT_CHECKED, h.row("llm").verification)
        assertEquals(TEST_METADATA, h.row("llm").metadata)
        assertEquals(DownloadPhase.PARTIAL, h.row("whisper").phase)
        assertEquals(200, h.row("whisper").downloadedBytes)
        assertEquals(DownloadPhase.NOT_DOWNLOADED, h.row("sd").phase)
        assertEquals(listOf("notes.txt"), h.state.otherFiles.map { it.fileName })
        assertEquals(1005, h.state.diskBytes)
    }

    @Test
    fun downloadReportsProgressThenVerifiesAndInspects() = runTest {
        val h = Harness(this)
        h.store.metadata[llmPath] = TEST_METADATA
        advanceUntilIdle()
        h.controller.download("llm")
        assertEquals(DownloadPhase.DOWNLOADING, h.row("llm").phase)
        val script = h.store.script("llm.gguf")
        script.send(DownloadEvent.Started(0, 1000))
        script.send(DownloadEvent.Progress(250, 1000))
        advanceUntilIdle()
        assertEquals(250, h.row("llm").downloadedBytes)
        assertEquals(0.25f, h.row("llm").fraction)
        script.send(DownloadEvent.Verifying(1000))
        advanceUntilIdle()
        assertEquals(DownloadPhase.VERIFYING, h.row("llm").phase)
        script.send(DownloadEvent.Completed(llmPath, 1000, "a".repeat(64), resumedFromBytes = 0, alreadyPresent = false))
        advanceUntilIdle()
        val row = h.row("llm")
        assertEquals(DownloadPhase.INSTALLED, row.phase)
        assertEquals(Verification.VERIFIED, row.verification)
        assertEquals(llmPath, row.installedPath)
        assertEquals(TEST_METADATA, row.metadata)
        assertEquals(1000, h.state.diskBytes)
        assertContains(h.state.status!!, "Downloaded Test LLM")
    }

    @Test
    fun catalogEntryWithoutPublishedDigestIsReportedUnverified() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        h.controller.download("whisper")
        h.store.script("ggml-tiny.bin").send(DownloadEvent.Completed("/data/models/ggml-tiny.bin", 500, "b".repeat(64), 0, false))
        advanceUntilIdle()
        assertEquals(Verification.UNVERIFIED, h.row("whisper").verification)
        assertEquals("b".repeat(64), h.row("whisper").sha256)
        // The fake has no header for it: the row reports why instead of failing the install.
        assertContains(h.row("whisper").metadataError!!, "ModelFormatException")
    }

    @Test
    fun pausingKeepsThePartialAndResumeContinuesFromIt() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        h.controller.download("llm")
        val script = h.store.script("llm.gguf")
        script.send(DownloadEvent.Started(0, 1000))
        script.send(DownloadEvent.Progress(400, 1000))
        advanceUntilIdle()
        h.controller.cancelDownload("llm")
        advanceUntilIdle()
        assertEquals(DownloadPhase.PARTIAL, h.row("llm").phase)
        assertEquals(400, h.row("llm").downloadedBytes)
        assertContains(h.state.status!!, "Paused")
        assertTrue(h.row("llm").canDownload)

        h.controller.download("llm")
        script.send(DownloadEvent.Started(resumedFromBytes = 400, totalBytes = 1000))
        advanceUntilIdle()
        assertEquals(DownloadPhase.DOWNLOADING, h.row("llm").phase)
        assertEquals(400, h.row("llm").downloadedBytes)
        script.send(DownloadEvent.Completed(llmPath, 1000, "a".repeat(64), resumedFromBytes = 400, alreadyPresent = false))
        advanceUntilIdle()
        assertEquals(DownloadPhase.INSTALLED, h.row("llm").phase)
        assertContains(h.state.status!!, "resumed at")
        assertEquals(listOf("llm.gguf", "llm.gguf"), h.store.downloadsStarted)
    }

    @Test
    fun checksumMismatchFailsTheDownload() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        h.controller.download("llm")
        val script = h.store.script("llm.gguf")
        script.send(DownloadEvent.Progress(1000, 1000))
        script.send(ChecksumMismatchException("a".repeat(64), "c".repeat(64)))
        advanceUntilIdle()
        val row = h.row("llm")
        assertEquals(DownloadPhase.FAILED, row.phase)
        assertEquals(Verification.MISMATCH, row.verification)
        assertContains(row.error!!, "SHA-256 mismatch")
        assertTrue(row.canDownload, "a failed download can be retried")
    }

    @Test
    fun networkFailureAfterProgressLeavesAResumablePartial() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        h.controller.download("sd")
        val script = h.store.script("sd.safetensors")
        script.send(DownloadEvent.Progress(100, 2000))
        script.send(DownloadHttpException(503, "https://example.invalid/sd.safetensors"))
        advanceUntilIdle()
        assertEquals(DownloadPhase.PARTIAL, h.row("sd").phase)
        assertContains(h.row("sd").error!!, "DownloadHttpException")
    }

    @Test
    fun verifyRecomputesTheDigest() = runTest {
        val store = FakeModelStore().apply {
            files["llm.gguf"] = 1000
            files["ggml-tiny.bin"] = 500
            digests["llm.gguf"] = "a".repeat(64)
            digests["ggml-tiny.bin"] = "d".repeat(64)
        }
        val h = Harness(this, store)
        advanceUntilIdle()
        h.controller.verify("llm")
        assertTrue(h.row("llm").working)
        advanceUntilIdle()
        assertFalse(h.row("llm").working)
        assertEquals(Verification.VERIFIED, h.row("llm").verification)
        h.controller.verify("whisper")
        advanceUntilIdle()
        assertEquals(Verification.UNVERIFIED, h.row("whisper").verification)

        store.digests["llm.gguf"] = "f".repeat(64)
        h.controller.verify("llm")
        advanceUntilIdle()
        assertEquals(Verification.MISMATCH, h.row("llm").verification)
    }

    @Test
    fun deleteAndClearPartialsUpdateTheCatalog() = runTest {
        val store = FakeModelStore().apply {
            files["llm.gguf"] = 1000
            partials["sd.safetensors"] = 30
        }
        val h = Harness(this, store)
        advanceUntilIdle()
        h.controller.delete("llm")
        advanceUntilIdle()
        assertEquals(DownloadPhase.NOT_DOWNLOADED, h.row("llm").phase)
        assertTrue(store.files.isEmpty())
        h.controller.clearPartialDownloads()
        advanceUntilIdle()
        assertEquals("Removed 1 partial download(s)", h.state.status)
        assertEquals(DownloadPhase.NOT_DOWNLOADED, h.row("sd").phase)
    }

    @Test
    fun clearPartialsIsRefusedWhileDownloading() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        h.controller.download("llm")
        h.controller.clearPartialDownloads()
        assertContains(h.state.error!!, "Pause or finish")
    }

    @Test
    fun useHandsTheInstalledPathToTheCallback() = runTest {
        val h = Harness(this, FakeModelStore().apply { files["llm.gguf"] = 1000 })
        advanceUntilIdle()
        h.controller.use("whisper") // not installed: ignored
        h.controller.use("llm")
        assertEquals(listOf("llm" to llmPath), h.used)
        assertEquals("Selected Test LLM", h.state.status)
    }

    @Test
    fun inspectArbitraryPathShowsMetadataOrError() = runTest {
        val h = Harness(this)
        h.store.metadata["/tmp/x.gguf"] = TEST_METADATA
        advanceUntilIdle()
        h.controller.setInspectPath("/tmp/x.gguf")
        h.controller.inspectPath()
        assertTrue(h.state.inspecting)
        advanceUntilIdle()
        assertEquals(TEST_METADATA, h.state.inspected)
        h.controller.setInspectPath("/tmp/unknown.bin")
        h.controller.inspectPath()
        advanceUntilIdle()
        assertNull(h.state.inspected)
        assertContains(h.state.error!!, "ModelFormatException")
    }

    @Test
    fun manualEvictionAndSimulatedOsSignalCloseIdleEngines() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        val a = FakeResource("a")
        val b = FakeResource("b")
        h.residents.lease("a") { a to 3000L }.close()
        val leasedB = h.residents.lease("b") { b to 3000L }
        assertEquals(listOf("a", "b"), h.residents.state.value.entries.map { it.key })
        assertEquals(6000, h.residents.state.value.residentBytes)

        // MODERATE trims idle engines to half the budget (5000): closing "a" is enough.
        h.controller.evictResident(MemoryPressureLevel.MODERATE)
        assertTrue(a.closed)
        assertFalse(b.closed, "leased engines are never evicted")
        assertEquals("Evicted 1 idle engine(s) (MODERATE)", h.state.status)

        // A simulated OS signal goes through the observer, exactly like onTrimMemory / a memory warning.
        h.controller.simulateLowMemory(MemoryPressureLevel.CRITICAL)
        assertFalse(b.closed)
        leasedB.close()
        h.controller.simulateLowMemory(MemoryPressureLevel.CRITICAL)
        assertTrue(b.closed)
        assertTrue(h.residents.state.value.entries.isEmpty())
        assertContains(h.state.status!!, "Simulated CRITICAL memory pressure")
        val latest = h.residents.state.value.evictions.first()
        assertEquals(EvictionSource.OS, latest.source)
        assertEquals(listOf("b"), latest.evictedKeys)
        assertEquals(MemoryPressureLevel.CRITICAL, h.residents.state.value.lastLevel)
    }

    @Test
    fun realOsSignalFromTheObserverEvictsWithoutTheController() = runTest {
        val observer = ManualMemoryPressureObserver()
        val h = Harness(this, observer = observer)
        val engine = FakeResource("llm")
        h.residents.lease("llm") { engine to 100L }.close()
        observer.emit(MemoryPressureLevel.CRITICAL)
        assertTrue(engine.closed)
        assertNotNull(h.residents.state.value.evictions.firstOrNull { it.evictedKeys == listOf("llm") })
        h.residents.close()
    }

    @Test
    fun unreadableModelDirectoryIsReported() = runTest {
        val store = FakeModelStore().apply { failures["files"] = IllegalStateException("permission denied") }
        val h = Harness(this, store)
        advanceUntilIdle()
        assertFalse(h.state.loading)
        assertEquals("Cannot read /data/models: IllegalStateException: permission denied", h.state.error)
        assertTrue(h.state.rows.all { it.phase == DownloadPhase.NOT_DOWNLOADED })
    }

    @Test
    fun failedHashingDeletionAndPartialCleanupAreReported() = runTest {
        val store = FakeModelStore().apply {
            files["llm.gguf"] = 1000
            partials["sd.safetensors"] = 30
        }
        val h = Harness(this, store)
        advanceUntilIdle()

        store.failures["sha256"] = IllegalStateException("I/O error")
        h.controller.verify("llm")
        advanceUntilIdle()
        assertEquals("IllegalStateException: I/O error", h.row("llm").error)
        assertFalse(h.row("llm").working)
        assertEquals(Verification.NOT_CHECKED, h.row("llm").verification)

        store.failures["delete"] = IllegalStateException("read-only file system")
        h.controller.delete("llm")
        advanceUntilIdle()
        assertEquals("IllegalStateException: read-only file system", h.state.error)
        assertEquals(DownloadPhase.INSTALLED, h.row("llm").phase, "the model is still installed")

        store.failures["clearPartialDownloads"] = IllegalStateException("busy")
        h.controller.clearPartialDownloads()
        advanceUntilIdle()
        assertEquals("IllegalStateException: busy", h.state.error)
        assertEquals(DownloadPhase.PARTIAL, h.row("sd").phase)
    }

    @Test
    fun refreshKeepsRunningDownloadsVerificationAndFailures() = runTest {
        val store = FakeModelStore().apply {
            files["ggml-tiny.bin"] = 500
            digests["ggml-tiny.bin"] = "d".repeat(64)
        }
        val h = Harness(this, store)
        advanceUntilIdle()
        h.controller.verify("whisper")
        advanceUntilIdle()
        assertEquals(Verification.UNVERIFIED, h.row("whisper").verification)

        h.controller.download("llm")
        h.store.script("llm.gguf").send(DownloadEvent.Progress(300, 1000))
        h.controller.download("sd")
        h.store.script("sd.safetensors").send(DownloadHttpException(404, "https://example.invalid/sd.safetensors"))
        advanceUntilIdle()
        assertEquals(DownloadPhase.FAILED, h.row("sd").phase, "nothing was downloaded, so nothing can be resumed")
        assertContains(h.row("sd").error!!, "DownloadHttpException")

        h.controller.refresh()
        advanceUntilIdle()
        assertEquals(DownloadPhase.DOWNLOADING, h.row("llm").phase, "a running download is not reset by a re-scan")
        assertEquals(300, h.row("llm").downloadedBytes)
        assertEquals(Verification.UNVERIFIED, h.row("whisper").verification, "a digest checked this session is kept")
        assertEquals(DownloadPhase.FAILED, h.row("sd").phase)
        assertContains(h.row("sd").error!!, "DownloadHttpException")
    }

    @Test
    fun downloadRequestsThatCannotStartAreIgnored() = runTest {
        val h = Harness(this, FakeModelStore().apply { files["llm.gguf"] = 1000 })
        advanceUntilIdle()
        h.controller.download("llm") // already installed
        h.controller.download("whisper")
        h.controller.download("whisper") // already running
        h.controller.cancelDownload("sd") // not running: no-op
        advanceUntilIdle()
        assertEquals(listOf("ggml-tiny.bin"), h.store.downloadsStarted)
        assertEquals(DownloadPhase.NOT_DOWNLOADED, h.row("sd").phase)

        h.controller.close()
        advanceUntilIdle()
        assertEquals(DownloadPhase.PARTIAL, h.row("whisper").phase, "close pauses running downloads")
        h.controller.close()
        h.controller.download("sd")
        h.controller.refresh()
        advanceUntilIdle()
        assertEquals(listOf("ggml-tiny.bin"), h.store.downloadsStarted, "a closed controller starts no download")
    }

    @Test
    fun actionsOnRowsInTheWrongStateAreIgnored() = runTest {
        val h = Harness(this)
        h.store.metadata[llmPath] = TEST_METADATA
        advanceUntilIdle()
        h.controller.verify("llm") // not installed
        h.controller.inspect("llm") // not installed
        h.controller.inspectPath() // no path
        advanceUntilIdle()
        assertNull(h.row("llm").metadata)
        assertNull(h.row("llm").verification)
        assertNull(h.state.inspected)
        assertFalse(h.state.inspecting)

        h.controller.download("llm")
        h.controller.delete("llm") // transferring
        advanceUntilIdle()
        assertEquals(DownloadPhase.DOWNLOADING, h.row("llm").phase)
        assertNull(h.state.status)
    }

    @Test
    fun progressWithoutATotalKeepsTheAnnouncedSize() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        h.controller.download("llm")
        val script = h.store.script("llm.gguf")
        script.send(DownloadEvent.Started(0, 1000))
        script.send(DownloadEvent.Progress(500, null))
        advanceUntilIdle()
        assertEquals(1000, h.row("llm").totalBytes)
        assertEquals(0.5f, h.row("llm").fraction)
    }

    @Test
    fun alreadyInstalledDownloadIsReportedAsSuch() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        h.controller.download("llm")
        h.store.script("llm.gguf").send(DownloadEvent.Completed(llmPath, 1000, "a".repeat(64), resumedFromBytes = 0, alreadyPresent = true))
        advanceUntilIdle()
        assertEquals("Test LLM was already installed", h.state.status)
        assertEquals(DownloadPhase.INSTALLED, h.row("llm").phase)
    }

    @Test
    fun diskUsageIsKeptWhenTheRescanAfterADownloadFails() = runTest {
        val h = Harness(this, FakeModelStore().apply { files["notes.txt"] = 5 })
        advanceUntilIdle()
        assertEquals(5, h.state.diskBytes)
        h.controller.download("llm")
        h.store.failures["files"] = IllegalStateException("gone")
        h.store.script("llm.gguf").send(DownloadEvent.Completed(llmPath, 1000, "a".repeat(64), 0, false))
        advanceUntilIdle()
        assertEquals(DownloadPhase.INSTALLED, h.row("llm").phase)
        assertEquals(5, h.state.diskBytes, "the last known usage is kept")
        assertNull(h.state.error)
    }

    @Test
    fun defaultLowMemorySimulationIsCriticalAndNeedsADrivableObserver() = runTest {
        val h = Harness(this)
        advanceUntilIdle()
        val engine = FakeResource("llm")
        h.residents.lease("llm") { engine to 100L }.close()
        h.controller.simulateLowMemory()
        assertTrue(engine.closed, "CRITICAL closes every idle engine")
        assertEquals("Simulated CRITICAL memory pressure: released 100 B", h.state.status)

        // A platform observer that only the OS can drive (not a ManualMemoryPressureObserver).
        val osOnly = object : MemoryPressureObserver {
            override val currentLevel = MemoryPressureLevel.NORMAL
            override fun addListener(listener: MemoryPressureListener) = AutoCloseable {}
        }
        val controller = ModelManagerController(
            scope = CoroutineScope(StandardTestDispatcher(testScheduler) + Job()),
            store = FakeModelStore(),
            residents = ResidentModels(10_000, osOnly),
            ioDispatcher = StandardTestDispatcher(testScheduler),
        )
        controller.simulateLowMemory()
        assertEquals("This platform's memory observer cannot be driven manually", controller.state.value.status)
    }

    @Test
    fun defaultCatalogAndCallbackAreUsedWithoutArguments() = runTest {
        val store = FakeModelStore()
        val controller = ModelManagerController(
            scope = CoroutineScope(StandardTestDispatcher(testScheduler) + Job()),
            store = store,
            residents = ResidentModels(10_000, ManualMemoryPressureObserver()),
            ioDispatcher = StandardTestDispatcher(testScheduler),
        )
        assertEquals(DEFAULT_MODEL_CATALOG.map { it.id }, controller.state.value.rows.map { it.model.id })
        val first = DEFAULT_MODEL_CATALOG.first()
        store.files[first.fileName] = 10
        advanceUntilIdle()
        controller.refresh()
        advanceUntilIdle()
        controller.use(first.id) // the default callback does nothing; the selection is still reported
        assertEquals("Selected ${first.displayName}", controller.state.value.status)
    }

    @Test
    fun residentLeaseReusesTheLoadedEngine() = runTest {
        val h = Harness(this)
        var loads = 0
        repeat(3) { h.residents.lease("k") { loads++; FakeResource("k") to 10L }.close() }
        assertEquals(1, loads)
        assertTrue(h.residents.remove("k"))
        h.residents.lease("k") { loads++; FakeResource("k") to 10L }.close()
        assertEquals(2, loads)
    }
}
