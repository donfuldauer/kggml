package com.donfuldauer.kggml.samples.showcase.models

import com.donfuldauer.kggml.core.MemoryPressureLevel
import com.donfuldauer.kggml.models.ChecksumMismatchException
import com.donfuldauer.kggml.models.DownloadEvent
import com.donfuldauer.kggml.samples.showcase.describe
import com.donfuldauer.kggml.samples.showcase.formatBytes
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Download / install state of a catalog model. */
enum class DownloadPhase(val label: String) {
    NOT_DOWNLOADED("Not downloaded"),
    PARTIAL("Paused (resumable)"),
    DOWNLOADING("Downloading"),
    VERIFYING("Verifying SHA-256…"),
    INSTALLED("Installed"),
    FAILED("Failed"),
}

/** Integrity status of an installed (or failed) model. */
enum class Verification(val label: String) {
    /** The file matches the catalog's published SHA-256. */
    VERIFIED("SHA-256 verified"),

    /** The catalog has no published SHA-256; the digest was computed and is shown, but not compared. */
    UNVERIFIED("Unverified (no published SHA-256)"),

    /** Installed before this session and not hashed yet. */
    NOT_CHECKED("Not checked this session"),

    /** The digest did not match: the file was rejected (download) or is corrupt (re-check). */
    MISMATCH("SHA-256 mismatch"),
}

/** One catalog entry with its on-device state. */
data class CatalogRow(
    val model: CatalogModel,
    val phase: DownloadPhase = DownloadPhase.NOT_DOWNLOADED,
    val downloadedBytes: Long = 0,
    val totalBytes: Long? = null,
    val installedPath: String? = null,
    val sha256: String? = null,
    val verification: Verification? = null,
    val metadata: ModelMetadata? = null,
    val metadataError: String? = null,
    /** Hashing or inspecting in the background. */
    val working: Boolean = false,
    val error: String? = null,
) {
    /** Completed fraction of a running / paused download, `null` if the size is unknown. */
    val fraction: Float? get() = totalBytes?.takeIf { it > 0 }?.let { (downloadedBytes.toDouble() / it).toFloat().coerceIn(0f, 1f) }
    val isTransferring: Boolean get() = phase == DownloadPhase.DOWNLOADING || phase == DownloadPhase.VERIFYING
    val canDownload: Boolean get() = !isTransferring && phase != DownloadPhase.INSTALLED
    val isInstalled: Boolean get() = phase == DownloadPhase.INSTALLED
    val tag: String get() = modelRowTag(model.id)
}

/** Test tag of the catalog row of [id]; its buttons use `"$tag-<action>"`. */
fun modelRowTag(id: String): String = "model-$id"

data class ModelManagerState(
    val directory: String,
    val rows: List<CatalogRow>,
    /** Files in the model directory that are not in the catalog. */
    val otherFiles: List<StoredFile> = emptyList(),
    val diskBytes: Long = 0,
    val loading: Boolean = true,
    val inspectPath: String = "",
    val inspected: ModelMetadata? = null,
    val inspecting: Boolean = false,
    val status: String? = null,
    val error: String? = null,
) {
    fun row(id: String): CatalogRow = rows.first { it.model.id == id }
}

/**
 * State holder of the Model Manager screen: catalog downloads with progress / resume / SHA-256 verification, header
 * inspection with a peak-RAM estimate, deletion, and resident-engine eviction. All public methods must be called
 * on [scope]'s thread (the UI thread); file I/O runs on [ioDispatcher].
 *
 * @param onUseModel hands an installed model to the screen that runs it.
 */
class ModelManagerController(
    private val scope: CoroutineScope,
    private val store: ModelStore,
    val residents: ResidentModels,
    private val ioDispatcher: CoroutineDispatcher,
    catalog: List<CatalogModel> = DEFAULT_MODEL_CATALOG,
    private val onUseModel: (CatalogModel, String) -> Unit = { _, _ -> },
) : AutoCloseable {
    private val mutableState = MutableStateFlow(ModelManagerState(store.directory, catalog.map { CatalogRow(it) }))
    val state: StateFlow<ModelManagerState> = mutableState.asStateFlow()

    private val downloads = HashMap<String, Job>()
    private var closed = false

    init {
        refresh()
    }

    /** Re-scans the model directory (installed files, `.part` files) and inspects installed models. */
    fun refresh() {
        if (closed) return
        mutableState.update { it.copy(loading = true) }
        scope.launch {
            try {
                val scan = withContext(ioDispatcher) {
                    val files = store.files()
                    val partials = mutableState.value.rows.associate { it.model.id to store.partialBytes(it.model.fileName) }
                    files to partials
                }
                val (files, partials) = scan
                val byName = files.associateBy { it.fileName }
                mutableState.update { state ->
                    state.copy(
                        loading = false,
                        diskBytes = files.sumOf { it.sizeBytes },
                        otherFiles = files.filter { f -> state.rows.none { it.model.fileName == f.fileName } },
                        rows = state.rows.map { row ->
                            if (row.isTransferring) return@map row
                            val file = byName[row.model.fileName]
                            val partial = partials[row.model.id] ?: 0L
                            when {
                                file != null -> row.copy(
                                    phase = DownloadPhase.INSTALLED,
                                    installedPath = file.path,
                                    downloadedBytes = file.sizeBytes,
                                    totalBytes = file.sizeBytes,
                                    verification = row.verification ?: Verification.NOT_CHECKED,
                                )

                                partial > 0 -> row.copy(
                                    phase = DownloadPhase.PARTIAL,
                                    installedPath = null,
                                    downloadedBytes = partial,
                                    totalBytes = row.totalBytes,
                                    metadata = null,
                                )

                                row.phase == DownloadPhase.FAILED -> row.copy(installedPath = null, downloadedBytes = 0)
                                else -> CatalogRow(row.model)
                            }
                        },
                    )
                }
                mutableState.value.rows.filter { it.isInstalled && it.metadata == null && it.metadataError == null }
                    .forEach { inspect(it.model.id) }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(loading = false, error = "Cannot read ${store.directory}: ${t.describe()}") }
            }
        }
    }

    /** Starts (or resumes, from the `.part` file) the download of catalog entry [id]. */
    fun download(id: String) {
        val row = mutableState.value.row(id)
        if (closed || !row.canDownload || downloads.containsKey(id)) return
        val model = row.model
        updateRow(id) {
            it.copy(phase = DownloadPhase.DOWNLOADING, error = null, verification = null, sha256 = null, metadata = null, metadataError = null)
        }
        mutableState.update { it.copy(error = null, status = null) }
        downloads[id] = scope.launch {
            try {
                store.download(model).collect { event -> onDownloadEvent(id, event) }
            } catch (e: CancellationException) {
                updateRow(id) { if (it.isTransferring) it.copy(phase = DownloadPhase.PARTIAL) else it }
                mutableState.update { it.copy(status = "Paused ${model.displayName}; the .part file is kept for resume") }
                throw e
            } catch (e: ChecksumMismatchException) {
                updateRow(id) {
                    it.copy(
                        phase = DownloadPhase.FAILED,
                        verification = Verification.MISMATCH,
                        sha256 = e.actualSha256,
                        downloadedBytes = 0,
                        error = "SHA-256 mismatch: expected ${e.expectedSha256.take(12)}…, got ${e.actualSha256.take(12)}… " +
                            "(partial file deleted)",
                    )
                }
            } catch (t: Throwable) {
                updateRow(id) { it.copy(phase = if (it.downloadedBytes > 0) DownloadPhase.PARTIAL else DownloadPhase.FAILED, error = t.describe()) }
            } finally {
                downloads.remove(id)
            }
        }
    }

    /** Pauses a running download; the `.part` file stays, so [download] resumes it. */
    fun cancelDownload(id: String) {
        downloads[id]?.cancel()
    }

    /** Re-computes the SHA-256 of an installed model and compares it to the catalog digest (if any). */
    fun verify(id: String) {
        val row = mutableState.value.row(id)
        if (!row.isInstalled || row.working) return
        updateRow(id) { it.copy(working = true, error = null) }
        scope.launch {
            try {
                val digest = withContext(ioDispatcher) { store.sha256(row.model.fileName) }
                val expected = row.model.sha256?.lowercase()
                updateRow(id) {
                    it.copy(
                        sha256 = digest,
                        verification = when {
                            expected == null -> Verification.UNVERIFIED
                            expected == digest -> Verification.VERIFIED
                            else -> Verification.MISMATCH
                        },
                    )
                }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                updateRow(id) { it.copy(error = t.describe()) }
            } finally {
                updateRow(id) { it.copy(working = false) }
            }
        }
    }

    /** Reads the header metadata and peak-RAM estimate of an installed catalog model. */
    fun inspect(id: String) {
        val row = mutableState.value.row(id)
        val path = row.installedPath ?: return
        scope.launch {
            val result = withContext(ioDispatcher) { runCatching { store.inspect(path) } }
            result.exceptionOrNull()?.let { if (it is CancellationException) throw it }
            updateRow(id) {
                it.copy(
                    metadata = result.getOrNull(),
                    metadataError = result.exceptionOrNull()?.let { e -> "Header not inspectable: ${e.describe()}" },
                )
            }
        }
    }

    /** Deletes an installed catalog model (and any `.part` file of it). */
    fun delete(id: String) {
        val row = mutableState.value.row(id)
        if (row.isTransferring) return
        scope.launch {
            try {
                withContext(ioDispatcher) { store.delete(row.model.fileName) }
                updateRow(id) { CatalogRow(it.model) }
                mutableState.update { it.copy(status = "Deleted ${row.model.fileName}") }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = t.describe()) }
            }
            refresh()
        }
    }

    /** Deletes every abandoned `.part` file (downloads that are not running). */
    fun clearPartialDownloads() {
        if (downloads.isNotEmpty()) {
            mutableState.update { it.copy(error = "Pause or finish running downloads first") }
            return
        }
        scope.launch {
            try {
                val removed = withContext(ioDispatcher) { store.clearPartialDownloads() }
                mutableState.update { it.copy(status = "Removed ${removed.size} partial download(s)", error = null) }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = t.describe()) }
            }
            refresh()
        }
    }

    /** Hands the installed catalog model [id] to the screen that runs it. */
    fun use(id: String) {
        val row = mutableState.value.row(id)
        val path = row.installedPath ?: return
        onUseModel(row.model, path)
        mutableState.update { it.copy(status = "Selected ${row.model.displayName}") }
    }

    fun setInspectPath(path: String) = mutableState.update { it.copy(inspectPath = path) }

    /** Inspects any model file by path (GGUF or Whisper ggml). */
    fun inspectPath() {
        val path = mutableState.value.inspectPath.trim()
        if (path.isEmpty() || mutableState.value.inspecting) return
        mutableState.update { it.copy(inspecting = true, inspected = null, error = null) }
        scope.launch {
            try {
                val metadata = withContext(ioDispatcher) { store.inspect(path) }
                mutableState.update { it.copy(inspected = metadata) }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = t.describe()) }
            } finally {
                mutableState.update { it.copy(inspecting = false) }
            }
        }
    }

    /** Evicts idle resident engines now (`CRITICAL` closes all of them, `MODERATE` trims to half the budget). */
    fun evictResident(level: MemoryPressureLevel) {
        val evicted = residents.evict(level)
        mutableState.update { it.copy(status = "Evicted ${evicted.size} idle engine(s) (${level.name})") }
    }

    /** Sends a synthetic OS low-memory signal through the platform observer. */
    fun simulateLowMemory(level: MemoryPressureLevel = MemoryPressureLevel.CRITICAL) {
        val before = residents.state.value.residentBytes
        val delivered = residents.simulateOsSignal(level)
        val after = residents.state.value.residentBytes
        mutableState.update {
            it.copy(
                status = if (delivered) "Simulated ${level.name} memory pressure: released ${formatBytes(before - after)}"
                else "This platform's memory observer cannot be driven manually",
            )
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        downloads.values.toList().forEach { it.cancel() }
    }

    private fun onDownloadEvent(id: String, event: DownloadEvent) {
        when (event) {
            is DownloadEvent.Started -> updateRow(id) {
                it.copy(phase = DownloadPhase.DOWNLOADING, downloadedBytes = event.resumedFromBytes, totalBytes = event.totalBytes)
            }

            is DownloadEvent.Progress -> updateRow(id) {
                it.copy(downloadedBytes = event.downloadedBytes, totalBytes = event.totalBytes ?: it.totalBytes)
            }

            is DownloadEvent.Verifying -> updateRow(id) { it.copy(phase = DownloadPhase.VERIFYING, downloadedBytes = event.sizeBytes) }

            is DownloadEvent.Completed -> {
                val model = mutableState.value.row(id).model
                updateRow(id) {
                    it.copy(
                        phase = DownloadPhase.INSTALLED,
                        installedPath = event.path,
                        downloadedBytes = event.sizeBytes,
                        totalBytes = event.sizeBytes,
                        sha256 = event.sha256,
                        verification = if (model.sha256 != null) Verification.VERIFIED else Verification.UNVERIFIED,
                    )
                }
                mutableState.update {
                    it.copy(
                        status = if (event.alreadyPresent) "${model.displayName} was already installed"
                        else "Downloaded ${model.displayName} (${formatBytes(event.sizeBytes)}" +
                            (if (event.resumedFromBytes > 0) ", resumed at ${formatBytes(event.resumedFromBytes)})" else ")"),
                    )
                }
                inspect(id)
                refreshDiskUsage()
            }
        }
    }

    private fun refreshDiskUsage() {
        scope.launch {
            val files = runCatching { withContext(ioDispatcher) { store.files() } }.getOrNull() ?: return@launch
            mutableState.update { it.copy(diskBytes = files.sumOf { f -> f.sizeBytes }) }
        }
    }

    private fun updateRow(id: String, transform: (CatalogRow) -> CatalogRow) = mutableState.update { state ->
        state.copy(rows = state.rows.map { if (it.model.id == id) transform(it) else it })
    }
}
