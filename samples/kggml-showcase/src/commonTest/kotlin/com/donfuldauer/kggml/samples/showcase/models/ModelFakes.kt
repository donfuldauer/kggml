package com.donfuldauer.kggml.samples.showcase.models

import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.models.DownloadEvent
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/** Two-entry test catalog: a GGUF LLM with a published SHA-256 and a Whisper model without one. */
val TEST_CATALOG: List<CatalogModel> = listOf(
    CatalogModel(
        id = "llm",
        displayName = "Test LLM",
        kind = ModelKind.LLM,
        url = "https://example.invalid/llm.gguf",
        fileName = "llm.gguf",
        approxSizeBytes = 1000,
        license = "MIT",
        sha256 = "a".repeat(64),
    ),
    CatalogModel(
        id = "whisper",
        displayName = "Test Whisper",
        kind = ModelKind.WHISPER,
        url = "https://example.invalid/ggml-tiny.bin",
        fileName = "ggml-tiny.bin",
        approxSizeBytes = 500,
        license = "MIT",
    ),
    CatalogModel(
        id = "sd",
        displayName = "Test SD",
        kind = ModelKind.DIFFUSION,
        url = "https://example.invalid/sd.safetensors",
        fileName = "sd.safetensors",
        approxSizeBytes = 2000,
        license = "MIT",
    ),
)

val TEST_METADATA = ModelMetadata(
    format = "GGUF v3",
    name = "Tiny Test LM",
    architecture = "llama",
    parameterCount = 494_000_000,
    quantization = "Q4_K_M",
    contextLength = 32_768,
    estimatedPeakRam = RamBreakdown(weightsBytes = 400L shl 20, kvCacheBytes = 48L shl 20, computeBytes = 300L shl 20, overheadBytes = 64L shl 20),
)

/**
 * In-memory [ModelStore]. Downloads are scripted per file name: the test sends [DownloadEvent]s (or a [Throwable]
 * to fail) through [script]; a [DownloadEvent.Completed] installs the file. Progress events are mirrored into the
 * `.part` size so a later download "resumes". [failures] makes an operation (`files`, `sha256`, `delete`,
 * `clearPartialDownloads`) throw, like an unreadable or read-only model directory.
 */
class FakeModelStore(override val directory: String = "/data/models") : ModelStore {
    val files = LinkedHashMap<String, Long>()
    val partials = HashMap<String, Long>()
    val metadata = HashMap<String, ModelMetadata>()
    val digests = HashMap<String, String>()
    val downloadsStarted = mutableListOf<String>()
    val failures = HashMap<String, Throwable>()
    private val scripts = HashMap<String, Channel<Any>>()

    fun script(fileName: String): Channel<Any> = scripts.getOrPut(fileName) { Channel(Channel.UNLIMITED) }

    override fun files(): List<StoredFile> = failures["files"]?.let { throw it } ?: files.map { (name, size) -> StoredFile(name, pathFor(name), size) }

    override fun partialBytes(fileName: String): Long = partials[fileName] ?: 0L

    override fun pathFor(fileName: String): String = "$directory/$fileName"

    override fun download(model: CatalogModel): Flow<DownloadEvent> = flow {
        downloadsStarted += model.fileName
        val channel = script(model.fileName)
        while (true) {
            when (val item = channel.receive()) {
                is Throwable -> {
                    if (item is com.donfuldauer.kggml.models.ChecksumMismatchException) partials.remove(model.fileName)
                    throw item
                }
                is DownloadEvent.Progress -> {
                    partials[model.fileName] = item.downloadedBytes
                    emit(item)
                }
                is DownloadEvent.Completed -> {
                    partials.remove(model.fileName)
                    files[model.fileName] = item.sizeBytes
                    emit(item)
                    return@flow
                }
                is DownloadEvent -> emit(item)
                else -> error("unexpected script item $item")
            }
        }
    }

    override fun inspect(path: String): ModelMetadata =
        metadata[path] ?: throw com.donfuldauer.kggml.models.ModelFormatException("unknown model format (magic 0x0)")

    override fun sha256(fileName: String): String = failures["sha256"]?.let { throw it } ?: digests[fileName] ?: error("no file $fileName")

    override fun delete(fileName: String): Boolean = failures["delete"]?.let { throw it } ?: (files.remove(fileName) != null)

    override fun clearPartialDownloads(): List<String> = failures["clearPartialDownloads"]?.let { throw it } ?: partials.keys.map { "$it.part" }.also { partials.clear() }
}

/** A [ResidentModels] driven by a manual observer (the test "OS"). */
fun testResidents(budgetBytes: Long = 10_000, observer: ManualMemoryPressureObserver = ManualMemoryPressureObserver()) =
    ResidentModels(budgetBytes, observer)

/** An [AutoCloseable] that records whether it was closed. */
class FakeResource(val name: String) : AutoCloseable {
    var closed = false
        private set

    override fun close() {
        closed = true
    }
}
