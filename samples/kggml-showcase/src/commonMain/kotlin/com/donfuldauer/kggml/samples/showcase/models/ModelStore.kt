package com.donfuldauer.kggml.samples.showcase.models

import com.donfuldauer.kggml.models.DownloadEvent
import com.donfuldauer.kggml.models.ModelCache
import com.donfuldauer.kggml.models.ModelDownloader
import com.donfuldauer.kggml.models.ModelFileSystem
import com.donfuldauer.kggml.models.ModelHeaderInfo
import com.donfuldauer.kggml.models.ModelInspector
import com.donfuldauer.kggml.models.PeakRamEstimate
import com.donfuldauer.kggml.models.Sha256
import com.donfuldauer.kggml.models.defaultModelFileSystem
import kotlinx.coroutines.flow.Flow

/** Pre-load peak RAM estimate of a model (see `PeakRamEstimate`). */
data class RamBreakdown(val weightsBytes: Long, val kvCacheBytes: Long, val computeBytes: Long, val overheadBytes: Long) {
    val totalBytes: Long get() = weightsBytes + kvCacheBytes + computeBytes + overheadBytes
}

/** Header metadata of a model file, read without loading its weights. */
data class ModelMetadata(
    /** `GGUF` or `Whisper ggml`. */
    val format: String,
    /** `general.name` / Whisper model size, when present. */
    val name: String?,
    /** `general.architecture` (GGUF) or `whisper`. */
    val architecture: String?,
    val parameterCount: Long,
    val quantization: String?,
    val contextLength: Long?,
    val estimatedPeakRam: RamBreakdown,
)

/** A complete file in the model directory. */
data class StoredFile(val fileName: String, val path: String, val sizeBytes: Long)

/**
 * The Model Manager's view of on-device storage: downloads, header inspection, hashing and deletion. Every method
 * except [download] blocks on file I/O and must run off the UI thread. [DiskModelStore] in the app, a fake in tests.
 */
interface ModelStore {
    /** Directory the models live in. */
    val directory: String

    /** Complete model files (no `.part` downloads), by name. */
    fun files(): List<StoredFile>

    /** Bytes of an interrupted download of [fileName] (its `.part` file), `0` if there is none. */
    fun partialBytes(fileName: String): Long

    /** Absolute path of [fileName] inside [directory]. */
    fun pathFor(fileName: String): String

    /** Resumable, verified download of [model] into [directory] (cold flow; cancelling keeps the `.part` file). */
    fun download(model: CatalogModel): Flow<DownloadEvent>

    /** Parses the GGUF / Whisper header at [path] (any path, not only inside [directory]). */
    fun inspect(path: String): ModelMetadata

    /** SHA-256 (lowercase hex) of [fileName]. */
    fun sha256(fileName: String): String

    /** Deletes [fileName]; `false` if it did not exist. */
    fun delete(fileName: String): Boolean

    /** Deletes abandoned `.part` files and returns their names. */
    fun clearPartialDownloads(): List<String>
}

/** [ModelStore] over `:kggml-model-manager`'s [ModelCache], [ModelDownloader] and [ModelInspector]. */
class DiskModelStore(
    override val directory: String,
    private val fileSystem: ModelFileSystem = defaultModelFileSystem(),
    private val downloader: ModelDownloader = ModelDownloader(fileSystem = fileSystem),
) : ModelStore {
    private val cache = ModelCache(directory, fileSystem)

    override fun files(): List<StoredFile> =
        cache.entries().map { StoredFile(it.fileName, it.path, it.sizeBytes) }.sortedBy { it.fileName }

    // Not cache.pathFor(): that creates the directory, and a scan must not have side effects.
    override fun partialBytes(fileName: String): Long =
        fileSystem.size(join(fileName) + PARTIAL_SUFFIX).coerceAtLeast(0)

    override fun pathFor(fileName: String): String = cache.pathFor(fileName)

    override fun download(model: CatalogModel): Flow<DownloadEvent> =
        downloader.download(cache.downloadRequest(url = model.url, fileName = model.fileName, expectedSha256 = model.sha256))

    override fun inspect(path: String): ModelMetadata = ModelInspector.inspectFile(path, fileSystem).toMetadata()

    override fun sha256(fileName: String): String = Sha256.hashFileHex(fileSystem, cache.pathFor(fileName))

    override fun delete(fileName: String): Boolean = cache.remove(fileName)

    override fun clearPartialDownloads(): List<String> = cache.clearPartialDownloads()

    private fun join(fileName: String): String =
        if (directory.endsWith("/") || directory.endsWith("\\")) directory + fileName else "$directory/$fileName"

    private companion object {
        const val PARTIAL_SUFFIX = ".part"
    }
}

/** Maps the model manager's header summary to the screen's [ModelMetadata]. */
fun ModelHeaderInfo.toMetadata(): ModelMetadata = when (this) {
    is ModelHeaderInfo.Gguf -> ModelMetadata(
        format = "GGUF v${header.version}",
        name = header.name,
        architecture = header.architecture,
        parameterCount = parameterCount,
        quantization = quantization,
        contextLength = contextLength,
        estimatedPeakRam = estimatedPeakRam.toBreakdown(),
    )

    is ModelHeaderInfo.Whisper -> ModelMetadata(
        format = "Whisper ggml",
        name = header.modelType?.let { "whisper-$it${if (header.isMultilingual) "" else ".en"}" },
        architecture = "whisper",
        parameterCount = parameterCount,
        quantization = quantization,
        contextLength = contextLength,
        estimatedPeakRam = estimatedPeakRam.toBreakdown(),
    )
}

private fun PeakRamEstimate.toBreakdown() = RamBreakdown(weightsBytes, kvCacheBytes, computeBytes, overheadBytes)
