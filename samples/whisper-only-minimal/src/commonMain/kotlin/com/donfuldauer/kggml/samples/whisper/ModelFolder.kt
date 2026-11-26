package com.donfuldauer.kggml.samples.whisper

/** A whisper model file already in the sample's model folder (the app's files directory / Documents). */
data class FolderModel(val fileName: String, val path: String)

/**
 * The whisper models among [fileNames], the entries of [directory]: `.bin` files (whisper.cpp ggml models), sorted by
 * name ignoring case. Hidden files and everything else (audio, notes) are left out.
 */
fun whisperModelsIn(directory: String, fileNames: List<String>): List<FolderModel> =
    fileNames
        .filter { !it.startsWith(".") && it.lowercase().endsWith(MODEL_EXTENSION) && it.length > MODEL_EXTENSION.length }
        .sortedBy { it.lowercase() }
        .map { FolderModel(it, joinPath(directory, it)) }

/** [directory] + [name], joined with the directory's own separator (`\\` for a Windows path without `/`). */
internal fun joinPath(directory: String, name: String): String {
    val separator = if ('\\' in directory && '/' !in directory) '\\' else '/'
    return directory.trimEnd('/', '\\') + separator + name
}

/** Whether [a] and [b] name the same path, whatever separators they use (`C:\\m\\a.bin` == `C:/m/a.bin`). */
internal fun samePath(a: String, b: String): Boolean = a.trim().replace('\\', '/') == b.trim().replace('\\', '/')

/** The whisper models in [directory] on this platform; empty when it does not exist or cannot be read. */
fun scanWhisperModels(directory: String): List<FolderModel> = whisperModelsIn(directory, listFileNames(directory))

/** Names of the regular files directly in [directory] (no subdirectories); empty when it cannot be listed. */
internal expect fun listFileNames(directory: String): List<String>

/**
 * The model path a new screen starts with: [requested] (e.g. an intent extra) when given; otherwise [defaultPath] when
 * that model is in the folder or the folder has none; otherwise the first model found in the folder.
 */
fun initialModelPath(requested: String?, defaultPath: String, models: List<FolderModel>): String = when {
    !requested.isNullOrBlank() -> requested
    models.isEmpty() || models.any { samePath(it.path, defaultPath) } -> defaultPath
    else -> models.first().path
}

/** Index in [models] of the model at [path] (surrounding blanks ignored), or `null` when [path] is a custom path. */
fun chosenModelIndex(models: List<FolderModel>, path: String): Int? =
    models.indexOfFirst { samePath(it.path, path) }.takeIf { it >= 0 }

/** The picker's hint when the folder has no whisper models. */
fun noModelsHint(directory: String): String =
    "No whisper models (.bin) in $directory yet: copy one there, or enter a path below."

private const val MODEL_EXTENSION = ".bin"
