package com.donfuldauer.kggml.models

/** Base class of every error raised by `:kggml-model-manager`. */
public open class ModelManagerException(message: String, cause: Throwable? = null) : Exception(message, cause)

/** A model file is not a valid GGUF / Whisper `ggml` file, or its header is truncated or corrupt. */
public class ModelFormatException(message: String, cause: Throwable? = null) : ModelManagerException(message, cause)

/** A file-system operation of a [ModelFileSystem] failed. */
public class ModelFileSystemException(message: String, cause: Throwable? = null) : ModelManagerException(message, cause)

/** Base class of download failures raised by [ModelDownloader]. */
public open class ModelDownloadException(message: String, cause: Throwable? = null) :
    ModelManagerException(message, cause)

/**
 * A network-level failure (connection reset, timeout, server error) that [ModelDownloader] retries, resuming from the
 * bytes already written. [HttpTransport] implementations throw it for I/O errors.
 */
public class DownloadTransportException(message: String, cause: Throwable? = null) :
    ModelDownloadException(message, cause)

/** The server answered with an HTTP status that cannot be retried (for example 403 or 404). */
public class DownloadHttpException(public val statusCode: Int, public val url: String) :
    ModelDownloadException("HTTP $statusCode for $url")

/** The server answered in a way that breaks the HTTP Range protocol (for example a `Content-Range` at the wrong offset). */
public class DownloadProtocolException(message: String) : ModelDownloadException(message)

/** The downloaded size differs from the expected size or from the size the server announced. */
public class DownloadSizeMismatchException(public val expectedBytes: Long, public val actualBytes: Long) :
    ModelDownloadException("Expected $expectedBytes bytes but got $actualBytes")

/** The SHA-256 of the downloaded file differs from the expected digest. The partial file has been deleted. */
public class ChecksumMismatchException(public val expectedSha256: String, public val actualSha256: String) :
    ModelDownloadException("SHA-256 mismatch: expected $expectedSha256, got $actualSha256")
