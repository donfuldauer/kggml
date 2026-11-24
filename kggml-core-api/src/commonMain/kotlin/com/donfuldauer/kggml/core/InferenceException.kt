package com.donfuldauer.kggml.core

/**
 * Structured error codes returned by Kggml's `extern "C"` native entry points.
 *
 * Native bridges never let C++ exceptions cross the FFI boundary; they catch them and return one of these
 * codes, which Kotlin maps to a typed [InferenceException] with [InferenceException.fromNativeError].
 *
 * @property code the integer value used on the native side. Must stay in sync with the C header.
 */
public enum class NativeErrorCode(public val code: Int) {
    OK(0),
    MODEL_LOAD_FAILED(1),
    INVALID_INPUT(2),
    BACKEND_UNAVAILABLE(3),
    OUT_OF_MEMORY(4),
    CANCELLED(5),
    NATIVE_FAILURE(6),
    ;

    public companion object {
        /** Maps a raw native [code] to its enum entry; unknown codes map to [NATIVE_FAILURE]. */
        public fun fromCode(code: Int): NativeErrorCode = entries.firstOrNull { it.code == code } ?: NATIVE_FAILURE
    }
}

/**
 * Base class of every error surfaced by Kggml engines.
 *
 * @property errorCode the native error category this exception represents.
 */
public sealed class InferenceException(
    public val errorCode: NativeErrorCode,
    message: String?,
    cause: Throwable? = null,
) : RuntimeException(message, cause) {

    public companion object {
        /**
         * Maps a non-[NativeErrorCode.OK] native result [code] to the matching typed exception.
         * Unknown codes produce a [NativeFailureException] that preserves the raw code.
         *
         * @param message optional native diagnostic message.
         * @param backend backend involved, reported by [BackendUnavailableException].
         * @throws IllegalArgumentException if [code] is [NativeErrorCode.OK] (`0`).
         */
        public fun fromNativeError(
            code: Int,
            message: String? = null,
            backend: ComputeBackend? = null,
        ): InferenceException {
            val errorCode = NativeErrorCode.fromCode(code)
            require(errorCode != NativeErrorCode.OK) { "Native result code 0 (OK) is not an error" }
            return when (errorCode) {
                NativeErrorCode.MODEL_LOAD_FAILED -> ModelLoadException(message ?: "Failed to load model")
                NativeErrorCode.INVALID_INPUT -> InvalidInputException(message ?: "Invalid input")
                NativeErrorCode.BACKEND_UNAVAILABLE -> BackendUnavailableException(backend, message)
                NativeErrorCode.OUT_OF_MEMORY -> OutOfMemoryException(message ?: "Native allocation failed")
                NativeErrorCode.CANCELLED -> InferenceCancelledException(message ?: "Inference cancelled")
                NativeErrorCode.NATIVE_FAILURE, NativeErrorCode.OK ->
                    NativeFailureException(nativeCode = code, message = message ?: "Native failure (code $code)")
            }
        }

        /**
         * Throws the typed exception for a native result [code], or returns normally if it is
         * [NativeErrorCode.OK].
         */
        public fun checkNativeResult(code: Int, message: String? = null, backend: ComputeBackend? = null) {
            if (code != NativeErrorCode.OK.code) throw fromNativeError(code, message, backend)
        }
    }
}

/** The model file is missing, unreadable, corrupt, or incompatible with the engine. */
public class ModelLoadException(message: String, cause: Throwable? = null) :
    InferenceException(NativeErrorCode.MODEL_LOAD_FAILED, message, cause)

/** The input (audio, prompt, image, grammar, …) was rejected as malformed or out of range. */
public class InvalidInputException(message: String, cause: Throwable? = null) :
    InferenceException(NativeErrorCode.INVALID_INPUT, message, cause)

/**
 * The requested compute backend cannot be used on this device or build.
 *
 * @property backend the unavailable backend, if known.
 */
public class BackendUnavailableException(
    public val backend: ComputeBackend?,
    message: String? = null,
    cause: Throwable? = null,
) : InferenceException(
    NativeErrorCode.BACKEND_UNAVAILABLE,
    message ?: "Compute backend ${backend ?: "(unknown)"} is not available",
    cause,
)

/** Native code failed to allocate memory (model weights, KV cache, compute buffers, …). */
public class OutOfMemoryException(message: String, cause: Throwable? = null) :
    InferenceException(NativeErrorCode.OUT_OF_MEMORY, message, cause)

/**
 * Native inference was aborted through its abort callback.
 *
 * When the abort was triggered by coroutine cancellation, engines rethrow the coroutine's own
 * `CancellationException` instead, so structured concurrency is preserved; this type is reported only for
 * aborts that did not originate from the caller's job.
 */
public class InferenceCancelledException(message: String, cause: Throwable? = null) :
    InferenceException(NativeErrorCode.CANCELLED, message, cause)

/**
 * An unexpected native failure (including C++ exceptions caught at the FFI boundary).
 *
 * @property nativeCode the raw native result code, which may be outside [NativeErrorCode]'s known values.
 */
public class NativeFailureException(
    public val nativeCode: Int = NativeErrorCode.NATIVE_FAILURE.code,
    message: String,
    cause: Throwable? = null,
) : InferenceException(NativeErrorCode.NATIVE_FAILURE, message, cause)
