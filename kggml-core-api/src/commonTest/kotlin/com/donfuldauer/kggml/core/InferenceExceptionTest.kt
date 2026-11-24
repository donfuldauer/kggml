package com.donfuldauer.kggml.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertSame

class InferenceExceptionTest {

    @Test
    fun nativeErrorCodesAreStableAndUnique() {
        assertEquals(
            listOf(0, 1, 2, 3, 4, 5, 6),
            NativeErrorCode.entries.map { it.code },
        )
    }

    @Test
    fun fromCodeRoundTripsKnownCodes() {
        for (entry in NativeErrorCode.entries) assertEquals(entry, NativeErrorCode.fromCode(entry.code))
    }

    @Test
    fun fromCodeMapsUnknownCodesToNativeFailure() {
        assertEquals(NativeErrorCode.NATIVE_FAILURE, NativeErrorCode.fromCode(-1))
        assertEquals(NativeErrorCode.NATIVE_FAILURE, NativeErrorCode.fromCode(42))
    }

    @Test
    fun mapsEachNativeCodeToTypedException() {
        assertIs<ModelLoadException>(InferenceException.fromNativeError(1))
        assertIs<InvalidInputException>(InferenceException.fromNativeError(2))
        assertIs<BackendUnavailableException>(InferenceException.fromNativeError(3))
        assertIs<OutOfMemoryException>(InferenceException.fromNativeError(4))
        assertIs<InferenceCancelledException>(InferenceException.fromNativeError(5))
        assertIs<NativeFailureException>(InferenceException.fromNativeError(6))
    }

    @Test
    fun mappedExceptionCarriesErrorCodeAndMessage() {
        for (entry in NativeErrorCode.entries - NativeErrorCode.OK) {
            val e = InferenceException.fromNativeError(entry.code, "detail-${entry.code}")
            assertEquals(entry, e.errorCode)
            assertEquals("detail-${entry.code}", e.message)
        }
    }

    @Test
    fun mappedExceptionHasDefaultMessageWhenNativeMessageMissing() {
        for (entry in NativeErrorCode.entries - NativeErrorCode.OK) {
            val message = InferenceException.fromNativeError(entry.code).message
            check(!message.isNullOrBlank()) { "No default message for $entry" }
        }
    }

    @Test
    fun backendUnavailableCarriesBackend() {
        val e = InferenceException.fromNativeError(3, backend = ComputeBackend.GPU_METAL)
        assertIs<BackendUnavailableException>(e)
        assertEquals(ComputeBackend.GPU_METAL, e.backend)
        assertEquals("Compute backend GPU_METAL is not available", e.message)
        assertEquals("Compute backend (unknown) is not available", BackendUnavailableException(null).message)
    }

    @Test
    fun unknownCodeIsPreservedOnNativeFailure() {
        val e = InferenceException.fromNativeError(99)
        assertIs<NativeFailureException>(e)
        assertEquals(99, e.nativeCode)
        assertEquals(NativeErrorCode.NATIVE_FAILURE, e.errorCode)
        assertEquals("Native failure (code 99)", e.message)
        assertEquals(6, NativeFailureException(message = "x").nativeCode)
    }

    @Test
    fun okCodeIsNotAnError() {
        assertFailsWith<IllegalArgumentException> { InferenceException.fromNativeError(0) }
    }

    @Test
    fun checkNativeResultPassesOnOkAndThrowsOtherwise() {
        InferenceException.checkNativeResult(0)
        val e = assertFailsWith<OutOfMemoryException> { InferenceException.checkNativeResult(4, "kv cache") }
        assertEquals("kv cache", e.message)
        val b = assertFailsWith<BackendUnavailableException> {
            InferenceException.checkNativeResult(3, backend = ComputeBackend.GPU_VULKAN)
        }
        assertEquals(ComputeBackend.GPU_VULKAN, b.backend)
    }

    @Test
    fun causesArePreserved() {
        val cause = IllegalStateException("root")
        assertSame(cause, ModelLoadException("m", cause).cause)
        assertSame(cause, InvalidInputException("m", cause).cause)
        assertSame(cause, BackendUnavailableException(ComputeBackend.CPU_ONLY, "m", cause).cause)
        assertSame(cause, OutOfMemoryException("m", cause).cause)
        assertSame(cause, InferenceCancelledException("m", cause).cause)
        assertSame(cause, NativeFailureException(6, "m", cause).cause)
        assertNull(ModelLoadException("m").cause)
    }

    @Test
    fun hierarchyIsExhaustive() {
        // Compile-time exhaustiveness: adding a subclass without updating callers fails to compile.
        fun describe(e: InferenceException): String = when (e) {
            is ModelLoadException -> "model"
            is InvalidInputException -> "input"
            is BackendUnavailableException -> "backend"
            is OutOfMemoryException -> "oom"
            is InferenceCancelledException -> "cancelled"
            is NativeFailureException -> "native"
        }
        assertEquals("oom", describe(OutOfMemoryException("x")))
    }
}
