package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.NativeErrorCode
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.OutOfMemoryException
import kotlin.reflect.KClass
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertSame
import kotlin.test.assertTrue

class RuntimeClientTest {
    private val fake = FakeRuntimeBridge()
    private val client = RuntimeClient { fake }

    // ---- lifecycle --------------------------------------------------------------------------------------

    @Test
    fun acquireAndCloseBalanceReferenceCount() {
        val first = client.acquire()
        val second = client.acquire()
        assertEquals(2, client.refCount())
        assertTrue(client.isInitialized())

        first.close()
        assertEquals(1, client.refCount())
        second.close()
        assertEquals(0, client.refCount())
        assertFalse(client.isInitialized())
    }

    @Test
    fun closingAHandleTwiceReleasesOnlyOnce() {
        val other = client.acquire()
        val handle = client.acquire()
        handle.close()
        handle.close()
        assertTrue(handle.isClosed)
        assertEquals(1, client.refCount(), "second close() must not release another caller's reference")
        assertEquals(1, fake.shutdownCalls)
        other.close()
    }

    @Test
    fun repeatedInitTeardownCyclesLeaveNoReferences() {
        repeat(100) { client.acquire().use { assertEquals(1, client.refCount()) } }
        assertEquals(0, client.refCount())
        assertEquals(100, fake.initCalls)
        assertEquals(100, fake.shutdownCalls)
    }

    @Test
    fun failedInitThrowsTypedExceptionAndHoldsNoReference() {
        fake.failInitWith = RuntimeStatus.OUT_OF_MEMORY
        fake.failureMessage = "kggml_runtime_init: out of memory: std::bad_alloc"
        val e = assertFailsWith<OutOfMemoryException> { client.acquire() }
        assertEquals(NativeErrorCode.OUT_OF_MEMORY, e.errorCode)
        assertEquals(
            "kggml_runtime_init failed: KGGML_STATUS_OUT_OF_MEMORY (4): kggml_runtime_init: out of memory: std::bad_alloc",
            e.message,
        )
        assertEquals(0, client.refCount())
    }

    @Test
    fun failedShutdownThrowsAndHandleStaysClosed() {
        val handle = client.acquire()
        fake.failShutdownWith = RuntimeStatus.NATIVE_FAILURE
        assertFailsWith<NativeFailureException> { handle.close() }
        assertTrue(handle.isClosed)
        handle.close() // no second native shutdown attempt
        assertEquals(1, fake.shutdownCalls)
    }

    // ---- status translation -----------------------------------------------------------------------------

    @Test
    fun everyNativeStatusMapsToItsTypedException() {
        val expected: Map<Int, KClass<out InferenceException>> = mapOf(
            1 to ModelLoadException::class,
            2 to InvalidInputException::class,
            3 to BackendUnavailableException::class,
            4 to OutOfMemoryException::class,
            5 to InferenceCancelledException::class,
            6 to NativeFailureException::class,
            RuntimeStatus.NOT_INITIALIZED to NativeFailureException::class,
            RuntimeStatus.BUFFER_TOO_SMALL to NativeFailureException::class,
            RuntimeStatus.UNKNOWN to NativeFailureException::class,
        )
        for ((status, type) in expected) {
            fake.failInitWith = status
            val e = assertFailsWith<InferenceException> { client.acquire() }
            assertEquals(type, e::class, "status $status")
            assertEquals(0, client.refCount())
        }
    }

    @Test
    fun runtimeSpecificCodesPreserveTheRawNativeCode() {
        for (status in listOf(RuntimeStatus.NOT_INITIALIZED, RuntimeStatus.BUFFER_TOO_SMALL, RuntimeStatus.UNKNOWN)) {
            fake.failInitWith = status
            val e = assertFailsWith<NativeFailureException> { client.acquire() }
            assertEquals(status, e.nativeCode)
        }
    }

    @Test
    fun messageOmitsDetailWhenNativeLastErrorIsEmpty() {
        fake.failInitWith = RuntimeStatus.UNKNOWN
        fake.failureMessage = ""
        val e = assertFailsWith<NativeFailureException> { client.acquire() }
        assertEquals("kggml_runtime_init failed: KGGML_STATUS_<102> (102)", e.message)
    }

    @Test
    fun bridgeLoadFailureBecomesNativeFailureWithCause() {
        val cause = IllegalStateException("libkggml_runtime_jni not found")
        val broken = RuntimeClient { throw cause }
        val e = assertFailsWith<NativeFailureException> { broken.availableBackends() }
        assertSame(cause, e.cause)
        assertTrue(e.message!!.contains("libkggml_runtime_jni not found"))
        assertEquals(NativeErrorCode.NATIVE_FAILURE.code, e.nativeCode)
    }

    @Test
    fun bridgeLoadFailureWithoutMessageIsDescribedByItsType() {
        val cause = IllegalStateException()
        val broken = RuntimeClient { throw cause }
        val e = assertFailsWith<NativeFailureException> { broken.info() }
        assertEquals("Failed to load the Kggml native runtime: IllegalStateException", e.message)
        assertSame(cause, e.cause)
    }

    @Test
    fun nativeErrorFallsBackToTheRawStatusWhenTheBridgeCannotDescribeIt() {
        val silent = object : NativeRuntimeBridge by fake {
            override fun lastError(): String = throw IllegalStateException("lastError unavailable")
            override fun statusName(status: Int): String = throw IllegalStateException("statusName unavailable")
        }
        val e = RuntimeClient.nativeError(silent, RuntimeStatus.UNKNOWN, "kggml_runtime_probe")
        assertIs<NativeFailureException>(e)
        assertEquals("kggml_runtime_probe failed: KGGML_STATUS_${RuntimeStatus.UNKNOWN} (${RuntimeStatus.UNKNOWN})", e.message)
        assertEquals(RuntimeStatus.UNKNOWN, e.nativeCode)
    }

    @Test
    fun nativeErrorReportsTheBackendOnlyWhenGiven() {
        fake.failureMessage = "no device"
        fake.failInitWith = RuntimeStatus.UNKNOWN // sets the fake's last error to `failureMessage`
        assertFailsWith<NativeFailureException> { client.acquire() }

        val withoutBackend = RuntimeClient.nativeError(fake, NativeErrorCode.BACKEND_UNAVAILABLE.code, "op")
        assertIs<BackendUnavailableException>(withoutBackend)
        assertEquals(null, withoutBackend.backend)
        assertEquals("op failed: KGGML_STATUS_<3> (3): no device", withoutBackend.message)

        val withBackend = RuntimeClient.nativeError(fake, NativeErrorCode.BACKEND_UNAVAILABLE.code, "op", ComputeBackend.GPU_VULKAN)
        assertIs<BackendUnavailableException>(withBackend)
        assertEquals(ComputeBackend.GPU_VULKAN, withBackend.backend)
    }

    @Test
    fun bridgeLoadInferenceExceptionPassesThroughUnchanged() {
        val original = InvalidInputException("bad configuration")
        val broken = RuntimeClient { throw original }
        assertSame(original, assertFailsWith<InvalidInputException> { broken.acquire() })
    }

    @Test
    fun bridgeIsCreatedOnceAndReused() {
        var created = 0
        val counting = RuntimeClient { created++; fake }
        counting.hardwareThreadCount()
        counting.recommendedThreadCount()
        counting.availableBackends()
        assertEquals(1, created)
    }

    @Test
    fun failedBridgeCreationIsRetriedOnNextCall() {
        var attempts = 0
        val flaky = RuntimeClient {
            attempts++
            if (attempts == 1) throw IllegalStateException("first load fails")
            fake
        }
        assertFailsWith<NativeFailureException> { flaky.hardwareThreadCount() }
        assertEquals(8, flaky.hardwareThreadCount())
        assertEquals(2, attempts)
    }

    // ---- backends -----------------------------------------------------------------------------------------

    @Test
    fun backendsMapNativeKindsOntoComputeBackends() {
        val backends = client.backends()
        assertEquals(
            listOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL, ComputeBackend.GPU_VULKAN, ComputeBackend.NPU_COREML),
            backends.map { it.backend },
        )
        assertEquals(BackendInfo(ComputeBackend.GPU_VULKAN, "Vulkan", compiled = true, available = true, deviceCount = 2), backends[2])
        assertEquals(0, client.refCount(), "queries must release the reference they take")
    }

    @Test
    fun availableBackendsContainsOnlyAvailableEntriesAndAlwaysCpu() {
        assertEquals(setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_VULKAN), client.availableBackends())

        fake.backends = fake.backends.map { it.copy(available = false) }
        assertEquals(setOf(ComputeBackend.CPU_ONLY), client.availableBackends())
    }

    @Test
    fun availableBackendsNeverContainsAuto() {
        assertFalse(ComputeBackend.AUTO in client.availableBackends())
    }

    @Test
    fun unknownFutureBackendKindsAreIgnored() {
        fake.backends = fake.backends + RawBackendInfo(kind = 42, compiled = true, available = true, deviceCount = 1, name = "NPU-X")
        val backends = client.backends()
        assertEquals(4, backends.size)
        assertEquals(setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_VULKAN), client.availableBackends())
    }

    @Test
    fun bufferTooSmallIsRetriedAtTheRequiredCapacity() {
        fake.backends = fake.backends + RawBackendInfo(kind = 7, compiled = false, available = false, deviceCount = 0, name = "Future")
        client.backends()
        assertEquals(listOf(BackendKind.COUNT, BackendKind.COUNT + 1), fake.getBackendsCapacities)
    }

    @Test
    fun bufferTooSmallWithoutALargerRequiredCountIsAnErrorNotARetry() {
        // A runtime that reports BUFFER_TOO_SMALL while needing no more than was offered is inconsistent;
        // retrying at the same size could not help, so the status surfaces as-is.
        fake.failBackendsWith = RuntimeStatus.BUFFER_TOO_SMALL
        val e = assertFailsWith<NativeFailureException> { client.backends() }
        assertEquals(RuntimeStatus.BUFFER_TOO_SMALL, e.nativeCode)
        assertTrue(e.message!!.startsWith("kggml_runtime_get_backends failed: "), e.message)
        assertEquals(listOf(BackendKind.COUNT), fake.getBackendsCapacities)
        assertEquals(0, client.refCount())
    }

    @Test
    fun negativeDeviceCountsAreClampedToZero() {
        fake.backends = listOf(RawBackendInfo(BackendKind.CPU, compiled = true, available = true, deviceCount = -1, name = "CPU"))
        assertEquals(0, client.backends().single().deviceCount)
    }

    @Test
    fun backendQueryFailureReleasesReferenceAndThrows() {
        fake.failBackendsWith = RuntimeStatus.NATIVE_FAILURE
        assertFailsWith<NativeFailureException> { client.backends() }
        assertEquals(0, client.refCount())
    }

    @Test
    fun backendQueryFailureKeepsOriginalExceptionWhenReleaseAlsoFails() {
        fake.failBackendsWith = RuntimeStatus.OUT_OF_MEMORY
        fake.failShutdownWith = RuntimeStatus.NATIVE_FAILURE
        val e = assertFailsWith<OutOfMemoryException> { client.backends() }
        assertIs<NativeFailureException>(e.suppressedExceptions.single())
    }

    @Test
    fun resolveBackendUsesAvailableBackends() {
        assertEquals(ComputeBackend.GPU_VULKAN, client.resolveBackend(ComputeBackend.AUTO, allowFallback = true))
        assertEquals(ComputeBackend.CPU_ONLY, client.resolveBackend(ComputeBackend.GPU_METAL, allowFallback = true))
        val e = assertFailsWith<BackendUnavailableException> {
            client.resolveBackend(ComputeBackend.GPU_METAL, allowFallback = false)
        }
        assertEquals(ComputeBackend.GPU_METAL, e.backend)
    }

    // ---- threads & info -----------------------------------------------------------------------------------

    @Test
    fun threadCountsAreReportedAndClampedToAtLeastOne() {
        assertEquals(4, client.recommendedThreadCount())
        assertEquals(8, client.hardwareThreadCount())

        fake.performanceThreads = 0
        fake.hardwareThreads = -3
        assertEquals(1, client.recommendedThreadCount())
        assertEquals(1, client.hardwareThreadCount())
    }

    @Test
    fun threadCountFailureIsTranslated() {
        fake.failThreadsWith = RuntimeStatus.INVALID_ARGUMENT
        assertFailsWith<InvalidInputException> { client.recommendedThreadCount() }
    }

    @Test
    fun infoReportsVersions() {
        assertEquals(
            RuntimeInfo(
                runtimeVersion = "1.0.0",
                ggmlVersion = "0.9.9",
                ggmlCommit = "a04eea0761a85d18f3f504d6ab970c5c9dce705f",
                ggmlBuildCommit = "unknown",
                hasCoreMl = false,
            ),
            client.info(),
        )
    }

    // ---- kind mapping -------------------------------------------------------------------------------------

    @Test
    fun backendKindMappingMatchesHeader() {
        assertEquals(ComputeBackend.CPU_ONLY, BackendKind.toComputeBackend(0))
        assertEquals(ComputeBackend.GPU_METAL, BackendKind.toComputeBackend(1))
        assertEquals(ComputeBackend.GPU_VULKAN, BackendKind.toComputeBackend(2))
        assertEquals(ComputeBackend.NPU_COREML, BackendKind.toComputeBackend(3))
        assertEquals(null, BackendKind.toComputeBackend(4))
        assertEquals(null, BackendKind.toComputeBackend(-1))
    }

    @Test
    fun sharedStatusCodesEqualNativeErrorCode() {
        assertEquals(NativeErrorCode.OK.code, RuntimeStatus.OK)
        assertEquals(NativeErrorCode.INVALID_INPUT.code, RuntimeStatus.INVALID_ARGUMENT)
        assertEquals(NativeErrorCode.OUT_OF_MEMORY.code, RuntimeStatus.OUT_OF_MEMORY)
        assertEquals(NativeErrorCode.NATIVE_FAILURE.code, RuntimeStatus.NATIVE_FAILURE)
    }
}
