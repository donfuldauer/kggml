package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.NativeFailureException
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * Without a configured native directory, bundled libraries or `java.library.path` entry, the real JVM wiring
 * ([KggmlRuntime] → [JniRuntimeBridge] → [JvmNativeLibraryLoader]) must fail with a typed, retryable error.
 * Runs only when the host native build is absent (the complementary case is [JvmNativeRuntimeTest]).
 */
class JvmMissingRuntimeTest {
    @BeforeTest
    fun requireNoNativeRuntime() {
        assumeTrue(System.getProperty(JvmNativeLibraryLoader.PATH_PROPERTY) == null, "native runtime configured")
        assumeTrue(System.getenv(JvmNativeLibraryLoader.PATH_ENV) == null, "native runtime configured")
        val platform = JvmNativeLibraryLoader.platformId()
        assumeTrue(javaClass.getResource("/kggml/native/$platform/index.txt") == null, "native runtime bundled")
    }

    @Test
    fun missingRuntimeSurfacesAsNativeFailureWithTheLinkErrorAsCause() {
        val e = assertFailsWith<NativeFailureException> { KggmlRuntime.info() }
        val cause = assertIs<UnsatisfiedLinkError>(e.cause)
        assertTrue(cause.message!!.startsWith("no kggml_runtime_jni in java.library.path"), cause.message)
        assertEquals("Failed to load the Kggml native runtime: ${cause.message}", e.message)
    }

    @Test
    fun everyFacadeEntryPointReportsTheTypedError() {
        val calls: Map<String, () -> Any> = mapOf(
            "acquire" to { KggmlRuntime.acquire() },
            "isInitialized" to { KggmlRuntime.isInitialized() },
            "backends" to { KggmlRuntime.backends() },
            "availableBackends" to { KggmlRuntime.availableBackends() },
            "resolveBackend" to { KggmlRuntime.resolveBackend(ComputeBackend.AUTO) },
            "recommendedThreadCount" to { KggmlRuntime.recommendedThreadCount() },
        )
        for ((name, call) in calls) {
            val e = assertFailsWith<NativeFailureException>(name) { call() }
            assertIs<UnsatisfiedLinkError>(e.cause, name)
        }
    }

    @Test
    fun loadFailureIsNotCachedSoLaterCallsRetry() {
        val first = assertFailsWith<NativeFailureException> { KggmlRuntime.hardwareThreadCount() }
        val second = assertFailsWith<NativeFailureException> { KggmlRuntime.hardwareThreadCount() }
        // A fresh load attempt produces a fresh link error each time.
        assertTrue(first.cause !== second.cause)
        assertNull(JvmNativeLibraryLoader.origin)
    }
}
