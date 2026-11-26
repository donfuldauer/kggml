package com.donfuldauer.kggml.models

import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureLevel
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class ResidentModelCacheTest {
    private class FakeModel(val name: String) : AutoCloseable {
        var closeCount = 0
        override fun close() { closeCount++ }
    }

    @Test
    fun budgetEvictsLeastRecentlyUsedIdleModels() {
        val cache = ResidentModelCache(budgetBytes = 100)
        val a = FakeModel("a"); val b = FakeModel("b"); val c = FakeModel("c")
        cache.put("a", 40, a).close()
        cache.put("b", 40, b).close()
        cache.acquire<FakeModel>("a")!!.close() // a is now more recent than b
        cache.put("c", 40, c).close()
        assertEquals(1, b.closeCount)
        assertEquals(0, a.closeCount)
        assertEquals(listOf("a", "c"), cache.keys)
        assertEquals(80L, cache.residentBytes)
    }

    @Test
    fun leasedModelsAreNeverEvicted() {
        val cache = ResidentModelCache(budgetBytes = 50)
        val a = FakeModel("a")
        val lease = cache.put("a", 40, a)
        val b = FakeModel("b")
        cache.put("b", 40, b).close()
        assertEquals(0, b.closeCount, "a model is leased while it is being put")
        assertEquals(listOf("b"), cache.evict(MemoryPressureLevel.NORMAL), "a is leased, so only b goes")
        assertEquals(emptyList(), cache.evict(MemoryPressureLevel.CRITICAL))
        assertEquals(0, a.closeCount)
        lease.close()
        lease.close()
        assertEquals(listOf("a"), cache.evict(MemoryPressureLevel.CRITICAL))
        assertEquals(1, a.closeCount)
        assertNull(cache.acquire<FakeModel>("a"))
    }

    @Test
    fun memoryPressureLevels() {
        val observer = ManualMemoryPressureObserver()
        val cache = ResidentModelCache(budgetBytes = 100)
        val models = (1..4).map { FakeModel("m$it") }
        models.forEach { cache.put(it.name, 25, it).close() }
        val subscription = cache.attachTo(observer)
        observer.emit(MemoryPressureLevel.NORMAL)
        assertEquals(4, cache.keys.size)
        observer.emit(MemoryPressureLevel.MODERATE)
        assertEquals(listOf("m3", "m4"), cache.keys, "trimmed to half the budget, LRU first")
        observer.emit(MemoryPressureLevel.CRITICAL)
        assertTrue(cache.keys.isEmpty())
        assertTrue(models.all { it.closeCount == 1 })
        subscription.close()
        val late = FakeModel("late")
        cache.put("late", 10, late).close()
        observer.emit(MemoryPressureLevel.CRITICAL)
        assertEquals(listOf("late"), cache.keys, "unsubscribed")
    }

    @Test
    fun moderateWithoutBudgetHalvesResidentBytes() {
        val cache = ResidentModelCache()
        (1..4).forEach { cache.put("m$it", 10, FakeModel("m$it")).close() }
        assertEquals(listOf("m1", "m2"), cache.evict(MemoryPressureLevel.MODERATE))
        assertEquals(20L, cache.residentBytes)
    }

    @Test
    fun getOrPutLoadsOnceAndReuses() = runTest {
        val cache = ResidentModelCache()
        var loads = 0
        val first = cache.getOrPut("k", 1) { loads++; FakeModel("k") }
        val second = cache.getOrPut("k", 1) { loads++; FakeModel("k2") }
        assertSame(first.resource, second.resource)
        assertEquals(1, loads)
        assertEquals("k", first.key)
        first.close(); second.close()
        // Resident: the loader is not called.
        val dup = FakeModel("dup")
        val resident = cache.getOrPut("k", 1) { dup }
        assertEquals(0, dup.closeCount)
        resident.close()
    }

    @Test
    fun putReplacesIdleAndRejectsLeased() {
        val cache = ResidentModelCache()
        val a = FakeModel("a")
        cache.put("k", 1, a).close()
        val a2 = FakeModel("a2")
        val lease = cache.put("k", 1, a2)
        assertEquals(1, a.closeCount)
        assertFailsWith<IllegalStateException> { cache.put("k", 1, FakeModel("x")) }
        assertFalse(cache.remove("k"))
        lease.close()
        assertTrue(cache.remove("k"))
        assertFalse(cache.remove("k"))
        assertEquals(1, a2.closeCount)
        assertFailsWith<IllegalArgumentException> { cache.put("n", -1, FakeModel("n")) }
        assertFailsWith<IllegalArgumentException> { ResidentModelCache(-1) }
    }

    @Test
    fun closeClosesEverythingOnce() {
        val cache = ResidentModelCache()
        val a = FakeModel("a"); val b = FakeModel("b")
        val lease = cache.put("a", 1, a)
        cache.put("b", 1, b).close()
        cache.close()
        cache.close()
        lease.close()
        assertEquals(1, a.closeCount)
        assertEquals(1, b.closeCount)
        assertFailsWith<IllegalStateException> { cache.put("c", 1, FakeModel("c")) }
    }

    @Test
    fun closeFailuresArePropagatedAfterClosingAll() {
        val cache = ResidentModelCache()
        val ok = FakeModel("ok")
        cache.put("bad", 1, AutoCloseable { throw IllegalStateException("boom") }).close()
        cache.put("ok", 1, ok).close()
        assertFailsWith<IllegalStateException> { cache.evict(MemoryPressureLevel.CRITICAL) }
        assertEquals(1, ok.closeCount)
        assertTrue(cache.keys.isEmpty())
    }

    @Test
    fun getOrPutClosesItsCopyWhenAnotherCallerLoadedTheKeyFirst() = runTest {
        val cache = ResidentModelCache()
        val winner = FakeModel("winner")
        val loser = FakeModel("loser")
        val lease = cache.getOrPut("k", 1) {
            // Another caller finishes loading the same key while this loader is running.
            cache.put("k", 1, winner).close()
            loser
        }
        assertSame(winner, lease.resource)
        assertEquals(1, loser.closeCount)
        assertEquals(0, winner.closeCount)
        assertEquals(listOf("k"), cache.keys)
        assertFalse(cache.remove("k"), "the returned lease is held")
        lease.close()
        assertTrue(cache.remove("k"))
    }

    @Test
    fun rePuttingTheSameResourceDoesNotCloseIt() {
        val cache = ResidentModelCache()
        val a = FakeModel("a")
        cache.put("k", 10, a).close()
        cache.put("k", 20, a).close()
        assertEquals(0, a.closeCount)
        assertEquals(20L, cache.residentBytes, "the new size estimate replaces the old one")
    }

    @Test
    fun laterCloseFailuresAreSuppressedIntoTheFirst() {
        val cache = ResidentModelCache()
        val first = IllegalStateException("first")
        val second = IllegalArgumentException("second")
        val ok = FakeModel("ok")
        cache.put("a", 1, AutoCloseable { throw first }).close()
        cache.put("b", 1, ok).close()
        cache.put("c", 1, AutoCloseable { throw second }).close()
        val thrown = assertFailsWith<IllegalStateException> { cache.close() }
        assertSame(first, thrown)
        assertEquals(listOf<Throwable>(second), thrown.suppressedExceptions)
        assertEquals(1, ok.closeCount)
    }
}
