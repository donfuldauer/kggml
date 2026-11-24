package com.donfuldauer.kggml.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BackendConfigTest {

    @Test
    fun defaultsAreAutoAllLayersAutoThreadsMmapNoFlashAttention() {
        val config = BackendConfig()
        assertEquals(ComputeBackend.AUTO, config.backend)
        assertEquals(BackendConfig.ALL_GPU_LAYERS, config.gpuLayers)
        assertEquals(BackendConfig.AUTO_THREADS, config.cpuThreads)
        assertFalse(config.useFlashAttention)
        assertTrue(config.useMmap)
        assertEquals(config, BackendConfig.DEFAULT)
    }

    @Test
    fun cpuPresetIsCpuOnlyWithZeroGpuLayers() {
        assertEquals(ComputeBackend.CPU_ONLY, BackendConfig.CPU.backend)
        assertEquals(0, BackendConfig.CPU.gpuLayers)
    }

    @Test
    fun acceptsBoundaryValues() {
        BackendConfig(gpuLayers = -1, cpuThreads = -1)
        BackendConfig(gpuLayers = 0, cpuThreads = 1)
        BackendConfig(gpuLayers = 999, cpuThreads = BackendConfig.MAX_CPU_THREADS)
        BackendConfig(backend = ComputeBackend.CPU_ONLY, gpuLayers = -1)
        BackendConfig(backend = ComputeBackend.CPU_ONLY, gpuLayers = 0, useFlashAttention = true)
    }

    @Test
    fun rejectsGpuLayersBelowMinusOne() {
        val e = assertFailsWith<IllegalArgumentException> { BackendConfig(gpuLayers = -2) }
        assertTrue(e.message!!.contains("gpuLayers"))
    }

    @Test
    fun rejectsNegativeThreadCountsOtherThanAuto() {
        assertFailsWith<IllegalArgumentException> { BackendConfig(cpuThreads = -2) }
        assertFailsWith<IllegalArgumentException> { BackendConfig(cpuThreads = Int.MIN_VALUE) }
    }

    @Test
    fun rejectsZeroAndExcessiveThreadCounts() {
        assertFailsWith<IllegalArgumentException> { BackendConfig(cpuThreads = 0) }
        assertFailsWith<IllegalArgumentException> { BackendConfig(cpuThreads = BackendConfig.MAX_CPU_THREADS + 1) }
    }

    @Test
    fun rejectsPositiveGpuLayersWithCpuOnly() {
        val e = assertFailsWith<IllegalArgumentException> {
            BackendConfig(backend = ComputeBackend.CPU_ONLY, gpuLayers = 10)
        }
        assertTrue(e.message!!.contains("CPU_ONLY"))
    }

    @Test
    fun copyIsValidatedToo() {
        assertFailsWith<IllegalArgumentException> { BackendConfig().copy(cpuThreads = -5) }
        assertFailsWith<IllegalArgumentException> { BackendConfig(gpuLayers = 4).copy(backend = ComputeBackend.CPU_ONLY) }
    }

    @Test
    fun effectiveGpuLayersIsZeroForCpuOnly() {
        val config = BackendConfig(backend = ComputeBackend.AUTO, gpuLayers = 20)
        assertEquals(20, config.effectiveGpuLayers())
        assertEquals(20, config.effectiveGpuLayers(ComputeBackend.GPU_METAL))
        assertEquals(0, config.effectiveGpuLayers(ComputeBackend.CPU_ONLY))
        assertEquals(0, BackendConfig(backend = ComputeBackend.CPU_ONLY).effectiveGpuLayers())
    }

    @Test
    fun resolveCpuThreadsKeepsExplicitValue() {
        assertEquals(3, BackendConfig(cpuThreads = 3).resolveCpuThreads(availableProcessors = 16))
        assertEquals(12, BackendConfig(cpuThreads = 12).resolveCpuThreads(availableProcessors = 2))
    }

    @Test
    fun resolveCpuThreadsAutoClampsToProcessorsAndMax() {
        val auto = BackendConfig()
        assertEquals(4, auto.resolveCpuThreads(availableProcessors = 4))
        assertEquals(BackendConfig.MAX_AUTO_THREADS, auto.resolveCpuThreads(availableProcessors = 64))
        assertEquals(1, auto.resolveCpuThreads(availableProcessors = 0))
        assertEquals(1, auto.resolveCpuThreads(availableProcessors = -3))
    }

    @Test
    fun equalityAndHashCodeAreStructural() {
        val a = BackendConfig(ComputeBackend.GPU_VULKAN, 8, 4, useFlashAttention = true, useMmap = false)
        val b = BackendConfig(ComputeBackend.GPU_VULKAN, 8, 4, useFlashAttention = true, useMmap = false)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
    }
}
