package com.donfuldauer.kggml.integration

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.diffusion.DiffusionEngine
import com.donfuldauer.kggml.llama.GenerationOptions
import com.donfuldauer.kggml.llama.GenerationResult
import com.donfuldauer.kggml.llama.LlamaContextConfig
import com.donfuldauer.kggml.llama.LlamaEngine
import com.donfuldauer.kggml.llama.TinyLlamaModel
import com.donfuldauer.kggml.runtime.KggmlRuntime
import com.donfuldauer.kggml.whisper.TinyWhisperModel
import com.donfuldauer.kggml.whisper.TranscriptionOptions
import com.donfuldauer.kggml.whisper.TranscriptionResult
import com.donfuldauer.kggml.whisper.WhisperEngine
import java.io.File
import java.nio.file.Files
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.minutes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * Multi-engine coexistence (issue #20): `:kggml-whisper`, `:kggml-llama` and `:kggml-diffusion` loaded into ONE JVM
 * process, all dynamically linked against the single `:kggml-runtime` ggml build.
 *
 * - Whisper and Llama run inference **concurrently** (each on its own serial dispatcher) and produce exactly the
 *   results they produce alone: no shared mutable ggml state, no symbol interposition between the engines.
 * - On Linux, `/proc/self/maps` proves every `libggml*` library is mapped from exactly one file — the runtime's —
 *   so no engine carries (or loaded) a second ggml copy.
 * - Diffusion loads next to both and reports a typed [ModelLoadException] for a bad model without disturbing them.
 * - Repeated open / infer / close cycles of both engines keep the resident set bounded (no leak, no OOM).
 *
 * Skipped unless Gradle found all four host builds; with `-Pkggml.requireNative=true` a missing build fails instead.
 */
class EngineCoexistenceTest {
    private val requireNative = System.getProperty("kggml.test.requireNative").toBoolean()
    private val tempDir: File = Files.createTempDirectory("kggml-coexistence-test").toFile().apply { deleteOnExit() }
    private lateinit var whisperModel: File
    private lateinit var llamaModel: File

    @BeforeTest
    fun setUp() {
        val properties = listOf("kggml.native.path", "kggml.whisper.native.path", "kggml.llama.native.path", "kggml.diffusion.native.path")
        val present = properties.all { System.getProperty(it) != null }
        if (requireNative) assertTrue(present, "-Pkggml.requireNative=true but a host native build was not found: $properties")
        assumeTrue(present, "host native builds not found; run kggml-{runtime,whisper,llama,diffusion}/native/scripts/build-host.sh")
        whisperModel = File(tempDir, "tiny-whisper.bin").apply { writeBytes(TinyWhisperModel.bytes()) }
        llamaModel = File(tempDir, "tiny-llama.gguf").apply { writeBytes(TinyLlamaModel.bytes()) }
    }

    @AfterTest
    fun tearDown() {
        tempDir.deleteRecursively()
    }

    private fun noise(samples: Int, seed: Int): FloatArray {
        val random = Random(seed)
        return FloatArray(samples) { random.nextFloat() * 0.2f - 0.1f }
    }

    private suspend fun whisperEngine(): WhisperEngine = WhisperEngine(whisperModel.absolutePath, CPU)

    private suspend fun llamaEngine(): LlamaEngine = LlamaEngine(llamaModel.absolutePath, CPU, SMALL)

    // Pure greedy decoding: whisper.cpp's temperature fallback samples from a per-state RNG that advances between
    // calls, which would make repeated runs of even a single engine differ.
    private suspend fun WhisperEngine.run(seed: Int): TranscriptionResult =
        transcribe(noise(16_000, seed), options = TranscriptionOptions(temperatureIncrement = 0f))

    private suspend fun LlamaEngine.run(prompt: String): GenerationResult =
        generateText(prompt, GenerationOptions.GREEDY.copy(maxTokens = 12, seed = 7))

    @Test
    fun whisperAndLlamaRunConcurrentlyWithTheResultsTheyProduceAlone() = runBlocking {
        withTimeout(5.minutes) {
            whisperEngine().use { whisper ->
                llamaEngine().use { llama ->
                    // Baselines: each engine alone (the other one is loaded but idle).
                    val soloWhisper = (1..3).map { whisper.run(it) }
                    val soloLlama = listOf("ab", "ab z", "z a").map { llama.run(it) }

                    // Both engines in flight at the same time, repeatedly, from different threads.
                    repeat(3) {
                        coroutineScope {
                            val transcriptions = async(Dispatchers.Default) { (1..3).map { whisper.run(it) } }
                            val generations = async(Dispatchers.Default) { listOf("ab", "ab z", "z a").map { llama.run(it) } }
                            assertEquals(soloWhisper, transcriptions.await(), "whisper output changed while llama was decoding")
                            assertEquals(soloLlama, generations.await(), "llama output changed while whisper was transcribing")
                        }
                    }
                    soloWhisper.forEach { assertEquals(1000L, it.audioDurationMs) }
                    soloLlama.forEach { assertTrue(it.tokens.isNotEmpty(), "llama generated no tokens") }
                    assertEquals(ComputeBackend.CPU_ONLY, whisper.activeBackend)
                }
            }
        }
    }

    @Test
    fun severalInstancesOfBothEnginesShareTheRuntime() = runBlocking {
        withTimeout(5.minutes) {
            val whispers = List(2) { whisperEngine() }
            val llamas = List(2) { llamaEngine() }
            try {
                val expectedWhisper = whispers[0].run(11)
                val expectedLlama = llamas[0].run("ab")
                val jobs = whispers.map { engine -> async(Dispatchers.Default) { engine.run(11) } } +
                    llamas.map { engine -> async(Dispatchers.Default) { engine.run("ab") } }
                val results = jobs.awaitAll()
                assertEquals(listOf(expectedWhisper, expectedWhisper), results.take(2))
                assertEquals(listOf(expectedLlama, expectedLlama), results.drop(2))
            } finally {
                (whispers + llamas).forEach { it.close() }
            }
        }
    }

    @Test
    fun diffusionLoadsAlongsideAndFailsTypedWithoutDisturbingTheOtherEngines() = runBlocking {
        withTimeout(5.minutes) {
            whisperEngine().use { whisper ->
                llamaEngine().use { llama ->
                    val before = llama.run("ab")
                    // A missing checkpoint and a non-diffusion GGUF (the llama model) are both typed load failures from
                    // stable-diffusion.cpp running on the same ggml runtime, while llama decodes on another thread.
                    val decoding = async(Dispatchers.Default) { llama.run("ab") }
                    assertFailsWith<ModelLoadException> {
                        DiffusionEngine(File(tempDir, "missing.gguf").absolutePath, CPU)
                    }
                    assertFailsWith<ModelLoadException> { DiffusionEngine(llamaModel.absolutePath, CPU) }
                    assertEquals(before, decoding.await())
                    assertTrue(DiffusionEngine.nativeInfo().stableDiffusionCppCommit.isNotEmpty())
                    assertEquals(1000L, whisper.run(3).audioDurationMs)
                    assertEquals(before, llama.run("ab"))
                }
            }
        }
    }

    @Test
    fun everyGgmlLibraryIsMappedFromTheSingleRuntimeBuild() = runBlocking {
        val maps = File("/proc/self/maps")
        assumeTrue(maps.isFile, "/proc/self/maps is Linux-only; the dynamic-linkage audits cover the other hosts")
        whisperEngine().use { whisper ->
            llamaEngine().use { llama ->
                whisper.run(1)
                llama.run("ab")
                runCatching { DiffusionEngine(File(tempDir, "missing.gguf").absolutePath, CPU) }
                assertTrue(KggmlRuntime.isInitialized())

                val mapped: Map<String, Set<String>> = maps.readLines()
                    .mapNotNull { line -> line.substringAfter('/', "").takeIf { it.isNotEmpty() }?.let { "/$it" } }
                    .filter { it.endsWith(".so") || ".so." in it }
                    .distinct()
                    .groupBy { File(it).name.substringBefore(".so") }
                    .mapValues { (_, paths) -> paths.map { File(it).canonicalPath }.toSet() }

                val runtimeDir = File(System.getProperty("kggml.native.path")).canonicalFile
                val ggml = mapped.filterKeys { it.startsWith("libggml") }
                assertTrue("libggml-base" in ggml, "libggml-base is not mapped: ${mapped.keys}")
                assertTrue("libggml-cpu" in ggml, "libggml-cpu is not mapped: ${mapped.keys}")
                for ((library, paths) in ggml) {
                    assertEquals(1, paths.size, "$library is mapped from more than one file: $paths")
                    assertEquals(runtimeDir, File(paths.single()).parentFile, "$library is not the :kggml-runtime build")
                }
                for (engine in listOf("libkggml_whisper", "libkggml_llama", "libkggml_diffusion")) {
                    assertEquals(1, mapped[engine]?.size, "$engine must be mapped exactly once: ${mapped[engine]}")
                }
            }
        }
    }

    @Test
    fun repeatedOpenInferCloseCyclesKeepTheResidentSetBounded() = runBlocking {
        val status = File("/proc/self/status")
        suspend fun cycle() = coroutineScope {
            val whisper = async(Dispatchers.Default) { whisperEngine().use { it.run(5) } }
            val llama = async(Dispatchers.Default) { llamaEngine().use { it.run("ab z") } }
            whisper.await()
            llama.await()
        }
        withTimeout(5.minutes) {
            // Warm-up: allocator pools and malloc arenas, the ggml backend registry, JIT, and the JVM heap growing to its
            // working size. That one-off growth depends on the host (cores, RAM) and is not a leak, so it is excluded.
            repeat(WARM_UP_CYCLES) { cycle() }
            if (residentKb(status) == null) return@withTimeout // not Linux: nothing to measure
            val samples = List(MEASURED_CYCLES) {
                cycle()
                checkNotNull(residentKb(status))
            }
            // A leak keeps growing every cycle; a plateau does not. Compare the medians of the first and last windows,
            // WINDOW_GAP cycles apart: leaking even 2 MB per cycle (far less than one engine's model plus contexts)
            // adds ~30 MB, while plateau noise stays well inside the bound.
            val growthKb = median(samples.takeLast(WINDOW)) - median(samples.take(WINDOW))
            assertTrue(
                growthKb < MAX_TREND_GROWTH_KB,
                "resident set kept growing after warm-up: ${growthKb / 1024} MB over $WINDOW_GAP cycles (median of the " +
                    "first vs last $WINDOW of $MEASURED_CYCLES, bound ${MAX_TREND_GROWTH_KB / 1024} MB); VmRSS per cycle in MB: " +
                    samples.joinToString { (it / 1024).toString() },
            )
        }
    }

    private fun median(values: List<Long>): Long = values.sorted().let { (it[(it.size - 1) / 2] + it[it.size / 2]) / 2 }

    private fun residentKb(status: File): Long? =
        if (!status.isFile) null
        else status.readLines().firstOrNull { it.startsWith("VmRSS:") }?.filter(Char::isDigit)?.toLongOrNull()

    private companion object {
        const val WARM_UP_CYCLES = 5
        const val MEASURED_CYCLES = 20
        const val WINDOW = 5
        const val WINDOW_GAP = MEASURED_CYCLES - WINDOW // cycles between the two windows' medians
        const val MAX_TREND_GROWTH_KB = 24L * 1024

        val CPU = BackendConfig(ComputeBackend.CPU_ONLY, gpuLayers = 0, cpuThreads = 2)
        val SMALL = LlamaContextConfig(contextLength = 256, batchSize = 16, embeddingContextLength = 64)
    }
}
