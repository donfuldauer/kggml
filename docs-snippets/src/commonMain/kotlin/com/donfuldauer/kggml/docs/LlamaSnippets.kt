package com.donfuldauer.kggml.docs

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.TokenEvent
import com.donfuldauer.kggml.llama.GenerationOptions
import com.donfuldauer.kggml.llama.LlamaContextConfig
import com.donfuldauer.kggml.llama.LlamaEngine
import com.donfuldauer.kggml.llama.LlamaGrammar

// README.md snippets for :kggml-llama. Compiled by :docs-snippets; see build.gradle.kts.

internal suspend fun llamaGenerate(modelPath: String) {
    // snippet:start llama-generate
    val engine = LlamaEngine(
        modelPath = modelPath,  // any llama.cpp GGUF, e.g. a 1–3 B Q4_K_M instruct model
        backendConfig = BackendConfig(backend = ComputeBackend.AUTO, gpuLayers = BackendConfig.ALL_GPU_LAYERS),
        contextConfig = LlamaContextConfig(contextLength = 4096),
    )
    engine.generate("Write a haiku about the sea.", GenerationOptions(maxTokens = 64)).collect { event ->
        when (event) {
            is TokenEvent.Token -> print(event.text)
            is TokenEvent.Completed -> println("\n[${event.stopReason}]")
            is TokenEvent.Error -> throw event.exception
        }
    }

    // Structured output: decoding is constrained by a grammar generated from a JSON Schema.
    val schema = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"""
    val json = engine.generateText(
        "Which city is the Eiffel Tower in? Answer in JSON.",
        GenerationOptions(maxTokens = 32, grammar = LlamaGrammar.jsonSchema(schema)),
    ).text

    val vector: FloatArray = engine.embed("on-device inference")  // L2-normalized by default
    engine.close()
    // snippet:end llama-generate
    println("$json ${vector.size}")
}
