package com.donfuldauer.kggml.docs

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.diffusion.DiffusionEngine
import com.donfuldauer.kggml.diffusion.DiffusionGenerationParams
import com.donfuldauer.kggml.diffusion.DiffusionImage

// README.md snippets for :kggml-diffusion. Compiled by :docs-snippets; see build.gradle.kts.

internal suspend fun diffusionGenerate(modelPath: String) {
    // snippet:start diffusion-generate
    DiffusionEngine(modelPath, BackendConfig.DEFAULT).use { engine ->  // e.g. an SD 1.5 q8_0 GGUF
        val image: DiffusionImage = engine.generateImage(
            DiffusionGenerationParams(
                prompt = "a lighthouse on a cliff at sunset, oil painting",
                negativePrompt = "blurry, low quality",
                width = 512,
                height = 512,
                sampleSteps = 20,
                seed = 42,
            ),
        ) { progress -> println("${progress.phase} ${progress.step}/${progress.totalSteps}") }
        // Packed RGB8, row-major: image.width * image.height * 3 bytes.
        check(image.pixels.size == image.width * image.height * image.channels)
    }
    // snippet:end diffusion-generate
}
