import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// README snippet compile check (issue #22). Never published and not part of kggml-bom.
//
// Every Kotlin / Gradle code block in the root README.md is a region of a file in this module, marked with
// `// snippet:start <name>` ... `// snippet:end <name>` (scripts/ci/verify-docs.sh checks that the README text and
// the region are identical). Compiling this module therefore compiles the documented quickstart code.
//
// The module consumes the SDK exactly like an external app: the published coordinates below resolve from the
// build-local `kggmlVerification` Maven repository (filled by `publishAllPublicationsToKggmlVerificationRepository`,
// the same repository `:kggml-bom:verifyBomConsumerResolution` uses), never through `project(...)` dependencies, so
// a wrong artifact id or BOM version in the README fails the build.
plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

val jvmTargetVersion: String = libs.versions.jvmTarget.get()

kotlin {
    jvm {
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
    }

    // snippet:start gradle-dependencies
    sourceSets {
        commonMain.dependencies {
            // One version for every Kggml module.
            implementation(project.dependencies.platform("com.donfuldauer:kggml-bom:1.0.0"))

            // Pick only the engines you ship; unused engines add no native code to your app.
            implementation("com.donfuldauer:kggml-whisper")        // speech-to-text
            implementation("com.donfuldauer:kggml-llama")          // text generation, embeddings, vision
            implementation("com.donfuldauer:kggml-diffusion")      // text-to-image
            implementation("com.donfuldauer:kggml-model-manager")  // optional: downloads, RAM estimates
            implementation("com.donfuldauer:kggml-runtime-vulkan") // optional: Android GPU (Vulkan); no-op elsewhere
        }
    }
    // snippet:end gradle-dependencies
}

// The Kggml coordinates above resolve from the build-local verification repository: publish the BOM and the JVM
// side of every library there (the root `kotlinMultiplatform` module that points at the `-jvm` variant, and that
// variant) before anything resolves this module's classpaths. Only JVM publications are needed, so this also works
// on macOS hosts without publishing iOS klibs.
val kggmlLibraryModules = listOf(
    ":kggml-core-api",
    ":kggml-runtime",
    ":kggml-runtime-vulkan",
    ":kggml-whisper",
    ":kggml-llama",
    ":kggml-diffusion",
    ":kggml-model-manager",
)
tasks.matching { it.name.startsWith("compile") }.configureEach {
    dependsOn(":kggml-bom:publishMavenPublicationToKggmlVerificationRepository")
    kggmlLibraryModules.forEach { module ->
        dependsOn("$module:publishKotlinMultiplatformPublicationToKggmlVerificationRepository")
        dependsOn("$module:publishJvmPublicationToKggmlVerificationRepository")
    }
}
