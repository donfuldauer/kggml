pluginManagement {
    includeBuild("build-logic")
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        // Build-local repository populated by `publishAllPublicationsToKggmlVerificationRepository`.
        // Used only by `:kggml-bom:verifyBomConsumerResolution` to resolve Kggml coordinates
        // exactly as an external consumer would.
        maven {
            name = "kggmlVerification"
            url = uri(rootDir.resolve("build/kggml-verification-repo"))
            content { includeGroup("com.donfuldauer") }
        }
    }
}

rootProject.name = "kggml"

// Published SDK coordinates (`com.donfuldauer:<name>`), version-aligned by :kggml-bom.
include(
    ":kggml-bom",
    ":kggml-core-api",
    ":kggml-runtime",
    ":kggml-runtime-vulkan",
    ":kggml-runtime-opencl",
    ":kggml-whisper",
    ":kggml-llama",
    ":kggml-diffusion",
    ":kggml-model-manager",
)

// Unpublished cross-engine integration tests (several engines in one process).
include(":integration-tests")

// Unpublished README snippet compile check: consumes the SDK through the BOM like an external app.
include(":docs-snippets")

// Samples (never published). Android apps live in separate androidApp modules (AGP 9 forbids combining
// com.android.application with org.jetbrains.kotlin.multiplatform in one module).
include(
    ":samples:kggml-showcase",
    ":samples:kggml-showcase:androidApp",
    ":samples:whisper-only-minimal",
    ":samples:whisper-only-minimal:androidApp",
)
