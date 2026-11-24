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
    ":kggml-whisper",
    ":kggml-llama",
    ":kggml-diffusion",
    ":kggml-model-manager",
)

// Samples (never published). Populated in later issues.
include(
    ":samples:kggml-showcase",
    ":samples:whisper-only-minimal",
)
