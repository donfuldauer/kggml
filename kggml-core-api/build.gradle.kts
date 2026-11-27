plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml core API: InferenceEngine, BackendConfig, event and exception types (pure Kotlin, no native code)."

kotlin {
    // Kotlin/Native macOS (issue #93), on top of the convention plugin's targets; shared Apple code lives in appleMain.
    macosArm64()

    sourceSets {
        all {
            languageSettings.optIn("kotlin.concurrent.atomics.ExperimentalAtomicApi")
            if (name.endsWith("Test")) {
                // kotlinx-coroutines-test scheduling APIs (advanceUntilIdle, runCurrent, …).
                languageSettings.optIn("kotlinx.coroutines.ExperimentalCoroutinesApi")
            }
        }
        commonMain.dependencies {
            // Exposed in the public API (CoroutineDispatcher / CloseableCoroutineDispatcher).
            api(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}

android {
    namespace = "com.donfuldauer.kggml.core"
}
