import com.android.build.api.withAndroid
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi

plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml optional model manager: resumable downloads, SHA-256 verification, RAM estimation."

kotlin {
    android {
        namespace = "com.donfuldauer.kggml.models"
    }

    // `jvmCommonMain` is shared by androidMain + jvmMain (java.io / java.nio file system, HttpURLConnection transport);
    // iosMain uses POSIX + NSURLSession instead. The module has no native C/C++ code of its own.
    @OptIn(ExperimentalKotlinGradlePluginApi::class)
    applyDefaultHierarchyTemplate {
        common {
            group("jvmCommon") {
                withAndroid()
                withJvm()
            }
        }
    }

    sourceSets {
        all {
            languageSettings.optIn("kotlin.concurrent.atomics.ExperimentalAtomicApi")
            if (name.endsWith("Test")) {
                languageSettings.optIn("kotlinx.coroutines.ExperimentalCoroutinesApi")
            }
        }
        commonMain.dependencies {
            // MemoryPressureObserver / MemoryPressureLevel are part of the public eviction API.
            api(project(":kggml-core-api"))
            api(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}
