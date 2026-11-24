plugins {
    `kotlin-dsl`
}

group = "com.donfuldauer.kggml.buildlogic"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    // compileOnly: the actual plugin implementations are put on the classpath by the root
    // build's `plugins { ... apply false }` block, keeping exactly one version in the build.
    compileOnly(libs.kotlin.gradle.plugin)
    compileOnly(libs.android.gradle.plugin)
}

gradlePlugin {
    plugins {
        register("kggmlKmpLibrary") {
            id = libs.plugins.kggml.kmp.library.get().pluginId
            implementationClass = "com.donfuldauer.kggml.buildlogic.KggmlKmpLibraryPlugin"
        }
        register("kggmlPublishing") {
            id = libs.plugins.kggml.publishing.get().pluginId
            implementationClass = "com.donfuldauer.kggml.buildlogic.KggmlPublishingPlugin"
        }
    }
}
