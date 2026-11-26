plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml Android GPU add-on: ggml's Vulkan backend for the shared :kggml-runtime (no-op on other platforms)."

// ---------------------------------------------------------------------------------------------------------
// Opt-in switch (like the engine modules; `./gradlew check` needs no C/C++ toolchain):
//   -Pkggml.bundleNative=true   build and package libggml-vulkan.so (arm64-v8a, x86_64) into the AAR jniLibs.
// The :kggml-runtime Android loader finds it in the app and registers it (kggml_runtime_load_backend), so
// ComputeBackend.GPU_VULKAN / AUTO can use the phone's GPU; without a usable Vulkan 1.1+ driver everything stays on
// the CPU. Desktop builds link Vulkan into the runtime instead (KGGML_ENABLE_VULKAN), iOS uses Metal.
// ---------------------------------------------------------------------------------------------------------

val bundleNative = providers.gradleProperty("kggml.bundleNative").map(String::toBoolean).getOrElse(false)

val nativeDir = layout.projectDirectory.dir("native")
val rootBuildDir = rootProject.layout.projectDirectory.dir("build")
val androidJniLibsDir = rootBuildDir.dir("native-android-vulkan-jniLibs")
val buildAndroidScript = nativeDir.file("scripts/build-android.sh").asFile.absolutePath

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(project(":kggml-runtime"))
        }
    }
}

android {
    namespace = "com.donfuldauer.kggml.runtime.vulkan"
}

val buildNativeAndroid by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Cross-compiles ggml's Vulkan backend module for Android (arm64-v8a, x86_64) into " +
        "<root>/build/native-android-vulkan-jniLibs."
    workingDir(rootProject.layout.projectDirectory)
    commandLine("bash", buildAndroidScript, "arm64-v8a", "x86_64")
    // Links libggml-base.so of the same ggml; keep the runtime's (parallel) build first.
    mustRunAfter(":kggml-runtime:buildNativeAndroid")
}

if (bundleNative) {
    android.sourceSets.getByName("main").jniLibs.srcDir(androidJniLibsDir)
    tasks.matching { it.name == "preBuild" }.configureEach { dependsOn(buildNativeAndroid) }
}
