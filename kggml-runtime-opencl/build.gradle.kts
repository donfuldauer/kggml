plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml Android OpenCL add-on: ggml's OpenCL backend (Adreno) for the shared :kggml-runtime (no-op on other platforms)."

// ---------------------------------------------------------------------------------------------------------
// Opt-in switch (like the engine modules; `./gradlew check` needs no C/C++ toolchain):
//   -Pkggml.bundleNative=true   build and package libggml-opencl.so (arm64-v8a, x86_64) into the AAR jniLibs.
// The :kggml-runtime Android loader finds it in the app and registers it (kggml_runtime_load_backend), so
// ComputeBackend.GPU_OPENCL / AUTO can use the phone's GPU through the vendor's libOpenCL.so (Qualcomm Adreno); on
// devices without OpenCL the module does not load and everything stays on the CPU (or Vulkan with that add-on).
// ---------------------------------------------------------------------------------------------------------

val bundleNative = providers.gradleProperty("kggml.bundleNative").map(String::toBoolean).getOrElse(false)

val nativeDir = layout.projectDirectory.dir("native")
val rootBuildDir = rootProject.layout.projectDirectory.dir("build")
val androidJniLibsDir = rootBuildDir.dir("native-android-opencl-jniLibs")
val buildAndroidScript = nativeDir.file("scripts/build-android.sh").asFile.absolutePath

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(project(":kggml-runtime"))
        }
    }
}

android {
    namespace = "com.donfuldauer.kggml.runtime.opencl"
}

val buildNativeAndroid by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Cross-compiles ggml's OpenCL backend module for Android (arm64-v8a, x86_64) into " +
        "<root>/build/native-android-opencl-jniLibs."
    workingDir(rootProject.layout.projectDirectory)
    commandLine("bash", buildAndroidScript, "arm64-v8a", "x86_64")
    // Links libggml-base.so of the same ggml; keep the runtime's (parallel) build first.
    mustRunAfter(":kggml-runtime:buildNativeAndroid")
}

if (bundleNative) {
    android.sourceSets.getByName("main").jniLibs.srcDir(androidJniLibsDir)
    tasks.matching { it.name == "preBuild" }.configureEach { dependsOn(buildNativeAndroid) }
}
