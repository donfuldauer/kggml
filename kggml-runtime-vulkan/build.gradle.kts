import com.donfuldauer.kggml.buildlogic.wireAndroidJniLibs

plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml GPU add-on: ggml's Vulkan backend for the shared :kggml-runtime on Android and JVM desktop " +
    "Linux x86_64 (no-op on other platforms)."

// ---------------------------------------------------------------------------------------------------------
// Opt-in switch (like the engine modules; `./gradlew check` needs no C/C++ toolchain):
//   -Pkggml.bundleNative=true   build and package libggml-vulkan.so (arm64-v8a, x86_64) into the AAR jniLibs, and the
//                               host's desktop module (Linux x86_64; Windows: prebuilt, see build-host.sh) into the jar.
//   -Pkggml.extraJvmAddonResources=<dir>[,<dir>...]  (with bundleNative) also package desktop modules staged on other
//                               hosts: each <dir> holds kggml/native-addons/<platform>/ggml-vulkan/ (the release workflow).
// The :kggml-runtime loaders find the module (Android: in the app; JVM: in this jar under
// /kggml/native-addons/<platform>/ggml-vulkan/) and register it (kggml_runtime_load_backend), so
// ComputeBackend.GPU_VULKAN / AUTO can use the GPU; without a usable Vulkan driver everything stays on the CPU.
// macOS and iOS use Metal.
// ---------------------------------------------------------------------------------------------------------

val bundleNative = providers.gradleProperty("kggml.bundleNative").map(String::toBoolean).getOrElse(false)

val nativeDir = layout.projectDirectory.dir("native")
val rootBuildDir = rootProject.layout.projectDirectory.dir("build")
val androidJniLibsDir = rootBuildDir.dir("native-android-vulkan-jniLibs")
val buildAndroidScript = nativeDir.file("scripts/build-android.sh").asFile.absolutePath
val buildHostScript = nativeDir.file("scripts/build-host.sh").asFile.absolutePath
val jvmAddonResourcesDir = rootBuildDir.dir("native-vulkan-jvm")
val hostOs = System.getProperty("os.name").lowercase()
val desktopModuleHost = hostOs.startsWith("linux") || hostOs.startsWith("windows")

kotlin {
    android {
        namespace = "com.donfuldauer.kggml.runtime.vulkan"
    }
    sourceSets {
        commonMain.dependencies {
            api(project(":kggml-runtime"))
        }
    }
}

val buildNativeAndroid = tasks.register<Exec>("buildNativeAndroid") {
    group = "kggml native"
    description = "Cross-compiles ggml's Vulkan backend module for Android (arm64-v8a, x86_64) into " +
        "<root>/build/native-android-vulkan-jniLibs."
    workingDir(rootProject.layout.projectDirectory)
    commandLine("bash", buildAndroidScript, "arm64-v8a", "x86_64")
    // Links libggml-base.so of the same ggml; keep the runtime's (parallel) build first.
    mustRunAfter(":kggml-runtime:buildNativeAndroid")
}

val buildNativeHost = tasks.register<Exec>("buildNativeHost") {
    group = "kggml native"
    description = "Builds ggml's Vulkan backend module for this desktop host (Linux; on Windows it only stages a " +
        "prebuilt windows-x64 preset build) into <root>/build/native-vulkan-jvm."
    workingDir(rootProject.layout.projectDirectory)
    val windows = hostOs.startsWith("windows")
    commandLine(listOf("bash", buildHostScript) + if (windows) listOf("--stage-only") else emptyList())
    outputs.dir(jvmAddonResourcesDir)
    mustRunAfter(":kggml-runtime:buildNativeHost", ":kggml-runtime:buildNativeHostRelease")
}

val extraJvmAddonResources: List<File> = providers.gradleProperty("kggml.extraJvmAddonResources").orNull
    ?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() }?.map { file(it) }.orEmpty()

if (bundleNative) {
    wireAndroidJniLibs(buildNativeAndroid, androidJniLibsDir)
    val jvmResources = kotlin.sourceSets.getByName("jvmMain").resources
    if (desktopModuleHost) jvmResources.srcDir(buildNativeHost)
    extraJvmAddonResources.forEach { dir ->
        require(dir.resolve("kggml/native-addons").isDirectory) {
            "kggml.extraJvmAddonResources: no kggml/native-addons/ under $dir"
        }
        jvmResources.srcDir(dir)
    }
}
