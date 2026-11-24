plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml shared dynamic ggml runtime (CPU, Metal, Vulkan backends)."

android {
    namespace = "com.donfuldauer.kggml.runtime"
}

// ---------------------------------------------------------------------------------------------------------
// Native (CMake) layer: kggml-runtime/native. Opt-in helper tasks only; deliberately NOT wired into `check`
// or `assemble` (needs CMake + a C/C++ toolchain + the ggml submodule). Kotlin bindings land in issue #4.
// ---------------------------------------------------------------------------------------------------------

val nativeDir = layout.projectDirectory.dir("native")
val nativeBuildDir = rootProject.layout.projectDirectory.dir("build/native")
val buildHostScript = nativeDir.file("scripts/build-host.sh").asFile.absolutePath

val buildNativeHost by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Configures and builds kggml-runtime/native (ggml + kggml_runtime + tests) into <root>/build/native."
    workingDir(rootProject.layout.projectDirectory)
    commandLine("bash", buildHostScript, "--skip-ctest")
}

val nativeCtest by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Runs the kggml-runtime native CTest suite (ctest --test-dir build/native --output-on-failure)."
    dependsOn(buildNativeHost)
    workingDir(rootProject.layout.projectDirectory)
    commandLine("ctest", "--test-dir", nativeBuildDir.asFile.absolutePath, "--output-on-failure")
}
