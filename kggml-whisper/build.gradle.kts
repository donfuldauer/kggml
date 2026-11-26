import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.konan.target.KonanTarget

plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml whisper.cpp speech-to-text engine."

// ---------------------------------------------------------------------------------------------------------
// Opt-in switches (same as :kggml-runtime; all default to false so `./gradlew check` needs no C/C++ toolchain):
//   -Pkggml.requireNative=true  build the host native runtime + whisper first and FAIL (instead of skip)
//                               native-backed jvmTest / iOS test cases when the libraries are missing.
//   -Pkggml.bundleNative=true   package the Android ABIs (arm64-v8a, x86_64) of libkggml_whisper(+_jni) into the
//                               AAR jniLibs.
// ---------------------------------------------------------------------------------------------------------

val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)
val bundleNative = providers.gradleProperty("kggml.bundleNative").map(String::toBoolean).getOrElse(false)

val nativeDir = layout.projectDirectory.dir("native")
val rootBuildDir = rootProject.layout.projectDirectory.dir("build")
val isWindowsHost = System.getProperty("os.name").lowercase().startsWith("windows")
val hostLibSubdir = if (isWindowsHost) "bin" else "lib"
val runtimeHostLibDir = rootBuildDir.dir("native/$hostLibSubdir")
val whisperHostLibDir = rootBuildDir.dir("native-whisper/$hostLibSubdir")
val androidJniLibsDir = rootBuildDir.dir("native-android-whisper-jniLibs")
val iosRuntimeXcframeworkDir = rootBuildDir.dir("native-ios/GgmlRuntime.xcframework")
val iosWhisperXcframeworkDir = rootBuildDir.dir("native-ios/KggmlWhisper.xcframework")
val buildHostScript = nativeDir.file("scripts/build-host.sh").asFile.absolutePath
val buildAndroidScript = nativeDir.file("scripts/build-android.sh").asFile.absolutePath

kotlin {
    // `jniMain` is shared by androidMain + jvmMain (the JNI bridge); iosMain uses cinterop instead.
    @OptIn(ExperimentalKotlinGradlePluginApi::class)
    applyDefaultHierarchyTemplate {
        common {
            group("jni") {
                withAndroidTarget()
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
            // BackendConfig / InferenceEngine / InferenceException (via :kggml-runtime's api) are public API.
            api(project(":kggml-runtime"))
            api(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
        }
    }

    targets.withType<KotlinNativeTarget>().configureEach {
        val slice = when (konanTarget) {
            KonanTarget.IOS_SIMULATOR_ARM64 -> "ios-arm64-simulator"
            else -> "ios-arm64"
        }
        compilations.getByName("main").cinterops.create("kggmlWhisper") {
            definitionFile.set(project.file("src/nativeInterop/cinterop/kggmlWhisper.def"))
            includeDirs(
                nativeDir.dir("include").asFile,
                rootProject.layout.projectDirectory.dir("kggml-runtime/native/include").asFile,
            )
        }
        // Test executables link against (and find at run time) the locally built dynamic frameworks.
        val searchPaths = listOf(iosWhisperXcframeworkDir, iosRuntimeXcframeworkDir).map { it.dir(slice).asFile.absolutePath }
        binaries.all {
            searchPaths.forEach { linkerOpts("-F", it, "-rpath", it) }
        }
    }
}

android {
    namespace = "com.donfuldauer.kggml.whisper"
    // Keeps the JNI entry points and the streaming segment-listener callback (called by name from C++).
    defaultConfig { consumerProguardFiles("consumer-rules.pro") }
}

// ---------------------------------------------------------------------------------------------------------
// Native (CMake) layer: kggml-whisper/native. Opt-in helper tasks, not wired into `check` / `assemble` unless
// one of the properties above is set.
// ---------------------------------------------------------------------------------------------------------

val buildNativeHost by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Builds kggml-whisper/native (whisper.cpp + kggml_whisper + JNI + tests) into <root>/build/native-whisper."
    workingDir(rootProject.layout.projectDirectory)
    commandLine("bash", buildHostScript, "--skip-ctest")
    // The engine script builds the shared runtime first when it is missing; run the runtime task before it so
    // parallel Gradle workers never configure the same runtime build directory at once.
    dependsOn(":kggml-runtime:buildNativeHost")
}

val nativeCtest by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Runs the kggml-whisper native CTest suite (ctest --test-dir build/native-whisper --output-on-failure)."
    dependsOn(buildNativeHost)
    workingDir(rootProject.layout.projectDirectory)
    commandLine("ctest", "--test-dir", rootBuildDir.dir("native-whisper").asFile.absolutePath, "--output-on-failure")
}

val buildNativeAndroid by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Cross-compiles whisper for Android (arm64-v8a, x86_64) into <root>/build/native-android-whisper-jniLibs."
    workingDir(rootProject.layout.projectDirectory)
    commandLine("bash", buildAndroidScript, "arm64-v8a", "x86_64")
    // The engine script builds the shared runtime first when it is missing; run the runtime task before it so
    // parallel Gradle workers never configure the same runtime build directory at once.
    dependsOn(":kggml-runtime:buildNativeAndroid")
}

if (bundleNative) {
    android.sourceSets.getByName("main").jniLibs.srcDir(androidJniLibsDir)
    tasks.matching { it.name == "preBuild" }.configureEach { dependsOn(buildNativeAndroid) }
}

// ---------------------------------------------------------------------------------------------------------
// Native-backed tests
// ---------------------------------------------------------------------------------------------------------

tasks.named<Test>("jvmTest") {
    if (requireNative) dependsOn(buildNativeHost)
    val runtimeDir = runtimeHostLibDir.asFile
    val whisperDir = whisperHostLibDir.asFile
    val runtimeJni = System.mapLibraryName("kggml_runtime_jni")
    val whisperJni = System.mapLibraryName("kggml_whisper_jni")
    systemProperty("kggml.test.requireNative", requireNative.toString())
    inputs.property("kggml.requireNative", requireNative)
    // The native-backed tests' results depend on the host libraries they load: track them, so a native-only
    // rebuild reruns the tests and a cached result is never reused for different (or missing) native builds.
    inputs.files(fileTree(runtimeDir)).withPropertyName("nativeLibs.runtimeDir").withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.files(fileTree(whisperDir)).withPropertyName("nativeLibs.whisperDir").withPathSensitivity(PathSensitivity.RELATIVE)
    doFirst {
        // Point the loaders at the host build trees only when they contain the JNI bridges; otherwise
        // native-backed tests are skipped (or fail when -Pkggml.requireNative=true).
        if (runtimeDir.resolve(runtimeJni).isFile && whisperDir.resolve(whisperJni).isFile) {
            systemProperty("kggml.native.path", runtimeDir.absolutePath)
            systemProperty("kggml.whisper.native.path", whisperDir.absolutePath)
        }
    }
}

// iOS test executables link against KggmlWhisper.xcframework + GgmlRuntime.xcframework (macOS only).
val iosFrameworksPresent = iosWhisperXcframeworkDir.asFile.isDirectory && iosRuntimeXcframeworkDir.asFile.isDirectory
tasks.matching { it.name.startsWith("linkDebugTestIos") || it.name.matches(Regex("ios.*Test")) }.configureEach {
    onlyIf("KggmlWhisper.xcframework is built (kggml-whisper/native/scripts/build-ios-xcframework.sh)") {
        if (!iosFrameworksPresent && requireNative) {
            throw GradleException("-Pkggml.requireNative=true but ${iosWhisperXcframeworkDir.asFile} does not exist")
        }
        iosFrameworksPresent
    }
}
