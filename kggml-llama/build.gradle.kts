import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.konan.target.KonanTarget

plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml llama.cpp + libmtmd text and vision-language engine."

// ---------------------------------------------------------------------------------------------------------
// Opt-in switches (same as :kggml-runtime; all default to false so `./gradlew check` needs no C/C++ toolchain):
//   -Pkggml.requireNative=true  build the host native runtime + llama first and FAIL (instead of skip)
//                               native-backed jvmTest / iOS test cases when the libraries are missing.
//   -Pkggml.bundleNative=true   package the Android ABIs (arm64-v8a, x86_64) of libkggml_llama(+_jni) into the
//                               AAR jniLibs.
// ---------------------------------------------------------------------------------------------------------

val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)
val bundleNative = providers.gradleProperty("kggml.bundleNative").map(String::toBoolean).getOrElse(false)

val nativeDir = layout.projectDirectory.dir("native")
val rootBuildDir = rootProject.layout.projectDirectory.dir("build")
val isWindowsHost = System.getProperty("os.name").lowercase().startsWith("windows")
val hostLibSubdir = if (isWindowsHost) "bin" else "lib"
val runtimeHostLibDir = rootBuildDir.dir("native/$hostLibSubdir")
val llamaHostLibDir = rootBuildDir.dir("native-llama/$hostLibSubdir")
val androidJniLibsDir = rootBuildDir.dir("native-android-llama-jniLibs")
val iosRuntimeXcframeworkDir = rootBuildDir.dir("native-ios/GgmlRuntime.xcframework")
val iosLlamaXcframeworkDir = rootBuildDir.dir("native-ios/KggmlLlama.xcframework")
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
        compilations.getByName("main").cinterops.create("kggmlLlama") {
            definitionFile.set(project.file("src/nativeInterop/cinterop/kggmlLlama.def"))
            includeDirs(
                nativeDir.dir("include").asFile,
                rootProject.layout.projectDirectory.dir("kggml-runtime/native/include").asFile,
            )
        }
        // Test executables link against (and find at run time) the locally built dynamic frameworks.
        val searchPaths = listOf(iosLlamaXcframeworkDir, iosRuntimeXcframeworkDir).map { it.dir(slice).asFile.absolutePath }
        binaries.all {
            searchPaths.forEach { linkerOpts("-F", it, "-rpath", it) }
        }
    }
}

android {
    namespace = "com.donfuldauer.kggml.llama"
    // Keeps the JNI entry points (bound by symbol name from C++).
    defaultConfig { consumerProguardFiles("consumer-rules.pro") }
}

// ---------------------------------------------------------------------------------------------------------
// Native (CMake) layer: kggml-llama/native. Opt-in helper tasks, not wired into `check` / `assemble` unless
// one of the properties above is set.
// ---------------------------------------------------------------------------------------------------------

val buildNativeHost by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Builds kggml-llama/native (llama.cpp + libmtmd + kggml_llama + JNI + tests) into <root>/build/native-llama."
    workingDir(rootProject.layout.projectDirectory)
    commandLine("bash", buildHostScript, "--skip-ctest")
    // The engine script builds the shared runtime first when it is missing; run the runtime task before it so
    // parallel Gradle workers never configure the same runtime build directory at once.
    dependsOn(":kggml-runtime:buildNativeHost")
}

val nativeCtest by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Runs the kggml-llama native CTest suite (ctest --test-dir build/native-llama --output-on-failure)."
    dependsOn(buildNativeHost)
    workingDir(rootProject.layout.projectDirectory)
    commandLine("ctest", "--test-dir", rootBuildDir.dir("native-llama").asFile.absolutePath, "--output-on-failure")
}

val buildNativeAndroid by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Cross-compiles llama for Android (arm64-v8a, x86_64) into <root>/build/native-android-llama-jniLibs."
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
    val llamaDir = llamaHostLibDir.asFile
    val runtimeJni = System.mapLibraryName("kggml_runtime_jni")
    val llamaJni = System.mapLibraryName("kggml_llama_jni")
    systemProperty("kggml.test.requireNative", requireNative.toString())
    inputs.property("kggml.requireNative", requireNative)
    // The native-backed tests' results depend on the host libraries they load: track them, so a native-only
    // rebuild reruns the tests and a cached result is never reused for different (or missing) native builds.
    inputs.files(fileTree(runtimeDir)).withPropertyName("nativeLibs.runtimeDir").withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.files(fileTree(llamaDir)).withPropertyName("nativeLibs.llamaDir").withPathSensitivity(PathSensitivity.RELATIVE)
    doFirst {
        // Point the loaders at the host build trees only when they contain the JNI bridges; otherwise
        // native-backed tests are skipped (or fail when -Pkggml.requireNative=true).
        if (runtimeDir.resolve(runtimeJni).isFile && llamaDir.resolve(llamaJni).isFile) {
            systemProperty("kggml.native.path", runtimeDir.absolutePath)
            systemProperty("kggml.llama.native.path", llamaDir.absolutePath)
        }
    }
}

// iOS test executables link against KggmlLlama.xcframework + GgmlRuntime.xcframework (macOS only).
val iosFrameworksPresent = iosLlamaXcframeworkDir.asFile.isDirectory && iosRuntimeXcframeworkDir.asFile.isDirectory
tasks.matching { it.name.startsWith("linkDebugTestIos") || it.name.matches(Regex("ios.*Test")) }.configureEach {
    onlyIf("KggmlLlama.xcframework is built (kggml-llama/native/scripts/build-ios-xcframework.sh)") {
        if (!iosFrameworksPresent && requireNative) {
            throw GradleException("-Pkggml.requireNative=true but ${iosLlamaXcframeworkDir.asFile} does not exist")
        }
        iosFrameworksPresent
    }
}
