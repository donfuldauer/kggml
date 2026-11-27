import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.konan.target.KonanTarget

plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml shared dynamic ggml runtime (CPU, Metal, Vulkan backends)."

// ---------------------------------------------------------------------------------------------------------
// Opt-in switches (Gradle properties; all default to false so `./gradlew check` needs no C/C++ toolchain):
//   -Pkggml.requireNative=true  build the host native runtime first and FAIL (instead of skip) native-backed
//                               jvmTest / iOS test cases when the libraries are missing.
//   -Pkggml.bundleNative=true   build release (no test hooks) native libraries and package them: host
//                               libraries into the JVM jar (/kggml/native/<platform>/) and Android ABIs
//                               (arm64-v8a, x86_64) into the AAR jniLibs.
//   -Pkggml.extraJvmNativeResources=<dir>[,<dir>...]  (with bundleNative) also package desktop runtimes staged on
//                               other hosts: each <dir> holds kggml/native/<platform>/ as written by
//                               stageJvmNativeResources there. The release workflow uses this to ship linux-x86_64,
//                               windows-x86_64 and macos-aarch64 in one jar.
//   -Pkggml.nativePlatform=<os>-<arch>  stage (and, for linux-aarch64, cross-compile) the release runtime for that JVM
//                               platform instead of the host's: linux-aarch64 is built by build-host.sh --target;
//                               windows-aarch64 is built beforehand with the windows-arm64 preset (needs an MSVC
//                               arm64 environment) and only staged here, like windows-x86_64 in the release.
// ---------------------------------------------------------------------------------------------------------

val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)
val bundleNative = providers.gradleProperty("kggml.bundleNative").map(String::toBoolean).getOrElse(false)
val nativePlatform: String? = providers.gradleProperty("kggml.nativePlatform").orNull?.trim()?.takeIf { it.isNotEmpty() }

val nativeDir = layout.projectDirectory.dir("native")
val rootBuildDir = rootProject.layout.projectDirectory.dir("build")
val nativeBuildDir = rootBuildDir.dir("native")
// CMake puts shared libraries in lib/ on Linux/macOS but DLLs (RUNTIME artifacts) in bin/ on Windows.
val isWindowsHost = System.getProperty("os.name").lowercase().startsWith("windows")
val hostNativeLibDir = nativeBuildDir.dir(if (isWindowsHost) "bin" else "lib")
val nativeReleaseBuildDir = rootBuildDir.dir("native-release")
val nativeReleaseInstallDir = rootBuildDir.dir("native-release-install")
val androidJniLibsDir = rootBuildDir.dir("native-android-jniLibs")
val iosXcframeworkDir = rootBuildDir.dir("native-ios/GgmlRuntime.xcframework")
val buildHostScript = nativeDir.file("scripts/build-host.sh").asFile.absolutePath
val buildAndroidScript = nativeDir.file("scripts/build-android.sh").asFile.absolutePath

/** Pinned ggml version / commit, read from the single source of truth in native/CMakeLists.txt. */
val nativeCmakeText = nativeDir.file("CMakeLists.txt").asFile.readText()
fun cmakePin(name: String): String =
    Regex("""set\($name "([^"]+)"\)""").find(nativeCmakeText)?.groupValues?.get(1)
        ?: error("$name not found in kggml-runtime/native/CMakeLists.txt")
val ggmlPinnedVersion = cmakePin("KGGML_GGML_PINNED_VERSION")
val ggmlPinnedCommit = cmakePin("KGGML_GGML_PINNED_COMMIT")

kotlin {
    // Kotlin/Native macOS (issue #93): links the macos-arm64 slice of GgmlRuntime.xcframework (Metal + Accelerate).
    macosArm64()

    // `jniMain` is shared by androidMain + jvmMain (the JNI bridge); appleMain (iOS + macOS) uses cinterop instead.
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
        }
        commonMain.dependencies {
            // ComputeBackend / InferenceException are part of this module's public API.
            api(project(":kggml-core-api"))
        }
    }

    targets.withType<KotlinNativeTarget>().configureEach {
        val slice = when (konanTarget) {
            KonanTarget.IOS_SIMULATOR_ARM64 -> "ios-arm64-simulator"
            KonanTarget.MACOS_ARM64 -> "macos-arm64"
            else -> "ios-arm64"
        }
        compilations.getByName("main").cinterops.create("kggmlRuntime") {
            definitionFile.set(project.file("src/nativeInterop/cinterop/kggmlRuntime.def"))
            includeDirs(nativeDir.dir("include").asFile)
        }
        // Test executables link against (and find at run time) the locally built dynamic framework.
        val frameworkSearchPath = iosXcframeworkDir.dir(slice).asFile.absolutePath
        binaries.all {
            linkerOpts("-F", frameworkSearchPath, "-rpath", frameworkSearchPath)
        }
    }
}

android {
    namespace = "com.donfuldauer.kggml.runtime"
}

// ---------------------------------------------------------------------------------------------------------
// Native (CMake) layer: kggml-runtime/native. Opt-in helper tasks only; deliberately NOT wired into `check`
// or `assemble` unless one of the properties above is set (needs CMake + a C/C++ toolchain + the ggml
// submodule).
// ---------------------------------------------------------------------------------------------------------

val buildNativeHost by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Configures and builds kggml-runtime/native (ggml + kggml_runtime + JNI + tests) into <root>/build/native."
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

val buildNativeHostRelease by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Builds and installs a release (no tests, no test hooks) host runtime into <root>/build/native-release-install."
    workingDir(rootProject.layout.projectDirectory)
    commandLine(
        listOf(
            "bash", buildHostScript, "--no-tests",
            "--build-dir", nativeReleaseBuildDir.asFile.absolutePath,
            "--install", nativeReleaseInstallDir.asFile.absolutePath,
        ) + listOfNotNull(nativePlatform?.let { "--target" }, nativePlatform),
    )
}

val buildNativeAndroid by tasks.registering(Exec::class) {
    group = "kggml native"
    description = "Cross-compiles the runtime for Android (arm64-v8a, x86_64) into <root>/build/native-android-jniLibs."
    workingDir(rootProject.layout.projectDirectory)
    commandLine("bash", buildAndroidScript, "arm64-v8a", "x86_64")
}

/** Host platform id; must match JvmNativeLibraryLoader.platformId(). */
fun hostPlatformId(): String {
    val os = System.getProperty("os.name").lowercase()
    val osPart = when {
        os.startsWith("mac") -> "macos"
        os.startsWith("windows") -> "windows"
        else -> "linux"
    }
    val archPart = when (System.getProperty("os.arch").lowercase()) {
        "amd64", "x86_64", "x64" -> "x86_64"
        "aarch64", "arm64" -> "aarch64"
        else -> System.getProperty("os.arch").lowercase()
    }
    return "$osPart-$archPart"
}

/** Library base names in load order; must match JvmNativeLibraryLoader.LOAD_ORDER. */
val nativeLoadOrder = listOf(
    "ggml-base", "ggml-cpu", "ggml-blas", "ggml-metal", "ggml-vulkan", "ggml", "kggml_runtime", "kggml_runtime_jni",
)

val jvmNativeResourcesDir = layout.buildDirectory.dir("generated/kggml-native/jvm")

val stageJvmNativeResources by tasks.registering {
    group = "kggml native"
    description = "Copies the release host runtime into JVM resources (/kggml/native/<platform>/) with a load-order index."
    dependsOn(buildNativeHostRelease)
    val installDir = nativeReleaseInstallDir.asFile
    val outRoot = jvmNativeResourcesDir
    outputs.dir(outRoot)
    doLast {
        val platform = nativePlatform ?: hostPlatformId()
        val out = outRoot.get().asFile.resolve("kggml/native/$platform")
        out.deleteRecursively()
        out.mkdirs()
        val candidates = listOf(installDir.resolve("lib"), installDir.resolve("bin"))
            .filter { it.isDirectory }
            .flatMap { dir -> dir.listFiles().orEmpty().filter { it.isFile } }
        // Copy each library under the name its dependents reference (Linux SONAME libX.so.N, macOS
        // install name libX.N.dylib, Windows X.dll) so sibling lookups resolve inside the extraction dir.
        val staged = nativeLoadOrder.mapNotNull { name ->
            val q = Regex.escape(name)
            val patterns = listOf(
                Regex("""^lib$q\.so\.\d+$"""), Regex("""^lib$q\.so$"""),
                Regex("""^lib$q\.\d+\.dylib$"""), Regex("""^lib$q\.dylib$"""),
                Regex("""^$q\.dll$"""),
            )
            patterns.firstNotNullOfOrNull { p -> candidates.firstOrNull { p.matches(it.name) } }
        }
        val names = staged.map { it.name }
        check(names.any { it.contains("kggml_runtime_jni") }) { "kggml_runtime_jni not found under $installDir" }
        staged.forEach { it.copyTo(out.resolve(it.name), overwrite = true) }
        out.resolve("index.txt").writeText(
            "# Kggml native runtime for $platform, in load order (generated by :kggml-runtime:stageJvmNativeResources)\n" +
                names.joinToString(separator = "\n", postfix = "\n"),
        )
    }
}

val extraJvmNativeResources: List<File> = providers.gradleProperty("kggml.extraJvmNativeResources").orNull
    ?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() }?.map { file(it) }.orEmpty()

if (bundleNative) {
    kotlin.sourceSets.getByName("jvmMain").resources.srcDir(stageJvmNativeResources)
    extraJvmNativeResources.forEach { dir ->
        val platforms = dir.resolve("kggml/native").listFiles()?.filter { it.resolve("index.txt").isFile }.orEmpty()
        require(platforms.isNotEmpty()) { "kggml.extraJvmNativeResources: no kggml/native/<platform>/index.txt under $dir" }
        kotlin.sourceSets.getByName("jvmMain").resources.srcDir(dir)
    }
    android.sourceSets.getByName("main").jniLibs.srcDir(androidJniLibsDir)
    tasks.matching { it.name == "preBuild" }.configureEach { dependsOn(buildNativeAndroid) }
}

// ---------------------------------------------------------------------------------------------------------
// Native-backed tests
// ---------------------------------------------------------------------------------------------------------

tasks.named<Test>("jvmTest") {
    if (requireNative) dependsOn(buildNativeHost)
    val libDir = hostNativeLibDir.asFile
    val jniLibraryName = System.mapLibraryName("kggml_runtime_jni")
    systemProperty("kggml.test.requireNative", requireNative.toString())
    systemProperty("kggml.test.expectedRuntimeVersion", project.version.toString())
    systemProperty("kggml.test.expectedGgmlVersion", ggmlPinnedVersion)
    systemProperty("kggml.test.expectedGgmlCommit", ggmlPinnedCommit)
    inputs.property("kggml.requireNative", requireNative)
    // Issue #92: -Pkggml.test.vulkanAddonDir=<dir holding the desktop ggml-vulkan module> registers it and expects
    // a usable Vulkan device (CI: Mesa lavapipe, which ggml only uses when GGML_VK_VISIBLE_DEVICES selects it).
    // Relative paths are from the repository root (where CI runs ./gradlew), not from this module.
    providers.gradleProperty("kggml.test.vulkanAddonDir").orNull?.let { arg -> rootProject.file(arg) }?.let { dir ->
        systemProperty("kggml.native.addons.path", dir.absolutePath)
        systemProperty("kggml.test.expectVulkanAddon", "true")
        environment("GGML_VK_VISIBLE_DEVICES", providers.gradleProperty("kggml.test.vulkanDevices").getOrElse("0"))
        inputs.files(fileTree(dir)).withPropertyName("vulkanAddonDir").withPathSensitivity(PathSensitivity.RELATIVE)
    }
    // The native-backed tests' results depend on the host libraries they load: track them, so a native-only
    // rebuild reruns the tests and a cached result is never reused for different (or missing) native builds.
    inputs.files(fileTree(libDir)).withPropertyName("nativeLibs.libDir").withPathSensitivity(PathSensitivity.RELATIVE)
    doFirst {
        // Point the loader at the host build tree only when it actually contains the JNI bridge; otherwise
        // native-backed tests are skipped (or fail when -Pkggml.requireNative=true).
        if (libDir.resolve(jniLibraryName).isFile) systemProperty("kggml.native.path", libDir.absolutePath)
    }
}

// Apple test executables link against GgmlRuntime.xcframework (build-ios-xcframework.sh, macOS only).
val iosFrameworkPresent = iosXcframeworkDir.asFile.isDirectory
tasks.matching {
    it.name.startsWith("linkDebugTestIos") || it.name.startsWith("linkDebugTestMacos") ||
        it.name.matches(Regex("(ios|macos).*Test"))
}.configureEach {
    onlyIf("GgmlRuntime.xcframework is built (kggml-runtime/native/scripts/build-ios-xcframework.sh)") {
        if (!iosFrameworkPresent && requireNative) {
            throw GradleException("-Pkggml.requireNative=true but ${iosXcframeworkDir.asFile} does not exist")
        }
        iosFrameworkPresent
    }
}

// The same pinned versions for the on-device runtime test (AndroidNativeRuntimeTest, InstrumentationRegistry arguments).
android.defaultConfig.testInstrumentationRunnerArguments.putAll(
    mapOf(
        "kggml.expectedRuntimeVersion" to project.version.toString(),
        "kggml.expectedGgmlVersion" to ggmlPinnedVersion,
        "kggml.expectedGgmlCommit" to ggmlPinnedCommit,
    ),
)
