import org.jetbrains.compose.desktop.application.dsl.TargetFormat
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.KotlinSourceSetTree

// Kggml showcase: Compose Multiplatform app (Android, iOS, JVM Desktop) consuming the SDK exclusively through
// the `kggml-bom` platform (no Kggml module versions are repeated here). Samples are never published.
plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.application)
    alias(libs.plugins.compose.multiplatform)
    alias(libs.plugins.kotlin.compose)
}

val jvmTargetVersion: String = libs.versions.jvmTarget.get()
// Samples are not published, so they do not get `project.version`; reuse the SDK version for app metadata.
val sampleVersion: String = providers.gradleProperty("VERSION_NAME").get()
// Built by the kggml-{runtime,whisper,llama,diffusion}/native/scripts/build-ios-xcframework.sh scripts (macOS only).
val iosXcframeworkDir: File = rootProject.layout.projectDirectory.dir("build/native-ios/GgmlRuntime.xcframework").asFile
val iosWhisperXcframeworkDir: File =
    rootProject.layout.projectDirectory.dir("build/native-ios/KggmlWhisper.xcframework").asFile
val iosLlamaXcframeworkDir: File =
    rootProject.layout.projectDirectory.dir("build/native-ios/KggmlLlama.xcframework").asFile
val iosDiffusionXcframeworkDir: File =
    rootProject.layout.projectDirectory.dir("build/native-ios/KggmlDiffusion.xcframework").asFile
val iosEngineXcframeworkDirs: List<File> =
    listOf(iosWhisperXcframeworkDir, iosLlamaXcframeworkDir, iosDiffusionXcframeworkDir, iosXcframeworkDir)

// -Pkggml.requireNative=true: native-backed tests (desktop, iOS, Android device) FAIL instead of skip when the engines'
// native libraries are missing.
val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)

// The engines' synthetic model fixtures (internal test sources of :kggml-whisper / :kggml-llama, MIT, see NOTICE) are
// compiled into this module's tests verbatim, like :integration-tests does, so the real-engine end-to-end UI tests load
// the exact models the engines' own native tests use.
val engineFixtureSources: List<File> = listOf(
    "kggml-whisper/src/commonTest/kotlin/com/donfuldauer/kggml/whisper/TinyWhisperModel.kt",
    "kggml-llama/src/commonTest/kotlin/com/donfuldauer/kggml/llama/TinyLlamaModel.kt",
).map { rootProject.layout.projectDirectory.file(it).asFile }
val syncEngineFixtures by tasks.registering(Sync::class) {
    description = "Copies the engines' synthetic model fixtures (TinyWhisperModel, TinyLlamaModel) into this module's tests."
    from(engineFixtureSources)
    into(layout.buildDirectory.dir("generated/engine-fixtures/kotlin"))
}

kotlin {
    androidTarget {
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
        // commonTest (fake-driven controller + Compose UI tests, real-engine end-to-end tests) runs everywhere:
        // desktop, iOS simulator, Android local unit tests (Robolectric) AND on an Android device / emulator.
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        instrumentedTestVariant.sourceSetTree.set(KotlinSourceSetTree.test)
    }
    jvm("desktop") {
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
    }
    listOf(iosArm64(), iosSimulatorArm64()).forEach { target ->
        target.binaries.framework {
            baseName = "KggmlShowcase"
            // Static: GgmlRuntime.framework and the KggmlWhisper / KggmlLlama / KggmlDiffusion frameworks are
            // embedded and linked by the Xcode host app, never duplicated here.
            isStatic = true
        }
        // Test executables are fully linked, so they need the locally built dynamic frameworks.
        val slice = if (target.name == "iosSimulatorArm64") "ios-arm64-simulator" else "ios-arm64"
        iosEngineXcframeworkDirs.map { it.resolve(slice).absolutePath }.forEach { path ->
            target.binaries.getTest("DEBUG").linkerOpts("-F", path, "-rpath", path)
        }
    }

    compilerOptions { freeCompilerArgs.add("-Xexpect-actual-classes") }

    sourceSets {
        commonMain.dependencies {
            // Kggml SDK: versions come from the BOM only.
            implementation(project.dependencies.platform(project(":kggml-bom")))
            implementation(project(":kggml-runtime"))
            implementation(project(":kggml-whisper"))
            implementation(project(":kggml-llama"))
            implementation(project(":kggml-diffusion"))
            implementation(project(":kggml-model-manager"))

            implementation(compose.runtime)
            implementation(compose.foundation)
            implementation(compose.material3)
            implementation(compose.ui)
            implementation(libs.kotlinx.coroutines.core)
        }
        commonTest {
            kotlin.srcDir(syncEngineFixtures)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
            @OptIn(org.jetbrains.compose.ExperimentalComposeLibrary::class)
            implementation(compose.uiTest)
        }
        androidMain.dependencies {
            implementation(libs.androidx.activity.compose)
        }
        // Robolectric runs the Android variant of commonTest (and the Android-only tests) on the host JVM.
        val androidUnitTest by getting {
            kotlin.srcDir("src/androidSharedTest/kotlin")
            dependencies {
                implementation(libs.robolectric)
                implementation(libs.junit4)
                implementation(libs.androidx.test.core)
                implementation(libs.androidx.test.ext.junit)
                implementation(libs.androidx.compose.ui.test.junit4)
            }
        }
        val androidInstrumentedTest by getting {
            kotlin.srcDir("src/androidSharedTest/kotlin")
            dependencies {
                implementation(libs.junit4)
                implementation(libs.androidx.test.core)
                implementation(libs.androidx.test.runner)
                implementation(libs.androidx.test.rules)
                implementation(libs.androidx.test.ext.junit)
                implementation(libs.androidx.compose.ui.test.junit4)
            }
        }
        val desktopMain by getting {
            dependencies {
                implementation(compose.desktop.currentOs)
                implementation(libs.kotlinx.coroutines.swing)
            }
        }
        val desktopTest by getting {
            dependencies {
                implementation(compose.desktop.currentOs)
                @OptIn(org.jetbrains.compose.ExperimentalComposeLibrary::class)
                implementation(compose.uiTest)
            }
        }
    }
}

android {
    namespace = "com.donfuldauer.kggml.samples.showcase"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    defaultConfig {
        applicationId = "com.donfuldauer.kggml.samples.showcase"
        minSdk = libs.versions.android.minSdk.get().toInt()
        targetSdk = libs.versions.android.compileSdk.get().toInt()
        versionCode = 1
        versionName = sampleVersion
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Read by the instrumented tests (InstrumentationRegistry arguments).
        testInstrumentationRunnerArguments["kggml.requireNative"] = requireNative.toString()
    }
    testOptions {
        unitTests.isIncludeAndroidResources = true
        unitTests.all { it.systemProperty("kggml.test.requireNative", "false") }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(jvmTargetVersion)
        targetCompatibility = JavaVersion.toVersion(jvmTargetVersion)
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

// Local unit tests (Robolectric) run on the debug variant only: the release variant has the same code but not the
// test activity from ui-test-manifest, which must stay out of release APKs.
androidComponents {
    beforeVariants(selector().withBuildType("release")) { variant ->
        (variant as com.android.build.api.variant.HasUnitTestBuilder).enableUnitTest = false
    }
}

dependencies {
    // Declares the empty ComponentActivity that runComposeUiTest / createComposeRule host content in (debug only,
    // never in release APKs).
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}

// Host runtime / engines built by build-host.sh: shared libraries land in build/native(-whisper|-llama|-diffusion)/lib,
// except Windows DLLs (bin).
val hostLibSubdir: String = if (System.getProperty("os.name").lowercase().startsWith("windows")) "bin" else "lib"
val hostNativeLibDir: File = rootProject.layout.projectDirectory.dir("build/native/$hostLibSubdir").asFile
val hostWhisperLibDir: File = rootProject.layout.projectDirectory.dir("build/native-whisper/$hostLibSubdir").asFile
val hostLlamaLibDir: File = rootProject.layout.projectDirectory.dir("build/native-llama/$hostLibSubdir").asFile
val hostDiffusionLibDir: File = rootProject.layout.projectDirectory.dir("build/native-diffusion/$hostLibSubdir").asFile

compose.desktop {
    application {
        mainClass = "com.donfuldauer.kggml.samples.showcase.MainKt"
        // During development, point the JVM loader at a locally built host runtime (build-host.sh) if present.
        // Release desktop builds instead bundle it into the :kggml-runtime jar (-Pkggml.bundleNative=true).
        if (hostNativeLibDir.isDirectory) jvmArgs += "-Dkggml.native.path=${hostNativeLibDir.absolutePath}"
        if (hostWhisperLibDir.isDirectory) jvmArgs += "-Dkggml.whisper.native.path=${hostWhisperLibDir.absolutePath}"
        if (hostLlamaLibDir.isDirectory) jvmArgs += "-Dkggml.llama.native.path=${hostLlamaLibDir.absolutePath}"
        if (hostDiffusionLibDir.isDirectory) jvmArgs += "-Dkggml.diffusion.native.path=${hostDiffusionLibDir.absolutePath}"
        // Optional default model for the Whisper Studio: ./gradlew :samples:kggml-showcase:run -Pkggml.whisper.model=...
        providers.gradleProperty("kggml.whisper.model").orNull?.let { jvmArgs += "-Dkggml.whisper.model=$it" }
        // Optional model directory of the Model Manager (default ~/.kggml/models): -Pkggml.models.dir=...
        providers.gradleProperty("kggml.models.dir").orNull?.let { jvmArgs += "-Dkggml.models.dir=$it" }
        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "KggmlShowcase"
            packageVersion = sampleVersion
        }
    }
}

// Native-backed desktop tests (KggmlRuntimeInspectorDesktopTest and the real-engine end-to-end UI tests in commonTest)
// use the host runtime / engines built by kggml-{runtime,whisper,llama,diffusion}/native/scripts/build-host.sh when
// present; otherwise they assert the graceful "runtime unavailable" state or skip. -Pkggml.requireNative=true makes the
// native path mandatory.
tasks.named<Test>("desktopTest") {
    val hostLibs = mapOf(
        "kggml.native.path" to (hostNativeLibDir to "kggml_runtime_jni"),
        "kggml.whisper.native.path" to (hostWhisperLibDir to "kggml_whisper_jni"),
        "kggml.llama.native.path" to (hostLlamaLibDir to "kggml_llama_jni"),
        "kggml.diffusion.native.path" to (hostDiffusionLibDir to "kggml_diffusion_jni"),
    )
    systemProperty("kggml.test.requireNative", requireNative.toString())
    inputs.property("kggml.requireNative", requireNative)
    doFirst {
        hostLibs.forEach { (property, lib) ->
            val (dir, jni) = lib
            if (dir.resolve(System.mapLibraryName(jni)).isFile) systemProperty(property, dir.absolutePath)
        }
    }
}

// iOS test executables link against GgmlRuntime.xcframework and the KggmlWhisper / KggmlLlama / KggmlDiffusion
// xcframeworks; skip them (like the engine modules) until all of them are built.
tasks.matching { it.name.startsWith("linkDebugTestIos") || it.name.matches(Regex("ios.*Test")) }.configureEach {
    onlyIf("GgmlRuntime, KggmlWhisper, KggmlLlama and KggmlDiffusion xcframeworks are built (native/scripts/build-ios-xcframework.sh)") {
        iosEngineXcframeworkDirs.all { it.isDirectory }
    }
}
