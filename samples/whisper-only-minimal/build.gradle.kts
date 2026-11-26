import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.KotlinSourceSetTree

// Minimal Android + iOS sample used for CI binary symbol leakage audits (AGENTS.md §2.1): it depends ONLY on the
// `kggml-bom` platform and `:kggml-whisper` (which brings `:kggml-runtime`). It must never depend on
// `:kggml-llama` or `:kggml-diffusion`, and uses no UI toolkit beyond the platform's own (android.widget /
// UIKit), so its binaries contain nothing but the app, the runtime and whisper. Never published.
//
// Binary audit (scripts/audit-binary-symbols.sh, run by CI):
//   ./gradlew :samples:whisper-only-minimal:assembleRelease -Pkggml.bundleNative=true
//   ./scripts/audit-binary-symbols.sh apk samples/whisper-only-minimal/build/outputs/apk/release/whisper-only-minimal-release-unsigned.apk
plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.application)
}

val jvmTargetVersion: String = libs.versions.jvmTarget.get()
// Samples are not published, so they do not get `project.version`; reuse the SDK version for app metadata.
val sampleVersion: String = providers.gradleProperty("VERSION_NAME").get()
// Built by kggml-runtime/native/scripts/build-ios-xcframework.sh and kggml-whisper/native/scripts/build-ios-xcframework.sh
// (macOS only).
val iosXcframeworkDir: File = rootProject.layout.projectDirectory.dir("build/native-ios/GgmlRuntime.xcframework").asFile
val iosWhisperXcframeworkDir: File =
    rootProject.layout.projectDirectory.dir("build/native-ios/KggmlWhisper.xcframework").asFile

// -Pkggml.requireNative=true: real-engine tests on an Android device FAIL instead of skip when the native libraries are
// missing (pair with -Pkggml.bundleNative=true, which packages them).
val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)

// :kggml-whisper's synthetic model fixture (internal test source, MIT, see NOTICE), compiled into this module's tests.
val syncWhisperFixture by tasks.registering(Sync::class) {
    description = "Copies :kggml-whisper's synthetic model fixture (TinyWhisperModel) into this module's tests."
    from(rootProject.layout.projectDirectory.file("kggml-whisper/src/commonTest/kotlin/com/donfuldauer/kggml/whisper/TinyWhisperModel.kt"))
    into(layout.buildDirectory.dir("generated/engine-fixtures/kotlin"))
}

kotlin {
    androidTarget {
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
        // commonTest (session logic, real-engine tests) also runs on an Android device / emulator.
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        instrumentedTestVariant.sourceSetTree.set(KotlinSourceSetTree.test)
    }
    listOf(iosArm64(), iosSimulatorArm64()).forEach { target ->
        target.binaries.framework {
            baseName = "WhisperOnlyMinimal"
            // Static: GgmlRuntime.framework and KggmlWhisper.framework are embedded and linked by the Xcode host
            // app, never duplicated here.
            isStatic = true
        }
        // Test executables are fully linked, so they need the locally built dynamic frameworks.
        val slice = if (target.name == "iosSimulatorArm64") "ios-arm64-simulator" else "ios-arm64"
        listOf(iosWhisperXcframeworkDir, iosXcframeworkDir).map { it.resolve(slice).absolutePath }.forEach { path ->
            target.binaries.getTest("DEBUG").linkerOpts("-F", path, "-rpath", path)
        }
    }

    compilerOptions { freeCompilerArgs.add("-Xexpect-actual-classes") }

    sourceSets {
        commonMain.dependencies {
            // Kggml SDK: versions come from the BOM only.
            implementation(project.dependencies.platform(project(":kggml-bom")))
            implementation(project(":kggml-whisper"))
        }
        commonTest {
            kotlin.srcDir(syncWhisperFixture)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
        // MainActivityTest (androidSharedTest) runs under Robolectric locally and on the device when instrumented.
        // Test-only dependencies: they never reach the audited app APK.
        val androidUnitTest by getting {
            kotlin.srcDir("src/androidSharedTest/kotlin")
            dependencies {
                implementation(libs.robolectric)
                implementation(libs.junit4)
                implementation(libs.androidx.test.core)
                implementation(libs.androidx.test.ext.junit)
                implementation(libs.androidx.test.espresso.core)
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
                implementation(libs.androidx.test.espresso.core)
            }
        }
    }
}

android {
    namespace = "com.donfuldauer.kggml.samples.whisper"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    defaultConfig {
        applicationId = "com.donfuldauer.kggml.samples.whisper"
        minSdk = libs.versions.android.minSdk.get().toInt()
        targetSdk = libs.versions.android.compileSdk.get().toInt()
        versionCode = 1
        versionName = sampleVersion
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Read by NativeTestEnvironment (InstrumentationRegistry arguments).
        testInstrumentationRunnerArguments["kggml.requireNative"] = requireNative.toString()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(jvmTargetVersion)
        targetCompatibility = JavaVersion.toVersion(jvmTargetVersion)
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
        unitTests.isIncludeAndroidResources = true
    }
}

// Guard: this sample must never (even transitively) pull in the llama or diffusion engines.
val forbiddenKggmlModules = setOf("kggml-llama", "kggml-diffusion")
val verifyMinimalDependencies by tasks.registering {
    group = "verification"
    description = "Fails if :samples:whisper-only-minimal resolves :kggml-llama or :kggml-diffusion."
    val roots = listOf("releaseRuntimeClasspath", "debugRuntimeClasspath")
        .map { configurations.getByName(it).incoming.resolutionResult.rootComponent }
    doLast {
        val offenders = roots.flatMap { root ->
            val seen = mutableSetOf<String>()
            fun visit(c: org.gradle.api.artifacts.result.ResolvedComponentResult) {
                val name = c.moduleVersion?.name ?: return
                if (!seen.add("${c.id}")) return
                c.dependencies.filterIsInstance<org.gradle.api.artifacts.result.ResolvedDependencyResult>()
                    .forEach { visit(it.selected) }
            }
            visit(root.get())
            seen.filter { id -> forbiddenKggmlModules.any { id.contains(it) } }
        }.distinct()
        check(offenders.isEmpty()) { "whisper-only-minimal must not depend on $forbiddenKggmlModules: $offenders" }
        logger.lifecycle("whisper-only-minimal dependency audit OK (no llama / diffusion modules)")
    }
}
tasks.named("check") { dependsOn(verifyMinimalDependencies) }

// iOS test executables link against the xcframeworks; skip them (like :kggml-whisper) until they are built.
tasks.matching { it.name.startsWith("linkDebugTestIos") || it.name.matches(Regex("ios.*Test")) }.configureEach {
    onlyIf("GgmlRuntime + KggmlWhisper xcframeworks are built (native/scripts/build-ios-xcframework.sh)") {
        iosXcframeworkDir.isDirectory && iosWhisperXcframeworkDir.isDirectory
    }
}
