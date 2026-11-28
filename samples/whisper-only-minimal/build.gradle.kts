import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// Minimal Android + iOS + JVM desktop sample used for CI binary symbol leakage audits (AGENTS.md §2.1): it depends
// ONLY on the `kggml-bom` platform and `:kggml-whisper` (which brings `:kggml-runtime`). It must never depend on
// `:kggml-llama`, `:kggml-diffusion` or a GPU add-on, and uses no UI toolkit beyond the platform's own
// (android.widget / UIKit / Swing), so its binaries contain nothing but the app, the runtime and whisper. The Android
// app shell lives in `:samples:whisper-only-minimal:androidApp` (AGP 9 forbids combining `com.android.application` with
// KMP in one module). Never published. Desktop: ./gradlew :samples:whisper-only-minimal:runDesktop.
//
// Binary audit (scripts/audit-binary-symbols.sh, run by CI):
//   ./gradlew :samples:whisper-only-minimal:androidApp:assembleRelease -Pkggml.bundleNative=true
//   ./scripts/audit-binary-symbols.sh apk samples/whisper-only-minimal/androidApp/build/outputs/apk/release/androidApp-release-unsigned.apk
plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.kotlin.multiplatform.library)
}

val jvmTargetVersion: String = libs.versions.jvmTarget.get()
// Built by kggml-runtime/native/scripts/build-ios-xcframework.sh and kggml-whisper/native/scripts/build-ios-xcframework.sh
// (macOS only).
val iosXcframeworkDir: File = rootProject.layout.projectDirectory.dir("build/native-ios/GgmlRuntime.xcframework").asFile
val iosWhisperXcframeworkDir: File =
    rootProject.layout.projectDirectory.dir("build/native-ios/KggmlWhisper.xcframework").asFile

// -Pkggml.requireNative=true: real-engine tests on an Android device FAIL instead of skip when the native libraries are
// missing (pair with -Pkggml.bundleNative=true, which packages them).
val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)

// :kggml-whisper's synthetic model fixture (internal test source, MIT, see NOTICE), compiled into this module's tests.
val syncWhisperFixture = tasks.register<Sync>("syncWhisperFixture") {
    description = "Copies :kggml-whisper's synthetic model fixture (TinyWhisperModel) into this module's tests."
    from(rootProject.layout.projectDirectory.file("kggml-whisper/src/commonTest/kotlin/com/donfuldauer/kggml/whisper/TinyWhisperModel.kt"))
    into(layout.buildDirectory.dir("generated/engine-fixtures/kotlin"))
}
tasks.matching { it.name.contains("Lint") }.configureEach { dependsOn(syncWhisperFixture) }

kotlin {
    android {
        namespace = "com.donfuldauer.kggml.samples.whisper.shared"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
        withHostTest {
            isReturnDefaultValues = true
            isIncludeAndroidResources = true
        }
        // commonTest (session logic, real-engine tests) also runs on an Android device / emulator.
        withDeviceTestBuilder {
            sourceSetTreeName = "test"
        }.configure {
            instrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
            // Read by NativeTestEnvironment (InstrumentationRegistry arguments).
            instrumentationRunnerArguments["kggml.requireNative"] = requireNative.toString()
        }
    }
    // Desktop (Swing): the same session logic and model picker on macOS / Linux / Windows.
    jvm("desktop") {
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
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
            api(project.dependencies.platform(project(":kggml-bom")))
            api(project(":kggml-whisper"))
        }
        commonTest {
            kotlin.srcDir(syncWhisperFixture)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
        val desktopTest by getting {
            dependencies {
                implementation(libs.junit4)
            }
        }
        // Test-only dependencies: they never reach the audited app APK.
        val androidHostTest by getting {
            kotlin.srcDir("src/androidSharedTest/kotlin")
            dependencies {
                implementation(libs.robolectric)
                implementation(libs.junit4)
                implementation(libs.androidx.test.core)
                implementation(libs.androidx.test.ext.junit)
            }
        }
        val androidDeviceTest by getting {
            kotlin.srcDir("src/androidSharedTest/kotlin")
            dependencies {
                implementation(libs.junit4)
                implementation(libs.androidx.test.core)
                implementation(libs.androidx.test.runner)
                implementation(libs.androidx.test.rules)
                implementation(libs.androidx.test.ext.junit)
            }
        }
    }
}

// Guard: this sample must never (even transitively) pull in the llama or diffusion engines.
val forbiddenKggmlModules = setOf("kggml-llama", "kggml-diffusion")
val verifyMinimalDependencies by tasks.registering {
    group = "verification"
    description = "Fails if :samples:whisper-only-minimal resolves :kggml-llama or :kggml-diffusion."
    val roots = listOf("androidRuntimeClasspath", "desktopRuntimeClasspath")
        .map { configurations.getByName(it).incoming.resolutionResult.rootComponent }
    doLast {
        val offenders = roots.flatMap { root ->
            val seen = mutableSetOf<String>()
            fun visit(c: org.gradle.api.artifacts.result.ResolvedComponentResult) {
                c.moduleVersion?.name ?: return
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

// ---------------------------------------------------------------------------------------------------------
// Desktop: the host native builds (kggml-runtime / kggml-whisper build-host.sh) for runDesktop and desktopTest.
// ---------------------------------------------------------------------------------------------------------
val hostLibSubdir: String = if (System.getProperty("os.name").lowercase().startsWith("windows")) "bin" else "lib"
val hostNativeLibDir: File = rootProject.layout.projectDirectory.dir("build/native/$hostLibSubdir").asFile
val hostWhisperLibDir: File = rootProject.layout.projectDirectory.dir("build/native-whisper/$hostLibSubdir").asFile
val hostLibs = mapOf(
    "kggml.native.path" to (hostNativeLibDir to "kggml_runtime_jni"),
    "kggml.whisper.native.path" to (hostWhisperLibDir to "kggml_whisper_jni"),
)

val runDesktop by tasks.registering(JavaExec::class) {
    group = "application"
    description = "Runs the Swing desktop app of the minimal sample (pass app arguments with --args=\"...\")."
    val main = kotlin.jvm("desktop").compilations.getByName("main") // the existing target
    classpath(main.output.allOutputs, main.runtimeDependencyFiles)
    mainClass.set("com.donfuldauer.kggml.samples.whisper.MainKt")
    hostLibs.forEach { (property, lib) -> if (lib.first.isDirectory) jvmArgs("-D$property=${lib.first.absolutePath}") }
    providers.gradleProperty("kggml.models.dir").orNull?.let { jvmArgs("-Dkggml.models.dir=$it") }
}

// What a desktop app of this sample ships: its jar plus the resolved runtime classpath (audited by
// scripts/ci/audit-minimal-sample.sh desktop with scripts/audit-binary-symbols.sh jars).
val desktopDistribution by tasks.registering(Sync::class) {
    group = "distribution"
    description = "Collects the desktop app jar and its runtime classpath into build/desktop-dist/lib."
    val main = kotlin.jvm("desktop").compilations.getByName("main")
    from(tasks.named("desktopJar"))
    from(main.runtimeDependencyFiles)
    into(layout.buildDirectory.dir("desktop-dist/lib"))
}

tasks.named<Test>("desktopTest") {
    systemProperty("kggml.test.requireNative", requireNative.toString())
    systemProperty("java.awt.headless", "true")
    inputs.property("kggml.requireNative", requireNative)
    // Real-engine results depend on the native libraries they load: never reuse a result for other native builds.
    hostLibs.forEach { (property, lib) ->
        inputs.files(fileTree(lib.first) { include("*.so*", "*.dylib", "*.dll") })
            .withPropertyName("nativeLibs.$property").withPathSensitivity(PathSensitivity.RELATIVE)
    }
    doFirst {
        hostLibs.forEach { (property, lib) ->
            val (dir, jni) = lib
            if (dir.resolve(System.mapLibraryName(jni)).isFile) systemProperty(property, dir.absolutePath)
        }
    }
}

// iOS test executables link against the xcframeworks; skip them (like :kggml-whisper) until they are built.
tasks.matching { it.name.startsWith("linkDebugTestIos") || it.name.matches(Regex("ios.*Test")) }.configureEach {
    onlyIf("GgmlRuntime + KggmlWhisper xcframeworks are built (native/scripts/build-ios-xcframework.sh)") {
        val present = iosXcframeworkDir.isDirectory && iosWhisperXcframeworkDir.isDirectory
        if (!present && requireNative) {
            throw GradleException("-Pkggml.requireNative=true but $iosXcframeworkDir or $iosWhisperXcframeworkDir is missing")
        }
        present
    }
}
