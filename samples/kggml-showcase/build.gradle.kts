import org.jetbrains.compose.desktop.application.dsl.TargetFormat
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

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
// Built by kggml-runtime/native/scripts/build-ios-xcframework.sh (macOS only).
val iosXcframeworkDir: File = rootProject.layout.projectDirectory.dir("build/native-ios/GgmlRuntime.xcframework").asFile

kotlin {
    androidTarget {
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
    }
    jvm("desktop") {
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
    }
    listOf(iosArm64(), iosSimulatorArm64()).forEach { target ->
        target.binaries.framework {
            baseName = "KggmlShowcase"
            // Static: GgmlRuntime.framework is embedded and linked by the Xcode host app, never duplicated here.
            isStatic = true
        }
        // Test executables are fully linked, so they need the locally built GgmlRuntime.framework.
        val slice = if (target.name == "iosSimulatorArm64") "ios-arm64-simulator" else "ios-arm64"
        val frameworkSearchPath = iosXcframeworkDir.resolve(slice).absolutePath
        target.binaries.getTest("DEBUG").linkerOpts("-F", frameworkSearchPath, "-rpath", frameworkSearchPath)
    }

    sourceSets {
        commonMain.dependencies {
            // Kggml SDK: versions come from the BOM only.
            implementation(project.dependencies.platform(project(":kggml-bom")))
            implementation(project(":kggml-runtime"))

            implementation(compose.runtime)
            implementation(compose.foundation)
            implementation(compose.material3)
            implementation(compose.ui)
            implementation(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
        androidMain.dependencies {
            implementation(libs.androidx.activity.compose)
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
    }
    compileOptions {
        sourceCompatibility = JavaVersion.toVersion(jvmTargetVersion)
        targetCompatibility = JavaVersion.toVersion(jvmTargetVersion)
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

// Host runtime built by build-host.sh: shared libraries land in build/native/lib, except Windows DLLs (build/native/bin).
val hostNativeLibDir: File = rootProject.layout.projectDirectory
    .dir(if (System.getProperty("os.name").lowercase().startsWith("windows")) "build/native/bin" else "build/native/lib")
    .asFile

compose.desktop {
    application {
        mainClass = "com.donfuldauer.kggml.samples.showcase.MainKt"
        // During development, point the JVM loader at a locally built host runtime (build-host.sh) if present.
        // Release desktop builds instead bundle it into the :kggml-runtime jar (-Pkggml.bundleNative=true).
        if (hostNativeLibDir.isDirectory) jvmArgs += "-Dkggml.native.path=${hostNativeLibDir.absolutePath}"
        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "KggmlShowcase"
            packageVersion = sampleVersion
        }
    }
}

// Native-backed desktop test (KggmlRuntimeInspectorDesktopTest): uses the host runtime built by
// kggml-runtime/native/scripts/build-host.sh when present; otherwise asserts the graceful "runtime unavailable"
// state. -Pkggml.requireNative=true makes the native path mandatory.
tasks.named<Test>("desktopTest") {
    val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)
    val libDir = hostNativeLibDir
    val jniLibraryName = System.mapLibraryName("kggml_runtime_jni")
    systemProperty("kggml.test.requireNative", requireNative.toString())
    inputs.property("kggml.requireNative", requireNative)
    doFirst {
        if (libDir.resolve(jniLibraryName).isFile) systemProperty("kggml.native.path", libDir.absolutePath)
    }
}

// iOS test executables link against GgmlRuntime.xcframework; skip them (like :kggml-runtime) until it is built.
tasks.matching { it.name.startsWith("linkDebugTestIos") || it.name.matches(Regex("ios.*Test")) }.configureEach {
    onlyIf("GgmlRuntime.xcframework is built (kggml-runtime/native/scripts/build-ios-xcframework.sh)") {
        iosXcframeworkDir.isDirectory
    }
}
