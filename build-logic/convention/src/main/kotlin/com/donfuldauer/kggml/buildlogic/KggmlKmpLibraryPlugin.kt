package com.donfuldauer.kggml.buildlogic

import com.android.build.gradle.LibraryExtension
import org.gradle.api.JavaVersion
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.testing.Test
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.withType
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.plugin.KotlinSourceSetTree

/**
 * `kggml.kmp.library` — the single source of truth for Kggml SDK library modules:
 *
 * - Kotlin Multiplatform targets: `androidTarget`, `iosArm64`, `iosSimulatorArm64`, `jvm`
 *   (iOS targets compile only on macOS hosts; they are skipped elsewhere). Modules that already support Kotlin/Native
 *   macOS add `macosArm64()` themselves (issue #93: core-api and runtime so far); it moves here once every module does.
 * - Explicit API mode (strict), JVM 17 bytecode, `expect`/`actual` classes enabled.
 * - `kotlin-test` wired into `commonTest`; JVM tests run on the JUnit Platform.
 * - Android instrumented tests (`connectedDebugAndroidTest`) join the test source-set tree: `commonTest` (incl. the
 *   shared native-engine contracts) also runs on a device / emulator next to `androidInstrumentedTest`. Run it with
 *   `-Pkggml.bundleNative=true` (the real native libraries are packaged): the native tests there never skip, so a
 *   missing library fails them.
 * - Android library defaults (compile/min SDK from the version catalog, release variant published).
 * - Applies `kggml.publishing` for `com.donfuldauer` coordinates and POM metadata.
 *
 * Each module sets its own `android.namespace` (see AGENTS.md §1 for the package table).
 */
public class KggmlKmpLibraryPlugin : Plugin<Project> {
    @OptIn(ExperimentalKotlinGradlePluginApi::class)
    override fun apply(target: Project): Unit = with(target) {
        pluginManager.apply("org.jetbrains.kotlin.multiplatform")
        pluginManager.apply("com.android.library")
        pluginManager.apply("kggml.publishing")

        val catalog = libs
        val jvmTarget = JvmTarget.fromTarget(catalog.versionOf("jvmTarget"))
        val javaVersion = JavaVersion.toVersion(catalog.versionOf("jvmTarget"))

        extensions.configure<KotlinMultiplatformExtension> {
            explicitApi()

            androidTarget {
                publishLibraryVariants("release")
                compilerOptions { this.jvmTarget.set(jvmTarget) }
                instrumentedTestVariant.sourceSetTree.set(KotlinSourceSetTree.test)
            }
            jvm {
                compilerOptions { this.jvmTarget.set(jvmTarget) }
            }
            iosArm64()
            iosSimulatorArm64()

            compilerOptions {
                freeCompilerArgs.add("-Xexpect-actual-classes")
            }

            sourceSets.getByName("commonTest").dependencies {
                implementation(catalog.findLibrary("kotlin-test").get())
            }
            sourceSets.getByName("androidInstrumentedTest").dependencies {
                listOf("junit4", "androidx-test-core", "androidx-test-runner", "androidx-test-ext-junit").forEach {
                    implementation(catalog.findLibrary(it).get())
                }
            }
        }

        extensions.configure<LibraryExtension> {
            compileSdk = catalog.versionOf("android-compileSdk").toInt()
            defaultConfig {
                minSdk = catalog.versionOf("android-minSdk").toInt()
                testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
            }
            compileOptions {
                sourceCompatibility = javaVersion
                targetCompatibility = javaVersion
            }
            testOptions {
                unitTests.isReturnDefaultValues = true
            }
        }

        tasks.withType<Test>().configureEach {
            if (name.startsWith("jvm")) useJUnitPlatform()
        }
    }
}
