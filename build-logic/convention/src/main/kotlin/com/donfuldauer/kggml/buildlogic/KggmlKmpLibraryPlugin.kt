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

/**
 * `kggml.kmp.library` — the single source of truth for Kggml SDK library modules:
 *
 * - Kotlin Multiplatform targets: `androidTarget`, `iosArm64`, `iosSimulatorArm64`, `jvm`
 *   (iOS targets compile only on macOS hosts; they are skipped elsewhere).
 * - Explicit API mode (strict), JVM 17 bytecode, `expect`/`actual` classes enabled.
 * - `kotlin-test` wired into `commonTest`; JVM tests run on the JUnit Platform.
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
        }

        extensions.configure<LibraryExtension> {
            compileSdk = catalog.versionOf("android-compileSdk").toInt()
            defaultConfig {
                minSdk = catalog.versionOf("android-minSdk").toInt()
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
