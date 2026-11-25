// Root build: plugins are declared here (not applied) so that every module and the
// `build-logic` convention plugins share a single classloader / version of each plugin.
plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.compose.multiplatform) apply false
    alias(libs.plugins.kotlin.compose) apply false
}

// `iosSimulatorArm64Test` boots Kotlin's default simulator device by name, which may not exist in a given Xcode.
// CI (and developers) can pin one by UDID or name: -Pkggml.iosSimulatorDevice=<udid|name>.
val iosSimulatorDevice: Provider<String> = providers.gradleProperty("kggml.iosSimulatorDevice")
// A hung simulator test must fail fast with the test's name in the log, not burn the CI job's whole
// timeout silently (PR #46: :kggml-llama:iosSimulatorArm64Test hung for 2.5 h with no output).
// Override per run with -Pkggml.iosTestTimeoutMinutes=<n>.
val iosTestTimeoutMinutes: Provider<Long> =
    providers.gradleProperty("kggml.iosTestTimeoutMinutes").map { it.toLong() }.orElse(20L)
subprojects {
    tasks.withType<org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeSimulatorTest>().configureEach {
        if (iosSimulatorDevice.isPresent) device.set(iosSimulatorDevice)
        timeout.set(iosTestTimeoutMinutes.map { java.time.Duration.ofMinutes(it) })
        testLogging {
            events("started", "passed", "skipped", "failed")
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        }
    }
}
