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
subprojects {
    tasks.withType<org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeSimulatorTest>().configureEach {
        if (iosSimulatorDevice.isPresent) device.set(iosSimulatorDevice)
    }
}
