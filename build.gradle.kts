// Root build: plugins are declared here (not applied) so that every module and the
// `build-logic` convention plugins share a single classloader / version of each plugin.
plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.kotlin.multiplatform.library) apply false
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.compose.multiplatform) apply false
    alias(libs.plugins.kotlin.compose) apply false
    // Code coverage of every JVM test task (commonTest + jvmTest / desktopTest, integration tests), merged here.
    alias(libs.plugins.kover)
}

// ---------------------------------------------------------------------------------------------------------
// Coverage (Kover). `./gradlew koverHtmlReport koverXmlReport koverLog` measures the JVM test tasks of every
// module below (commonTest + jvmTest / desktopTest; with the host native builds present, the JNI paths too) and
// merges them into build/reports/kover. Android-only (androidUnitTest / connected) and iOS code is exercised by
// its own test tasks but not instrumented by Kover. Per-module reports: `./gradlew :<module>:koverHtmlReport`.
// ---------------------------------------------------------------------------------------------------------
val coveredProjects = listOf(
    ":kggml-core-api",
    ":kggml-runtime",
    ":kggml-whisper",
    ":kggml-llama",
    ":kggml-diffusion",
    ":kggml-model-manager",
    ":integration-tests",
    ":samples:kggml-showcase",
)
subprojects {
    if (path in coveredProjects) pluginManager.apply("org.jetbrains.kotlinx.kover")
}
dependencies {
    coveredProjects.forEach { kover(project(it)) }
}

// `iosSimulatorArm64Test` boots Kotlin's default simulator device by name, which may not exist in a given Xcode.
// CI (and developers) can pin one by UDID or name: -Pkggml.iosSimulatorDevice=<udid|name>.
val iosSimulatorDevice: Provider<String> = providers.gradleProperty("kggml.iosSimulatorDevice")
// A hung simulator test must fail fast with the test's name in the log, not burn the CI job's whole
// timeout silently (PR #46: :kggml-llama:iosSimulatorArm64Test hung for 2.5 h with no output).
// Override per run with -Pkggml.iosTestTimeoutMinutes=<n>.
val iosTestTimeoutMinutes: Provider<Long> =
    providers.gradleProperty("kggml.iosTestTimeoutMinutes").map { it.toLong() }.orElse(20L)
// Pre-stage Robolectric's API 34 android-all-instrumented jar via Gradle's cached dependency resolution so
// Robolectric runs offline (`robolectric.offline=true`) instead of fetching from repo1.maven.org over raw
// HttpURLConnection inside test JVMs (which hits Maven Central HTTP 429 rate limits on GitHub Actions runners).
val robolectricSdk = configurations.create("robolectricSdk") {
    isCanBeConsumed = false
    isCanBeResolved = true
    isTransitive = false
}
dependencies {
    robolectricSdk(libs.robolectric.android.all.instrumented)
}
val robolectricSdkDir = layout.buildDirectory.dir("robolectric-sdk")
val syncRobolectricSdk = tasks.register<Sync>("syncRobolectricSdk") {
    description = "Stages Robolectric's pre-instrumented Android SDK jar from Gradle's dependency cache."
    from(robolectricSdk)
    into(robolectricSdkDir)
}

// Every JVM test process gets its own java.io.tmpdir (build/tmp/<task>): Gradle runs test tasks of different modules in
// parallel, and tests that assert "no temporary files left behind" (e.g. JvmNativeWhisperStreamTest) must not see the
// files another module's tests are creating at the same moment in the shared system temp directory.
subprojects {
    val isSampleProject = path.startsWith(":samples:")
    tasks.withType<Test>().configureEach {
        if (isSampleProject && (name == "testAndroidHostTest" || (name.startsWith("test") && name.endsWith("UnitTest")))) {
            dependsOn(syncRobolectricSdk)
            inputs.dir(robolectricSdkDir).withPathSensitivity(PathSensitivity.RELATIVE)
            systemProperty("robolectric.offline", "true")
            systemProperty("robolectric.dependency.dir", robolectricSdkDir.get().asFile.absolutePath)
        }
        doFirst {
            val dir = temporaryDir.resolve("java-tmp").apply { mkdirs() }
            systemProperty("java.io.tmpdir", dir.absolutePath)
        }
    }
}
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
