import javax.inject.Inject
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// Minimal Android application shell for `:samples:whisper-only-minimal` (AGP 9 built-in Kotlin; plain android.widget
// Activity with zero Compose / AndroidX in the main APK, audited by scripts/ci/audit-minimal-sample.sh).
plugins {
    alias(libs.plugins.android.application)
}

val jvmTargetVersion: String = libs.versions.jvmTarget.get()
val sampleVersion: String = providers.gradleProperty("VERSION_NAME").get()
val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)

// :kggml-whisper's synthetic model fixture (internal test source, MIT, see NOTICE), compiled into this app's tests.
abstract class SyncWhisperFixtureTask : DefaultTask() {
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val fixtureFile: RegularFileProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @get:Inject
    protected abstract val fileSystemOperations: FileSystemOperations

    @TaskAction
    fun copy() {
        fileSystemOperations.sync {
            from(fixtureFile)
            into(outputDir)
        }
    }
}

val whisperFixtureFile =
    rootProject.layout.projectDirectory.file("kggml-whisper/src/commonTest/kotlin/com/donfuldauer/kggml/whisper/TinyWhisperModel.kt")

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
        testInstrumentationRunnerArguments["kggml.requireNative"] = requireNative.toString()
    }
    sourceSets {
        getByName("test").kotlin.directories.add("src/sharedTest/kotlin")
        getByName("androidTest").kotlin.directories.add("src/sharedTest/kotlin")
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

androidComponents {
    onVariants { variant ->
        val cap = variant.name.replaceFirstChar { it.uppercase() }
        variant.hostTests[com.android.build.api.variant.HostTestBuilder.UNIT_TEST_TYPE]?.sources?.kotlin?.let { kotlinSources ->
            val syncTask = tasks.register<SyncWhisperFixtureTask>("syncWhisperFixture${cap}UnitTest") {
                fixtureFile.set(whisperFixtureFile)
            }
            kotlinSources.addGeneratedSourceDirectory(syncTask, SyncWhisperFixtureTask::outputDir)
        }
        variant.deviceTests[com.android.build.api.variant.DeviceTestBuilder.ANDROID_TEST_TYPE]?.sources?.kotlin?.let { kotlinSources ->
            val syncTask = tasks.register<SyncWhisperFixtureTask>("syncWhisperFixture${cap}AndroidTest") {
                fixtureFile.set(whisperFixtureFile)
            }
            kotlinSources.addGeneratedSourceDirectory(syncTask, SyncWhisperFixtureTask::outputDir)
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion))
    }
}

dependencies {
    implementation(platform(project(":kggml-bom")))
    implementation(project(":samples:whisper-only-minimal"))

    // Test-only dependencies: they never reach the audited app APK.
    testImplementation(libs.robolectric)
    testImplementation(libs.junit4)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.androidx.test.ext.junit)
    testImplementation(libs.androidx.test.espresso.core)

    androidTestImplementation(libs.junit4)
    androidTestImplementation(libs.androidx.test.core)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.test.espresso.core)
}

// Guard: this sample APK must never (even transitively) pull in the llama or diffusion engines or GPU add-ons.
val forbiddenKggmlModules = setOf("kggml-llama", "kggml-diffusion", "kggml-runtime-vulkan", "kggml-runtime-opencl")
val verifyMinimalDependencies = tasks.register("verifyMinimalDependencies") {
    group = "verification"
    description = "Fails if :samples:whisper-only-minimal:androidApp resolves forbidden Kggml modules."
    val roots = listOf("releaseRuntimeClasspath", "debugRuntimeClasspath")
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
        check(offenders.isEmpty()) { "whisper-only-minimal:androidApp must not depend on $forbiddenKggmlModules: $offenders" }
        logger.lifecycle("whisper-only-minimal:androidApp dependency audit OK (no llama / diffusion / GPU add-on modules)")
    }
}
tasks.named("check") { dependsOn(verifyMinimalDependencies) }
