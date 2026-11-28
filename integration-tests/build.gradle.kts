import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// Cross-engine integration tests (issue #20). Never published and not part of kggml-bom: it only hosts tests that
// load several engine modules into ONE process, which no single engine module can do on its own.
//
//   -Pkggml.requireNative=true  FAIL (instead of skip) when the host native builds are missing.
//
// The host builds are produced by kggml-{runtime,whisper,llama,diffusion}/native/scripts/build-host.sh (CI runs them
// as explicit steps before Gradle).
plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

val jvmTargetVersion: String = libs.versions.jvmTarget.get()
val requireNative = providers.gradleProperty("kggml.requireNative").map(String::toBoolean).getOrElse(false)

// The engines' synthetic model fixtures are `internal` test sources of their modules. They are compiled into this
// module verbatim (same files, same packages) so the coexistence test drives the exact models whose bytes the
// native CTest suites pin by SHA-256, without publishing test fixtures.
val fixtureSources: List<File> = listOf(
    "kggml-whisper/src/commonTest/kotlin/com/donfuldauer/kggml/whisper/TinyWhisperModel.kt",
    "kggml-llama/src/commonTest/kotlin/com/donfuldauer/kggml/llama/TinyLlamaModel.kt",
).map { rootProject.layout.projectDirectory.file(it).asFile }
val fixtureDir = layout.buildDirectory.dir("generated/engine-fixtures/kotlin")
val syncEngineFixtures = tasks.register<Sync>("syncEngineFixtures") {
    description = "Copies the engines' synthetic model fixtures (TinyWhisperModel, TinyLlamaModel) into this module's tests."
    from(fixtureSources)
    into(fixtureDir)
}

kotlin {
    jvm {
        compilerOptions { jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion)) }
    }

    sourceSets {
        jvmTest {
            kotlin.srcDir(syncEngineFixtures)
            dependencies {
                implementation(project(":kggml-core-api"))
                implementation(project(":kggml-runtime"))
                implementation(project(":kggml-whisper"))
                implementation(project(":kggml-llama"))
                implementation(project(":kggml-diffusion"))
                implementation(libs.kotlin.test)
                implementation(libs.kotlinx.coroutines.core)
            }
        }
    }
}

// Host runtime / engines built by build-host.sh: shared libraries land in build/native(-whisper|-llama|-diffusion)/lib,
// except Windows DLLs (bin).
val hostLibSubdir: String = if (System.getProperty("os.name").lowercase().startsWith("windows")) "bin" else "lib"
val rootBuild = rootProject.layout.projectDirectory.dir("build")
val hostLibDirs: Map<String, Pair<File, String>> = mapOf(
    "kggml.native.path" to (rootBuild.dir("native/$hostLibSubdir").asFile to "kggml_runtime_jni"),
    "kggml.whisper.native.path" to (rootBuild.dir("native-whisper/$hostLibSubdir").asFile to "kggml_whisper_jni"),
    "kggml.llama.native.path" to (rootBuild.dir("native-llama/$hostLibSubdir").asFile to "kggml_llama_jni"),
    "kggml.diffusion.native.path" to (rootBuild.dir("native-diffusion/$hostLibSubdir").asFile to "kggml_diffusion_jni"),
)

tasks.named<Test>("jvmTest") {
    useJUnitPlatform()
    // Like the engine modules: build the host natives first when they are required (CI excludes these with -x
    // because it builds them in explicit steps). The libraries are inputs, so a native-only rebuild reruns the tests.
    if (requireNative) {
        dependsOn(":kggml-runtime:buildNativeHost", ":kggml-whisper:buildNativeHost", ":kggml-llama:buildNativeHost", ":kggml-diffusion:buildNativeHost")
    }
    hostLibDirs.values.forEach { (dir, _) -> inputs.files(fileTree(dir)).withPropertyName("nativeLibs-${dir.parentFile.name}").optional() }
    // One test JVM loads every engine: one process == the one shared ggml runtime under test.
    maxParallelForks = 1
    // Cap glibc per-thread malloc arenas on Linux: each EngineCoexistenceTest cycle spawns fresh single-threaded
    // engine executors that allocate and free ~32 MB of ggml compute buffers (below glibc's 32 MB dynamic mmap
    // threshold), which otherwise retains freed pages across up to 8*nproc arenas and looks like RSS growth.
    if (System.getenv("MALLOC_ARENA_MAX") == null) {
        environment("MALLOC_ARENA_MAX", "2")
    }
    systemProperty("kggml.test.requireNative", requireNative.toString())
    inputs.property("kggml.requireNative", requireNative)
    testLogging {
        events("passed", "skipped", "failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
    doFirst {
        // Only when every host build tree holds its JNI bridge; otherwise the tests skip (or fail when required).
        if (hostLibDirs.values.all { (dir, jni) -> dir.resolve(System.mapLibraryName(jni)).isFile }) {
            hostLibDirs.forEach { (property, entry) -> systemProperty(property, entry.first.absolutePath) }
        }
    }
}
