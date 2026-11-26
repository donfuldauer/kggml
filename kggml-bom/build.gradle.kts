import com.donfuldauer.kggml.buildlogic.KggmlPublishingPlugin
import javax.xml.parsers.DocumentBuilderFactory
import org.gradle.api.artifacts.result.ResolvedComponentResult
import org.gradle.api.artifacts.result.ResolvedDependencyResult
import org.gradle.api.attributes.java.TargetJvmEnvironment
import org.w3c.dom.Element

plugins {
    `java-platform`
    alias(libs.plugins.kggml.publishing)
}

description = "Kggml Bill of Materials: aligns versions of all Kggml SDK artifacts."

/**
 * Every published Kggml library coordinate. Adding a new SDK module means adding it here;
 * `verifyBomPom` fails if a module applying `kggml.kmp.library` is missing from this list.
 */
val kggmlLibraryModules: List<String> = listOf(
    ":kggml-core-api",
    ":kggml-runtime",
    ":kggml-runtime-vulkan",
    ":kggml-whisper",
    ":kggml-llama",
    ":kggml-diffusion",
    ":kggml-model-manager",
)

dependencies {
    constraints {
        // Project constraints resolve to each module's root KMP publication coordinates
        // (`com.donfuldauer:<name>:<version>`), so versions are never duplicated here.
        kggmlLibraryModules.forEach { api(project(it)) }
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["javaPlatform"])
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Verification (wired into `check`)
// ---------------------------------------------------------------------------------------------

val expectedGroup = project.group.toString()
val expectedVersion = project.version.toString()
val expectedArtifactIds: List<String> = kggmlLibraryModules.map { it.removePrefix(":") }

val generatePom = tasks.named<GenerateMavenPom>("generatePomFileForMavenPublication")

val kmpLibraryPluginUsage = Regex("""(?m)^\s*alias\(libs\.plugins\.kggml\.kmp\.library\)""")

val verifyBomPom by tasks.registering {
    group = "verification"
    description = "Verifies the generated BOM POM declares <dependencyManagement> constraints for every Kggml library."
    val pomFile = generatePom.map { it.destination }
    inputs.file(pomFile)
    val libraryProjects = rootProject.subprojects
        .filter { it != project && it.buildFile.isFile && kmpLibraryPluginUsage.containsMatchIn(it.buildFile.readText()) }
        .map { it.path }
        .sorted()
    val declared = kggmlLibraryModules.sorted()
    doLast {
        check(libraryProjects == declared) {
            "BOM module list $declared does not match projects applying kggml.kmp.library: $libraryProjects"
        }
        val doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(pomFile.get())
        val root = doc.documentElement
        fun Element.child(name: String): String? =
            (getElementsByTagName(name).item(0) as? Element)?.textContent?.trim()

        check(root.child("groupId") == expectedGroup) { "BOM groupId is ${root.child("groupId")}, expected $expectedGroup" }
        check(root.child("artifactId") == "kggml-bom") { "BOM artifactId is ${root.child("artifactId")}" }
        check(root.child("version") == expectedVersion) { "BOM version is ${root.child("version")}, expected $expectedVersion" }
        check(root.child("packaging") == "pom") { "BOM packaging is ${root.child("packaging")}, expected pom" }
        val license = root.getElementsByTagName("license").item(0) as? Element
            ?: error("BOM POM is missing <licenses> metadata")
        check(license.child("name") == "MIT License") { "BOM POM license must be MIT, was ${license.child("name")}" }

        val dm = root.getElementsByTagName("dependencyManagement").item(0) as? Element
            ?: error("BOM POM has no <dependencyManagement> section")
        val deps = dm.getElementsByTagName("dependency")
        val found = (0 until deps.length).map { i ->
            val d = deps.item(i) as Element
            Triple(d.child("groupId"), d.child("artifactId"), d.child("version"))
        }
        val expected = expectedArtifactIds.map { Triple(expectedGroup, it, expectedVersion) }
        check(found.sortedBy { it.second } == expected.sortedBy { it.second }) {
            "BOM <dependencyManagement> mismatch.\n  expected: $expected\n  found:    $found"
        }
        logger.lifecycle("kggml-bom POM OK: ${found.size} constraints at $expectedGroup:*:$expectedVersion")
    }
}

/**
 * Simulates an external JVM consumer: imports `platform("com.donfuldauer:kggml-bom:<v>")` and
 * declares every Kggml library WITHOUT a version, resolving against the build-local Maven repo.
 * Every library must resolve (via Gradle module metadata) to the BOM-aligned version.
 */
val bomConsumer: Configuration by configurations.creating {
    isCanBeConsumed = false
    isCanBeResolved = true
    attributes {
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
        attribute(Category.CATEGORY_ATTRIBUTE, objects.named(Category.LIBRARY))
        attribute(LibraryElements.LIBRARY_ELEMENTS_ATTRIBUTE, objects.named(LibraryElements.JAR))
        attribute(TargetJvmEnvironment.TARGET_JVM_ENVIRONMENT_ATTRIBUTE, objects.named(TargetJvmEnvironment.STANDARD_JVM))
        attribute(Attribute.of("org.jetbrains.kotlin.platform.type", String::class.java), "jvm")
    }
}

dependencies {
    bomConsumer(platform("$expectedGroup:kggml-bom:$expectedVersion"))
    expectedArtifactIds.forEach { bomConsumer("$expectedGroup:$it") }
}

val publishToVerificationRepo = "publishAllPublicationsTo${KggmlPublishingPlugin.VERIFICATION_REPOSITORY_NAME.replaceFirstChar { it.uppercase() }}Repository"

val verifyBomConsumerResolution by tasks.registering {
    group = "verification"
    description = "Resolves every Kggml library through the BOM without explicit versions, as an external consumer."
    dependsOn(tasks.named(publishToVerificationRepo))
    kggmlLibraryModules.forEach { dependsOn("$it:$publishToVerificationRepo") }
    val rootComponent = bomConsumer.incoming.resolutionResult.rootComponent
    doLast {
        val resolved = rootComponent.get().dependencies
            .map { dep ->
                check(dep is ResolvedDependencyResult) { "Unresolved dependency: ${dep.requested} (${dep})" }
                dep.selected
            }
        fun ResolvedComponentResult.coords() = moduleVersion!!.let { Triple(it.group, it.name, it.version) }
        val libraries = resolved.filter { it.moduleVersion?.name != "kggml-bom" }.distinctBy { it.id }
        val platformTypeKey = Attribute.of("org.jetbrains.kotlin.platform.type", String::class.java)
        libraries.forEach { component ->
            val platformTypes = component.variants.map { v ->
                v.attributes.keySet().firstOrNull { it.name == platformTypeKey.name }?.let { v.attributes.getAttribute(it).toString() }
            }
            check(platformTypes.isNotEmpty() && platformTypes.all { it == "jvm" }) {
                "${component.id} resolved to non-JVM variant(s): ${component.variants.map { it.displayName }}"
            }
        }
        val libs = libraries.map { it.coords() }
        val expected = expectedArtifactIds.map { Triple(expectedGroup, it, expectedVersion) }
        check(libs.sortedBy { it.second } == expected.sortedBy { it.second }) {
            "Consumer resolution mismatch.\n  expected: $expected\n  resolved: $libs"
        }
        logger.lifecycle("kggml-bom consumer resolution OK: ${libs.joinToString { "${it.second}:${it.third}" }}")
    }
}

tasks.named("check") {
    dependsOn(verifyBomPom, verifyBomConsumerResolution)
}
