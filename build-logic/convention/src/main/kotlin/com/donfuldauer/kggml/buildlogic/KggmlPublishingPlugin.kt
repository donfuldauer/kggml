package com.donfuldauer.kggml.buildlogic

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.publish.PublishingExtension
import org.gradle.api.publish.maven.MavenPublication
import org.gradle.api.publish.maven.tasks.AbstractPublishToMaven
import org.gradle.jvm.tasks.Jar
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.register
import org.gradle.kotlin.dsl.withType
import org.gradle.plugins.signing.Sign
import org.gradle.plugins.signing.SigningExtension

/**
 * `kggml.publishing` — applies `maven-publish`, assigns the `com.donfuldauer` group and the
 * shared SDK version, and stamps consistent POM metadata (MIT license, SCM, developer) onto
 * every Maven publication of the module.
 *
 * Also registers two build-local repositories:
 * - `kggmlVerification` (`build/kggml-verification-repo`): the BOM consumer-resolution check and `:docs-snippets`
 *   resolve artifacts from it exactly as an external consumer would.
 * - `kggmlStaging` (`build/kggml-staging-repo`): what `scripts/ci/stage-release.sh` bundles for Maven Central.
 *
 * Maven Central requirements are met for every publication:
 * - a `-javadoc.jar` (an empty one with a pointer to the KDoc in the sources jar; Central accepts that for
 *   Kotlin libraries), one per publication so each gets its own signature;
 * - PGP signatures (`.asc`) via the `signing` plugin with an in-memory ASCII-armored key from the Gradle
 *   properties `signingInMemoryKey`, `signingInMemoryKeyPassword` and optional `signingInMemoryKeyId`
 *   (`ORG_GRADLE_PROJECT_<name>` in CI). With `-Pkggml.release=true` a missing key fails the build instead of
 *   publishing unsigned artifacts.
 */
public class KggmlPublishingPlugin : Plugin<Project> {
    override fun apply(target: Project): Unit = with(target) {
        pluginManager.apply("maven-publish")

        group = requiredProperty("GROUP")
        version = requiredProperty("VERSION_NAME")

        val pomUrl = requiredProperty("POM_URL")
        val licenseName = requiredProperty("POM_LICENSE_NAME")
        val licenseUrl = requiredProperty("POM_LICENSE_URL")
        val scmConnection = requiredProperty("POM_SCM_CONNECTION")
        val scmDevConnection = requiredProperty("POM_SCM_DEV_CONNECTION")
        val developerId = requiredProperty("POM_DEVELOPER_ID")
        val developerName = requiredProperty("POM_DEVELOPER_NAME")
        val developerUrl = requiredProperty("POM_DEVELOPER_URL")
        val verificationRepo = rootProject.layout.buildDirectory.dir("kggml-verification-repo")
        val stagingRepo = rootProject.layout.buildDirectory.dir("kggml-staging-repo")
        val release = providers.gradleProperty(RELEASE_PROPERTY).map(String::toBoolean).getOrElse(false)
        val signingKey = providers.gradleProperty("signingInMemoryKey").orNull?.takeIf { it.isNotBlank() }
        val signingKeyId = providers.gradleProperty("signingInMemoryKeyId").orNull?.takeIf { it.isNotBlank() }
        val signingPassword = providers.gradleProperty("signingInMemoryKeyPassword").orNull ?: ""
        if (release && signingKey == null) {
            throw org.gradle.api.GradleException(
                "-P$RELEASE_PROPERTY=true needs a PGP key: set signingInMemoryKey (ORG_GRADLE_PROJECT_signingInMemoryKey).",
            )
        }

        extensions.configure<PublishingExtension> {
            repositories {
                maven {
                    name = VERIFICATION_REPOSITORY_NAME
                    url = uri(verificationRepo)
                }
                maven {
                    name = STAGING_REPOSITORY_NAME
                    url = uri(stagingRepo)
                }
            }
            val javadocReadme = tasks.register(JAVADOC_README_TASK) {
                description = "Writes the README placed in every javadoc jar of this project."
                val out = layout.buildDirectory.dir("kggml-javadoc/content")
                val text = "API documentation for ${project.name} is the KDoc in the -sources.jar of this " +
                    "artifact and at $pomUrl.\n"
                inputs.property("text", text)
                outputs.dir(out)
                doLast { out.get().file("README.txt").asFile.apply { parentFile.mkdirs(); writeText(text) } }
            }
            publications.withType<MavenPublication>().configureEach {
                val publicationName = name
                // A java-platform BOM has `pom` packaging: Central requires no javadoc/sources jar for it.
                if (!project.plugins.hasPlugin("java-platform")) {
                    val javadocJar = tasks.register<Jar>("${publicationName}JavadocJar") {
                        description = "Builds the javadoc jar Maven Central requires for the $publicationName publication."
                        archiveBaseName.set("${project.name}-$publicationName")
                        archiveClassifier.set("javadoc")
                        destinationDirectory.set(layout.buildDirectory.dir("kggml-javadoc/$publicationName"))
                        from(javadocReadme)
                    }
                    artifact(javadocJar)
                }
                pom {
                    name.set(provider { "${project.name}${if (artifactId != project.name) " ($artifactId)" else ""}" })
                    description.set(provider { project.description ?: "Kggml ${project.name}" })
                    url.set(pomUrl)
                    inceptionYear.set("2026")
                    licenses {
                        license {
                            name.set(licenseName)
                            url.set(licenseUrl)
                            distribution.set("repo")
                        }
                    }
                    developers {
                        developer {
                            id.set(developerId)
                            name.set(developerName)
                            url.set(developerUrl)
                        }
                    }
                    scm {
                        url.set(pomUrl)
                        connection.set(scmConnection)
                        developerConnection.set(scmDevConnection)
                    }
                }
            }
        }

        if (signingKey != null) {
            pluginManager.apply("signing")
            extensions.configure<SigningExtension> {
                if (signingKeyId != null) {
                    useInMemoryPgpKeys(signingKeyId, signingKey, signingPassword)
                } else {
                    useInMemoryPgpKeys(signingKey, signingPassword)
                }
                sign(extensions.getByType(PublishingExtension::class.java).publications)
            }
            // KMP publications share task outputs (e.g. the root module metadata), so Gradle's implicit-dependency
            // validation needs every publish task to run after every signing task of the project.
            tasks.withType<AbstractPublishToMaven>().configureEach { dependsOn(tasks.withType<Sign>()) }
        }
    }

    public companion object {
        /** Name of the build-local repository used by the BOM consumer-resolution check and `:docs-snippets`. */
        public const val VERIFICATION_REPOSITORY_NAME: String = "kggmlVerification"

        /** Name of the build-local repository that `scripts/ci/stage-release.sh` bundles for Maven Central. */
        public const val STAGING_REPOSITORY_NAME: String = "kggmlStaging"

        /** Task writing the README that every (KDoc pointer) javadoc jar contains. */
        public const val JAVADOC_README_TASK: String = "kggmlJavadocReadme"

        /** Gradle property (`-Pkggml.release=true`) that makes signing mandatory. */
        public const val RELEASE_PROPERTY: String = "kggml.release"
    }
}
