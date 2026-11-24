package com.donfuldauer.kggml.buildlogic

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.publish.PublishingExtension
import org.gradle.api.publish.maven.MavenPublication
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.withType

/**
 * `kggml.publishing` — applies `maven-publish`, assigns the `com.donfuldauer` group and the
 * shared SDK version, and stamps consistent POM metadata (MIT license, SCM, developer) onto
 * every Maven publication of the module.
 *
 * Also registers the build-local `kggmlVerification` repository so the BOM consumer-resolution
 * check can resolve artifacts exactly as an external consumer would.
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

        extensions.configure<PublishingExtension> {
            repositories {
                maven {
                    name = VERIFICATION_REPOSITORY_NAME
                    url = uri(verificationRepo)
                }
            }
            publications.withType<MavenPublication>().configureEach {
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
    }

    public companion object {
        public const val VERIFICATION_REPOSITORY_NAME: String = "kggmlVerification"
    }
}
