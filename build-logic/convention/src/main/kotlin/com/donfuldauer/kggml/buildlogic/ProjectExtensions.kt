package com.donfuldauer.kggml.buildlogic

import org.gradle.api.Project
import org.gradle.api.artifacts.VersionCatalog
import org.gradle.api.artifacts.VersionCatalogsExtension
import org.gradle.kotlin.dsl.getByType

/** The shared `gradle/libs.versions.toml` catalog. */
internal val Project.libs: VersionCatalog
    get() = extensions.getByType<VersionCatalogsExtension>().named("libs")

internal fun VersionCatalog.versionOf(alias: String): String =
    findVersion(alias).orElseThrow { IllegalStateException("Missing version '$alias' in libs.versions.toml") }.requiredVersion

/** Reads a required Gradle property (e.g. from `gradle.properties`). */
internal fun Project.requiredProperty(name: String): String =
    providers.gradleProperty(name).orNull
        ?: error("Missing required Gradle property '$name' (expected in gradle.properties)")
