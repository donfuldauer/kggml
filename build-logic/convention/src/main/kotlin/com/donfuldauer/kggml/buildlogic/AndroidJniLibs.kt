package com.donfuldauer.kggml.buildlogic

import com.android.build.api.variant.KotlinMultiplatformAndroidComponentsExtension
import javax.inject.Inject
import org.gradle.api.DefaultTask
import org.gradle.api.Project
import org.gradle.api.file.Directory
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.FileSystemOperations
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.api.tasks.TaskProvider
import org.gradle.kotlin.dsl.getByType
import org.gradle.kotlin.dsl.register

/**
 * Copies cross-compiled Android native libraries from `<root>/build/native-android-*-jniLibs` into a task-owned
 * directory so AGP's `com.android.kotlin.multiplatform.library` Variant API (`variant.sources.jniLibs`) packages
 * them into the AAR and tracks the native build task dependency automatically.
 */
public abstract class StageAndroidJniLibsTask : DefaultTask() {
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    public abstract val sourceDir: DirectoryProperty

    @get:OutputDirectory
    public abstract val outputDir: DirectoryProperty

    @get:Inject
    protected abstract val fileSystemOperations: FileSystemOperations

    @TaskAction
    public fun stage() {
        fileSystemOperations.sync {
            from(sourceDir)
            into(outputDir)
        }
    }
}

/**
 * Wires [jniLibsDir] (produced by [buildTask]) into the KMP Android library variant's `jniLibs` so the `.so` files
 * are packaged into the module's AAR and any consuming APK / device test APK.
 */
public fun Project.wireAndroidJniLibs(buildTask: TaskProvider<*>, jniLibsDir: Directory) {
    val stageTask = tasks.register<StageAndroidJniLibsTask>("stageAndroidJniLibs") {
        group = "kggml native"
        description = "Stages cross-compiled Android JNI libraries into the KMP Android variant."
        dependsOn(buildTask)
        sourceDir.set(jniLibsDir)
    }
    extensions.getByType<KotlinMultiplatformAndroidComponentsExtension>().onVariants { variant ->
        variant.sources.jniLibs?.addGeneratedSourceDirectory(stageTask, StageAndroidJniLibsTask::outputDir)
    }
}
