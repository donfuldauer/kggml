// Root build: plugins are declared here (not applied) so that every module and the
// `build-logic` convention plugins share a single classloader / version of each plugin.
plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.library) apply false
}
