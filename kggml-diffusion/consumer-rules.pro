# Consumer R8 / ProGuard rules shipped with the kggml-diffusion AAR.
#
# kggml_diffusion_jni calls back into Kotlin by name during generation
# (GetMethodID(listener class, "onProgress", "(IIF)V")), and binds the native methods of KggmlDiffusionJni by their
# JNI symbol names. Neither may be renamed or removed by minification in the consuming app.
-keep class com.donfuldauer.kggml.diffusion.KggmlDiffusionJni { native <methods>; }
-keep interface com.donfuldauer.kggml.diffusion.DiffusionProgressListener { *; }
-keep class * implements com.donfuldauer.kggml.diffusion.DiffusionProgressListener {
    void onProgress(int, int, float);
}
