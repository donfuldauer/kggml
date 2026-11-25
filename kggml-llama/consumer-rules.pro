# Consumer R8 / ProGuard rules shipped with the kggml-llama AAR.
#
# kggml_llama_jni binds the native methods of KggmlLlamaJni by their JNI symbol names, so the class and its native
# methods may not be renamed or removed by minification in the consuming app.
-keep class com.donfuldauer.kggml.llama.KggmlLlamaJni { native <methods>; }
