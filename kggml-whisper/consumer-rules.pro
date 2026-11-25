# Consumer R8 / ProGuard rules shipped with the kggml-whisper AAR.
#
# kggml_whisper_jni calls back into Kotlin by name during streaming transcription
# (GetMethodID(listener class, "onNewSegments", "(II)I")), and binds the native methods of KggmlWhisperJni by their
# JNI symbol names. Neither may be renamed or removed by minification in the consuming app.
-keep class com.donfuldauer.kggml.whisper.KggmlWhisperJni { native <methods>; }
-keep interface com.donfuldauer.kggml.whisper.WhisperSegmentListener { *; }
-keep class * implements com.donfuldauer.kggml.whisper.WhisperSegmentListener {
    int onNewSegments(int, int);
}
