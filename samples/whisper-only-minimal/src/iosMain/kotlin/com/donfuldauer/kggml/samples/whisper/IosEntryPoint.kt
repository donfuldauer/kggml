package com.donfuldauer.kggml.samples.whisper

/**
 * Entry point for the Swift host app (`WhisperOnlyMinimal.framework`): returns the status text to display.
 * The host app embeds `GgmlRuntime.xcframework`; the UI itself lands with the whisper engine in Phase 2.
 */
@Suppress("unused")
fun runtimeStatusText(): String = describeRuntime()
