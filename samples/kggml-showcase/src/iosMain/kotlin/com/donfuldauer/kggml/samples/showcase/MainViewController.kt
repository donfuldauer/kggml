package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.window.ComposeUIViewController
import platform.UIKit.UIViewController

/** Entry point for the iOS host app (embeds `KggmlShowcase.framework` and `GgmlRuntime.xcframework`). */
@Suppress("FunctionName", "unused")
fun MainViewController(): UIViewController = ComposeUIViewController { ShowcaseApp() }
