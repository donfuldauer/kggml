package com.donfuldauer.kggml.samples.showcase

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.runner.RunWith

/**
 * Shared by `androidUnitTest` (Robolectric) and `androidInstrumentedTest` (device): `AndroidJUnit4` picks the right
 * runner for each. `@RunWith` is `@Inherited`, so every subclass runs with it.
 */
@RunWith(AndroidJUnit4::class)
actual abstract class UiTestBase actual constructor()
