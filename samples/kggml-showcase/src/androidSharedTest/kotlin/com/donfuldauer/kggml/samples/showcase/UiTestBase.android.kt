package com.donfuldauer.kggml.samples.showcase

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Before
import org.junit.runner.RunWith

/**
 * Shared by `androidHostTest` (Robolectric) and `androidDeviceTest` (device): `AndroidJUnit4` picks the right
 * runner for each. `@RunWith` is `@Inherited`, so every subclass runs with it.
 */
@RunWith(AndroidJUnit4::class)
actual abstract class UiTestBase actual constructor() {
    /** What `MainActivity.onCreate` does first: the app's platform defaults need the application context. */
    @Before
    fun initShowcaseAndroid() {
        ShowcaseAndroid.init(ApplicationProvider.getApplicationContext())
    }
}
