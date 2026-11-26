package com.donfuldauer.kggml.samples.showcase

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Shadows.shadowOf
import kotlin.test.assertEquals

/** The resident-engine budget on Android, from a scripted `ActivityManager.MemoryInfo` (Robolectric shadow). */
@RunWith(AndroidJUnit4::class)
class ResidentBudgetAndroidTest {
    @Test
    fun residentBudgetIsHalfThePhysicalMemoryWithA512MiBFloor() {
        val app: Application = ApplicationProvider.getApplicationContext()
        ShowcaseAndroid.init(app)
        val activityManager = app.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        fun budgetWithTotalMemory(bytes: Long): Long {
            shadowOf(activityManager).setMemoryInfo(ActivityManager.MemoryInfo().apply { totalMem = bytes })
            return defaultResidentBudgetBytes()
        }
        assertEquals(4L shl 30, budgetWithTotalMemory(8L shl 30))
        assertEquals(1536L shl 20, budgetWithTotalMemory(3L shl 30))
        assertEquals(512L shl 20, budgetWithTotalMemory(768L shl 20), "small devices still get 512 MiB")
    }
}
