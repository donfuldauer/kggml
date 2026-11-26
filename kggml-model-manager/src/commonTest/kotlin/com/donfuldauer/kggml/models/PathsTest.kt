package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class PathsTest {
    @Test
    fun parentOfPosixPaths() {
        assertEquals("/models/nested", parentOf("/models/nested/model.bin"))
        assertEquals("/models", parentOf("/models/nested/"))
        assertEquals("/", parentOf("/model.bin"))
        assertEquals("models", parentOf("models/model.bin"))
        assertNull(parentOf("model.bin"))
        assertNull(parentOf("/"))
        assertEquals("ab", parentOf("ab/model.bin"), "a two-letter directory is not a drive root")
    }

    @Test
    fun parentOfWindowsPaths() {
        assertEquals("C:\\Users\\me\\nested", parentOf("C:\\Users\\me\\nested\\model.bin"))
        assertEquals("C:\\Users\\me", parentOf("C:\\Users\\me\\nested\\"))
        assertEquals("C:\\", parentOf("C:\\model.bin"))
        assertEquals("C:/models", parentOf("C:/models/model.bin"))
        assertEquals("C:\\models", parentOf("C:\\models/model.bin"))
        assertNull(parentOf("C:\\"))
        assertNull(parentOf("model.bin"))
    }

    @Test
    fun joinPathKeepsOneSeparator() {
        assertEquals("/models/a.bin", joinPath("/models", "a.bin"))
        assertEquals("/models/a.bin", joinPath("/models/", "a.bin"))
        assertEquals("C:\\models\\a.bin", joinPath("C:\\models\\", "a.bin"))
        assertEquals("C:\\models\\a.bin", joinPath("C:\\models", "a.bin"))
        assertEquals("C:/models/a.bin", joinPath("C:/models", "a.bin"))
        assertEquals("C:\\models/x/a.bin", joinPath("C:\\models/x", "a.bin"))
        assertEquals("models/a.bin", joinPath("models", "a.bin"))
    }
}
