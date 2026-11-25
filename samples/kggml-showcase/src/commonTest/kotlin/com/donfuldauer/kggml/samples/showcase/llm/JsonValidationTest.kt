package com.donfuldauer.kggml.samples.showcase.llm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class JsonValidationTest {
    @Test
    fun parsesAllValueKinds() {
        val value = MiniJson.parse(""" { "s": "a\"b\u00e9\n", "i": -12, "d": 1.5e2, "t": true, "f": false, "n": null, "a": [1, [], {}] } """)
        assertEquals(
            mapOf("s" to "a\"bé\n", "i" to -12L, "d" to 150.0, "t" to true, "f" to false, "n" to null, "a" to listOf(1L, emptyList<Any?>(), emptyMap<String, Any?>())),
            value,
        )
    }

    @Test
    fun rejectsMalformedJson() {
        listOf("{,}", "[1,]", "01", "1.", "\"\\x\"", "tru e", "{\"a\" 1}", "[1] 2", "\"a\u0001\"").forEach { text ->
            val e = assertFailsWith<MiniJson.ParseException>(text) { MiniJson.parse(text) }
            assertTrue(e.position >= 0)
        }
    }

    @Test
    fun truncatedInputIsIncompleteNotInvalid() {
        listOf("{", "{\"name\": \"Ma", "[1, 2", "{\"a\": tr", "{\"a\": 1").forEach { text ->
            assertIs<JsonValidation.Incomplete>(validateJson(text, DEFAULT_SCHEMA), text)
        }
        assertIs<JsonValidation.Invalid>(validateJson("{\"a\": x", DEFAULT_SCHEMA))
        assertIs<JsonValidation.Empty>(validateJson("  ", DEFAULT_SCHEMA))
    }

    @Test
    fun checksTypesRequiredItemsAndEnums() {
        val schema = """
            {"type": "object", "required": ["tags", "level"], "properties": {
              "tags": {"type": "array", "items": {"type": "string"}},
              "level": {"enum": ["low", "high"]},
              "score": {"type": "number"}, "ok": {"type": "boolean"}, "none": {"type": "null"}
            }}
        """.trimIndent()
        assertIs<JsonValidation.Valid>(validateJson("""{"tags": ["a"], "level": "low", "score": 2, "ok": true, "none": null}""", schema))
        val violation = assertIs<JsonValidation.SchemaViolation>(
            validateJson("""{"tags": ["a", 3], "level": "mid", "score": "x", "ok": 1, "none": 0}""", schema),
        )
        assertEquals(
            listOf(
                "$.tags[1] must be a string",
                "$.level is not one of low, high",
                "$.score must be a number",
                "$.ok must be a boolean",
                "$.none must be null",
            ),
            violation.problems,
        )
        assertIs<JsonValidation.SchemaViolation>(validateJson("[1]", schema))
        // Without a (parseable) schema, any well-formed JSON is valid.
        assertIs<JsonValidation.Valid>(validateJson("[1]", null))
        assertIs<JsonValidation.Valid>(validateJson("[1]", "{oops"))
    }

    @Test
    fun schemaProblemDetectsNonObjects() {
        assertNull(schemaProblem(DEFAULT_SCHEMA))
        assertNotNull(schemaProblem("[]"))
        assertNotNull(schemaProblem("{"))
    }
}
