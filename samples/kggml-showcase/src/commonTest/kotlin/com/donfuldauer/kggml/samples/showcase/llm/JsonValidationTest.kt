package com.donfuldauer.kggml.samples.showcase.llm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
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
        listOf("{,}", "[1,]", "01", "\"\\x\"", "tru e", "{\"a\" 1}", "[1] 2", "\"a\u0001\"", "\u0663", "[\u0663]").forEach { text ->
            val e = assertFailsWith<MiniJson.ParseException>(text) { MiniJson.parse(text) }
            // An error inside the text (not at its end) is invalid JSON, never "incomplete".
            assertTrue(e.position in 0 until text.length, "position ${e.position} of '$text'")
            assertFalse(e.atEnd, "'$text' must not be reported as incomplete")
        }
        // A number cut off at the end is incomplete (more digits may still stream in).
        assertTrue(assertFailsWith<MiniJson.ParseException> { MiniJson.parse("1.") }.atEnd)
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
    fun decodesEveryStringEscape() {
        assertEquals(listOf("q\" s\\ /\b\u000C\n\r\tAé"), MiniJson.parse("""["q\" s\\ \/\b\f\n\r\tAé"]"""))
    }

    @Test
    fun invalidUnicodeEscapeIsReportedWithItsPosition() {
        val e = assertFailsWith<MiniJson.ParseException> { MiniJson.parse("\"\\uZZZZ\"") }
        assertEquals("invalid \\u escape at offset 3", e.message)
        assertFalse(e.atEnd)
        assertIs<JsonValidation.Invalid>(validateJson("""{"a": "\uZZZZ"}""", null))
    }

    /** RFC 8259: exactly four hex digits after `\u`; a sign (accepted by `toIntOrNull(16)`) is invalid, not incomplete. */
    @Test
    fun signedUnicodeEscapeIsInvalid() {
        assertIs<JsonValidation.Invalid>(validateJson("{\"a\": \"\\u+041\"}", null))
        assertIs<JsonValidation.Invalid>(validateJson("{\"a\": \"\\u-041\"}", null))
        // A non-hex digit is invalid even when the text ends before four characters.
        assertIs<JsonValidation.Invalid>(validateJson("{\"a\": \"\\uZ", null))
        assertEquals(mapOf("a" to "A"), MiniJson.parse("{\"a\": \"\\u0041\"}"))
    }

    /**
     * Regression: a `\u` escape cut off by the end of the streamed text failed with `atEnd = false` (the position was
     * the escape start), so live validation flashed "Invalid JSON" while the model was still streaming.
     */
    @Test
    fun unicodeEscapeCutOffByTheEndOfTheStreamIsIncomplete() {
        assertIs<JsonValidation.Incomplete>(validateJson("{\"a\": \"\\", null), "cut-off escape")
        assertIs<JsonValidation.Incomplete>(validateJson("{\"a\": \"\\u00", null), "cut-off \\u escape")
    }

    @Test
    fun parsesNumberForms() {
        assertEquals(listOf(0L, -7L, 100000.0, 0.2, -0.5), MiniJson.parse("[0, -7, 1e+5, 2E-1, -0.5]"))
        // Integers beyond Long fall back to Double instead of failing.
        assertEquals(12345678901234567890.0, MiniJson.parse("12345678901234567890"))
    }

    @Test
    fun malformedNumbersAreInvalidMidTextAndIncompleteAtTheEnd() {
        assertEquals("expected a digit at offset 1", assertFailsWith<MiniJson.ParseException> { MiniJson.parse("-x") }.message)
        assertEquals("expected an exponent digit at offset 3", assertFailsWith<MiniJson.ParseException> { MiniJson.parse("1e+]") }.message)
        assertIs<JsonValidation.Invalid>(validateJson("[-x]", null))
        assertIs<JsonValidation.Invalid>(validateJson("[1.5e]", null))
        // The model may still be streaming the number or the property.
        listOf("[-", "[1.5e", "[1.5e-", """{"a"""").forEach { text ->
            assertIs<JsonValidation.Incomplete>(validateJson(text, null), text)
        }
    }

    @Test
    fun arraysIntegersEnumsOfNumbersAndUnknownTypes() {
        val schema = """
            {"type": "object", "properties": {
              "list": {"type": "array"},
              "count": {"type": "integer"},
              "level": {"enum": [1, 2]},
              "id": {"type": "uuid"}
            }}
        """.trimIndent()
        // An array without "items" accepts any elements; 2.0 is an integer; 1.0 equals the enum's 1; unknown types
        // are not checked.
        assertIs<JsonValidation.Valid>(validateJson("""{"list": [1, "a", null], "count": 2.0, "level": 1.0, "id": 5}""", schema))
        val violation = assertIs<JsonValidation.SchemaViolation>(validateJson("""{"list": {}, "count": 2.5, "level": 3}""", schema))
        assertEquals(listOf("$.list must be an array", "$.count must be an integer", "$.level is not one of 1, 2"), violation.problems)
        assertEquals("Schema violations: $.list must be an array; $.count must be an integer; $.level is not one of 1, 2", violation.message)
    }

    @Test
    fun malformedSchemaKeywordsAreIgnored() {
        // "required" that is not a list, "properties" that is not an object and a property schema that is not an
        // object constrain nothing; a blank schema validates syntax only.
        assertIs<JsonValidation.Valid>(validateJson("""{"a": 1}""", """{"type": "object", "required": "a", "properties": []}"""))
        assertIs<JsonValidation.Valid>(validateJson("""{"a": 1}""", """{"type": "object", "properties": {"a": "string"}}"""))
        assertIs<JsonValidation.Valid>(validateJson("""{"a": 1}""", "   "))
    }

    @Test
    fun schemaProblemDetectsNonObjects() {
        assertNull(schemaProblem(DEFAULT_SCHEMA))
        assertNotNull(schemaProblem("[]"))
        assertNotNull(schemaProblem("{"))
    }
}
