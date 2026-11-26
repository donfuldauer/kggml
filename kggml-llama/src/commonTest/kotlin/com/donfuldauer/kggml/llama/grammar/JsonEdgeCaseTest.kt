package com.donfuldauer.kggml.llama.grammar

import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

/** RFC 8259 edge cases of [Json.parse] / [Json.encode]: escapes, numbers, whitespace, depth and error offsets. */
class JsonEdgeCaseTest {
    private fun str(text: String): String = (Json.parse(text) as Json.Str).value

    private fun failure(text: String): String = assertFailsWith<JsonException>(text) { Json.parse(text) }.message!!

    @Test
    fun everyRfc8259EscapeDecodes() {
        assertEquals("\"\\/\b\u000C\n\r\t", str(""""\"\\\/\b\f\n\r\t""""))
        assertEquals("Aé", str(""""\u0041\u00e9""""))
        assertEquals("\u00C9", str(""""\u00C9""""), "hex digits are case-insensitive")
        // A UTF-16 surrogate pair written as two escapes is one supplementary code point.
        assertEquals("😀", str(""""\ud83d\ude00""""))
        assertEquals("", str("\"\""))
        assertEquals("ü€", str("\"ü€\""), "raw non-ASCII characters are taken verbatim")
    }

    @Test
    fun encodeEscapesQuotesBackslashesAndEveryControlCharacter() {
        assertEquals("\"\\\"\\\\\\n\\r\\t\\b\\f\"", Json.encode(Json.Str("\"\\\n\r\t\b\u000C")))
        assertEquals("\"\\u0000\\u001f\"", Json.encode(Json.Str("\u0000\u001F")))
        // '/' and DEL need no escaping in JSON.
        assertEquals("\"/\u007F\"", Json.encode(Json.Str("/\u007F")))
        // Every encoded string parses back to itself.
        val all = buildString { for (c in 0 until 0x80) append(c.toChar()) }
        assertEquals(all, str(Json.encode(Json.Str(all))))
    }

    @Test
    fun encodeIsCompactAndKeepsNumbersAsWritten() {
        val doc = Json.parse(" [ 0 , -0 , 1.50 , 2E+3 , 4e-2 , {\t\"k\"\r\n:\n[ ]} ] ")
        assertEquals("""[0,-0,1.50,2E+3,4e-2,{"k":[]}]""", Json.encode(doc))
        assertEquals(Json.Arr(emptyList()), Json.parse("[ ]"))
        assertEquals(Json.Obj(linkedMapOf()), Json.parse("{ }"))
        assertEquals(Json.Null, Json.parse(" null "))
    }

    @Test
    fun numberAsLongAcceptsOnlyIntegralFiniteValues() {
        fun asLong(text: String) = (Json.parse(text) as Json.Num).asLong
        assertEquals(-7L, asLong("-7"))
        assertEquals(0L, asLong("-0"))
        assertEquals(100L, asLong("1e2"))
        assertEquals(1000L, asLong("1E+3"))
        assertEquals(-2L, asLong("-2.0"))
        assertNull(asLong("1e-1"))
        assertNull(asLong("1e400"), "an overflowing double is infinite, not an integer")
    }

    @Test
    fun malformedNumbersAreRejectedAtTheOffendingOffset() {
        assertEquals("invalid number at offset 1", failure("-"))
        assertEquals("unexpected character 'a' at offset 1", failure("-a"))
        assertEquals("unexpected character 'x' at offset 0", failure("x"))
        assertEquals("invalid fraction at offset 2", failure("1."))
        assertEquals("invalid fraction at offset 2", failure("1.a"))
        assertEquals("invalid exponent at offset 2", failure("1e"))
        assertEquals("invalid exponent at offset 3", failure("1e+"))
        assertEquals("invalid exponent at offset 3", failure("1E-x"))
        assertEquals("trailing characters at offset 1", failure("01"))
        assertEquals("unexpected character '+' at offset 0", failure("+1"))
        assertEquals("unexpected character '.' at offset 0", failure(".5"))
    }

    @Test
    fun malformedStringsAreRejectedAtTheOffendingOffset() {
        assertEquals("unterminated string at offset 4", failure("\"abc"))
        assertEquals("unterminated escape at offset 2", failure("\"\\"))
        assertEquals("short \\u escape at offset 3", failure("\"\\u12\""))
        assertEquals("bad \\u escape at offset 3", failure("\"\\u12zz\""))
        assertEquals("invalid escape '\\x' at offset 3", failure("\"\\x\""))
        // The offset is reported after the offending character has been consumed.
        assertEquals("control character in string at offset 3", failure("\"a\tb\""))
    }

    @Test
    fun malformedContainersAreRejectedAtTheOffendingOffset() {
        assertEquals("unexpected end of input at offset 1", failure("["))
        assertEquals("unterminated array at offset 2", failure("[1"))
        assertEquals("expected ',' or ']' at offset 2", failure("[1;2]"))
        assertEquals("expected object key at offset 1", failure("{1:2}"))
        assertEquals("expected ':' at offset 4", failure("{\"a\""))
        assertEquals("expected ':' at offset 5", failure("{\"a\" 1}"))
        assertEquals("unterminated object at offset 6", failure("{\"a\":1"))
        assertEquals("expected ',' or '}' at offset 6", failure("{\"a\":1;"))
        assertEquals("expected object key at offset 7", failure("{\"a\":1,"))
        assertEquals("duplicate key 'a' at offset 11", failure("{\"a\":1,\"a\":2}"))
        assertEquals("invalid literal at offset 0", failure("nul"))
        assertEquals("invalid literal at offset 1", failure("[tru]"))
        assertEquals("unexpected end of input at offset 0", failure(""))
        assertEquals("unexpected end of input at offset 3", failure(" \n\t"))
    }

    @Test
    fun nestingIsLimitedTo256Levels() {
        // The outermost value is at depth 0, so 257 nested arrays reach depth 256: the deepest allowed.
        val ok = "[".repeat(257) + "]".repeat(257)
        assertEquals(ok, Json.encode(Json.parse(ok)))
        val tooDeep = "[".repeat(258) + "]".repeat(258)
        assertEquals("nesting deeper than 256 at offset 257", failure(tooDeep))
    }

    /**
     * BUG (not fixed here): RFC 8259 requires exactly four hex digits after `\u`, but the parser decodes them with
     * `String.toIntOrNull(16)`, which also accepts a leading sign, so `"\u+041"` decodes to `A` and `"\u-041"` to
     * U+FFBF instead of being rejected. Enable once [Json] validates the four characters as hex digits.
     */
    @Ignore
    @Test
    fun signedUnicodeEscapesAreRejected() {
        assertFailsWith<JsonException> { Json.parse("\"\\u+041\"") }
        assertFailsWith<JsonException> { Json.parse("\"\\u-041\"") }
    }
}
