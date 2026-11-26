package com.donfuldauer.kggml.llama.grammar

import com.donfuldauer.kggml.core.InvalidInputException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** Keyword combinations, `$ref` pointers, bounds and every rejection path of [JsonSchemaToGbnf]. */
class JsonSchemaToGbnfEdgeCaseTest {
    private val d = '$'

    private fun rules(schema: String): Map<String, String> =
        JsonSchemaToGbnf.convert(schema).lines().filter { it.isNotBlank() }.associate {
            val (name, body) = it.split(" ::= ", limit = 2)
            name to body
        }

    /** The schema is rejected with an [InvalidInputException] whose message ends with [reason]. */
    private fun assertRejected(schema: String, reason: String) {
        val e = assertFailsWith<InvalidInputException>(schema) { JsonSchemaToGbnf.convert(schema) }
        assertEquals("Unsupported JSON schema: $reason", e.message, schema)
    }

    @Test
    fun invalidJsonIsWrappedWithTheParserDiagnosticAndCause() {
        val e = assertFailsWith<InvalidInputException> { JsonSchemaToGbnf.convert("""{"type":}""") }
        assertEquals("JSON schema is not valid JSON: unexpected character '}' at offset 8", e.message)
        assertIs<JsonException>(e.cause)
    }

    @Test
    fun everyUnsupportedKeywordIsRejectedNeverIgnored() {
        for (keyword in listOf("pattern", "patternProperties", "not", "if", "then", "else", "dependentSchemas")) {
            assertRejected("""{"type":"string","$keyword":{}}""", "keyword '$keyword' is not supported")
        }
    }

    @Test
    fun schemaShapesAreValidated() {
        assertRejected("""{"type":5}""", "'type' must be a string or an array of strings")
        assertRejected("""{"type":[1]}""", "'type' must be a string or an array of strings")
        assertRejected("""{"type":[]}""", "'type' must not be empty")
        assertRejected("""{"properties":{"a":false}}""", "'false' schema accepts nothing")
        assertRejected("""{"properties":{"a":1}}""", "a schema must be an object or a boolean")
        assertRejected("""{"properties":[1]}""", "'properties' must be an object")
        assertRejected("""{"properties":{"a":{}},"required":"a"}""", "'required' must be an array")
        assertRejected("""{"properties":{"a":{}},"required":[1]}""", "'required' must contain strings")
        assertRejected("""{"enum":"a"}""", "'enum' must be an array")
        assertRejected("""{"enum":[]}""", "'enum' must not be empty")
        assertRejected("""{"anyOf":{}}""", "'anyOf' / 'oneOf' must be an array")
        assertRejected("""{"oneOf":[]}""", "'anyOf' / 'oneOf' must not be empty")
    }

    @Test
    fun integerKeywordsMustBeIntegralAndWithinTheRepetitionLimit() {
        assertRejected("""{"type":"array","minItems":"2"}""", "'minItems' must be an integer")
        assertRejected("""{"type":"array","minItems":2.5}""", "'minItems' must be an integer")
        assertRejected("""{"type":"array","minItems":-1}""", "'minItems' must be in 0..4096, was -1")
        assertRejected("""{"type":"string","maxLength":4097}""", "'maxLength' must be in 0..4096, was 4097")
        assertRejected("""{"type":"string","minLength":2,"maxLength":1}""", "maxLength (1) < minLength (2)")
        assertEquals(""""\"" char{0,4096} "\"" space""", rules("""{"type":"string","maxLength":4096}""")["root-1"])
        // An integral number written with a fraction or exponent is accepted.
        assertEquals(
            """"[" space integer ( "," space integer )+ "]" space""",
            rules("""{"type":"array","items":{"type":"integer"},"minItems":2.0}""")["root-1"],
        )
        assertEquals(""""\"" char{10,} "\"" space""", rules("""{"type":"string","minLength":1e1}""")["root-1"])
    }

    @Test
    fun stringLengthBounds() {
        assertEquals(""""\"" char{3,} "\"" space""", rules("""{"type":"string","minLength":3}""")["root-1"])
        assertEquals(""""\"" char{0,4} "\"" space""", rules("""{"type":"string","maxLength":4}""")["root-1"])
        assertEquals(""""\"" char{2} "\"" space""", rules("""{"type":"string","minLength":2,"maxLength":2}""")["root-1"])
        // minLength 0 without maxLength is the unconstrained primitive.
        assertEquals("string", rules("""{"type":"string","minLength":0}""")["root"])
    }

    @Test
    fun arrayShapesWithoutExplicitType() {
        assertEquals(
            """"[" space ( boolean ( "," space boolean )* )? "]" space""",
            rules("""{"items":{"type":"boolean"}}""")["root-1"],
        )
        assertEquals(""""[" space "]" space""", rules("""{"prefixItems":[]}""")["root-1"])
        assertEquals(
            """"[" space value ( "," space value )* "]" space""",
            rules("""{"type":"array","minItems":1}""")["root-1"],
        )
    }

    @Test
    fun mapObjectsWithoutExplicitType() {
        val g = rules("""{"additionalProperties":{"type":"integer"}}""")
        assertEquals("""string ":" space integer""", g["root-kv"])
        assertEquals(""""{" space ( root-kv ( "," space root-kv )* )? "}" space""", g["root-1"])
        assertEquals("""string ":" space value""", rules("""{"type":"object","additionalProperties":true}""")["root-kv"])
    }

    @Test
    fun oneOfConstAndNullableTypeLists() {
        assertEquals("""integer | "\"x\"" space""", rules("""{"oneOf":[{"type":"integer"},{"const":"x"}]}""")["root-1"])
        assertEquals(""""1.5e2" space""", rules("""{"const":1.5e2}""")["root"])
        // Sibling keywords apply to every member of a type list.
        val g = rules("""{"type":["string","null"],"maxLength":3}""")
        assertEquals("root-0 | null", g["root-1"])
        assertEquals(""""\"" char{0,3} "\"" space""", g["root-0"])
    }

    @Test
    fun refsResolveIntoArraysAndUnescapeJsonPointerTokens() {
        val g = rules("""{"defs":[{"type":"integer"},{"type":"boolean"}],"${d}ref":"#/defs/1"}""")
        assertEquals("ref-1", g["root"])
        assertEquals("boolean", g["ref-1"])

        // RFC 6901: "~1" is '/', "~0" is '~'.
        val escaped = rules(
            """{"defs":{"a/b":{"type":"null"},"c~d":{"type":"string"}},
               "properties":{"x":{"${d}ref":"#/defs/a~1b"},"y":{"${d}ref":"#/defs/c~0d"}},"required":["x","y"]}""",
        )
        assertEquals("null", escaped["ref-a-1b"])
        assertEquals("string", escaped["ref-c-0d"])
        assertEquals(""""\"x\"" space ":" space ref-a-1b""", escaped["root-x-kv"])
        assertEquals(""""\"y\"" space ":" space ref-c-0d""", escaped["root-y-kv"])
    }

    @Test
    fun aRefUsedTwiceIsConvertedOnceAndDistinctTargetsWithTheSameNameAreUniquified() {
        val shared = rules(
            """{"${d}defs":{"n":{"type":"number"}},"properties":{"a":{"${d}ref":"#/${d}defs/n"},"b":{"${d}ref":"#/${d}defs/n"}}}""",
        )
        assertEquals("number", shared["ref-n"])
        assertTrue(shared.getValue("root-a-kv").endsWith(" ref-n"))
        assertTrue(shared.getValue("root-b-kv").endsWith(" ref-n"))
        assertFalse("ref-n-1" in shared)

        val clash = rules(
            """{"${d}defs":{"p":{"x":{"type":"integer"}},"q":{"x":{"type":"string"}}},
               "properties":{"a":{"${d}ref":"#/${d}defs/p/x"},"b":{"${d}ref":"#/${d}defs/q/x"}}}""",
        )
        assertEquals("integer", clash["ref-x"])
        assertEquals("string", clash["ref-x-1"])
    }

    @Test
    fun badRefsAreRejected() {
        assertRejected("""{"${d}ref":5}""", "'${d}ref' must be a string")
        assertRejected("""{"defs":[{}],"${d}ref":"#/defs/x"}""", "'${d}ref' target '#/defs/x' does not exist")
        assertRejected("""{"defs":[{}],"${d}ref":"#/defs/1"}""", "'${d}ref' target '#/defs/1' does not exist")
        assertRejected("""{"n":1,"${d}ref":"#/n/0"}""", "'${d}ref' target '#/n/0' does not exist")
    }

    @Test
    fun identicalRulesAreSharedAndClashingNamesUniquified() {
        // "x_y" and "x-y" map to the same rule name: the identical enum body is reused, the different kv bodies are not.
        val g = rules("""{"properties":{"x_y":{"enum":[1]},"x-y":{"enum":[1]}}}""")
        assertEquals(""""1" space""", g["root-x-y"])
        assertFalse("root-x-y-1" in g)
        assertEquals(""""\"x_y\"" space ":" space root-x-y""", g["root-x-y-kv"])
        assertEquals(""""\"x-y\"" space ":" space root-x-y""", g["root-x-y-kv-1"])
        assertEquals("""root-x-y-kv ( "," space root-x-y-kv-1 )? | root-x-y-kv-1""", g["root-rest-0"])
        assertEquals(""""{" space ( root-rest-0 )? "}" space""", g["root-1"])
    }

    @Test
    fun allOfFollowsRefsAndNestedAllOf() {
        val schema = """
            {"${d}defs":{"base":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}},
             "allOf":[{"${d}ref":"#/${d}defs/base"},{"allOf":[{"properties":{"name":{"type":"string"}}}]}]}
        """
        val g = rules(schema)
        assertEquals(""""{" space root-id-kv ( "," space root-name-kv )? "}" space""", g["root-1"])
        assertEquals(""""\"name\"" space ":" space string""", g["root-name-kv"])
    }

    @Test
    fun allOfRejectsNonObjectComponents() {
        assertRejected("""{"allOf":[1]}""", "'allOf' items must be object schemas")
        assertRejected("""{"allOf":[{"type":"string"}]}""", "'allOf' is only supported for object schemas")
        assertRejected("""{"allOf":[{"required":[1]}]}""", "bad 'required'")
        assertRejected("""{"allOf":[{"${d}ref":1}]}""", "bad '${d}ref'")
    }

    @Test
    fun schemaNestingIsLimitedTo64Levels() {
        fun nestedArrays(n: Int): String = if (n == 0) "{}" else """{"type":"array","items":${nestedArrays(n - 1)}}"""
        // The root is depth 0, so 64 nested `items` put the innermost schema at the deepest allowed depth.
        val deepest = rules(nestedArrays(64))["root" + "-item".repeat(63)]
        assertEquals(""""[" space ( value ( "," space value )* )? "]" space""", deepest)
        assertRejected(nestedArrays(65), "schema nesting deeper than 64")

        fun nestedAllOf(n: Int): String = if (n == 0) """{"properties":{"a":{}}}""" else """{"allOf":[${nestedAllOf(n - 1)}]}"""
        assertEquals(""""{" space ( root-a-kv )? "}" space""", rules(nestedAllOf(64))["root-1"])
        assertRejected(nestedAllOf(65), "allOf nesting deeper than 64")
    }

    @Test
    fun literalAndNameHelpers() {
        assertEquals(""""\r\t\x7F\x1F"""", JsonSchemaToGbnf.gbnfLiteral("\r\t\u007F\u001F"))
        assertEquals("\"é\"", JsonSchemaToGbnf.gbnfLiteral("é"), "non-ASCII passes through")
        assertEquals("rule", JsonSchemaToGbnf.ruleName(""))
        assertEquals("a-b-C9", JsonSchemaToGbnf.ruleName("a b_C9"))
        assertEquals("{1,3}", JsonSchemaToGbnf.repetition(1, 3))
        assertEquals("{0}", JsonSchemaToGbnf.repetition(0, 0))
    }
}
