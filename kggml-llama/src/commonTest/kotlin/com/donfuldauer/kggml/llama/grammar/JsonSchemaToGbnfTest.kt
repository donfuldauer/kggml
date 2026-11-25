package com.donfuldauer.kggml.llama.grammar

import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.llama.LlamaGrammar
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JsonSchemaToGbnfTest {
    private fun rules(gbnf: String): Map<String, String> = gbnf.lines().filter { it.isNotBlank() }.associate {
        val (name, body) = it.split(" ::= ", limit = 2)
        name to body
    }

    /** Every rule referenced by name is defined (a cheap structural validity check of the grammar). */
    private fun assertClosed(gbnf: String) {
        val defined = rules(gbnf).keys
        val stripped = rules(gbnf).values.joinToString(" ") { body ->
            body.replace(Regex("\"(\\\\.|[^\"\\\\])*\""), " ").replace(Regex("\\[(\\\\.|[^\\]\\\\])*\\]"), " ")
        }
        val referenced = Regex("[a-zA-Z][a-zA-Z0-9-]*").findAll(stripped).map { it.value }.toSet()
        val missing = referenced - defined
        assertTrue(missing.isEmpty(), "undefined rules $missing in\n$gbnf")
        assertTrue("root" in defined)
    }

    @Test
    fun primitivesMatchLlamaCppDefinitions() {
        val g = rules(JsonSchemaToGbnf.convert("""{"type":"integer"}"""))
        assertEquals("integer", g["root"])
        assertEquals("""("-"? integral-part) space""", g["integer"])
        assertEquals("""[0] | [1-9] [0-9]{0,15}""", g["integral-part"])
        assertEquals("""| " " | "\n" [ \t]{0,20}""", g["space"])
        for (type in listOf("string", "number", "boolean", "null")) {
            assertEquals(type, rules(JsonSchemaToGbnf.convert("""{"type":"$type"}"""))["root"])
        }
        assertEquals("value", rules(JsonSchemaToGbnf.convert("{}"))["root"])
        assertEquals("value", rules(JsonSchemaToGbnf.convert("true"))["root"])
    }

    @Test
    fun objectsKeepDeclaredOrderWithOptionalTail() {
        val schema = """
            {"type":"object","properties":{
               "name":{"type":"string"},
               "age":{"type":"integer"},
               "nick":{"type":"string"},
               "tags":{"type":"array","items":{"type":"string"},"maxItems":3}
            },"required":["name","age"]}
        """
        val gbnf = JsonSchemaToGbnf.convert(schema)
        assertClosed(gbnf)
        val g = rules(gbnf)
        assertEquals("""root-1""", g["root"])
        assertEquals(""""{" space root-name-kv "," space root-age-kv ( "," space root-rest-0 )? "}" space""", g["root-1"])
        assertEquals("""root-nick-kv ( "," space root-tags-kv )? | root-tags-kv""", g["root-rest-0"])
        assertEquals(""""\"name\"" space ":" space string""", g["root-name-kv"])
        assertEquals(""""[" space ( string ( "," space string ){0,2} )? "]" space""", g["root-tags"])
    }

    @Test
    fun allOptionalObjectMayBeEmptyAndClosedObjectsAreSupported() {
        val g = rules(JsonSchemaToGbnf.convert("""{"properties":{"a":{"type":"boolean"}}}"""))
        assertEquals(""""{" space ( root-a-kv )? "}" space""", g["root-1"])
        val closed = rules(JsonSchemaToGbnf.convert("""{"type":"object","additionalProperties":false}"""))
        assertEquals(""""{" space "}" space""", closed["root-1"])
        val map = rules(JsonSchemaToGbnf.convert("""{"type":"object","additionalProperties":{"type":"number"}}"""))
        assertEquals("""string ":" space number""", map["root-kv"])
    }

    @Test
    fun enumsConstsAndUnionsBecomeAlternations() {
        val g = rules(JsonSchemaToGbnf.convert("""{"enum":["red","say \"hi\"",3,null]}"""))
        assertEquals(""""\"red\"" space | "\"say \\\"hi\\\"\"" space | "3" space | "null" space""", g["root-1"])
        assertEquals(""""true" space""", rules(JsonSchemaToGbnf.convert("""{"const":true}"""))["root"])
        val union = rules(JsonSchemaToGbnf.convert("""{"anyOf":[{"type":"string"},{"type":"null"}]}"""))
        assertEquals("string | null", union["root-1"])
        val typeList = rules(JsonSchemaToGbnf.convert("""{"type":["integer","boolean"]}"""))
        assertEquals("integer | boolean", typeList["root-1"])
    }

    @Test
    fun arraysHonourBounds() {
        fun body(schema: String) = rules(JsonSchemaToGbnf.convert(schema))["root-1"]
        assertEquals(""""[" space integer ( "," space integer ){1,3} "]" space""", body("""{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":4}"""))
        assertEquals(""""[" space integer ( "," space integer )* "]" space""", body("""{"type":"array","items":{"type":"integer"},"minItems":1}"""))
        assertEquals(""""[" space "]" space""", body("""{"type":"array","maxItems":0}"""))
        assertEquals(""""[" space string "," space number "]" space""", body("""{"prefixItems":[{"type":"string"},{"type":"number"}]}"""))
        assertEquals(""""\"" char{2,5} "\"" space""", body("""{"type":"string","minLength":2,"maxLength":5}"""))
    }

    @Test
    fun localRefsIncludingRecursionResolve() {
        val schema = """
            {"${'$'}defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},"next":{"anyOf":[{"${'$'}ref":"#/${'$'}defs/node"},{"type":"null"}]}},"required":["v","next"]}},
             "${'$'}ref":"#/${'$'}defs/node"}
        """
        val gbnf = JsonSchemaToGbnf.convert(schema)
        assertClosed(gbnf)
        assertEquals("ref-node", rules(gbnf)["root"])
    }

    @Test
    fun allOfMergesObjectComponents() {
        val schema = """{"allOf":[{"properties":{"a":{"type":"integer"}},"required":["a"]},{"properties":{"b":{"type":"string"}}}]}"""
        val g = rules(JsonSchemaToGbnf.convert(schema))
        assertEquals(""""{" space root-a-kv ( "," space root-b-kv )? "}" space""", g["root-1"])
    }

    @Test
    fun unsupportedOrMalformedSchemasAreRejected() {
        for (bad in listOf(
            "{not json",
            """{"type":"string","pattern":"^a+$"}""",
            """{"type":"wat"}""",
            """{"${'$'}ref":"https://example.com/schema.json"}""",
            """{"${'$'}ref":"#/${'$'}defs/missing"}""",
            """{"type":"object","properties":{},"required":["x"]}""".replace("{}", """{"y":{}}"""),
            """{"type":"array","minItems":3,"maxItems":1}""",
            "false",
            "[1]",
        )) {
            assertFailsWith<InvalidInputException>(bad) { JsonSchemaToGbnf.convert(bad) }
        }
    }

    @Test
    fun literalEscapingAndRuleNames() {
        assertEquals(""""a\"b\\c\n\x01"""", JsonSchemaToGbnf.gbnfLiteral("a\"b\\c\n\u0001"))
        assertEquals("my-prop-x", JsonSchemaToGbnf.ruleName("my_prop.x"))
        assertEquals("*", JsonSchemaToGbnf.repetition(0, null))
        assertEquals("+", JsonSchemaToGbnf.repetition(1, null))
        assertEquals("{3,}", JsonSchemaToGbnf.repetition(3, null))
        assertEquals("{2}", JsonSchemaToGbnf.repetition(2, 2))
    }

    @Test
    fun grammarFactories() {
        val json = LlamaGrammar.json()
        assertEquals("root", json.root)
        assertClosed(json.gbnf)
        assertEquals(LlamaGrammar.gbnf("root ::= \"x\""), LlamaGrammar.gbnf("root ::= \"x\""))
        assertFailsWith<IllegalArgumentException> { LlamaGrammar.gbnf("root ::= \"x\"", root = "bad_name") }
        assertFailsWith<IllegalArgumentException> { LlamaGrammar.gbnf(" ") }
        assertFailsWith<InvalidInputException> { LlamaGrammar.jsonSchema("""{"type":"string","pattern":"x"}""") }
    }
}
