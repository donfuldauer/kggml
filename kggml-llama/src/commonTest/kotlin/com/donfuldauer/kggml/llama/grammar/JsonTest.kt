package com.donfuldauer.kggml.llama.grammar

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class JsonTest {
    @Test
    fun parsesAndReencodesDocuments() {
        val text = """ { "a" : [1, -2.5e3, true, false, null], "b\n\u00e9" : {"c": "x\"y"}, "d": {} } """
        val doc = Json.parse(text)
        assertEquals("""{"a":[1,-2.5e3,true,false,null],"b\né":{"c":"x\"y"},"d":{}}""", Json.encode(doc))
        val obj = doc as Json.Obj
        assertEquals(listOf("a", "b\né", "d"), obj.members.keys.toList())
        assertEquals(3L, (Json.parse("3.0") as Json.Num).asLong)
        assertEquals(null, (Json.parse("3.5") as Json.Num).asLong)
        assertEquals("\"\\u0001\\t\"", Json.encode(Json.Str("\u0001\t")))
    }

    @Test
    fun rejectsMalformedInput() {
        for (bad in listOf("", "{", "[1,]", "{\"a\" 1}", "01", "1.", "tru", "\"\\x\"", "{\"a\":1,\"a\":2}", "1 2", "\"a\nb\"")) {
            assertFailsWith<JsonException>(bad) { Json.parse(bad) }
        }
        assertFailsWith<JsonException> { Json.parse("[".repeat(1000)) }
    }
}
