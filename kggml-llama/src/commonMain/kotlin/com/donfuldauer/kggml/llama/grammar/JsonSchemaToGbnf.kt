package com.donfuldauer.kggml.llama.grammar

import com.donfuldauer.kggml.core.InvalidInputException

/**
 * Converts a JSON Schema into a llama.cpp GBNF grammar whose language is the set of (compact, lightly
 * whitespaced) JSON documents valid against the schema. Modelled on llama.cpp's `json-schema-to-grammar`, with the
 * same primitive rules, reimplemented in common Kotlin so it works identically on every platform.
 *
 * See [com.donfuldauer.kggml.llama.LlamaGrammar.jsonSchema] for the supported keywords.
 */
internal object JsonSchemaToGbnf {
    /** Keywords that restrict the accepted language in ways this converter cannot express: rejected, never ignored. */
    private val UNSUPPORTED = listOf("pattern", "patternProperties", "not", "if", "then", "else", "dependentSchemas")

    /**
     * @throws InvalidInputException if [schema] is not valid JSON, is not a schema, or uses an unsupported keyword.
     */
    fun convert(schema: String): String {
        val root = try {
            Json.parse(schema)
        } catch (e: JsonException) {
            throw InvalidInputException("JSON schema is not valid JSON: ${e.message}", e)
        }
        return try {
            Converter(root).run()
        } catch (e: SchemaException) {
            throw InvalidInputException("Unsupported JSON schema: ${e.message}", e)
        }
    }

    /** Quotes [text] as a GBNF string literal. */
    fun gbnfLiteral(text: String): String = buildString {
        append('"')
        for (c in text) {
            when {
                c == '"' -> append("\\\"")
                c == '\\' -> append("\\\\")
                c == '\n' -> append("\\n")
                c == '\r' -> append("\\r")
                c == '\t' -> append("\\t")
                c < ' ' || c == '\u007F' -> append("\\x").append(c.code.toString(16).uppercase().padStart(2, '0'))
                else -> append(c)
            }
        }
        append('"')
    }

    /** A GBNF rule name (`[a-zA-Z0-9-]+`) derived from [hint]. */
    fun ruleName(hint: String): String {
        val name = hint.map { if (it in 'a'..'z' || it in 'A'..'Z' || it in '0'..'9' || it == '-') it else '-' }.joinToString("")
        return name.ifEmpty { "rule" }
    }

    /** Primitive rules (name -> body), shared with llama.cpp's converter. */
    internal val PRIMITIVES: Map<String, String> = linkedMapOf(
        "space" to """| " " | "\n" [ \t]{0,20}""",
        "boolean" to """("true" | "false") space""",
        "null" to """"null" space""",
        "integral-part" to """[0] | [1-9] [0-9]{0,15}""",
        "decimal-part" to """[0-9]{1,16}""",
        "integer" to """("-"? integral-part) space""",
        "number" to """("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space""",
        "char" to """[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})""",
        "string" to """"\"" char* "\"" space""",
        "value" to """object | array | string | number | boolean | null""",
        "object" to """"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space""",
        "array" to """"[" space ( value ("," space value)* )? "]" space""",
    )

    private val PRIMITIVE_DEPS: Map<String, List<String>> = mapOf(
        "integer" to listOf("integral-part"),
        "number" to listOf("integral-part", "decimal-part"),
        "string" to listOf("char"),
        "value" to listOf("object", "array", "string", "number", "boolean", "null"),
        "object" to listOf("string", "value"),
        "array" to listOf("value"),
    )

    private class SchemaException(message: String) : Exception(message)

    private class Converter(private val rootSchema: Json) {
        private val rules = LinkedHashMap<String, String>()
        private val refRules = HashMap<String, String>()

        fun run(): String {
            usePrimitive("space")
            val expr = visit(rootSchema, "root", depth = 0)
            val out = StringBuilder()
            out.append("root ::= ").append(expr).append('\n')
            for ((name, body) in rules) if (name != "root") out.append(name).append(" ::= ").append(body).append('\n')
            return out.toString()
        }

        private fun fail(message: String): Nothing = throw SchemaException(message)

        private fun usePrimitive(name: String): String {
            if (name !in rules) {
                rules[name] = PRIMITIVES.getValue(name)
                PRIMITIVE_DEPS[name]?.forEach(::usePrimitive)
            }
            return name
        }

        /** Registers rule [hint] = [body], reusing an identical rule and uniquifying clashing names. */
        private fun addRule(hint: String, body: String): String {
            val base = ruleName(hint)
            var name = base
            var n = 1
            while (true) {
                val existing = rules[name]
                if (existing == null && name !in PRIMITIVES && name != "root") {
                    rules[name] = body
                    return name
                }
                if (existing == body) return name
                name = "$base-${n++}"
            }
        }

        /** Claims a fresh rule name for [hint] (never shared with an existing rule). */
        private fun reserveName(hint: String): String {
            val base = ruleName(hint)
            var name = base
            var n = 1
            while (name in rules || name in PRIMITIVES || name == "root") name = "$base-${n++}"
            rules[name] = PENDING
            return name
        }

        /** Returns a GBNF expression matching [schema] (a rule name or literal). */
        fun visit(schema: Json, hint: String, depth: Int): String {
            if (depth > MAX_DEPTH) fail("schema nesting deeper than $MAX_DEPTH")
            when (schema) {
                is Json.Bool -> return if (schema.value) usePrimitive("value") else fail("'false' schema accepts nothing")
                is Json.Obj -> {}
                else -> fail("a schema must be an object or a boolean")
            }
            for (k in UNSUPPORTED) if (schema[k] != null) fail("keyword '$k' is not supported")

            schema["\$ref"]?.let { ref -> return visitRef(ref, depth) }
            schema["const"]?.let { return literal(it) }
            schema["enum"]?.let { e ->
                val values = (e as? Json.Arr)?.items ?: fail("'enum' must be an array")
                if (values.isEmpty()) fail("'enum' must not be empty")
                return addRule(hint, values.joinToString(" | ") { literal(it) })
            }
            (schema["anyOf"] ?: schema["oneOf"])?.let { alts ->
                val items = (alts as? Json.Arr)?.items ?: fail("'anyOf' / 'oneOf' must be an array")
                if (items.isEmpty()) fail("'anyOf' / 'oneOf' must not be empty")
                return addRule(hint, items.mapIndexed { i, s -> visit(s, "$hint-$i", depth + 1) }.joinToString(" | "))
            }
            schema["allOf"]?.let { return visitObject(mergeAllOf(schema, depth), hint, depth) }

            return when (val type = schema["type"]) {
                null -> when {
                    schema["properties"] != null || schema["additionalProperties"] != null -> visitObject(schema, hint, depth)
                    schema["items"] != null || schema["prefixItems"] != null -> visitArray(schema, hint, depth)
                    else -> usePrimitive("value")
                }
                is Json.Arr -> {
                    if (type.items.isEmpty()) fail("'type' must not be empty")
                    val alts = type.items.mapIndexed { i, t ->
                        visit(Json.Obj(LinkedHashMap(schema.members).apply { put("type", t) }), "$hint-$i", depth + 1)
                    }
                    addRule(hint, alts.joinToString(" | "))
                }
                is Json.Str -> when (type.value) {
                    "object" -> visitObject(schema, hint, depth)
                    "array" -> visitArray(schema, hint, depth)
                    "string" -> visitString(schema, hint)
                    "number" -> usePrimitive("number")
                    "integer" -> usePrimitive("integer")
                    "boolean" -> usePrimitive("boolean")
                    "null" -> usePrimitive("null")
                    else -> fail("unknown type '${type.value}'")
                }
                else -> fail("'type' must be a string or an array of strings")
            }
        }

        private fun visitRef(ref: Json, depth: Int): String {
            val path = (ref as? Json.Str)?.value ?: fail("'\$ref' must be a string")
            refRules[path]?.let { return it }
            if (!path.startsWith("#/")) fail("only local '\$ref's (\"#/...\") are supported, was '$path'")
            val target = resolvePointer(path)
            val name = reserveName("ref-" + path.substringAfterLast('/'))
            refRules[path] = name // registered before visiting so recursive schemas terminate
            rules[name] = visit(target, name, depth + 1)
            return name
        }

        private fun resolvePointer(path: String): Json {
            var node: Json = rootSchema
            for (raw in path.removePrefix("#/").split('/')) {
                val key = raw.replace("~1", "/").replace("~0", "~")
                node = when (node) {
                    is Json.Obj -> node[key]
                    is Json.Arr -> key.toIntOrNull()?.let { node.items.getOrNull(it) }
                    else -> null
                } ?: fail("'\$ref' target '$path' does not exist")
            }
            return node
        }

        /** Merges the object components of `allOf` (and the schema's own properties) into one object schema. */
        private fun mergeAllOf(schema: Json.Obj, depth: Int): Json.Obj {
            val properties = LinkedHashMap<String, Json>()
            val required = LinkedHashSet<String>()
            fun absorb(s: Json, level: Int) {
                if (level > MAX_DEPTH) fail("allOf nesting deeper than $MAX_DEPTH")
                val obj = s as? Json.Obj ?: fail("'allOf' items must be object schemas")
                obj["\$ref"]?.let { r -> absorb(resolvePointer((r as? Json.Str)?.value ?: fail("bad '\$ref'")), level + 1) }
                (obj["allOf"] as? Json.Arr)?.items?.forEach { absorb(it, level + 1) }
                val type = obj["type"]
                if (type != null && type != Json.Str("object")) fail("'allOf' is only supported for object schemas")
                (obj["properties"] as? Json.Obj)?.members?.let(properties::putAll)
                (obj["required"] as? Json.Arr)?.items?.forEach { required += (it as? Json.Str)?.value ?: fail("bad 'required'") }
            }
            absorb(schema, depth)
            return Json.Obj(
                linkedMapOf(
                    "type" to Json.Str("object"),
                    "properties" to Json.Obj(properties),
                    "required" to Json.Arr(required.map(Json::Str)),
                ),
            )
        }

        private fun visitObject(schema: Json.Obj, hint: String, depth: Int): String {
            val properties = when (val p = schema["properties"]) {
                null -> emptyMap()
                is Json.Obj -> p.members
                else -> fail("'properties' must be an object")
            }
            val required = when (val r = schema["required"]) {
                null -> emptySet()
                is Json.Arr -> r.items.map { (it as? Json.Str)?.value ?: fail("'required' must contain strings") }.toSet()
                else -> fail("'required' must be an array")
            }
            usePrimitive("space")
            if (properties.isEmpty()) {
                val additional = schema["additionalProperties"]
                if (additional == Json.Bool(false)) return addRule(hint, """"{" space "}" space""")
                val valueExpr = if (additional is Json.Obj) visit(additional, "$hint-value", depth + 1) else usePrimitive("value")
                usePrimitive("string")
                val kv = addRule("$hint-kv", """string ":" space $valueExpr""")
                return addRule(hint, """"{" space ( $kv ( "," space $kv )* )? "}" space""")
            }
            val missing = required - properties.keys
            if (missing.isNotEmpty()) fail("required properties without a schema: ${missing.joinToString()}")

            val kvs = properties.entries.map { (key, propSchema) ->
                val valueExpr = visit(propSchema, "$hint-$key", depth + 1)
                key to addRule("$hint-$key-kv", "${gbnfLiteral(Json.encode(Json.Str(key)))} space \":\" space $valueExpr")
            }
            val req = kvs.filter { it.first in required }.map { it.second }
            val opt = kvs.filter { it.first !in required }.map { it.second }

            // Optional properties in declared order, any subset: R(i) = kv_i ("," space R(i+1))? | R(i+1).
            var rest: String? = null
            for (i in opt.indices.reversed()) {
                val next = rest
                rest = if (next == null) {
                    opt[i]
                } else {
                    addRule("$hint-rest-$i", "${opt[i]} ( \",\" space $next )? | $next")
                }
            }
            val body = StringBuilder("\"{\" space ")
            if (req.isNotEmpty()) {
                body.append(req.joinToString(" \",\" space "))
                if (rest != null) body.append(" ( \",\" space ").append(rest).append(" )?")
            } else if (rest != null) {
                body.append("( ").append(rest).append(" )?")
            }
            body.append(" \"}\" space")
            return addRule(hint, body.toString())
        }

        private fun visitArray(schema: Json.Obj, hint: String, depth: Int): String {
            usePrimitive("space")
            (schema["prefixItems"] as? Json.Arr)?.let { tuple ->
                val items = tuple.items.mapIndexed { i, s -> visit(s, "$hint-tuple-$i", depth + 1) }
                val inner = if (items.isEmpty()) "" else items.joinToString(" \",\" space ") + " "
                return addRule(hint, "\"[\" space $inner\"]\" space")
            }
            val item = when (val items = schema["items"]) {
                null -> usePrimitive("value")
                else -> visit(items, "$hint-item", depth + 1)
            }
            val min = intKeyword(schema, "minItems") ?: 0
            val max = intKeyword(schema, "maxItems")
            if (max != null && max < min) fail("maxItems ($max) < minItems ($min)")
            if (max == 0) return addRule(hint, """"[" space "]" space""")
            val more = "( \",\" space $item )"
            val body = if (min == 0) {
                "\"[\" space ( $item ${more}${repetition(0, max?.minus(1))} )? \"]\" space"
            } else {
                "\"[\" space $item ${more}${repetition(min - 1, max?.minus(1))} \"]\" space"
            }
            return addRule(hint, body)
        }

        private fun visitString(schema: Json.Obj, hint: String): String {
            val min = intKeyword(schema, "minLength") ?: 0
            val max = intKeyword(schema, "maxLength")
            if (min == 0 && max == null) return usePrimitive("string")
            if (max != null && max < min) fail("maxLength ($max) < minLength ($min)")
            usePrimitive("char")
            return addRule(hint, "\"\\\"\" char${repetition(min, max)} \"\\\"\" space")
        }

        private fun intKeyword(schema: Json.Obj, key: String): Int? {
            val v = schema[key] ?: return null
            val n = (v as? Json.Num)?.asLong ?: fail("'$key' must be an integer")
            if (n < 0 || n > MAX_REPETITION) fail("'$key' must be in 0..$MAX_REPETITION, was $n")
            return n.toInt()
        }

        private fun literal(value: Json): String {
            usePrimitive("space")
            return "${gbnfLiteral(Json.encode(value))} space"
        }
    }

    /** GBNF repetition suffix for `min..max` occurrences (`max == null`: unbounded). */
    internal fun repetition(min: Int, max: Int?): String = when {
        max == null && min == 0 -> "*"
        max == null && min == 1 -> "+"
        max == null -> "{$min,}"
        min == max -> "{$min}"
        else -> "{$min,$max}"
    }

    private const val PENDING = "\"\"" // placeholder body while a \$ref target is being converted
    private const val MAX_DEPTH = 64
    private const val MAX_REPETITION = 4096L
}
