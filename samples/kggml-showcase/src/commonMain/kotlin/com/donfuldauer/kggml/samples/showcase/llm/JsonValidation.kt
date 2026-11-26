package com.donfuldauer.kggml.samples.showcase.llm

/** A small, strict JSON (RFC 8259) parser for the structured-output playground's live validation. */
object MiniJson {
    /** Thrown for malformed JSON; [position] is the character offset of the problem. */
    class ParseException(message: String, val position: Int, val atEnd: Boolean) : IllegalArgumentException(message)

    /**
     * Parses [text] into `Map<String, Any?>` (objects, insertion-ordered), `List<Any?>`, `String`, `Double` / `Long`,
     * `Boolean` or `null`.
     */
    fun parse(text: String): Any? {
        val parser = Parser(text)
        parser.skipWhitespace()
        val value = parser.value()
        parser.skipWhitespace()
        if (!parser.atEnd()) parser.fail("unexpected trailing content")
        return value
    }

    private class Parser(private val s: String) {
        private var i = 0

        fun atEnd() = i >= s.length

        fun fail(message: String): Nothing =
            throw ParseException("$message at offset $i", i, atEnd())

        fun skipWhitespace() {
            while (i < s.length && s[i] in " \t\r\n") i++
        }

        fun value(): Any? {
            if (atEnd()) fail("unexpected end of input")
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> array()
                '"' -> string()
                't' -> literal("true", true)
                'f' -> literal("false", false)
                'n' -> literal("null", null)
                else -> if (c == '-' || c.isAsciiDigit()) number() else fail("unexpected character '$c'")
            }
        }

        private fun obj(): Map<String, Any?> {
            i++
            val out = LinkedHashMap<String, Any?>()
            skipWhitespace()
            if (peek() == '}') {
                i++
                return out
            }
            while (true) {
                skipWhitespace()
                if (peek() != '"') fail("expected a property name")
                val key = string()
                skipWhitespace()
                expect(':')
                skipWhitespace()
                out[key] = value()
                skipWhitespace()
                when (peek()) {
                    ',' -> i++
                    '}' -> {
                        i++
                        return out
                    }
                    else -> fail("expected ',' or '}'")
                }
            }
        }

        private fun array(): List<Any?> {
            i++
            val out = ArrayList<Any?>()
            skipWhitespace()
            if (peek() == ']') {
                i++
                return out
            }
            while (true) {
                skipWhitespace()
                out += value()
                skipWhitespace()
                when (peek()) {
                    ',' -> i++
                    ']' -> {
                        i++
                        return out
                    }
                    else -> fail("expected ',' or ']'")
                }
            }
        }

        private fun string(): String {
            i++
            val sb = StringBuilder()
            while (true) {
                if (atEnd()) fail("unterminated string")
                val c = s[i++]
                when {
                    c == '"' -> return sb.toString()
                    c == '\\' -> {
                        if (atEnd()) fail("unterminated escape")
                        when (val e = s[i++]) {
                            '"', '\\', '/' -> sb.append(e)
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> {
                                val hex = s.substring(i, minOf(i + 4, s.length))
                                if (!hex.all { it.isAsciiHexDigit() }) fail("invalid \\u escape")
                                if (hex.length < 4) {
                                    // Cut off by the end of the (possibly still streaming) text: incomplete, not invalid.
                                    i = s.length
                                    fail("truncated \\u escape")
                                }
                                val code = hex.toInt(16)
                                sb.append(code.toChar())
                                i += 4
                            }
                            else -> fail("invalid escape '\\$e'")
                        }
                    }
                    c < ' ' -> {
                        i--
                        fail("control character in string")
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun number(): Any {
            val start = i
            if (peek() == '-') i++
            if (peek() == '0') {
                i++
            } else {
                if (peek()?.isAsciiDigit() != true) fail("expected a digit")
                while (peek()?.isAsciiDigit() == true) i++
            }
            var integral = true
            if (peek() == '.') {
                integral = false
                i++
                if (peek()?.isAsciiDigit() != true) fail("expected a digit after '.'")
                while (peek()?.isAsciiDigit() == true) i++
            }
            if (peek() == 'e' || peek() == 'E') {
                integral = false
                i++
                if (peek() == '+' || peek() == '-') i++
                if (peek()?.isAsciiDigit() != true) fail("expected an exponent digit")
                while (peek()?.isAsciiDigit() == true) i++
            }
            val text = s.substring(start, i)
            return if (integral) text.toLongOrNull() ?: text.toDouble() else text.toDouble()
        }

        private fun literal(word: String, value: Any?): Any? {
            if (!s.startsWith(word, i)) {
                if (word.startsWith(s.substring(i))) {
                    i = s.length
                    fail("unexpected end of input")
                }
                fail("invalid literal")
            }
            i += word.length
            return value
        }

        private fun expect(c: Char) {
            if (peek() != c) fail("expected '$c'")
            i++
        }

        private fun peek(): Char? = if (i < s.length) s[i] else null
    }
}

/** Live validation result of the extraction output. */
sealed interface JsonValidation {
    val message: String

    /** Nothing to validate yet. */
    data object Empty : JsonValidation {
        override val message: String = "No output yet"
    }

    /** Well-formed so far but not complete (streaming in progress). */
    data class Incomplete(override val message: String) : JsonValidation

    /** Malformed JSON. */
    data class Invalid(override val message: String) : JsonValidation

    /** Well-formed JSON that does not satisfy the schema. */
    data class SchemaViolation(val problems: List<String>) : JsonValidation {
        override val message: String get() = "Schema violations: ${problems.joinToString("; ")}"
    }

    /** Well-formed JSON matching the schema. */
    data class Valid(val value: Any?) : JsonValidation {
        override val message: String = "Valid JSON matching the schema"
    }
}

/**
 * Validates [json] against a JSON [schema] subset: `type` (`object`, `array`, `string`, `integer`, `number`,
 * `boolean`, `null`), `properties`, `required`, `items`, `enum`. A parse error at the very end of the text is
 * [JsonValidation.Incomplete] (the model is still streaming), anywhere else [JsonValidation.Invalid].
 */
fun validateJson(json: String, schema: String?): JsonValidation {
    if (json.isBlank()) return JsonValidation.Empty
    val value = try {
        MiniJson.parse(json)
    } catch (e: MiniJson.ParseException) {
        return if (e.atEnd) JsonValidation.Incomplete("Incomplete JSON: ${e.message}") else JsonValidation.Invalid("Invalid JSON: ${e.message}")
    }
    val schemaValue = schema?.takeIf { it.isNotBlank() }?.let { runCatching { MiniJson.parse(it) }.getOrNull() }
    val problems = ArrayList<String>()
    if (schemaValue is Map<*, *>) checkSchema(value, schemaValue, "$", problems)
    return if (problems.isEmpty()) JsonValidation.Valid(value) else JsonValidation.SchemaViolation(problems)
}

/** `null` if [schema] is a well-formed JSON object, otherwise the problem. */
fun schemaProblem(schema: String): String? = try {
    if (MiniJson.parse(schema) is Map<*, *>) null else "The schema must be a JSON object"
} catch (e: MiniJson.ParseException) {
    "Schema is not valid JSON: ${e.message}"
}

private fun checkSchema(value: Any?, schema: Map<*, *>, path: String, problems: MutableList<String>) {
    (schema["enum"] as? List<*>)?.let { options ->
        if (options.none { sameJson(it, value) }) problems += "$path is not one of ${options.joinToString()}"
    }
    when (val type = schema["type"] as? String) {
        null -> Unit
        "object" -> {
            if (value !is Map<*, *>) {
                problems += "$path must be an object"
                return
            }
            (schema["required"] as? List<*>)?.filterIsInstance<String>()?.forEach { key ->
                if (!value.containsKey(key)) problems += "$path.$key is required"
            }
            (schema["properties"] as? Map<*, *>)?.forEach { (key, sub) ->
                if (key is String && sub is Map<*, *> && value.containsKey(key)) checkSchema(value[key], sub, "$path.$key", problems)
            }
        }
        "array" -> {
            if (value !is List<*>) {
                problems += "$path must be an array"
                return
            }
            val items = schema["items"] as? Map<*, *> ?: return
            value.forEachIndexed { index, item -> checkSchema(item, items, "$path[$index]", problems) }
        }
        "string" -> if (value !is String) problems += "$path must be a string"
        "integer" -> if (!(value is Long || (value is Double && value % 1.0 == 0.0))) problems += "$path must be an integer"
        "number" -> if (value !is Long && value !is Double) problems += "$path must be a number"
        "boolean" -> if (value !is Boolean) problems += "$path must be a boolean"
        "null" -> if (value != null) problems += "$path must be null"
        else -> Unit
    }
}

private fun sameJson(a: Any?, b: Any?): Boolean = when {
    a is Number && b is Number -> a.toDouble() == b.toDouble()
    else -> a == b
}

/** JSON (RFC 8259) digits are ASCII only; [Char.isDigit] would also accept e.g. Arabic-Indic digits. */
private fun Char.isAsciiDigit(): Boolean = this in '0'..'9'

private fun Char.isAsciiHexDigit(): Boolean = this in '0'..'9' || this in 'a'..'f' || this in 'A'..'F'
