package com.donfuldauer.kggml.llama.grammar

/**
 * Minimal JSON document model and strict RFC 8259 parser, used to read JSON Schemas (and by tests to check
 * constrained output). Object member order is preserved.
 */
internal sealed interface Json {
    data class Obj(val members: LinkedHashMap<String, Json>) : Json {
        operator fun get(key: String): Json? = members[key]
    }

    data class Arr(val items: List<Json>) : Json

    data class Str(val value: String) : Json

    /** [raw] is the number exactly as written (canonical for [encode]). */
    data class Num(val raw: String) : Json {
        val asLong: Long? get() = raw.toLongOrNull() ?: raw.toDoubleOrNull()?.takeIf { it == kotlin.math.floor(it) && !it.isInfinite() }?.toLong()
    }

    data class Bool(val value: Boolean) : Json

    data object Null : Json

    companion object {
        /** @throws JsonException on malformed input. */
        fun parse(text: String): Json = Parser(text).parseDocument()

        /** Compact JSON serialization of [value] (no whitespace). */
        fun encode(value: Json): String = buildString { write(value) }

        private fun StringBuilder.write(value: Json) {
            when (value) {
                is Obj -> {
                    append('{')
                    value.members.entries.forEachIndexed { i, (k, v) ->
                        if (i > 0) append(',')
                        writeString(k)
                        append(':')
                        write(v)
                    }
                    append('}')
                }
                is Arr -> {
                    append('[')
                    value.items.forEachIndexed { i, v ->
                        if (i > 0) append(',')
                        write(v)
                    }
                    append(']')
                }
                is Str -> writeString(value.value)
                is Num -> append(value.raw)
                is Bool -> append(value.value)
                Null -> append("null")
            }
        }

        private fun StringBuilder.writeString(s: String) {
            append('"')
            for (c in s) {
                when {
                    c == '"' -> append("\\\"")
                    c == '\\' -> append("\\\\")
                    c == '\n' -> append("\\n")
                    c == '\r' -> append("\\r")
                    c == '\t' -> append("\\t")
                    c == '\b' -> append("\\b")
                    c == '\u000C' -> append("\\f")
                    c < ' ' -> append("\\u").append(c.code.toString(16).padStart(4, '0'))
                    else -> append(c)
                }
            }
            append('"')
        }
    }
}

internal class JsonException(message: String) : IllegalArgumentException(message)

private class Parser(private val s: String) {
    private var i = 0

    fun parseDocument(): Json {
        val v = value(0)
        ws()
        if (i != s.length) fail("trailing characters")
        return v
    }

    private fun fail(msg: String): Nothing = throw JsonException("$msg at offset $i")

    private fun ws() {
        while (i < s.length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++
    }

    private fun value(depth: Int): Json {
        if (depth > MAX_DEPTH) fail("nesting deeper than $MAX_DEPTH")
        ws()
        if (i >= s.length) fail("unexpected end of input")
        return when (s[i]) {
            '{' -> obj(depth)
            '[' -> arr(depth)
            '"' -> Json.Str(string())
            't' -> literal("true", Json.Bool(true))
            'f' -> literal("false", Json.Bool(false))
            'n' -> literal("null", Json.Null)
            else -> number()
        }
    }

    private fun literal(word: String, v: Json): Json {
        if (!s.startsWith(word, i)) fail("invalid literal")
        i += word.length
        return v
    }

    private fun obj(depth: Int): Json {
        i++ // {
        val members = LinkedHashMap<String, Json>()
        ws()
        if (i < s.length && s[i] == '}') {
            i++
            return Json.Obj(members)
        }
        while (true) {
            ws()
            if (i >= s.length || s[i] != '"') fail("expected object key")
            val key = string()
            ws()
            if (i >= s.length || s[i] != ':') fail("expected ':'")
            i++
            if (members.containsKey(key)) fail("duplicate key '$key'")
            members[key] = value(depth + 1)
            ws()
            if (i >= s.length) fail("unterminated object")
            when (s[i++]) {
                ',' -> continue
                '}' -> return Json.Obj(members)
                else -> { i--; fail("expected ',' or '}'") }
            }
        }
    }

    private fun arr(depth: Int): Json {
        i++ // [
        val items = ArrayList<Json>()
        ws()
        if (i < s.length && s[i] == ']') {
            i++
            return Json.Arr(items)
        }
        while (true) {
            items += value(depth + 1)
            ws()
            if (i >= s.length) fail("unterminated array")
            when (s[i++]) {
                ',' -> continue
                ']' -> return Json.Arr(items)
                else -> { i--; fail("expected ',' or ']'") }
            }
        }
    }

    private fun string(): String {
        i++ // opening quote
        val sb = StringBuilder()
        while (true) {
            if (i >= s.length) fail("unterminated string")
            val c = s[i++]
            when {
                c == '"' -> return sb.toString()
                c == '\\' -> {
                    if (i >= s.length) fail("unterminated escape")
                    when (val e = s[i++]) {
                        '"' -> sb.append('"')
                        '\\' -> sb.append('\\')
                        '/' -> sb.append('/')
                        'b' -> sb.append('\b')
                        'f' -> sb.append('\u000C')
                        'n' -> sb.append('\n')
                        'r' -> sb.append('\r')
                        't' -> sb.append('\t')
                        'u' -> {
                            if (i + 4 > s.length) fail("short \\u escape")
                            val hex = s.substring(i, i + 4)
                            // Exactly four hex digits (RFC 8259); toIntOrNull alone would also accept a sign.
                            if (!hex.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }) fail("bad \\u escape")
                            val code = hex.toInt(16)
                            sb.append(code.toChar())
                            i += 4
                        }
                        else -> fail("invalid escape '\\$e'")
                    }
                }
                c < ' ' -> fail("control character in string")
                else -> sb.append(c)
            }
        }
    }

    private fun number(): Json {
        val start = i
        if (i < s.length && s[i] == '-') i++
        if (i >= s.length) fail("invalid number")
        if (s[i] == '0') {
            i++
        } else if (s[i] in '1'..'9') {
            while (i < s.length && s[i].isAsciiDigit()) i++
        } else {
            fail("unexpected character '${s[i]}'")
        }
        if (i < s.length && s[i] == '.') {
            i++
            if (i >= s.length || !s[i].isAsciiDigit()) fail("invalid fraction")
            while (i < s.length && s[i].isAsciiDigit()) i++
        }
        if (i < s.length && (s[i] == 'e' || s[i] == 'E')) {
            i++
            if (i < s.length && (s[i] == '+' || s[i] == '-')) i++
            if (i >= s.length || !s[i].isAsciiDigit()) fail("invalid exponent")
            while (i < s.length && s[i].isAsciiDigit()) i++
        }
        return Json.Num(s.substring(start, i))
    }

    private fun Char.isAsciiDigit() = this in '0'..'9'

    private companion object {
        const val MAX_DEPTH = 256
    }
}
