package com.donfuldauer.kggml.samples.showcase

import kotlin.math.abs
import kotlin.math.roundToLong

/** `"ModelLoadException: no such file"` for any failure shown on screen. */
fun Throwable.describe(): String = "${this::class.simpleName ?: "Error"}: ${message ?: "(no message)"}"

/** Human-readable binary size: `512 B`, `1.5 KiB`, `74.1 MiB`, `3.86 GiB`. */
fun formatBytes(bytes: Long): String {
    val units = listOf("B", "KiB", "MiB", "GiB", "TiB")
    var value = abs(bytes).toDouble()
    var unit = 0
    while (value >= 1024 && unit < units.lastIndex) {
        value /= 1024
        unit++
    }
    val sign = if (bytes < 0) "-" else ""
    return if (unit == 0) "$sign${abs(bytes)} B" else "$sign${formatDecimal(value, if (value < 10) 2 else 1)} ${units[unit]}"
}

/** Fixed-point formatting without platform `String.format` (unavailable in common code). */
fun formatDecimal(value: Double, decimals: Int): String {
    if (value.isNaN()) return "NaN"
    var factor = 1L
    repeat(decimals) { factor *= 10 }
    val scaled = (abs(value) * factor).roundToLong()
    val whole = scaled / factor
    val fraction = (scaled % factor).toString().padStart(decimals, '0')
    val sign = if (value < 0 && scaled != 0L) "-" else ""
    return if (decimals == 0) "$sign$whole" else "$sign$whole.$fraction"
}

/** `1.2 B`, `494 M`, `39 K` parameter counts. */
fun formatCount(count: Long): String = when {
    count >= 1_000_000_000 -> "${formatDecimal(count / 1e9, 2)} B"
    count >= 1_000_000 -> "${formatDecimal(count / 1e6, 1)} M"
    count >= 1_000 -> "${formatDecimal(count / 1e3, 1)} K"
    else -> count.toString()
}
