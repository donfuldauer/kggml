package com.donfuldauer.kggml.samples.showcase

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.donfuldauer.kggml.core.ComputeBackend

/** `label: [-] value [+]` integer control. [tag] is on the value text; the buttons use `"$tag-dec"` / `"$tag-inc"`. */
@Composable
fun Stepper(label: String, valueText: String, tag: String, enabled: Boolean, onDecrement: () -> Unit, onIncrement: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(label, modifier = Modifier.widthIn(min = 110.dp))
        OutlinedButton(onClick = onDecrement, enabled = enabled, modifier = Modifier.testTag("$tag-dec")) { Text("−") }
        Text(valueText, textAlign = TextAlign.Center, modifier = Modifier.widthIn(min = 56.dp).testTag(tag))
        OutlinedButton(onClick = onIncrement, enabled = enabled, modifier = Modifier.testTag("$tag-inc")) { Text("+") }
    }
}

/** Compute backend selector chips; each chip is tagged `"$tagPrefix-<backend>"`. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun BackendChips(selected: ComputeBackend, tagPrefix: String, enabled: Boolean, onSelect: (ComputeBackend) -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ENGINE_BACKENDS.forEach { backend ->
            FilterChip(
                selected = selected == backend,
                onClick = { onSelect(backend) },
                enabled = enabled,
                label = { Text(backend.displayName()) },
                modifier = Modifier.testTag("$tagPrefix-${backend.name.lowercase()}"),
            )
        }
    }
}

/** Backends llama.cpp / stable-diffusion.cpp can be asked for (Core ML falls back to Metal / CPU in both). */
val ENGINE_BACKENDS: List<ComputeBackend> =
    listOf(ComputeBackend.AUTO, ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL, ComputeBackend.GPU_VULKAN)

@Composable
fun ErrorCard(message: String, tag: String) {
    Card(
        modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {}.testTag(tag),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
    ) {
        Text(message, modifier = Modifier.padding(12.dp))
    }
}

@Composable
fun SectionTitle(text: String) {
    Text(text, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
}

/** A card with a title and content, merged for accessibility and tests when [merge] is set. */
@Composable
fun InfoCard(tag: String, merge: Boolean = true, content: @Composable () -> Unit) {
    val modifier = Modifier.fillMaxWidth().let { if (merge) it.semantics(mergeDescendants = true) {} else it }.testTag(tag)
    Card(modifier = modifier) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) { content() }
    }
}
