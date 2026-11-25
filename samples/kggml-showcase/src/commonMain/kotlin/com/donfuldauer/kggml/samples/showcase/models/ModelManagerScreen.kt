package com.donfuldauer.kggml.samples.showcase.models

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.donfuldauer.kggml.core.MemoryPressureLevel
import com.donfuldauer.kggml.samples.showcase.ErrorCard
import com.donfuldauer.kggml.samples.showcase.InfoCard
import com.donfuldauer.kggml.samples.showcase.SectionTitle
import com.donfuldauer.kggml.samples.showcase.ShowcaseDestination
import com.donfuldauer.kggml.samples.showcase.formatBytes
import com.donfuldauer.kggml.samples.showcase.formatCount

/** Test tags of the Model Manager controls (catalog rows use [modelRowTag]). */
object ModelManagerTags {
    const val DIRECTORY = "models-directory"
    const val REFRESH = "models-refresh"
    const val CLEAR_PARTIALS = "models-clear-partials"
    const val INSPECT_PATH = "models-inspect-path"
    const val INSPECT = "models-inspect"
    const val INSPECTED = "models-inspected"
    const val RESIDENT = "models-resident"
    const val RESIDENT_SUMMARY = "models-resident-summary"
    const val EVICT_MODERATE = "models-evict-moderate"
    const val EVICT_CRITICAL = "models-evict-critical"
    const val SIMULATE_LOW_MEMORY = "models-simulate-low-memory"
    const val EVICTION_LOG = "models-eviction-log"
    const val STATUS = "models-status"
    const val ERROR = "models-error"

    fun residentEntry(index: Int) = "resident-$index"
}

@Composable
fun ModelManagerScreen(controller: ModelManagerController) {
    val state by controller.state.collectAsState()
    val resident by controller.residents.state.collectAsState()
    ModelManagerContent(
        state = state,
        resident = resident,
        onRefresh = controller::refresh,
        onDownload = controller::download,
        onCancel = controller::cancelDownload,
        onVerify = controller::verify,
        onDelete = controller::delete,
        onUse = controller::use,
        onClearPartials = controller::clearPartialDownloads,
        onInspectPathChange = controller::setInspectPath,
        onInspectPath = controller::inspectPath,
        onEvict = controller::evictResident,
        onSimulateLowMemory = { controller.simulateLowMemory(MemoryPressureLevel.CRITICAL) },
    )
}

@Composable
fun ModelManagerContent(
    state: ModelManagerState,
    resident: ResidentState,
    onRefresh: () -> Unit = {},
    onDownload: (String) -> Unit = {},
    onCancel: (String) -> Unit = {},
    onVerify: (String) -> Unit = {},
    onDelete: (String) -> Unit = {},
    onUse: (String) -> Unit = {},
    onClearPartials: () -> Unit = {},
    onInspectPathChange: (String) -> Unit = {},
    onInspectPath: () -> Unit = {},
    onEvict: (MemoryPressureLevel) -> Unit = {},
    onSimulateLowMemory: () -> Unit = {},
) {
    val destination = ShowcaseDestination.MODEL_MANAGER
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp).testTag(destination.screenTag),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(destination.title, style = MaterialTheme.typography.headlineSmall)
        Text(destination.summary, style = MaterialTheme.typography.bodyMedium)
        Text(
            "Storage: ${state.directory} · ${formatBytes(state.diskBytes)} on disk" + if (state.loading) " · scanning…" else "",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.testTag(ModelManagerTags.DIRECTORY),
        )
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onRefresh, modifier = Modifier.testTag(ModelManagerTags.REFRESH)) { Text("Refresh") }
            OutlinedButton(onClick = onClearPartials, modifier = Modifier.testTag(ModelManagerTags.CLEAR_PARTIALS)) {
                Text("Delete partial downloads")
            }
        }
        state.status?.let { Text(it, style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag(ModelManagerTags.STATUS)) }
        state.error?.let { ErrorCard(it, ModelManagerTags.ERROR) }

        SectionTitle("Catalog")
        state.rows.forEach { CatalogRowCard(it, onDownload, onCancel, onVerify, onDelete, onUse) }

        if (state.otherFiles.isNotEmpty()) {
            SectionTitle("Other files in the model directory")
            state.otherFiles.forEach { Text("${it.fileName} · ${formatBytes(it.sizeBytes)}", style = MaterialTheme.typography.bodySmall) }
        }

        SectionTitle("Inspect a model file")
        OutlinedTextField(
            value = state.inspectPath,
            onValueChange = onInspectPathChange,
            label = { Text("GGUF or Whisper .bin path") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth().testTag(ModelManagerTags.INSPECT_PATH),
        )
        Button(
            onClick = onInspectPath,
            enabled = state.inspectPath.isNotBlank() && !state.inspecting,
            modifier = Modifier.testTag(ModelManagerTags.INSPECT),
        ) { Text(if (state.inspecting) "Inspecting…" else "Inspect header") }
        state.inspected?.let { InfoCard(ModelManagerTags.INSPECTED) { MetadataLines(it) } }

        ResidentPanel(resident, onEvict, onSimulateLowMemory)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun CatalogRowCard(
    row: CatalogRow,
    onDownload: (String) -> Unit,
    onCancel: (String) -> Unit,
    onVerify: (String) -> Unit,
    onDelete: (String) -> Unit,
    onUse: (String) -> Unit,
) {
    val id = row.model.id
    val tag = row.tag
    Card(modifier = Modifier.fillMaxWidth().testTag(tag)) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(row.model.displayName, fontWeight = FontWeight.Bold)
            Text(
                "${row.model.kind.label} · ${row.model.license} · ≈${formatBytes(row.model.approxSizeBytes)} · ${row.model.fileName}",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(phaseText(row), modifier = Modifier.testTag("$tag-phase"))
            if (row.isTransferring || row.phase == DownloadPhase.PARTIAL) {
                val fraction = row.fraction
                if (fraction != null) {
                    LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth().testTag("$tag-progress"))
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth().testTag("$tag-progress"))
                }
            }
            row.verification?.let { verification ->
                Text(
                    verification.label + (row.sha256?.let { " · sha256 ${it.take(16)}…" } ?: ""),
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.testTag("$tag-verification"),
                )
            }
            row.metadata?.let { metadata ->
                Column(Modifier.semantics(mergeDescendants = true) {}.testTag("$tag-metadata")) { MetadataLines(metadata) }
            }
            row.metadataError?.let { Text(it, style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag("$tag-metadata-error")) }
            row.error?.let { Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.testTag("$tag-error")) }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                when {
                    row.isTransferring -> OutlinedButton(onClick = { onCancel(id) }, modifier = Modifier.testTag("$tag-cancel")) { Text("Pause") }
                    row.canDownload -> Button(onClick = { onDownload(id) }, modifier = Modifier.testTag("$tag-download")) {
                        Text(if (row.phase == DownloadPhase.PARTIAL) "Resume" else "Download")
                    }
                }
                if (row.isInstalled) {
                    Button(onClick = { onUse(id) }, modifier = Modifier.testTag("$tag-use")) { Text(useLabel(row.model.kind)) }
                    OutlinedButton(onClick = { onVerify(id) }, enabled = !row.working, modifier = Modifier.testTag("$tag-verify")) {
                        Text(if (row.working) "Hashing…" else "Verify SHA-256")
                    }
                    OutlinedButton(onClick = { onDelete(id) }, modifier = Modifier.testTag("$tag-delete")) { Text("Delete") }
                }
            }
        }
    }
}

private fun phaseText(row: CatalogRow): String = when (row.phase) {
    DownloadPhase.DOWNLOADING, DownloadPhase.PARTIAL -> {
        val total = row.totalBytes?.let { " / ${formatBytes(it)}" } ?: ""
        val percent = row.fraction?.let { " (${(it * 100).toInt()}%)" } ?: ""
        "${row.phase.label}: ${formatBytes(row.downloadedBytes)}$total$percent"
    }
    DownloadPhase.INSTALLED -> "Installed · ${formatBytes(row.downloadedBytes)}"
    else -> row.phase.label
}

private fun useLabel(kind: ModelKind): String = when (kind) {
    ModelKind.WHISPER -> "Use in Whisper"
    ModelKind.LLM, ModelKind.VISION_LLM -> "Use in Chat"
    ModelKind.VISION_PROJECTOR -> "Use as projector"
    ModelKind.DIFFUSION -> "Use in Diffusion"
}

@Composable
private fun MetadataLines(metadata: ModelMetadata) {
    val ram = metadata.estimatedPeakRam
    Text(
        listOfNotNull(metadata.format, metadata.architecture, metadata.name).joinToString(" · "),
        fontWeight = FontWeight.Medium,
    )
    Text(
        "${formatCount(metadata.parameterCount)} parameters · ${metadata.quantization ?: "unknown quantization"}" +
            (metadata.contextLength?.let { " · context $it" } ?: ""),
        style = MaterialTheme.typography.bodySmall,
    )
    Text("Estimated peak RAM: ${formatBytes(ram.totalBytes)}", fontWeight = FontWeight.Medium)
    Text(
        "weights ${formatBytes(ram.weightsBytes)} · KV cache ${formatBytes(ram.kvCacheBytes)} · " +
            "compute ${formatBytes(ram.computeBytes)} · overhead ${formatBytes(ram.overheadBytes)}",
        style = MaterialTheme.typography.bodySmall,
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ResidentPanel(resident: ResidentState, onEvict: (MemoryPressureLevel) -> Unit, onSimulateLowMemory: () -> Unit) {
    SectionTitle("Loaded engines & low-memory eviction")
    InfoCard(ModelManagerTags.RESIDENT, merge = false) {
        Text(
            "${resident.entries.size} engine(s) resident · ${formatBytes(resident.residentBytes)} of " +
                "${formatBytes(resident.budgetBytes)} budget · last OS level ${resident.lastLevel.name}",
            modifier = Modifier.testTag(ModelManagerTags.RESIDENT_SUMMARY),
        )
        if (resident.entries.isEmpty()) {
            Text("No engine loaded. Chat and Diffusion engines appear here while resident.", style = MaterialTheme.typography.bodySmall)
        }
        resident.entries.forEachIndexed { index, entry ->
            Text(
                "${entry.key} · ≈${formatBytes(entry.estimatedBytes)}",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.testTag(ModelManagerTags.residentEntry(index)),
            )
        }
        Text(
            "OS signals evict idle engines automatically (MODERATE trims to half the budget, CRITICAL closes every idle " +
                "engine); engines in use are never closed.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = { onEvict(MemoryPressureLevel.MODERATE) }, modifier = Modifier.testTag(ModelManagerTags.EVICT_MODERATE)) {
            Text("Trim to half budget")
        }
        OutlinedButton(onClick = { onEvict(MemoryPressureLevel.CRITICAL) }, modifier = Modifier.testTag(ModelManagerTags.EVICT_CRITICAL)) {
            Text("Evict all idle engines")
        }
        Button(onClick = onSimulateLowMemory, modifier = Modifier.testTag(ModelManagerTags.SIMULATE_LOW_MEMORY)) {
            Text("Simulate OS low-memory warning")
        }
    }
    if (resident.evictions.isNotEmpty()) {
        Column(Modifier.semantics(mergeDescendants = true) {}.testTag(ModelManagerTags.EVICTION_LOG)) {
            resident.evictions.forEach { record ->
                Text(
                    "${record.source.label}: ${record.level.name} → evicted " +
                        (if (record.evictedKeys.isEmpty()) "nothing" else record.evictedKeys.joinToString()),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}
