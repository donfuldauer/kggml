package com.donfuldauer.kggml.samples.showcase.whisper

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.samples.showcase.ShowcaseDestination
import com.donfuldauer.kggml.samples.showcase.displayName

/** Test tags of the Whisper Studio controls. */
object WhisperStudioTags {
    const val MODEL_PATH = "whisper-model-path"
    const val FILE_PATH = "whisper-file-path"
    const val START = "whisper-start"
    const val STOP = "whisper-stop"
    const val CANCEL = "whisper-cancel"
    const val TRANSCRIBE_FILE = "whisper-transcribe-file"
    const val BACKGROUND_STRESS = "whisper-background-stress"
    const val WAVEFORM = "whisper-waveform"
    const val ACTIVITY = "whisper-activity"
    const val ACTIVE_BACKEND = "whisper-active-backend"
    const val PARTIAL = "whisper-partial"
    const val STATUS = "whisper-status"
    const val ERROR = "whisper-error"
    const val MIC_UNAVAILABLE = "whisper-mic-unavailable"

    fun backendChip(backend: ComputeBackend): String = "whisper-backend-${backend.name.lowercase()}"
}

@Composable
fun WhisperStudioScreen(controller: WhisperStudioController) {
    val state by controller.state.collectAsState()
    WhisperStudioContent(
        state = state,
        onModelPathChange = controller::setModelPath,
        onFilePathChange = controller::setFilePath,
        onSelectBackend = controller::selectBackend,
        onBackgroundStressChange = controller::setBackgroundStress,
        onStart = controller::startLive,
        onStop = controller::stop,
        onCancel = controller::cancel,
        onTranscribeFile = controller::transcribeFile,
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun WhisperStudioContent(
    state: WhisperStudioState,
    onModelPathChange: (String) -> Unit = {},
    onFilePathChange: (String) -> Unit = {},
    onSelectBackend: (ComputeBackend) -> Unit = {},
    onBackgroundStressChange: (Boolean) -> Unit = {},
    onStart: () -> Unit = {},
    onStop: () -> Unit = {},
    onCancel: () -> Unit = {},
    onTranscribeFile: () -> Unit = {},
) {
    val destination = ShowcaseDestination.WHISPER_STUDIO
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
            .testTag(destination.screenTag),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(destination.title, style = MaterialTheme.typography.headlineSmall)
        Text(destination.summary, style = MaterialTheme.typography.bodyMedium)

        OutlinedTextField(
            value = state.modelPath,
            onValueChange = onModelPathChange,
            label = { Text("Whisper model (.bin)") },
            singleLine = true,
            enabled = !state.isBusy,
            modifier = Modifier.fillMaxWidth().testTag(WhisperStudioTags.MODEL_PATH),
        )

        Text("Compute backend", fontWeight = FontWeight.Bold)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            STUDIO_BACKENDS.forEach { backend ->
                FilterChip(
                    selected = state.selectedBackend == backend,
                    onClick = { onSelectBackend(backend) },
                    enabled = !state.isBusy,
                    label = { Text(backend.displayName()) },
                    modifier = Modifier.testTag(WhisperStudioTags.backendChip(backend)),
                )
            }
        }
        Text(
            text = state.activeBackend?.let { "Active: ${it.displayName()} (use_gpu=${state.usesGpu})" }
                ?: "Model not loaded",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.testTag(WhisperStudioTags.ACTIVE_BACKEND),
        )

        if (state.backgroundStressSupported) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Switch(
                    checked = state.backgroundStress,
                    onCheckedChange = onBackgroundStressChange,
                    enabled = !state.isBusy,
                    modifier = Modifier.testTag(WhisperStudioTags.BACKGROUND_STRESS),
                )
                Column {
                    Text("Background stress test", fontWeight = FontWeight.Bold)
                    Text(
                        "Keeps the selected backend in the background (no automatic CPU fallback): CPU stays " +
                            "transcribing, Metal is cut off by iOS.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Button(onClick = onStart, enabled = state.canStartLive, modifier = Modifier.testTag(WhisperStudioTags.START)) {
                Text("Start dictation")
            }
            OutlinedButton(onClick = onStop, enabled = state.canStop, modifier = Modifier.testTag(WhisperStudioTags.STOP)) {
                Text("Stop")
            }
            OutlinedButton(onClick = onCancel, enabled = state.canCancel, modifier = Modifier.testTag(WhisperStudioTags.CANCEL)) {
                Text("Cancel")
            }
        }
        if (!state.micAvailable) {
            Text(
                "Live capture is not available on this platform.",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.testTag(WhisperStudioTags.MIC_UNAVAILABLE),
            )
        }
        Text(
            text = buildString {
                append(state.activity.label)
                state.captureFormat?.takeIf { state.source == TranscriptSource.LIVE }?.let {
                    append(" · ${it.sampleRate} Hz × ${it.channels} ch · ${formatDuration(state.capturedMs)} captured")
                }
            },
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.testTag(WhisperStudioTags.ACTIVITY),
        )

        Waveform(state.levels)

        state.partial?.let { partial ->
            Card(
                modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {}.testTag(WhisperStudioTags.PARTIAL),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            ) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(partial.text.ifEmpty { "…" }, fontStyle = FontStyle.Italic)
                    Text(
                        "Partial · ${formatTimestamp(partial.startMs)} → ${formatTimestamp(partial.endMs)} · " +
                            "lag ${partial.lagMs} ms (target < 400 ms)",
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
        }

        if (state.committed.isNotEmpty()) {
            Text(
                if (state.source == TranscriptSource.FILE) "File transcript" else "Committed segments",
                fontWeight = FontWeight.Bold,
            )
        }
        state.committed.forEach { line ->
            Row(
                modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {}.testTag(line.testTag),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(line.timeLabel, style = MaterialTheme.typography.labelMedium)
                Text(line.text, fontWeight = FontWeight.Medium)
            }
        }

        state.status?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag(WhisperStudioTags.STATUS))
        }
        state.error?.let {
            Card(
                modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {}.testTag(WhisperStudioTags.ERROR),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
            ) {
                Text(it, modifier = Modifier.padding(12.dp))
            }
        }

        Text("Audio file", fontWeight = FontWeight.Bold)
        OutlinedTextField(
            value = state.filePath,
            onValueChange = onFilePathChange,
            label = { Text("WAV (16-bit / float) or raw 16 kHz PCM") },
            singleLine = true,
            enabled = !state.isBusy,
            modifier = Modifier.fillMaxWidth().testTag(WhisperStudioTags.FILE_PATH),
        )
        Button(
            onClick = onTranscribeFile,
            enabled = state.canTranscribeFile,
            modifier = Modifier.testTag(WhisperStudioTags.TRANSCRIBE_FILE),
        ) {
            Text("Transcribe file")
        }
    }
}

/** Scrolling bar visualizer of recent microphone RMS levels. */
@Composable
private fun Waveform(levels: List<Float>) {
    val color = MaterialTheme.colorScheme.primary
    val baseline = MaterialTheme.colorScheme.outlineVariant
    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp)
            .testTag(WhisperStudioTags.WAVEFORM)
            .semantics { contentDescription = "Microphone level, ${levels.size} samples" },
    ) {
        val mid = size.height / 2
        drawLine(baseline, Offset(0f, mid), Offset(size.width, mid), strokeWidth = 1f)
        if (levels.isEmpty()) return@Canvas
        val step = size.width / WhisperStudioController.DEFAULT_MAX_LEVELS
        val stroke = (step * 0.6f).coerceAtLeast(1f)
        val start = size.width - levels.size * step
        levels.forEachIndexed { i, level ->
            // Speech RMS is small; a square-root curve keeps quiet input visible.
            val half = (kotlin.math.sqrt(level) * mid).coerceAtLeast(1f)
            val x = start + i * step + step / 2
            drawLine(color, Offset(x, mid - half), Offset(x, mid + half), strokeWidth = stroke)
        }
    }
}
