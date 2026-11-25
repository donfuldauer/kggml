package com.donfuldauer.kggml.samples.showcase.llm

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.samples.showcase.BackendChips
import com.donfuldauer.kggml.samples.showcase.ErrorCard
import com.donfuldauer.kggml.samples.showcase.InfoCard
import com.donfuldauer.kggml.samples.showcase.SectionTitle
import com.donfuldauer.kggml.samples.showcase.ShowcaseDestination
import com.donfuldauer.kggml.samples.showcase.Stepper
import com.donfuldauer.kggml.samples.showcase.displayName
import com.donfuldauer.kggml.samples.showcase.formatDecimal
import com.donfuldauer.kggml.samples.showcase.models.ResidentState

/** Panels of the LLM screen below the model controls. */
enum class LlmPanel(val label: String) { CHAT("Chat"), STRUCTURED("Structured output"), EMBEDDINGS("Embeddings"), VISION("Vision") }

/** Test tags of the LLM Chat, Structured Output & Vision controls. */
object LlmChatTags {
    const val MODEL_PATH = "llm-model-path"
    const val PROJECTOR_PATH = "llm-projector-path"
    const val SYSTEM_PROMPT = "llm-system-prompt"
    const val BACKEND = "llm-backend"
    const val THREADS = "llm-threads"
    const val GPU_LAYERS = "llm-gpu-layers"
    const val MAX_TOKENS = "llm-max-tokens"
    const val LOAD = "llm-load"
    const val LOADED = "llm-loaded"
    const val ACTIVITY = "llm-activity"
    const val CANCEL = "llm-cancel"
    const val STATUS = "llm-status"
    const val ERROR = "llm-error"

    const val DRAFT = "llm-draft"
    const val SEND = "llm-send"
    const val CLEAR = "llm-clear"
    const val STREAMING = "chat-streaming"
    const val SAVE_SNAPSHOT = "llm-save-snapshot"

    const val SCHEMA = "llm-schema"
    const val SCHEMA_ERROR = "llm-schema-error"
    const val EXTRACT_INPUT = "llm-extract-input"
    const val EXTRACT = "llm-extract"
    const val EXTRACT_OUTPUT = "llm-extract-output"
    const val VALIDATION = "llm-validation"

    const val EMBED_A = "llm-embed-a"
    const val EMBED_B = "llm-embed-b"
    const val EMBED = "llm-embed"
    const val SIMILARITY = "llm-similarity"

    const val GALLERY = "llm-gallery"
    const val CAMERA = "llm-camera"
    const val IMAGE_PATH = "llm-image-path"
    const val IMAGE_LOAD = "llm-image-load"
    const val IMAGE = "llm-image"
    const val VISION_QUESTION = "llm-vision-question"
    const val ANALYZE = "llm-analyze"
    const val VISION_ANSWER = "llm-vision-answer"

    fun panel(panel: LlmPanel) = "llm-tab-${panel.name.lowercase()}"
    fun template(template: PromptTemplate) = "llm-template-${template.name.lowercase()}"
    fun message(index: Int) = "chat-msg-$index"
    fun snapshotRestore(id: Int) = "snapshot-$id-restore"
    fun snapshotDelete(id: Int) = "snapshot-$id-delete"
    fun snapshot(id: Int) = "snapshot-$id"
}

@Composable
fun LlmChatScreen(controller: LlmChatController, resident: ResidentState) {
    val state by controller.state.collectAsState()
    LlmChatContent(
        state = state,
        residentKeys = resident.entries.map { it.key }.toSet(),
        canPickFromGallery = controller.canPickFromGallery,
        canCapture = controller.canCapture,
        actions = controller,
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun LlmChatContent(
    state: LlmChatState,
    residentKeys: Set<String>,
    canPickFromGallery: Boolean,
    canCapture: Boolean,
    actions: LlmChatController,
) {
    val destination = ShowcaseDestination.LLM_CHAT
    var panelName by rememberSaveable { mutableStateOf(LlmPanel.CHAT.name) }
    val panel = LlmPanel.valueOf(panelName)
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp).testTag(destination.screenTag),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(destination.title, style = MaterialTheme.typography.headlineSmall)
        Text(destination.summary, style = MaterialTheme.typography.bodyMedium)

        ModelControls(state, residentKeys, actions)

        Text(state.activity.label, style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag(LlmChatTags.ACTIVITY))
        OutlinedButton(onClick = actions::cancel, enabled = state.isBusy, modifier = Modifier.testTag(LlmChatTags.CANCEL)) { Text("Cancel") }
        state.status?.let { Text(it, style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag(LlmChatTags.STATUS)) }
        state.error?.let { ErrorCard(it, LlmChatTags.ERROR) }

        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            LlmPanel.entries.forEach { entry ->
                FilterChip(
                    selected = entry == panel,
                    onClick = { panelName = entry.name },
                    label = { Text(entry.label) },
                    modifier = Modifier.testTag(LlmChatTags.panel(entry)),
                )
            }
        }
        when (panel) {
            LlmPanel.CHAT -> ChatPanel(state, actions)
            LlmPanel.STRUCTURED -> StructuredPanel(state, actions)
            LlmPanel.EMBEDDINGS -> EmbeddingsPanel(state, actions)
            LlmPanel.VISION -> VisionPanel(state, canPickFromGallery, canCapture, actions)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ModelControls(state: LlmChatState, residentKeys: Set<String>, actions: LlmChatController) {
    val editable = !state.isBusy
    OutlinedTextField(
        value = state.modelPath,
        onValueChange = actions::setModelPath,
        label = { Text("GGUF model") },
        singleLine = true,
        enabled = editable,
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.MODEL_PATH),
    )
    OutlinedTextField(
        value = state.projectorPath,
        onValueChange = actions::setProjectorPath,
        label = { Text("Vision projector (mmproj GGUF, optional)") },
        singleLine = true,
        enabled = editable,
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.PROJECTOR_PATH),
    )
    Text("Prompt template", fontWeight = FontWeight.Bold)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        PromptTemplate.entries.forEach { template ->
            FilterChip(
                selected = state.template == template,
                onClick = { actions.setTemplate(template) },
                enabled = editable,
                label = { Text(template.label) },
                modifier = Modifier.testTag(LlmChatTags.template(template)),
            )
        }
    }
    OutlinedTextField(
        value = state.systemPrompt,
        onValueChange = actions::setSystemPrompt,
        label = { Text("System prompt") },
        enabled = editable,
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.SYSTEM_PROMPT),
    )
    Text("Compute backend", fontWeight = FontWeight.Bold)
    BackendChips(state.backend, LlmChatTags.BACKEND, editable, actions::setBackend)
    Stepper(
        label = "CPU threads",
        valueText = if (state.cpuThreads == BackendConfig.AUTO_THREADS) "Auto" else state.cpuThreads.toString(),
        tag = LlmChatTags.THREADS,
        enabled = editable,
        onDecrement = { actions.setCpuThreads(if (state.cpuThreads <= 1) BackendConfig.AUTO_THREADS else state.cpuThreads - 1) },
        onIncrement = { actions.setCpuThreads(if (state.cpuThreads == BackendConfig.AUTO_THREADS) 1 else state.cpuThreads + 1) },
    )
    Stepper(
        label = "GPU layers",
        valueText = when {
            state.backend == ComputeBackend.CPU_ONLY -> "0 (CPU)"
            state.gpuLayers == BackendConfig.ALL_GPU_LAYERS -> "All"
            else -> state.gpuLayers.toString()
        },
        tag = LlmChatTags.GPU_LAYERS,
        enabled = editable && state.backend != ComputeBackend.CPU_ONLY,
        onDecrement = {
            actions.setGpuLayers(
                if (state.gpuLayers == BackendConfig.ALL_GPU_LAYERS) GPU_LAYER_STEP * 8 else (state.gpuLayers - GPU_LAYER_STEP).coerceAtLeast(0),
            )
        },
        onIncrement = {
            actions.setGpuLayers(if (state.gpuLayers == BackendConfig.ALL_GPU_LAYERS) BackendConfig.ALL_GPU_LAYERS else state.gpuLayers + GPU_LAYER_STEP)
        },
    )
    Stepper(
        label = "Max tokens",
        valueText = state.maxTokens.toString(),
        tag = LlmChatTags.MAX_TOKENS,
        enabled = editable,
        onDecrement = { actions.setMaxTokens(state.maxTokens / 2) },
        onIncrement = { actions.setMaxTokens(state.maxTokens * 2) },
    )
    Button(onClick = actions::loadModel, enabled = editable && state.hasModel, modifier = Modifier.testTag(LlmChatTags.LOAD)) {
        Text(if (state.loaded == null) "Load model" else "Reload with these settings")
    }
    val loaded = state.loaded
    InfoCard(LlmChatTags.LOADED) {
        if (loaded == null) {
            Text("Model not loaded (loads on first use)")
        } else {
            val threads = if (loaded.config.cpuThreads == BackendConfig.AUTO_THREADS) "auto" else loaded.config.cpuThreads.toString()
            val layers = if (loaded.config.gpuLayers == BackendConfig.ALL_GPU_LAYERS) "all" else loaded.config.gpuLayers.toString()
            Text("${loaded.description} · ${loaded.backend.displayName()} (GPU ${if (loaded.usesGpu) "on" else "off"})", fontWeight = FontWeight.Medium)
            Text(
                "threads $threads · GPU layers $layers · context ${loaded.contextLength} · vision ${if (loaded.supportsVision) "yes" else "no"}",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                when {
                    loaded.key !in residentKeys -> "Evicted from memory: reloads on next use"
                    state.needsReload -> "Settings changed: reloads on next use"
                    else -> "Resident"
                },
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

private const val GPU_LAYER_STEP = 4

@Composable
private fun ChatPanel(state: LlmChatState, actions: LlmChatController) {
    SectionTitle("Conversation")
    if (state.messages.isEmpty() && state.streaming == null) {
        Text("Say hello to start a multi-turn conversation.", style = MaterialTheme.typography.bodySmall)
    }
    state.messages.forEachIndexed { index, message -> MessageBubble(message, LlmChatTags.message(index)) }
    state.streaming?.let { text ->
        Card(
            modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {}.testTag(LlmChatTags.STREAMING),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
        ) {
            Column(Modifier.padding(12.dp)) {
                Text("Assistant (streaming)", style = MaterialTheme.typography.labelSmall)
                Text(text.ifEmpty { "…" }, fontStyle = FontStyle.Italic)
            }
        }
    }
    OutlinedTextField(
        value = state.draft,
        onValueChange = actions::setDraft,
        label = { Text("Message") },
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.DRAFT),
    )
    FlowRowButtons {
        Button(onClick = actions::send, enabled = state.canSend, modifier = Modifier.testTag(LlmChatTags.SEND)) { Text("Send") }
        OutlinedButton(onClick = actions::clearConversation, enabled = !state.isBusy, modifier = Modifier.testTag(LlmChatTags.CLEAR)) {
            Text("Clear")
        }
        OutlinedButton(onClick = actions::saveSnapshot, enabled = state.canSaveSnapshot, modifier = Modifier.testTag(LlmChatTags.SAVE_SNAPSHOT)) {
            Text("Save KV snapshot")
        }
    }
    if (state.snapshots.isNotEmpty()) SectionTitle("KV-cache snapshots")
    state.snapshots.forEach { snapshot ->
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(snapshot.label, modifier = Modifier.testTag(LlmChatTags.snapshot(snapshot.id)))
                FlowRowButtons {
                    OutlinedButton(
                        onClick = { actions.restoreSnapshot(snapshot.id) },
                        enabled = !state.isBusy,
                        modifier = Modifier.testTag(LlmChatTags.snapshotRestore(snapshot.id)),
                    ) { Text("Restore") }
                    OutlinedButton(onClick = { actions.deleteSnapshot(snapshot.id) }, modifier = Modifier.testTag(LlmChatTags.snapshotDelete(snapshot.id))) {
                        Text("Delete")
                    }
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(message: ChatMessage, tag: String) {
    val user = message.role == ChatRole.USER
    Card(
        modifier = Modifier.fillMaxWidth().semantics(mergeDescendants = true) {}.testTag(tag),
        colors = CardDefaults.cardColors(
            containerColor = if (user) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(if (user) "You" else "Assistant", style = MaterialTheme.typography.labelSmall)
            Text(message.text)
        }
    }
}

@Composable
private fun StructuredPanel(state: LlmChatState, actions: LlmChatController) {
    SectionTitle("JSON-schema constrained extraction")
    Text(
        "The schema is compiled to a GBNF grammar, so every sampled token keeps the output valid JSON for it.",
        style = MaterialTheme.typography.bodySmall,
    )
    OutlinedTextField(
        value = state.schema,
        onValueChange = actions::setSchema,
        label = { Text("JSON schema") },
        textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
        modifier = Modifier.fillMaxWidth().heightIn(min = 120.dp).testTag(LlmChatTags.SCHEMA),
    )
    state.schemaError?.let { Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.testTag(LlmChatTags.SCHEMA_ERROR)) }
    OutlinedTextField(
        value = state.extractionInput,
        onValueChange = actions::setExtractionInput,
        label = { Text("Text to extract from") },
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.EXTRACT_INPUT),
    )
    Button(onClick = actions::extract, enabled = state.canExtract && state.schemaError == null, modifier = Modifier.testTag(LlmChatTags.EXTRACT)) {
        Text("Extract JSON")
    }
    OutlinedTextField(
        value = state.extractionOutput,
        onValueChange = actions::setExtractionOutput,
        label = { Text("Output (editable, validated live)") },
        textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
        enabled = state.activity != ChatActivity.EXTRACTING,
        modifier = Modifier.fillMaxWidth().heightIn(min = 80.dp).testTag(LlmChatTags.EXTRACT_OUTPUT),
    )
    val validation = state.validation
    val color = when (validation) {
        is JsonValidation.Valid -> MaterialTheme.colorScheme.primary
        is JsonValidation.Invalid, is JsonValidation.SchemaViolation -> MaterialTheme.colorScheme.error
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }
    Text(validation.message, color = color, modifier = Modifier.testTag(LlmChatTags.VALIDATION))
}

@Composable
private fun EmbeddingsPanel(state: LlmChatState, actions: LlmChatController) {
    SectionTitle("Embedding similarity")
    OutlinedTextField(
        value = state.embedTextA,
        onValueChange = actions::setEmbedTextA,
        label = { Text("Text A") },
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.EMBED_A),
    )
    OutlinedTextField(
        value = state.embedTextB,
        onValueChange = actions::setEmbedTextB,
        label = { Text("Text B") },
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.EMBED_B),
    )
    Button(onClick = actions::compareEmbeddings, enabled = state.canEmbed, modifier = Modifier.testTag(LlmChatTags.EMBED)) {
        Text("Compare")
    }
    state.similarity?.let { similarity ->
        Text(
            "Cosine similarity: ${formatDecimal(similarity.toDouble(), 4)} (${state.embeddingDimensions} dimensions)",
            fontWeight = FontWeight.Medium,
            modifier = Modifier.testTag(LlmChatTags.SIMILARITY),
        )
    }
}

@Composable
private fun VisionPanel(state: LlmChatState, canPickFromGallery: Boolean, canCapture: Boolean, actions: LlmChatController) {
    SectionTitle("Image understanding (libmtmd)")
    Text(
        "Images stay in memory: camera / library photos and files are passed to the projector as encoded bytes, " +
            "never written to a temporary file.",
        style = MaterialTheme.typography.bodySmall,
    )
    if (canPickFromGallery || canCapture) {
        FlowRowButtons {
            if (canPickFromGallery) {
                OutlinedButton(onClick = actions::pickFromGallery, enabled = !state.isBusy, modifier = Modifier.testTag(LlmChatTags.GALLERY)) {
                    Text("Photo library")
                }
            }
            if (canCapture) {
                OutlinedButton(onClick = actions::captureFromCamera, enabled = !state.isBusy, modifier = Modifier.testTag(LlmChatTags.CAMERA)) {
                    Text("Camera")
                }
            }
        }
    }
    OutlinedTextField(
        value = state.imagePath,
        onValueChange = actions::setImagePath,
        label = { Text("Image file (PNG / JPEG / …)") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.IMAGE_PATH),
    )
    OutlinedButton(
        onClick = actions::loadImageFromPath,
        enabled = !state.isBusy && state.imagePath.isNotBlank(),
        modifier = Modifier.testTag(LlmChatTags.IMAGE_LOAD),
    ) { Text("Load image into memory") }
    Text(state.image?.label ?: "No image attached", modifier = Modifier.testTag(LlmChatTags.IMAGE))
    OutlinedTextField(
        value = state.visionQuestion,
        onValueChange = actions::setVisionQuestion,
        label = { Text("Question") },
        modifier = Modifier.fillMaxWidth().testTag(LlmChatTags.VISION_QUESTION),
    )
    Button(onClick = actions::analyzeImage, enabled = state.canAnalyzeImage, modifier = Modifier.testTag(LlmChatTags.ANALYZE)) {
        Text("Analyze image")
    }
    if (state.visionAnswer.isNotEmpty()) {
        Text(state.visionAnswer, modifier = Modifier.testTag(LlmChatTags.VISION_ANSWER))
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun FlowRowButtons(content: @Composable () -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) { content() }
}
