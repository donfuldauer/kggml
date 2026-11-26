package com.donfuldauer.kggml.samples.showcase.diffusion

import com.donfuldauer.kggml.samples.showcase.models.ModelChoice
import com.donfuldauer.kggml.samples.showcase.models.ModelPathPicker
import com.donfuldauer.kggml.samples.showcase.models.ModelSlot
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.sizeIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import com.donfuldauer.kggml.samples.showcase.BackendChips
import com.donfuldauer.kggml.samples.showcase.ErrorCard
import com.donfuldauer.kggml.samples.showcase.SectionTitle
import com.donfuldauer.kggml.samples.showcase.ShowcaseDestination
import com.donfuldauer.kggml.samples.showcase.displayName
import com.donfuldauer.kggml.samples.showcase.formatDecimal
import com.donfuldauer.kggml.samples.showcase.pixelsToImageBitmap

/** Test tags of the Diffusion Image Studio controls. */
object DiffusionStudioTags {
    const val MODEL_PATH = "sd-model-path"
    const val BACKEND = "sd-backend"
    const val PROMPT = "sd-prompt"
    const val NEGATIVE_PROMPT = "sd-negative"
    const val STEPS = "sd-steps"
    const val STEPS_LABEL = "sd-steps-label"
    const val CFG = "sd-cfg"
    const val CFG_LABEL = "sd-cfg-label"
    const val SEED = "sd-seed"
    const val RANDOM_SEED = "sd-random-seed"
    const val RANDOMIZE = "sd-randomize"
    const val GENERATE = "sd-generate"
    const val CANCEL = "sd-cancel"
    const val ACTIVITY = "sd-activity"
    const val PROGRESS = "sd-progress"
    const val PROGRESS_LABEL = "sd-progress-label"
    const val IMAGE = "sd-image"
    const val IMAGE_INFO = "sd-image-info"
    const val EXPORT_PATH = "sd-export-path"
    const val EXPORT = "sd-export"
    const val STATUS = "sd-status"
    const val ERROR = "sd-error"

    fun size(size: Int) = "sd-size-$size"
}

@Composable
fun DiffusionStudioScreen(controller: DiffusionStudioController, modelChoices: List<ModelChoice> = emptyList()) {
    val state by controller.state.collectAsState()
    DiffusionStudioContent(state, controller, modelChoices)
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun DiffusionStudioContent(
    state: DiffusionStudioState,
    actions: DiffusionStudioController,
    /** Downloaded diffusion models to offer next to the custom path. */
    modelChoices: List<ModelChoice> = emptyList(),
) {
    val destination = ShowcaseDestination.DIFFUSION_STUDIO
    val editable = !state.isBusy
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp).testTag(destination.screenTag),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(destination.title, style = MaterialTheme.typography.headlineSmall)
        Text(destination.summary, style = MaterialTheme.typography.bodyMedium)

        ModelPathPicker(
            title = "Model",
            fieldLabel = "Diffusion model (.gguf / .safetensors / .ckpt)",
            slot = ModelSlot.DIFFUSION,
            choices = modelChoices,
            path = state.modelPath,
            enabled = editable,
            fieldTag = DiffusionStudioTags.MODEL_PATH,
            onPathChange = actions::setModelPath,
            onChoose = { actions.useModel(it.path, it.diffusionPreset) },
        )
        BackendChips(state.backend, DiffusionStudioTags.BACKEND, editable, actions::setBackend)
        state.loadedBackend?.let {
            Text("Active: ${it.displayName()} (GPU ${if (state.usesGpu) "on" else "off"})", style = MaterialTheme.typography.bodySmall)
        }

        OutlinedTextField(
            value = state.prompt,
            onValueChange = actions::setPrompt,
            label = { Text("Prompt") },
            enabled = editable,
            modifier = Modifier.fillMaxWidth().testTag(DiffusionStudioTags.PROMPT),
        )
        OutlinedTextField(
            value = state.negativePrompt,
            onValueChange = actions::setNegativePrompt,
            label = { Text("Negative prompt") },
            enabled = editable,
            modifier = Modifier.fillMaxWidth().testTag(DiffusionStudioTags.NEGATIVE_PROMPT),
        )

        Text("Sampling steps: ${state.steps}", modifier = Modifier.testTag(DiffusionStudioTags.STEPS_LABEL))
        Slider(
            value = state.steps.toFloat(),
            onValueChange = { actions.setSteps(kotlin.math.round(it).toInt()) },
            valueRange = 1f..DiffusionStudioController.MAX_STEPS.toFloat(),
            steps = DiffusionStudioController.MAX_STEPS - 2,
            enabled = editable,
            modifier = Modifier.fillMaxWidth().testTag(DiffusionStudioTags.STEPS),
        )
        Text("CFG scale: ${formatDecimal(state.cfgScale.toDouble(), 1)}", modifier = Modifier.testTag(DiffusionStudioTags.CFG_LABEL))
        Slider(
            value = state.cfgScale,
            onValueChange = actions::setCfgScale,
            valueRange = 1f..DiffusionStudioController.MAX_CFG,
            enabled = editable,
            modifier = Modifier.fillMaxWidth().testTag(DiffusionStudioTags.CFG),
        )
        Text("Size")
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            DiffusionStudioController.SIZES.forEach { size ->
                FilterChip(
                    selected = state.size == size,
                    onClick = { actions.setSize(size) },
                    enabled = editable,
                    label = { Text("${size}×$size") },
                    modifier = Modifier.testTag(DiffusionStudioTags.size(size)),
                )
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = state.seedText,
                onValueChange = actions::setSeedText,
                label = { Text("Seed") },
                singleLine = true,
                enabled = editable && !state.randomSeed,
                modifier = Modifier.weight(1f).testTag(DiffusionStudioTags.SEED),
            )
            OutlinedButton(onClick = actions::randomizeSeed, enabled = editable, modifier = Modifier.testTag(DiffusionStudioTags.RANDOMIZE)) {
                Text("Randomize")
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Switch(
                checked = state.randomSeed,
                onCheckedChange = actions::setRandomSeed,
                enabled = editable,
                modifier = Modifier.testTag(DiffusionStudioTags.RANDOM_SEED),
            )
            Text("New random seed for every image")
        }

        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = actions::generate, enabled = state.canGenerate, modifier = Modifier.testTag(DiffusionStudioTags.GENERATE)) {
                Text("Generate")
            }
            OutlinedButton(onClick = actions::cancel, enabled = state.canCancel, modifier = Modifier.testTag(DiffusionStudioTags.CANCEL)) {
                Text("Cancel")
            }
        }
        Text(state.activity.label, style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag(DiffusionStudioTags.ACTIVITY))
        state.progress?.let { progress ->
            LinearProgressIndicator(progress = { progress.fraction }, modifier = Modifier.fillMaxWidth().testTag(DiffusionStudioTags.PROGRESS))
            Text(
                progress.label + if (progress.secondsPerStep > 0f) " · ${formatDecimal(progress.secondsPerStep.toDouble(), 2)} s/step" else "",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.testTag(DiffusionStudioTags.PROGRESS_LABEL),
            )
        }
        state.status?.let { Text(it, style = MaterialTheme.typography.bodySmall, modifier = Modifier.testTag(DiffusionStudioTags.STATUS)) }
        state.error?.let { ErrorCard(it, DiffusionStudioTags.ERROR) }

        state.result?.let { result ->
            SectionTitle("Result")
            val bitmap = remember(result) {
                pixelsToImageBitmap(result.image.width, result.image.height, result.image.channels, result.image.pixels)
            }
            Image(
                bitmap = bitmap,
                contentDescription = "Generated image: ${result.prompt}",
                contentScale = ContentScale.Fit,
                modifier = Modifier.sizeIn(maxWidth = 512.dp, maxHeight = 512.dp).testTag(DiffusionStudioTags.IMAGE),
            )
            Text(
                "${result.image.width}×${result.image.height} · seed ${result.seed} · ${result.steps} steps · " +
                    "CFG ${formatDecimal(result.cfgScale.toDouble(), 1)}",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.testTag(DiffusionStudioTags.IMAGE_INFO),
            )
            OutlinedTextField(
                value = state.exportPath,
                onValueChange = actions::setExportPath,
                label = { Text("Export PNG to") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth().testTag(DiffusionStudioTags.EXPORT_PATH),
            )
            Button(onClick = actions::export, enabled = state.canExport, modifier = Modifier.testTag(DiffusionStudioTags.EXPORT)) {
                Text("Export PNG")
            }
        }
    }
}
