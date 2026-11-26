package com.donfuldauer.kggml.samples.showcase.models

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/** A model input of a feature screen, and which downloaded files can fill it. */
enum class ModelSlot(
    /** Catalog kinds that fit this slot. */
    val kinds: Set<ModelKind>,
    /** Extensions (lowercase, no dot) of non-catalog files in the model folder that fit this slot. */
    val extensions: Set<String>,
    /** What the slot holds, for the "nothing downloaded" hint. */
    val noun: String,
) {
    WHISPER(setOf(ModelKind.WHISPER), setOf("bin"), "speech"),
    LLM(setOf(ModelKind.LLM, ModelKind.VISION_LLM), setOf("gguf"), "LLM"),
    VISION_PROJECTOR(setOf(ModelKind.VISION_PROJECTOR), setOf("gguf"), "vision projector"),

    // GGUF is left out of the other-file match: a stray .gguf is far more likely an LLM than a diffusion model.
    DIFFUSION(setOf(ModelKind.DIFFUSION), setOf("safetensors", "ckpt"), "image generation"),
    ;

    /** Whether a non-catalog file named [fileName] fits this slot (by extension; projectors are named `mmproj…`). */
    fun accepts(fileName: String): Boolean {
        val name = fileName.lowercase()
        if (name.substringAfterLast('.', "") !in extensions) return false
        val projector = "mmproj" in name
        return when (this) {
            LLM -> !projector
            VISION_PROJECTOR -> projector
            else -> true
        }
    }
}

/**
 * A downloaded model a screen can use.
 *
 * @property id catalog id, or the file name of a non-catalog file (unique within the model folder).
 * @property projectorPath for a vision LLM, the path of its installed projector, if it is downloaded too.
 */
data class ModelChoice(
    val id: String,
    val label: String,
    val detail: String,
    val path: String,
    val projectorPath: String? = null,
)

/**
 * The installed models that fit [slot]: installed catalog models of the slot's kinds first (catalog order), then other
 * files in the model folder whose extension fits (by name).
 */
fun ModelManagerState.choicesFor(slot: ModelSlot): List<ModelChoice> {
    val installed = rows.filter { it.isInstalled && it.installedPath != null }
    val catalog = installed.filter { it.model.kind in slot.kinds }.map { row ->
        val projector = row.model.projectorId?.let { id -> installed.firstOrNull { it.model.id == id }?.installedPath }
        val note = when {
            row.model.projectorId == null -> ""
            projector != null -> " · with its vision projector"
            else -> " · vision projector not downloaded"
        }
        ModelChoice(row.model.id, row.model.displayName, row.model.fileName + note, row.installedPath!!, projector)
    }
    val others = otherFiles.filter { slot.accepts(it.fileName) }.sortedBy { it.fileName }.map { file ->
        ModelChoice(file.fileName, file.fileName, "Other file in the model folder", file.path)
    }
    return catalog + others
}

/** Test tags of a [ModelPathPicker] whose text field has tag [fieldTag]. */
object ModelPickerTags {
    fun group(fieldTag: String) = "$fieldTag-picker"
    fun choice(fieldTag: String, id: String) = "$fieldTag-choice-$id"
    fun custom(fieldTag: String) = "$fieldTag-custom"
    fun empty(fieldTag: String) = "$fieldTag-none-downloaded"
}

/**
 * Picks a model for a screen: one of the downloaded [choices] that fit, or a custom path typed into the field below.
 * The field always shows the path in use, so a chosen model can still be edited.
 *
 * @param fieldTag test tag of the path text field (the other tags derive from it, see [ModelPickerTags]).
 * @param onChoose a downloaded model was picked (screens may use more than its path, e.g. a vision projector).
 */
@Composable
fun ModelPathPicker(
    title: String,
    fieldLabel: String,
    slot: ModelSlot,
    choices: List<ModelChoice>,
    path: String,
    enabled: Boolean,
    fieldTag: String,
    onPathChange: (String) -> Unit,
    onChoose: (ModelChoice) -> Unit,
) {
    val chosen = choices.firstOrNull { it.path == path }
    Column(
        modifier = Modifier.fillMaxWidth().selectableGroup().testTag(ModelPickerTags.group(fieldTag)),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(title, fontWeight = FontWeight.Bold)
        if (choices.isEmpty()) {
            Text(
                "No downloaded ${slot.noun} models yet: download one in the Models tab, or enter a path below.",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.testTag(ModelPickerTags.empty(fieldTag)),
            )
        }
        choices.forEach { choice ->
            PickerRow(
                selected = choice == chosen,
                enabled = enabled,
                tag = ModelPickerTags.choice(fieldTag, choice.id),
                onClick = { onChoose(choice) },
            ) {
                Column {
                    Text(choice.label)
                    Text(choice.detail, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        PickerRow(
            selected = chosen == null,
            enabled = enabled,
            tag = ModelPickerTags.custom(fieldTag),
            // Leaving a downloaded model for a custom path starts from an empty field.
            onClick = { if (chosen != null) onPathChange("") },
        ) {
            Text("Custom path")
        }
        OutlinedTextField(
            value = path,
            onValueChange = onPathChange,
            label = { Text(fieldLabel) },
            singleLine = true,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth().testTag(fieldTag),
        )
    }
}

@Composable
private fun PickerRow(selected: Boolean, enabled: Boolean, tag: String, onClick: () -> Unit, content: @Composable () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .selectable(selected = selected, enabled = enabled, role = Role.RadioButton, onClick = onClick)
            .padding(vertical = 4.dp)
            .testTag(tag),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(selected = selected, onClick = null, enabled = enabled)
        content()
    }
}
