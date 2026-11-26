package com.donfuldauer.kggml.samples.whisper

import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.Font
import java.io.File
import javax.swing.BorderFactory
import javax.swing.Box
import javax.swing.BoxLayout
import javax.swing.ButtonGroup
import javax.swing.JButton
import javax.swing.JComponent
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JRadioButton
import javax.swing.JScrollPane
import javax.swing.JTextArea
import javax.swing.JTextField
import javax.swing.SwingUtilities
import javax.swing.event.DocumentEvent
import javax.swing.event.DocumentListener
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * The desktop screen of the minimal sample: plain Swing (the JVM's own toolkit, no Compose), mirroring the Android
 * `MainActivity` and the iOS UIKit screen: runtime status, a model picker over the whisper models (`.bin`) in
 * [modelFolder] plus a custom path, the audio file, the compute backend, Transcribe and the report.
 *
 * Component names ([Names]) let tests drive it headlessly (`java.awt.headless=true`: panels, buttons and fields work
 * without a window).
 */
class WhisperSamplePanel(
    val modelFolder: String = desktopModelFolder(),
    requestedModel: String? = null,
    requestedAudio: String? = null,
    initialBackend: BackendChoice = BackendChoice.CPU_ONLY,
) : JPanel(BorderLayout()), AutoCloseable {
    object Names {
        const val STATUS = "status"
        const val NO_MODELS = "model-none"
        const val MODEL_CUSTOM = "model-custom"
        const val MODEL_PATH = "model-path"
        const val AUDIO_PATH = "audio-path"
        const val RESCAN = "rescan"
        const val TRANSCRIBE = "transcribe"
        const val OUTPUT = "output"
        fun modelChoice(fileName: String) = "model-choice:$fileName"
        fun backend(choice: BackendChoice) = "backend:${choice.name}"
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val session = WhisperSampleSession()
    private var running: Job? = null

    private var folderModels: List<FolderModel> = scanWhisperModels(modelFolder)
    private val modelChoices = JPanel().apply { layout = BoxLayout(this, BoxLayout.Y_AXIS) }
    private val modelGroup = ButtonGroup()
    private val customChoice = JRadioButton("Custom path").apply { name = Names.MODEL_CUSTOM }
    /** Set while the picker and the path field update each other, so neither reacts to the other's change. */
    private var syncing = false

    val status = JTextArea("Loading Kggml runtime…").apply { name = Names.STATUS; isEditable = false; isOpaque = false }
    val noModels = JLabel(noModelsHint(modelFolder)).apply { name = Names.NO_MODELS }
    val modelPath = JTextField(initialModelPath(requestedModel, File(modelFolder, DEFAULT_MODEL).path, folderModels), 48)
        .apply { name = Names.MODEL_PATH }
    val audioPath = JTextField(requestedAudio ?: File(modelFolder, DEFAULT_AUDIO).path, 48).apply { name = Names.AUDIO_PATH }
    private val backendButtons: Map<BackendChoice, JRadioButton> = BackendChoice.entries.associateWith { choice ->
        JRadioButton(choice.label(), choice == initialBackend).apply { name = Names.backend(choice) }
    }
    val transcribeButton = JButton("Transcribe").apply { name = Names.TRANSCRIBE }
    val output = JTextArea(8, 60).apply {
        name = Names.OUTPUT
        isEditable = false
        font = Font(Font.MONOSPACED, Font.PLAIN, 12)
    }

    init {
        val form = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = BorderFactory.createEmptyBorder(12, 12, 12, 12)
            add(status.left())
            add(section("Model (.bin)"))
            add(noModels.left())
            add(modelChoices.left())
            add(row(modelPath, JButton("Rescan folder").apply { name = Names.RESCAN; addActionListener { rescan() } }))
            add(section("Audio (WAV or 16 kHz mono s16le .pcm)"))
            add(audioPath.left())
            add(section("Compute backend"))
            val group = ButtonGroup()
            add(JPanel(FlowLayout(FlowLayout.LEFT, 0, 0)).apply { backendButtons.values.forEach { group.add(it); add(it) } }.left())
            add(transcribeButton.left())
        }
        add(form, BorderLayout.NORTH)
        add(JScrollPane(output), BorderLayout.CENTER)

        customChoice.addActionListener { onCustomChosen() }
        modelPath.document.addDocumentListener(object : DocumentListener {
            override fun insertUpdate(e: DocumentEvent) = syncPickerToPath()
            override fun removeUpdate(e: DocumentEvent) = syncPickerToPath()
            override fun changedUpdate(e: DocumentEvent) = syncPickerToPath()
        })
        transcribeButton.addActionListener { transcribe() }
        populateModelChoices()

        scope.launch {
            val text = describeRuntime()
            SwingUtilities.invokeLater { status.text = text }
        }
    }

    /** Re-reads the model folder (models copied in while the app runs; also called when the window is activated). */
    fun rescan() {
        val rescanned = scanWhisperModels(modelFolder)
        if (rescanned != folderModels) {
            folderModels = rescanned
            populateModelChoices()
        }
    }

    fun selectedBackend(): BackendChoice = backendButtons.entries.first { it.value.isSelected }.key

    /** Transcribes the audio file with the chosen model and backend; the report (or failure) lands in [output]. */
    fun transcribe() {
        if (running?.isActive == true) return
        val model = modelPath.text.trim()
        val audio = audioPath.text.trim()
        val choice = selectedBackend()
        transcribeButton.isEnabled = false
        output.text = "Transcribing on ${choice.label()}…"
        running = scope.launch {
            val text = try {
                val pcm = parsePcmFile(File(audio).readBytes())
                formatReport(session.transcribe(model, pcm, choice.toComputeBackend()))
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                "Failed: ${t::class.simpleName}: ${t.message}"
            }
            SwingUtilities.invokeLater {
                output.text = text
                transcribeButton.isEnabled = true
            }
        }
    }

    /** One radio button per model in the folder, then "Custom path"; the hint when the folder has none. */
    private fun populateModelChoices() {
        syncing = true
        try {
            modelGroup.elements.toList().forEach { modelGroup.remove(it) }
            modelChoices.removeAll()
            folderModels.forEach { model ->
                val button = JRadioButton(model.fileName).apply {
                    name = Names.modelChoice(model.fileName)
                    addActionListener { onModelChosen(model) }
                }
                modelGroup.add(button)
                modelChoices.add(button)
            }
            modelGroup.add(customChoice)
            modelChoices.add(customChoice)
            noModels.isVisible = folderModels.isEmpty()
            modelChoices.revalidate()
        } finally {
            syncing = false
        }
        syncPickerToPath()
    }

    private fun onModelChosen(model: FolderModel) {
        if (syncing) return
        syncing = true
        try {
            modelPath.text = model.path
        } finally {
            syncing = false
        }
    }

    // Leaving a folder model for a custom path starts from an empty field.
    private fun onCustomChosen() {
        if (syncing) return
        if (chosenModelIndex(folderModels, modelPath.text) != null) {
            syncing = true
            try {
                modelPath.text = ""
            } finally {
                syncing = false
            }
        }
    }

    /** Selects the radio button of the model in the path field, or "Custom path" for any other path. */
    private fun syncPickerToPath() {
        if (syncing) return
        val index = chosenModelIndex(folderModels, modelPath.text)
        val target = if (index == null) customChoice else modelChoices.components
            .filterIsInstance<JRadioButton>().first { it.name == Names.modelChoice(folderModels[index].fileName) }
        if (!target.isSelected) target.isSelected = true
    }

    override fun close() {
        scope.cancel()
        session.close()
    }

    private companion object {
        const val DEFAULT_MODEL = "ggml-tiny.en.bin"
        const val DEFAULT_AUDIO = "jfk.wav"

        fun <T : JComponent> T.left(): T = apply { alignmentX = LEFT_ALIGNMENT }
        fun section(text: String) = JLabel(text).apply {
            font = font.deriveFont(Font.BOLD)
            border = BorderFactory.createEmptyBorder(10, 0, 2, 0)
            alignmentX = LEFT_ALIGNMENT
        }
        fun row(vararg components: JComponent) = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.X_AXIS)
            components.forEachIndexed { i, c -> if (i > 0) add(Box.createHorizontalStrut(8)); add(c) }
            alignmentX = LEFT_ALIGNMENT
        }
    }
}
