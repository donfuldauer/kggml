@file:OptIn(ExperimentalForeignApi::class, ExperimentalAtomicApi::class)

package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.whisper.observeAppLifecycle
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.time.Duration.Companion.seconds
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.useContents
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import platform.Foundation.NSData
import platform.Foundation.NSHomeDirectory
import platform.Foundation.NSLog
import platform.Foundation.NSNotificationCenter
import platform.Foundation.NSOperationQueue
import platform.Foundation.dataWithContentsOfFile
import platform.UIKit.UIAction
import platform.UIKit.UIApplication
import platform.UIKit.UIApplicationDidEnterBackgroundNotification
import platform.UIKit.UIApplicationWillEnterForegroundNotification
import platform.UIKit.UIBackgroundTaskInvalid
import platform.UIKit.UIButton
import platform.UIKit.UIColor
import platform.UIKit.UIControlContentHorizontalAlignmentLeft
import platform.UIKit.UIControlEventEditingChanged
import platform.UIKit.UIControlEventTouchUpInside
import platform.UIKit.UIControlStateDisabled
import platform.UIKit.UIControlStateNormal
import platform.UIKit.UIFont
import platform.UIKit.UILabel
import platform.UIKit.UIMenu
import platform.UIKit.UIMenuElementState
import platform.UIKit.UISegmentedControl
import platform.UIKit.UITextField
import platform.UIKit.UITextView
import platform.UIKit.UIView
import platform.UIKit.UIViewAutoresizingFlexibleHeight
import platform.UIKit.UIViewAutoresizingFlexibleWidth
import platform.UIKit.UIViewController
import platform.UIKit.secondarySystemBackgroundColor
import platform.UIKit.systemBackgroundColor
import platform.UIKit.systemBlueColor
import platform.UIKit.systemGrayColor
import platform.CoreGraphics.CGRectMake
import platform.posix.memcpy

/**
 * Entry point for the Swift host app (`WhisperOnlyMinimal.framework`): returns the runtime status text.
 */
@Suppress("unused")
fun runtimeStatusText(): String = describeRuntime()

/**
 * Root view controller for the Swift host app (`IosEntryPointKt.MainViewController()`): plain UIKit, no Compose,
 * so the audited binary contains only the app, the Kggml runtime and whisper.
 *
 * - **Transcribe** runs one transcription of a PCM file (WAV or raw 16 kHz mono s16le) on the selected backend.
 * - **Background test** repeats the transcription for ~20 s inside a `UIApplication` background task; send the
 *   app to the background while it runs. The log (also written with `NSLog`) marks every run as
 *   `[background]` / `[foreground]` and ends with `PASS` when background runs completed with `use_gpu=false`.
 *
 * The engine follows the app lifecycle (`WhisperEngine.observeAppLifecycle`), so GPU runs are switched to
 * `CPU_ONLY` in the background automatically; select **CPU only** to verify the explicit `CPU_ONLY` path.
 * The **Model** menu lists the whisper models (`.bin`) in the app's Documents directory (rescanned when the app
 * returns to the foreground) and **Custom path**; the field below always holds the path in use. Defaults read
 * `ggml-tiny.en.bin` and `jfk.wav` from Documents.
 */
@Suppress("FunctionName", "unused")
fun MainViewController(): UIViewController = WhisperSampleScreen().controller

private class WhisperSampleScreen {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val inBackground = AtomicBoolean(false)
    private var lifecycleHandle: AutoCloseable? = null
    private val session = WhisperSampleSession(
        onEngineOpened = { engine -> lifecycleHandle = engine.whisperEngine?.observeAppLifecycle(scope) },
        onEngineClosing = { lifecycleHandle?.close(); lifecycleHandle = null },
    )
    private var job: Job? = null

    val controller = UIViewController(nibName = null, bundle = null)
    private val documents = NSHomeDirectory() + "/Documents"
    private val status = UILabel().apply { numberOfLines = 0; text = "Loading Kggml runtime…" }
    private var folderModels: List<FolderModel> = scanWhisperModels(documents)
    private val modelField = textField(initialModelPath(null, "$documents/ggml-tiny.en.bin", folderModels)).apply {
        addAction(UIAction.actionWithHandler { refreshModelPicker() }, forControlEvents = UIControlEventEditingChanged)
    }
    private val modelPicker = UIButton().apply {
        showsMenuAsPrimaryAction = true
        contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeft
        setTitleColor(UIColor.systemBlueColor, forState = UIControlStateNormal)
    }
    private val audioField = textField("$documents/jfk.wav")
    private val backendControl = UISegmentedControl(items = BackendChoice.entries.map { it.label() }).apply {
        selectedSegmentIndex = BackendChoice.CPU_ONLY.ordinal.toLong()
    }
    private val transcribeButton = button("Transcribe") { transcribeOnce() }
    private val backgroundButton = button("Background test (20 s)") { backgroundTest() }
    // Left editable: `editable` is read-only in the Kotlin/Native UIKit mapping, and every update replaces the text.
    private val output = UITextView().apply {
        font = UIFont.systemFontOfSize(13.0)
    }

    init {
        val center = NSNotificationCenter.defaultCenter
        center.addObserverForName(UIApplicationDidEnterBackgroundNotification, null, null as NSOperationQueue?) {
            inBackground.store(true)
        }
        center.addObserverForName(UIApplicationWillEnterForegroundNotification, null, null as NSOperationQueue?) {
            inBackground.store(false)
            // Models copied into Documents (Finder / Files) while the app was away show up on return.
            folderModels = scanWhisperModels(documents)
            refreshModelPicker()
        }
        refreshModelPicker()
        layout(controller.view)
        scope.launch {
            val text = describeRuntime()
            withContext(Dispatchers.Main) { status.text = text }
        }
    }

    private fun layout(root: UIView) {
        root.backgroundColor = UIColor.systemBackgroundColor
        val width = root.bounds.useContents { size.width } - 2 * MARGIN
        var y = 60.0
        fun place(view: UIView, height: Double) {
            view.setFrame(CGRectMake(MARGIN, y, width, height))
            view.autoresizingMask = UIViewAutoresizingFlexibleWidth
            root.addSubview(view)
            y += height + 8.0
        }
        place(status, 88.0)
        place(modelPicker, 32.0)
        place(modelField, 36.0)
        place(audioField, 36.0)
        place(backendControl, 32.0)
        place(transcribeButton, 40.0)
        place(backgroundButton, 40.0)
        val height = root.bounds.useContents { size.height } - y - MARGIN
        output.setFrame(CGRectMake(MARGIN, y, width, height))
        output.autoresizingMask = UIViewAutoresizingFlexibleWidth or UIViewAutoresizingFlexibleHeight
        root.addSubview(output)
    }

    /** Rebuilds the Model menu (folder models + Custom path, the one in use checked) and its title. */
    private fun refreshModelPicker() {
        val chosen = chosenModelIndex(folderModels, modelField.text.orEmpty())
        val items = folderModels.mapIndexed { index, model ->
            UIAction.actionWithTitle(model.fileName, image = null, identifier = null) { _ ->
                modelField.text = model.path
                refreshModelPicker()
            }.apply { state = if (index == chosen) UIMenuElementState.UIMenuElementStateOn else UIMenuElementState.UIMenuElementStateOff }
        } + UIAction.actionWithTitle("Custom path", image = null, identifier = null) { _ ->
            // Leaving a folder model for a custom path starts from an empty field.
            if (chosenModelIndex(folderModels, modelField.text.orEmpty()) != null) modelField.text = ""
            refreshModelPicker()
            modelField.becomeFirstResponder()
        }.apply { state = if (chosen == null) UIMenuElementState.UIMenuElementStateOn else UIMenuElementState.UIMenuElementStateOff }
        modelPicker.menu = UIMenu.menuWithTitle(
            title = if (folderModels.isEmpty()) noModelsHint(documents) else "Whisper models in Documents",
            children = items,
        )
        val name = chosen?.let { folderModels[it].fileName } ?: "Custom path"
        val count = if (folderModels.isEmpty()) "none in Documents" else "${folderModels.size} in Documents"
        modelPicker.setTitle("Model: $name ($count) ▾", forState = UIControlStateNormal)
    }

    private fun selectedBackend(): ComputeBackend =
        BackendChoice.entries[backendControl.selectedSegmentIndex.toInt().coerceIn(0, BackendChoice.entries.size - 1)]
            .toComputeBackend()

    private fun transcribeOnce() {
        start { model, audio, backend ->
            setOutput("Transcribing on $backend…")
            setOutput(formatReport(session.transcribe(model, audio, backend)))
        }
    }

    private fun backgroundTest() {
        start { model, audio, backend ->
            val app = UIApplication.sharedApplication
            val log = StringBuilder("Background test on $backend: send the app to the background now.\n")
            setOutput(log.toString())
            val current = currentCoroutineContext()[Job]
            // Ended exactly once, on the main thread: by the expiration handler (which Apple requires to end the
            // task synchronously, or the app is terminated) or by the `finally` block below.
            val ended = AtomicBoolean(false)
            var task = UIBackgroundTaskInvalid
            fun endTask() {
                if (task != UIBackgroundTaskInvalid && ended.compareAndSet(false, true)) app.endBackgroundTask(task)
            }
            withContext(Dispatchers.Main) {
                task = app.beginBackgroundTaskWithName("kggml-whisper-background-test") {
                    current?.cancel()
                    endTask()
                }
            }
            try {
                val summary = runBackgroundTranscriptionTest(
                    session = session,
                    modelPath = model,
                    audio = audio,
                    backend = backend,
                    isInBackground = { inBackground.load() },
                    duration = BACKGROUND_TEST_DURATION,
                ) { run ->
                    NSLog("kggml-whisper-bg-test %@", run.describe())
                    log.appendLine(run.describe())
                    setOutput(log.toString())
                }
                NSLog("kggml-whisper-bg-test %@", summary.describe())
                log.appendLine(summary.describe())
                setOutput(log.toString())
            } finally {
                withContext(NonCancellable + Dispatchers.Main) { endTask() }
            }
        }
    }

    private fun start(block: suspend (model: String, audio: PcmAudio, backend: ComputeBackend) -> Unit) {
        if (job?.isActive == true) return
        val model = modelField.text.orEmpty().trim()
        val audioPath = audioField.text.orEmpty().trim()
        val backend = selectedBackend()
        setButtonsEnabled(false)
        job = scope.launch {
            try {
                block(model, parsePcmFile(readFileBytes(audioPath)), backend)
            } catch (e: CancellationException) {
                withContext(NonCancellable + Dispatchers.Main) { output.text = "Cancelled (background time expired)" }
                throw e
            } catch (t: Throwable) {
                setOutput("Failed: ${t::class.simpleName}: ${t.message}")
            } finally {
                withContext(NonCancellable + Dispatchers.Main) { setButtonsEnabled(true) }
            }
        }
    }

    private suspend fun setOutput(text: String) = withContext(Dispatchers.Main) { output.text = text }

    private fun setButtonsEnabled(enabled: Boolean) {
        transcribeButton.enabled = enabled
        backgroundButton.enabled = enabled
    }

    private fun textField(value: String) = UITextField().apply {
        text = value
        font = UIFont.systemFontOfSize(12.0)
        backgroundColor = UIColor.secondarySystemBackgroundColor
    }

    private fun button(title: String, onTap: () -> Unit) = UIButton().apply {
        setTitle(title, forState = UIControlStateNormal)
        setTitleColor(UIColor.systemBlueColor, forState = UIControlStateNormal)
        setTitleColor(UIColor.systemGrayColor, forState = UIControlStateDisabled)
        addAction(UIAction.actionWithHandler { onTap() }, forControlEvents = UIControlEventTouchUpInside)
    }

    private companion object {
        const val MARGIN = 16.0

        /** Stays well inside the ~30 s of background execution time iOS grants a background task. */
        val BACKGROUND_TEST_DURATION = 20.seconds
    }
}

/** Reads a whole file into memory (the sample's audio files are small). */
internal fun readFileBytes(path: String): ByteArray {
    val data = NSData.dataWithContentsOfFile(path) ?: throw IllegalArgumentException("Cannot read $path")
    val size = data.length.toInt()
    val bytes = ByteArray(size)
    if (size > 0) bytes.usePinned { memcpy(it.addressOf(0), data.bytes, data.length) }
    return bytes
}
