package com.donfuldauer.kggml.samples.whisper

import java.awt.event.WindowAdapter
import java.awt.event.WindowEvent
import javax.swing.JFrame
import javax.swing.SwingUtilities
import javax.swing.WindowConstants

/**
 * Desktop entry point of the minimal sample (`./gradlew :samples:whisper-only-minimal:runDesktop`).
 *
 * Optional arguments, like the Android intent extras: `--model <path>`, `--audio <path>`,
 * `--backend CPU_ONLY|GPU|AUTO`, `--autostart`.
 */
fun main(args: Array<String>) {
    fun arg(name: String): String? = args.indexOf(name).takeIf { it >= 0 && it + 1 < args.size }?.let { args[it + 1] }
    val backend = arg("--backend")?.let { name -> BackendChoice.entries.firstOrNull { it.name == name } } ?: BackendChoice.CPU_ONLY
    SwingUtilities.invokeLater {
        val panel = WhisperSamplePanel(requestedModel = arg("--model"), requestedAudio = arg("--audio"), initialBackend = backend)
        JFrame("Kggml Whisper Minimal").apply {
            defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
            contentPane = panel
            addWindowListener(object : WindowAdapter() {
                // Models copied into the folder while the app was in the background show up on return.
                override fun windowActivated(e: WindowEvent) = panel.rescan()
                override fun windowClosing(e: WindowEvent) = panel.close()
            })
            setSize(860, 640)
            setLocationRelativeTo(null)
            isVisible = true
        }
        if ("--autostart" in args) panel.transcribe()
    }
}
