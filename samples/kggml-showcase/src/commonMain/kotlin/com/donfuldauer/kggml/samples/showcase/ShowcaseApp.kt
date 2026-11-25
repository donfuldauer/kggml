package com.donfuldauer.kggml.samples.showcase

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.rememberCoroutineScope
import com.donfuldauer.kggml.samples.showcase.whisper.AudioCapture
import com.donfuldauer.kggml.samples.showcase.whisper.TranscriberFactory
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperStudioController
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperStudioScreen
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperTranscriberFactory
import com.donfuldauer.kggml.samples.showcase.whisper.defaultWhisperModelPath
import com.donfuldauer.kggml.samples.showcase.whisper.platformAudioCapture
import com.donfuldauer.kggml.samples.showcase.whisper.readAudioFile
import com.donfuldauer.kggml.samples.showcase.whisper.supportsBackgroundStress
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Platform services used by the Whisper Studio; replaced by deterministic fakes in UI tests. */
data class WhisperStudioEnvironment(
    val transcriberFactory: TranscriberFactory = WhisperTranscriberFactory,
    val audioCapture: AudioCapture? = platformAudioCapture(),
    val readFile: (String) -> ByteArray = ::readAudioFile,
    val initialModelPath: String = defaultWhisperModelPath(),
    val backgroundStressSupported: Boolean = supportsBackgroundStress,
)

/**
 * Root of the showcase: a navigation shell over [ShowcaseDestination]s.
 *
 * @param inspector runtime source for the Hardware Backend Inspector (faked in UI tests).
 * @param ioDispatcher dispatcher for blocking native runtime calls (never the UI thread).
 * @param whisperEnvironment engine factory, microphone and file access of the Whisper Studio.
 */
@Composable
fun ShowcaseApp(
    inspector: RuntimeInspector = KggmlRuntimeInspector,
    ioDispatcher: CoroutineDispatcher = Dispatchers.Default,
    initialDestination: ShowcaseDestination = ShowcaseDestination.BACKEND_INSPECTOR,
    whisperEnvironment: WhisperStudioEnvironment = remember { WhisperStudioEnvironment() },
) {
    // Hoisted above the navigation so a live dictation keeps running while other screens are shown.
    val scope = rememberCoroutineScope()
    val whisperStudio = remember(whisperEnvironment, ioDispatcher) {
        WhisperStudioController(
            scope = scope,
            factory = whisperEnvironment.transcriberFactory,
            capture = whisperEnvironment.audioCapture,
            readFile = whisperEnvironment.readFile,
            ioDispatcher = ioDispatcher,
            initialModelPath = whisperEnvironment.initialModelPath,
            backgroundStressSupported = whisperEnvironment.backgroundStressSupported,
        )
    }
    DisposableEffect(whisperStudio) { onDispose { whisperStudio.close() } }
    MaterialTheme {
        var selectedName by rememberSaveable { mutableStateOf(initialDestination.name) }
        val selected = ShowcaseDestination.valueOf(selectedName)
        Scaffold(
            bottomBar = {
                NavigationBar {
                    ShowcaseDestination.entries.forEach { destination ->
                        NavigationBarItem(
                            selected = destination == selected,
                            onClick = { selectedName = destination.name },
                            icon = { Text(destination.shortLabel.take(1)) },
                            label = { Text(destination.shortLabel) },
                            modifier = Modifier.testTag(destination.navTag),
                        )
                    }
                }
            },
        ) { padding ->
            Surface(modifier = Modifier.fillMaxSize().padding(padding)) {
                when (selected) {
                    ShowcaseDestination.BACKEND_INSPECTOR -> BackendInspectorScreen(inspector, ioDispatcher)
                    ShowcaseDestination.WHISPER_STUDIO -> WhisperStudioScreen(whisperStudio)
                    else -> PlaceholderScreen(selected)
                }
            }
        }
    }
}

@Composable
fun BackendInspectorScreen(inspector: RuntimeInspector, ioDispatcher: CoroutineDispatcher) {
    var refreshCount by remember { mutableIntStateOf(0) }
    var state by remember { mutableStateOf<BackendInspectorState>(BackendInspectorState.Loading) }
    LaunchedEffect(inspector, refreshCount) {
        state = BackendInspectorState.Loading
        state = withContext(ioDispatcher) { loadInspectorState(inspector) }
    }
    BackendInspectorContent(state = state, onRefresh = { refreshCount++ })
}

@Composable
fun BackendInspectorContent(state: BackendInspectorState, onRefresh: () -> Unit) {
    val destination = ShowcaseDestination.BACKEND_INSPECTOR
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
        when (state) {
            BackendInspectorState.Loading -> Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.testTag("inspector-loading"))
                Spacer(Modifier.width(12.dp))
                Text("Querying the Kggml runtime…")
            }

            is BackendInspectorState.Failed -> Card(modifier = Modifier.fillMaxWidth().testTag("inspector-error")) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Native runtime unavailable", fontWeight = FontWeight.Bold)
                    Text(state.message, style = MaterialTheme.typography.bodySmall)
                    Text(
                        "Bundle the native libraries (-Pkggml.bundleNative=true) or point kggml.native.path at a " +
                            "host build of kggml-runtime/native.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            is BackendInspectorState.Ready -> {
                state.rows.forEach { row -> BackendRowCard(row) }
                Card(modifier = Modifier.fillMaxWidth().testTag("inspector-metadata")) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Runtime", fontWeight = FontWeight.Bold)
                        Text("Recommended threads (P-cores): ${state.recommendedThreadCount}")
                        Text("Hardware threads: ${state.hardwareThreadCount}")
                        Text("Kggml runtime: ${state.info.runtimeVersion}")
                        Text("ggml: ${state.info.ggmlVersion} (${state.info.ggmlCommit.take(12)})")
                        Text("Core ML capability: ${if (state.info.hasCoreMl) "yes" else "no"}")
                    }
                }
            }
        }
        Button(onClick = onRefresh, enabled = state !is BackendInspectorState.Loading, modifier = Modifier.testTag("inspector-refresh")) {
            Text("Refresh")
        }
    }
}

@Composable
private fun BackendRowCard(row: BackendRow) {
    Card(modifier = Modifier.fillMaxWidth().testTag(row.testTag)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                Text(row.label, fontWeight = FontWeight.Bold)
                Text(row.status.label, style = MaterialTheme.typography.bodySmall)
            }
            if (row.deviceCount > 0) Text("${row.deviceCount} device(s)")
        }
    }
}

@Composable
fun PlaceholderScreen(destination: ShowcaseDestination) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp).testTag(destination.screenTag),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(destination.title, style = MaterialTheme.typography.headlineSmall)
        Text(destination.summary, style = MaterialTheme.typography.bodyMedium)
        Text(
            "Coming in Phase ${destination.plannedPhase}.",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.testTag("placeholder-${destination.name.lowercase()}"),
        )
    }
}
