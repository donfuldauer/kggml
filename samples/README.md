# Kggml samples

Both samples consume the SDK exclusively through the BOM (`platform(project(":kggml-bom"))` + versionless
module dependencies). They are never published.

| Sample | Targets | Purpose |
| --- | --- | --- |
| `:samples:kggml-showcase` | Android, iOS, JVM Desktop (Compose Multiplatform) | Five functional screens: **Hardware Backend Inspector** (`KggmlRuntime`), **Model Manager** (catalog, resumable / verified downloads, header metadata + peak-RAM estimate, manual / OS low-memory eviction), **Live Whisper Dictation & Audio Studio**, **LLM Chat, Structured Output & Vision** (streaming chat, thread / GPU-layer controls, KV snapshots, JSON-schema extraction, embeddings, in-memory VLM images) and **Diffusion Image Studio** (steps / CFG / seed, live progress, cancel, PNG export); see [its README](kggml-showcase/README.md). |
| `:samples:whisper-only-minimal` | Android, iOS | Minimal app depending only on `kggml-bom` + `:kggml-whisper`: PCM file transcription, runtime `ComputeBackend` toggle and an iOS background `CPU_ONLY` test mode (see [its README](whisper-only-minimal/README.md)). Used for CI binary symbol leakage audits (`scripts/audit-binary-symbols.sh`); `verifyMinimalDependencies` (part of `check`) fails if `:kggml-llama` or `:kggml-diffusion` is ever resolved. |

## Running

```bash
# JVM Desktop (uses build/native/lib automatically if kggml-runtime/native/scripts/build-host.sh was run)
./gradlew :samples:kggml-showcase:run

# Android APKs (add -Pkggml.bundleNative=true to package the native runtime for arm64-v8a / x86_64)
./gradlew :samples:kggml-showcase:assembleDebug :samples:whisper-only-minimal:assembleDebug

# iOS static frameworks (macOS). The Xcode host app embeds GgmlRuntime.xcframework and KggmlWhisper.xcframework
# (*/native/scripts/build-ios-xcframework.sh) next to KggmlShowcase.framework / WhisperOnlyMinimal.framework.
./gradlew :samples:kggml-showcase:linkDebugFrameworkIosSimulatorArm64
```

Without native libraries the inspector shows a "Native runtime unavailable" state instead of crashing.

## Tests

```bash
./gradlew :samples:kggml-showcase:desktopTest          # state + Compose UI tests (+ real JNI runtime if built)
./gradlew :samples:kggml-showcase:testDebugUnitTest    # common state tests on Android
./gradlew :samples:whisper-only-minimal:check          # unit tests + dependency audit
```
