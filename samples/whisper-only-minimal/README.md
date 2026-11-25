# `:samples:whisper-only-minimal`

Minimal Android + iOS app that depends **only** on `com.donfuldauer:kggml-bom` and
`com.donfuldauer:kggml-whisper` (which brings `:kggml-runtime`). It exists for two reasons:

1. **Binary symbol leakage audits** (AGENTS.md §2.1): its shipped binaries must contain zero `llama.cpp`,
   `libmtmd` or `stable-diffusion.cpp` symbols, and exactly one copy of every `ggml` symbol (in the
   `:kggml-runtime` libraries). `verifyMinimalDependencies` (part of `check`) fails if `:kggml-llama` or
   `:kggml-diffusion` is ever resolved; `scripts/audit-binary-symbols.sh` checks the built artifacts.
2. **Manual verification** of `WhisperEngine` on devices: PCM file transcription, the runtime
   `ComputeBackend` toggle and the iOS background `CPU_ONLY` path.

The UI uses only platform toolkits (`android.widget`, UIKit), no Compose/AndroidX.

| Feature | Android | iOS |
| --- | --- | --- |
| Transcribe a PCM file (16-bit / float WAV, or raw 16 kHz mono s16le `.pcm`) | ✓ | ✓ |
| Backend toggle: CPU only / GPU (Vulkan on Android, Metal on iOS) / Auto | ✓ | ✓ |
| Backend switch reuses the engine (`WhisperEngine.reinitialize`) | ✓ | ✓ |
| Engine follows the app lifecycle (GPU → `CPU_ONLY` in the background) | — | ✓ (`observeAppLifecycle`) |
| Background transcription test mode | — | ✓ |

Shared logic (WAV/PCM parsing, the engine session, the background test runner) lives in `commonMain` and is
covered by `commonTest` with fake engines.

## Models and audio

Not committed (license / size). Use e.g. `ggml-tiny.en.bin` from the whisper.cpp model repository (MIT) and
`jfk.wav` from `kggml-whisper/native/third_party/whisper.cpp/samples/` (16 kHz mono 16-bit).

## Android

```bash
./gradlew :samples:whisper-only-minimal:installDebug -Pkggml.bundleNative=true   # needs ANDROID_NDK_HOME
adb push ggml-tiny.en.bin jfk.wav /sdcard/Android/data/com.donfuldauer.kggml.samples.whisper/files/
# Optional scripted run:
adb shell am start -n com.donfuldauer.kggml.samples.whisper/.MainActivity --es backend CPU_ONLY --ez autostart true
```

## iOS

`WhisperOnlyMinimal.framework` is a static Kotlin framework; the host app embeds the two dynamic
xcframeworks next to it.

```bash
./kggml-runtime/native/scripts/build-ios-xcframework.sh   # build/native-ios/GgmlRuntime.xcframework
./kggml-whisper/native/scripts/build-ios-xcframework.sh   # build/native-ios/KggmlWhisper.xcframework
./gradlew :samples:whisper-only-minimal:linkDebugFrameworkIosArm64
```

Host app (Xcode → new iOS App, Swift, UIKit lifecycle or SwiftUI):

1. Add `samples/whisper-only-minimal/build/bin/iosArm64/debugFramework/WhisperOnlyMinimal.framework` under
   *Frameworks, Libraries, and Embedded Content* as **Do Not Embed** (static), and both
   `GgmlRuntime.xcframework` and `KggmlWhisper.xcframework` as **Embed & Sign**.
2. Link `Accelerate`, `Metal`, `MetalKit`, `CoreML` and `Foundation`, and add `-lc++` to *Other Linker Flags*.
3. Use the Kotlin view controller as the root:
   ```swift
   import WhisperOnlyMinimal
   // SceneDelegate.scene(_:willConnectTo:options:)
   window.rootViewController = IosEntryPointKt.MainViewController()
   ```
4. Enable *UIFileSharingEnabled* and *LSSupportsOpeningDocumentsInPlace* in `Info.plist`, then copy
   `ggml-tiny.en.bin` and `jfk.wav` into the app's Documents folder with Finder (or `xcrun devicectl`).

## Physical iOS background `CPU_ONLY` verification (Phase 2 exit criterion)

Goal: confirm that `ComputeBackend.CPU_ONLY` (`whisper_context_params.use_gpu = false`) keeps transcribing
while the app is in the background, without Metal `MTLCommandBufferErrorDomain` / `accessRevoked` aborts. This
needs a **physical device** (the simulator does not revoke GPU access).

1. Install the host app on an iPhone (Release or Debug), with the model and audio in Documents.
2. Select **CPU only**, tap **Transcribe** once and confirm the result shows
   `ran on CPU_ONLY (use_gpu=false)` and the expected text.
3. Tap **Background test (20 s)** and immediately press the Home button / swipe home (or lock the device).
4. Wait ~30 s, return to the app and read the log. Each line is
   `#n [background|foreground] CPU_ONLY use_gpu=false <ms> ms: <text>`; the last line must start with
   `PASS:` and report at least one background run completed. The same lines are written with `NSLog`
   (`kggml-whisper-bg-test`), visible in Console.app while the device is attached.
5. In Xcode's device logs / Console.app, confirm there is no `MTLCommandBufferErrorDomain`,
   `accessRevoked` or `Execution of the command buffer was aborted` entry and that the app did not crash
   (no new crash report under *Settings → Privacy → Analytics Data*).
6. Optional (lifecycle fallback): select **GPU (Metal)**, repeat steps 3–5. The engine is switched to
   `CPU_ONLY` on `UIApplicationDidEnterBackgroundNotification`, so background runs must also report
   `use_gpu=false`. A run that was executing on the GPU at the transition may be reported as
   `ABORTED (GPU run at background transition)`; it does not fail the test, but every other background run
   must succeed.

Record the device model, iOS version, model file, the summary line and the commit SHA in the Phase 2 gate
issue / PR.

## Binary symbol audit

```bash
# Android (Linux CI): self-test of the audit + release APK with bundled native libraries + audit of every ABI
./scripts/ci/audit-minimal-sample.sh android
# iOS (macOS CI, after linkDebugFrameworkIos* and both xcframework builds)
./scripts/ci/audit-minimal-sample.sh ios
# Direct use
./scripts/audit-binary-symbols.sh apk <app.apk> [abi ...]
./scripts/audit-binary-symbols.sh ios <x.framework | x.xcframework | mach-o> ...
./scripts/audit-binary-symbols.sh elf <dir> ...
```

## Tests

```bash
./gradlew :samples:whisper-only-minimal:check                    # commonTest on Android (debug + release) + dependency audit
./gradlew :samples:whisper-only-minimal:iosSimulatorArm64Test    # macOS, needs both xcframeworks
./scripts/tests/test-audit-binary-symbols.sh                     # audit script incl. negative (forbidden symbol) cases
```
