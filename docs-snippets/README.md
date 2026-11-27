# `:docs-snippets`

The code examples of the root [`README.md`](../README.md), as Kotlin that the build compiles. Not published.

Each README code block in Kotlin, Gradle (`kts` / `groovy`), Java or Swift is a region of a file in this module:

```kotlin
// snippet:start whisper-transcribe
WhisperEngine(modelPath, BackendConfig.DEFAULT).use { engine -> /* ... */ }
// snippet:end whisper-transcribe
```

and the README marks the block with `<!-- snippet: whisper-transcribe -->` on the line before its opening fence.
`scripts/ci/verify-docs.sh` fails if such a README block has no marker, differs from its region (after removing
the common indentation), or if a region is not used by the README.

The module depends on the SDK only through its published coordinates (`com.donfuldauer:kggml-bom` and the engine
artifacts, with no versions), resolved from the build-local verification repository that
`publish...ToKggmlVerificationRepository` fills, exactly as an app would. `build.gradle.kts` holds the README's Gradle
snippet, so a wrong artifact id, a wrong BOM version or an API change that breaks an example fails
`./gradlew :docs-snippets:build` (part of `./gradlew check`).

Only the JVM target is compiled: the snippets use common API only. Platform-specific calls in the README
(`AndroidMemoryPressureObserver`, `IosMemoryPressureObserver`, `observeAppLifecycle`) are named in comments and covered
by their modules' own tests.
