# `:integration-tests`

Cross-engine integration tests. This module is **not published** and is not part of `kggml-bom`. It holds tests that
load several engine modules into one process, which no single engine module can do on its own.

## Dual-engine coexistence (`EngineCoexistenceTest`, issue #20)

`:kggml-whisper`, `:kggml-llama` and `:kggml-diffusion` are loaded into one JVM, each dynamically linked against the
single `:kggml-runtime` ggml build. The tests use the engines' synthetic CC0/MIT fixtures (`TinyWhisperModel`,
`TinyLlamaModel`), which are compiled in verbatim from the engine modules' `commonTest` sources, plus generated PCM.

| Test | What it proves |
| --- | --- |
| `whisperAndLlamaRunConcurrentlyWithTheResultsTheyProduceAlone` | Whisper transcription and Llama greedy generation run at the same time on different threads, three rounds in a row. Each engine returns exactly the result it returns alone, so the engines share no mutable ggml state and no symbol is interposed between them. |
| `severalInstancesOfBothEnginesShareTheRuntime` | Two `WhisperEngine`s and two `LlamaEngine`s infer concurrently in one process. |
| `diffusionLoadsAlongsideAndFailsTypedWithoutDisturbingTheOtherEngines` | `stable-diffusion.cpp` loads next to both engines. A missing or non-diffusion model fails with a typed `ModelLoadException` while Llama decodes, and both other engines keep working. |
| `everyGgmlLibraryIsMappedFromTheSingleRuntimeBuild` | Linux only: `/proc/self/maps` shows every `libggml*` library mapped from exactly one file, inside the `:kggml-runtime` build, and each engine library mapped once. On other hosts the dynamic-linkage audits (`scripts/ci/audit-*-linkage.sh`) cover this. |
| `repeatedOpenInferCloseCyclesKeepTheResidentSetBounded` | Fifteen concurrent open, infer and close cycles of both engines grow the resident set by less than 64 MB (Linux `VmRSS`), so nothing leaks and nothing runs out of memory. |

### Running

```bash
./kggml-runtime/native/scripts/build-host.sh
./kggml-whisper/native/scripts/build-host.sh
./kggml-llama/native/scripts/build-host.sh
./kggml-diffusion/native/scripts/build-host.sh
./gradlew :integration-tests:jvmTest -Pkggml.requireNative=true
```

Without the host builds the tests are skipped. With `-Pkggml.requireNative=true` a missing build is a failure. CI
runs the suite in the Ubuntu job (`./gradlew check`) and in the label-gated macOS and Windows jobs (`./gradlew jvmTest`),
and `./scripts/ci-local.sh` runs it in its `jvm` step.

## Manual verification

### Whisper and Llama together on devices

This is deferred to the owner's final acceptance pass (`docs/MANUAL_VERIFICATION.md`).

1. Install `:samples:kggml-showcase` on a physical Android phone and an iPhone, and on desktop run
   `./gradlew :samples:kggml-showcase:run`.
2. In **Model Manager**, download a Whisper model (for example `ggml-base.en`) and a small instruct GGUF (for example
   a 0.5B to 1.5B Q4 model).
3. Start live dictation in **Whisper Studio** and keep talking. While it transcribes, switch to **LLM Chat** and send a
   prompt that produces a long answer.
4. Expect both engines to keep producing output: transcript segments keep arriving and tokens keep streaming. The app
   must not crash, and neither engine may fail with an out-of-memory or native error.
5. On Android, run `adb shell cat /proc/$(adb shell pidof com.donfuldauer.kggml.samples.showcase)/maps | grep libggml`.
   Each `libggml*.so` must appear from exactly one path.
6. Close both screens and confirm in the Model Manager's resident-engine list that both engines are released.
