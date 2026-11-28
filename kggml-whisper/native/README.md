# `:kggml-whisper` native layer

`libkggml_whisper` (`KggmlWhisper.framework` on iOS) is one shared library that contains:

- `whisper.cpp` **v1.9.6** (`79b11911d9d38c4be2fc7dc8b4efeff3324c090f`, `v1.9.6-181-g79b11911`, git submodule
  `third_party/whisper.cpp`, recorded in `NOTICE`). Only `src/whisper.cpp` and `src/aneforge/whisper-aneforge.cpp`
  (plus `src/coreml/*` on iOS) are compiled. The vendored `ggml/` directory and whisper.cpp's own `CMakeLists.txt` are
  never used.
- The Kggml C ABI in [`include/kggml_whisper.h`](include/kggml_whisper.h): backend toggle, CoreML gate, abort
  plumbing, PCM conversion, streaming states, and FFI error mapping.

ggml is **not** built here. The library links the shared ggml of `:kggml-runtime` dynamically through the
runtime's CMake package (`find_package(kggml-runtime CONFIG)` → `kggml::runtime`, `kggml::ggml`):

| Platform | Links | Output |
| --- | --- | --- |
| Linux / Android | `NEEDED libggml.so, libggml-base.so, libkggml_runtime.so` | `libkggml_whisper.so` (unversioned on Android) |
| macOS | `LC_LOAD_DYLIB libggml*.dylib, libkggml_runtime.dylib` | `libkggml_whisper.dylib` |
| Windows | `ggml.dll, ggml-base.dll, kggml_runtime.dll` imports | `kggml_whisper.dll` |
| iOS | `GgmlRuntime.framework` (monolithic runtime) + CoreML | `KggmlWhisper.xcframework` |

`scripts/ci/audit-whisper-linkage.sh` checks that none of the `ggml_*` / `gguf_*` symbols exported by the runtime is
*defined* in `libkggml_whisper`, at any binding, so a hidden static copy is caught too. The iOS script runs the
same check against `GgmlRuntime.framework`.

## Build

```bash
# Host (Linux / macOS): builds :kggml-runtime into build/native first if needed, then build/native-whisper + CTest.
kggml-whisper/native/scripts/build-host.sh
ctest --test-dir build/native-whisper --output-on-failure
./scripts/ci/audit-whisper-linkage.sh host

# Android arm64-v8a + x86_64 (needs ANDROID_NDK_HOME). Stages libkggml_whisper.so + libkggml_whisper_jni.so in build/native-android-whisper-jniLibs/<abi>.
kggml-whisper/native/scripts/build-android.sh
./scripts/ci/audit-whisper-linkage.sh android

# iOS (macOS + Xcode): KggmlWhisper.xcframework (device + simulator, CoreML) linked to GgmlRuntime.framework.
kggml-whisper/native/scripts/build-ios-xcframework.sh

# Windows (MSVC developer prompt, after building the runtime with its windows-x64 preset):
cmake -S kggml-whisper/native --preset windows-x64 -B build/native-whisper
cmake --build build/native-whisper --config Release
ctest --test-dir build/native-whisper --build-config Release --output-on-failure
```

With an installed runtime, point CMake at its package: `-Dkggml-runtime_DIR=<prefix>/lib/cmake/kggml-runtime`.

## Backend toggle

`kggml_whisper_context_params.backend` (a `kggml_backend_kind`) is mapped to `whisper_context_params.use_gpu`
**before** `whisper_init_from_file_with_params_no_state()` runs:

| Backend | `use_gpu` | CoreML encoder | Notes |
| --- | --- | --- | --- |
| `KGGML_BACKEND_CPU` | `false` | never loaded | Accelerate / NEON / AVX only. Safe for iOS background work (no Metal command buffers). |
| `KGGML_BACKEND_METAL`, `KGGML_BACKEND_VULKAN`, `KGGML_BACKEND_OPENCL` | `true` | never loaded | `BACKEND_UNAVAILABLE` if the runtime has no such device. |
| `KGGML_BACKEND_COREML` | `true` | loaded if `<model>-encoder.mlmodelc` exists | `BACKEND_UNAVAILABLE` if the build has no CoreML. A missing bundle falls back to Metal, then CPU. |

On iOS, whisper.cpp is built with `WHISPER_USE_COREML` (the define behind upstream's `WHISPER_COREML` option) and
`WHISPER_COREML_ALLOW_FALLBACK`. Upstream always tries to
load the `.mlmodelc`, so the upstream loader is compiled under a different name and `src/coreml_gate.cpp` supplies
`whisper_coreml_init()`. That gate loads CoreML only for contexts created with `KGGML_BACKEND_COREML`.
`kggml_whisper_get_backend()` reports the backend actually in use.

## Abort, threading, and errors

- `kggml_whisper_request_abort()` stores an atomic flag and is safe from any thread. The flag is polled by
  `whisper_full_params.abort_callback`, which runs inside every ggml graph compute, and by
  `encoder_begin_callback`. The interrupted `kggml_whisper_transcribe()` returns `KGGML_STATUS_CANCELLED` and
  discards partial segments. The flag stays set until `kggml_whisper_clear_abort()`.
- A context is not thread-safe. The Kotlin `WhisperEngine` runs init, transcribe, and free on its own
  single-threaded dispatcher and calls `kggml_whisper_request_abort()` from a coroutine cancellation handler.
- Every `extern "C"` function catches all C++ exceptions, including ones thrown inside whisper.cpp, and maps them:
  `std::bad_alloc` → `OUT_OF_MEMORY`, `std::exception` → `NATIVE_FAILURE`, anything else → `UNKNOWN`. The
  message is stored thread-locally and is read with `kggml_whisper_last_error()`. Status codes `0..6` equal
  `NativeErrorCode`.
- Input shorter than 1000 ms is zero-padded to 16000 samples before `whisper_full_with_state()`. Longer input is
  passed through without a copy.

## Streaming states

Live transcription (`WhisperEngine.transcribeStream`) does not use the context's built-in `whisper_state`. Each
stream owns its own:

| Function | Wraps | Notes |
| --- | --- | --- |
| `kggml_whisper_state_create(ctx, &state)` | `whisper_init_state` | Same backend / CoreML choice as the context. Free before the context. |
| `kggml_whisper_state_free(state)` | `whisper_free_state` | `NULL` is a no-op. |
| `kggml_whisper_transcribe_with_state(ctx, state, pcm, n, params, prompt_tokens, n_prompt, callback, user_data)` | `whisper_full_with_state` | Same 1000 ms padding and abort flag as `kggml_whisper_transcribe()`. Sets `prompt_tokens` / `prompt_n_tokens` and `new_segment_callback`. |
| `kggml_whisper_state_segment_count` / `_get_segment` | `whisper_full_*_from_state` | Results of the state's last successful run. |
| `kggml_whisper_state_get_segment_tokens` | `whisper_full_get_token_id_from_state` | Text tokens only (`id < whisper_token_eot`). Uses the `BUFFER_TOO_SMALL` size query. |
| `kggml_whisper_max_prompt_tokens(ctx)` | `n_text_ctx / 2 - 1` | How many prompt tokens whisper.cpp actually uses (223 for released models). |

- `prompt_tokens` must be vocabulary ids (`0 <= id < n_vocab`); anything else returns `INVALID_ARGUMENT` before
  whisper.cpp can index with it. With `no_context = 1` (what the Kotlin stream passes) they are the decoder's
  only history, so carry-over is explicit and deterministic. Upstream ignores `initial_prompt` when
  `prompt_tokens` is set.
- The `new_segment_callback` (`int32_t (*)(state, first_index, n_new, user_data)`) runs synchronously on the thread
  inside `kggml_whisper_transcribe_with_state()`. The new segments can be read from inside the callback. A
  non-`OK` return value stops decoding at the next abort poll and becomes the call's status (an abort request
  wins and gives `CANCELLED`). Segments of a stopped, cancelled, or failed run are discarded.
- The JNI bridge calls `WhisperSegmentListener.onNewSegments(int, int)` from that callback. Because JNI calls are
  illegal while a `GetPrimitiveArrayCritical` region is open, this path copies the PCM once with
  `GetFloatArrayRegion` and pins nothing. A Java exception thrown by the listener is cleared and returned as
  `NATIVE_FAILURE`. Without a listener, the PCM is pinned zero-copy as usual. On iOS the callback is a
  `staticCFunction`, and the listener travels as a `StableRef` that is disposed on every path.
  `consumer-rules.pro` keeps the callback method and the JNI natives from being renamed by R8.

## Kotlin bindings (`WhisperEngine`)

- **Android / JVM Desktop:** `jni/kggml_whisper_jni.cpp` builds `libkggml_whisper_jni` (`KGGML_WHISPER_BUILD_JNI`,
  on by default when JNI headers are found; never on iOS). It links `libkggml_whisper` dynamically, exports only
  `Java_*` symbols (checked by `audit-whisper-linkage.sh`), and runs every entry point behind a `try/catch` that
  returns a `kggml_status`. `float[]` / `short[]` PCM is pinned with `GetPrimitiveArrayCritical` and read in place;
  an RAII guard releases it on every path, and no other JNI call happens while it is held. Segment text is
  returned as raw UTF-8 bytes and decoded leniently in Kotlin.
- **iOS:** Kotlin/Native calls this C ABI directly through cinterop (`src/nativeInterop/cinterop/kggmlWhisper.def`),
  pinning arrays with `usePinned { it.addressOf(0) }`, and links `KggmlWhisper.framework` + `GgmlRuntime.framework`.
- On the JVM, `JvmWhisperLibraryLoader` loads the runtime first (via `KggmlRuntime`), then `kggml_whisper` and
  `kggml_whisper_jni` from `-Dkggml.whisper.native.path` (Gradle points it at `build/native-whisper/lib` for
  `:kggml-whisper:jvmTest`), the `kggml.native.path` directory, or `java.library.path`.

## PCM conversion

whisper.cpp accepts only 16 kHz mono Float32. `kggml_whisper_pcm_to_whisper()` (one-shot) and
`kggml_whisper_resampler_*` (streaming) convert interleaved Float32 or Int16 input (Int16 is divided by 32768)
with 1–8 channels (averaged to mono) at 1–384 kHz:

- `KGGML_RESAMPLE_SINC` (default) is a rational polyphase Kaiser-windowed sinc filter. Its cutoff sits just below
  the output Nyquist frequency, and every phase is normalized to unity DC gain. Against an ideal 16 kHz sine, the
  maximum error is below 2·10⁻³ for 44.1 and 48 kHz input. A 12 kHz tone at 48 kHz is suppressed to an RMS below
  10⁻³.
- `KGGML_RESAMPLE_LINEAR` is 2-tap interpolation without anti-aliasing.
- Streaming output is bit-identical to one-shot output for any chunking. Converting N input frames produces
  exactly `ceil(N · 16000 / rate)` output frames.

## Tests

`tests/` is a GoogleTest suite run by CTest (82 tests). It needs no model download: `tests/tiny_model.cpp` writes a
structurally valid 1-layer whisper model with deterministic random weights at test time, so the real whisper.cpp
encoder and decoder run. The suite covers:

- resampling accuracy (44.1 / 48 kHz × mono / stereo × Float32 / Int16), aliasing, DC gain, and chunked streaming;
- Int16 normalization and downmixing;
- the `use_gpu` mapping and backend availability;
- abort before and during inference, including abort from another thread;
- streaming states: create / free balance, short-buffer padding through `whisper_full_with_state`, the
  `new_segment_callback` (ordering, readability from inside the callback, stop-by-status), `prompt_tokens`
  validation and carry-over (checked against what reached `whisper_full_params`), multi-window input, and
  isolation from the context's own results;
- exception-to-status mapping through test-only fault injection (`KGGML_WHISPER_TEST_HOOKS`, never in release
  builds).
