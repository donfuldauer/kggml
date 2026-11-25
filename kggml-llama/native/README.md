# `:kggml-llama` native layer

`libkggml_llama` (`KggmlLlama.framework` on iOS) is one shared library that contains:

- `llama.cpp` release **b8598** (`9281dd135d6f175ceacca7d3a3ec440012264cfe`, git submodule `third_party/llama.cpp`,
  recorded in `NOTICE`). That commit is the llama.cpp "sync : ggml" for ggml **v0.9.9**, the exact ggml the
  `:kggml-runtime` pins, so llama.cpp is compiled against the same public ggml API it was released with. Every
  `src/*.cpp` and `src/models/*.cpp` translation unit of upstream `src/CMakeLists.txt` is compiled (= `libllama`).
- `libmtmd` from the same commit: the library sources of `tools/mtmd/` (`mtmd*.cpp`, `clip.cpp`, `models/*.cpp`,
  plus the vendored `stb_image` / `miniaudio` headers it decodes media with). The CLI / debug tools are not built.
- The Kggml C ABI in [`include/kggml_llama.h`](include/kggml_llama.h): model / context lifecycle, backend selection,
  abort plumbing, batched decoding, KV-cache state serialization, GBNF grammar samplers, embeddings, in-memory
  multimodal evaluation and FFI error mapping.

llama.cpp's vendored `ggml/` directory, its `common/` helpers and its own `CMakeLists.txt` are never used. ggml is
**not** built here: the library links the shared ggml of `:kggml-runtime` dynamically through the runtime's CMake
package (`find_package(kggml-runtime CONFIG)` → `kggml::runtime`, `kggml::ggml`):

| Platform | Links | Output |
| --- | --- | --- |
| Linux / Android | `NEEDED libggml.so, libggml-base.so, libkggml_runtime.so` | `libkggml_llama.so` (unversioned on Android) |
| macOS | `LC_LOAD_DYLIB libggml*.dylib, libkggml_runtime.dylib` | `libkggml_llama.dylib` |
| Windows | `ggml.dll, ggml-base.dll, kggml_runtime.dll` imports | `kggml_llama.dll` |
| iOS | `GgmlRuntime.framework` (monolithic runtime) | `KggmlLlama.xcframework` |

`scripts/ci/audit-llama-linkage.sh` checks that none of the `ggml_*` / `gguf_*` symbols exported by the runtime is
*defined* in `libkggml_llama`, at any binding (so a hidden static copy is caught too), that no `whisper_*` symbol is
defined, and that the llama.cpp, libmtmd and Kggml entry points are exported. The iOS script runs the same check
against `GgmlRuntime.framework`. `:kggml-llama` and `:kggml-whisper` are independent libraries: an app that only
depends on `:kggml-whisper` ships no llama.cpp code at all (audited on `:samples:whisper-only-minimal`).

## Build

```bash
# Host (Linux / macOS): builds :kggml-runtime into build/native first if needed, then build/native-llama + CTest.
kggml-llama/native/scripts/build-host.sh
ctest --test-dir build/native-llama --output-on-failure
./scripts/ci/verify-llama-pin.sh
./scripts/ci/audit-llama-linkage.sh host

# Android arm64-v8a + x86_64 (needs ANDROID_NDK_HOME). Stages libkggml_llama.so + libkggml_llama_jni.so in
# build/native-android-llama-jniLibs/<abi>.
kggml-llama/native/scripts/build-android.sh
./scripts/ci/audit-llama-linkage.sh android

# iOS (macOS + Xcode): KggmlLlama.xcframework (device + simulator) linked to GgmlRuntime.framework.
kggml-llama/native/scripts/build-ios-xcframework.sh

# Windows (MSVC developer prompt, after building the runtime with its windows-x64 preset):
cmake -S kggml-llama/native --preset windows-x64 -B build/native-llama
cmake --build build/native-llama --config Release
ctest --test-dir build/native-llama --build-config Release --output-on-failure
```

With an installed runtime, point CMake at its package: `-Dkggml-runtime_DIR=<prefix>/lib/cmake/kggml-runtime`.
The installed `kggml-llama` package exports `kggml::llama` (headers `kggml_llama.h`, `llama.h`, `mtmd.h`,
`mtmd-helper.h`).

The Kotlin `LlamaEngine` (see [`../README.md`](../README.md)) is built on this C ABI: through `libkggml_llama_jni`
(`jni/kggml_llama_jni.cpp`, built when JNI headers are found or on Android; `-DKGGML_LLAMA_BUILD_JNI=OFF` skips it)
on Android / JVM Desktop, and through Kotlin/Native cinterop (`src/nativeInterop/cinterop/kggmlLlama.def`) on iOS.
The JNI bridge links `libkggml_llama` dynamically, pins token arrays with `GetPrimitiveArrayCritical` (zero-copy),
passes text as UTF-8 `byte[]`, runs every entry point behind a `try/catch` firewall and exports only `Java_*`
symbols (version script on ELF; audited by `scripts/ci/audit-llama-linkage.sh`). Image `byte[]`s of `nativeMtmdTokenize` are
pinned the same way, only for `kggml_llama_mtmd_tokenize()`: the (long) image encode runs with nothing pinned.

## Backend selection

`kggml_llama_model_params.backend` (a `kggml_backend_kind`) decides, **before** `llama_model_load_from_file()` runs,
which ggml devices llama.cpp may use:

| Backend | `llama_model_params` | Context | Notes |
| --- | --- | --- | --- |
| `KGGML_BACKEND_CPU` | `devices = {NULL}`, `main_gpu = -1`, `n_gpu_layers = 0` | `offload_kqv = false`, `op_offload = false` | No GPU backend is ever created: weights, KV cache and every op stay on Accelerate / NEON / AVX. Safe for iOS background work (no Metal command buffers). |
| `KGGML_BACKEND_METAL`, `KGGML_BACKEND_VULKAN` | `devices = {<gpu_device>-th device of that backend, NULL}`, `n_gpu_layers = gpu_layers` (-1 = all) | llama.cpp defaults | `BACKEND_UNAVAILABLE` if the runtime has no such device. |
| `KGGML_BACKEND_COREML` | — | — | `BACKEND_UNAVAILABLE` (llama.cpp has no CoreML backend). |

`kggml_llama_mtmd_params.backend` does the same for the vision / audio encoder (`mtmd_context_params.use_gpu`); by
default it follows the model.

## Error handling, threading and cancellation

- Every `kggml_status`-returning entry point runs behind `kggml::llama::guarded()` (`src/guard.hpp`):
  `std::bad_alloc` → `KGGML_STATUS_OUT_OF_MEMORY`, other `std::exception` → `KGGML_STATUS_NATIVE_FAILURE`, anything
  else → `KGGML_STATUS_UNKNOWN`, with a thread-local message in `kggml_llama_last_error()`. No C++ exception crosses
  the FFI boundary. User callbacks (load progress) are wrapped too.
- llama.cpp aborts the process (`GGML_ABORT` / `GGML_ASSERT`) on a few invalid inputs instead of throwing, so the
  wrapper validates first: token ids are range-checked before `llama_decode()` / `llama_token_to_piece()`, sequence
  ids against `n_seq_max`, grammar sampling is applied on its own so "no admissible token" becomes an error instead
  of an assertion, and `kggml_llama_sampler_accept(..., apply_grammar = 1)` probes the grammar before accepting
  (accepting an end-of-generation token the grammar does not admit would otherwise abort).
- Objects are not thread-safe: the Kotlin engines serialize every call on one dedicated single-threaded dispatcher.
  `kggml_llama_request_abort()` is the exception: it sets an atomic flag from any thread. The flag is polled by the
  callback registered with `llama_set_abort_callback()` (inside every ggml graph compute, so a running decode stops
  mid-graph), by `kggml_llama_decode()` between batches and by `kggml_llama_mtmd_eval[_chunks]()` between chunks (an image
  encode itself is not interruptible). The interrupted call returns `KGGML_STATUS_CANCELLED`; the flag stays latched
  until `kggml_llama_clear_abort()`.
- A full KV cache returns `KGGML_STATUS_BUFFER_TOO_SMALL` (make room with `kggml_llama_memory_seq_rm()`).

## State serialization, grammars, embeddings, multimodal

- `kggml_llama_state_size/save/load` wrap `llama_state_*` (whole context, `seq_id < 0`) and `llama_state_seq_*`
  (one sequence, restorable into another sequence id). Corrupt or truncated data is rejected with
  `KGGML_STATUS_INVALID_ARGUMENT`. Logits are not part of the restored state: decode the next token before sampling.
- `kggml_llama_sampler_create` builds `[grammar] → [penalties] → top-k → top-p → min-p → temperature → dist`
  (or greedy for `temperature <= 0`). The GBNF grammar (`llama_sampler_init_grammar`) constrains every sampled token;
  JSON output is a GBNF grammar (JSON-schema → GBNF conversion lives in the Kotlin layer, `LlamaGrammar.jsonSchema()`).
- `kggml_llama_get_embeddings` returns the pooled sequence embedding (or the last token's for `POOLING_NONE`),
  optionally L2-normalized.
- `kggml_llama_mtmd_eval` takes images as raw RGB888 buffers (wrapped with `mtmd_bitmap_init`) or encoded bytes
  (PNG / JPEG / ..., decoded in memory by `mtmd_helper_bitmap_init_from_buf`). No temporary image file is ever
  written. The prompt contains one `kggml_llama_mtmd_default_marker()` per image. It is
  `kggml_llama_mtmd_tokenize()` (wrap / decode + preprocess; the caller's bytes are not referenced afterwards) followed
  by `kggml_llama_mtmd_eval_chunks()` (encode + evaluate, `KGGML_STATUS_BUFFER_TOO_SMALL` before evaluating anything
  when the prompt does not fit the context); the bindings use the two halves so image buffers stay pinned only
  briefly. `kggml_llama_mtmd_chunks_info()` reports token / position / image counts.

## Tests

`tests/` is a GoogleTest suite registered with CTest (67 cases, labelled `kggml-llama`). It needs no model download:
`tests/tiny_model.cpp` writes a 1-block, 32-wide llama-architecture GGUF with a synthetic 288-token SentencePiece
vocabulary through the ggml `gguf_*` API at test time (see `NOTICE`). The suite covers model-load error handling
(missing, garbage, truncated, directory, cancelled by progress callback, unavailable backends), `extern "C"`
exception trapping (fault injection through test-only hooks compiled in with `KGGML_LLAMA_TEST_HOOKS`, never in
release builds), abort-callback triggering (before decode, mid-graph through a poll countdown, between batches,
from another thread), batched decoding, KV-cache state round trips, embeddings, GBNF-constrained sampling, libmtmd
projector load errors, in-memory image evaluation with a synthetic LLaVA-style projector (`tiny_mmproj_path()`: RGB
and BMP inputs give identical logits, invalid images / marker mismatches / a too-small context return typed errors,
abort, no temporary files) and deterministic resource release (live-object counters and the runtime reference count
return to their baseline).
