# `:kggml-diffusion` native layer

`libkggml_diffusion` (`KggmlDiffusion.framework` on iOS) is one shared library that contains:

- `stable-diffusion.cpp` at **`3b4d26f3d92c8db0199d7326f66260f4c18f4b41`** (2026-05-17, no release tag; `git describe`
  = `master-612-d7ecbe1-2-g3b4d26f`; git submodule `third_party/stable-diffusion.cpp`, recorded in `NOTICE`). It is
  the newest commit whose own `ggml` submodule is still **upstream** ggml (`404fcb9d`, two commits behind the
  `:kggml-runtime` pin ggml v0.9.9 `a04eea07`); later commits switch to leejet's ggml fork with custom ops that the
  shared runtime does not have. Every translation unit of upstream's `stable-diffusion` library target is compiled
  (`src/*.cpp`, `src/model_io/*.cpp`, `src/tokenizers/*.cpp`, `src/tokenizers/vocab/*.cpp`, the same GLOB as its
  `CMakeLists.txt`) plus `thirdparty/zip.c` (kuba--/zip + miniz, for `.ckpt` files); `json.hpp` and `darts.h` are
  header-only. `sd_version()` / `sd_commit()` are baked from the pin, not from the checkout.
- The Kggml C ABI in [`include/kggml_diffusion.h`](include/kggml_diffusion.h): context lifecycle, backend selection,
  txt2img generation with a per-step progress callback and a mid-generation abort, RGB8 output, and FFI error
  mapping.

stable-diffusion.cpp's own `CMakeLists.txt`, its nested submodules (its `ggml` copy, `libwebp`, `libwebm`, the web
UI) and its examples / server are never initialised or built (`git submodule update --init` of this submodule only;
CI checks out first-level submodules). ggml is **not** built here: the library links the shared ggml of
`:kggml-runtime` dynamically through the runtime's CMake package (`find_package(kggml-runtime CONFIG)` →
`kggml::runtime`, `kggml::ggml`):

| Platform | Links | Output |
| --- | --- | --- |
| Linux / Android | `NEEDED libggml.so, libggml-base.so, libggml-cpu.so, libkggml_runtime.so` | `libkggml_diffusion.so` (unversioned on Android) |
| macOS | `LC_LOAD_DYLIB libggml*.dylib, libkggml_runtime.dylib` | `libkggml_diffusion.dylib` |
| Windows | `ggml.dll, ggml-base.dll, ggml-cpu.dll, kggml_runtime.dll` imports | `kggml_diffusion.dll` |
| iOS | `GgmlRuntime.framework` (monolithic runtime) | `KggmlDiffusion.xcframework` |

`scripts/ci/audit-diffusion-linkage.sh` checks that none of the `ggml_*` / `gguf_*` symbols exported by the runtime
is *defined* in `libkggml_diffusion`, at any binding (so a hidden static copy is caught too), that no `whisper_*` /
`llama_*` / `mtmd_*` symbol is defined, that the vendored zip / miniz symbols stay hidden, and that the
stable-diffusion.cpp and Kggml entry points are exported (Android: unversioned soname, no test hooks). The iOS
script runs the same check against `GgmlRuntime.framework`. `:kggml-diffusion`, `:kggml-llama` and `:kggml-whisper`
are independent libraries: an app that only depends on `:kggml-whisper` ships no stable-diffusion.cpp code.

### ggml internals: `ggml-impl.h`

`src/ggml_graph_cut.cpp` includes `"../ggml/src/ggml-impl.h"` (its vendored ggml) for `struct ggml_cgraph`'s
internals. Those must be the internals of the ggml actually linked, so `CMakeLists.txt` generates a one-line shim
(`<build>/sd-ggml-shim/ggml/src/ggml-impl.h`) that includes `:kggml-runtime`'s `third_party/ggml/src/ggml-impl.h`, and
fails to configure if that checkout is not the commit the linked runtime was built from
(`KGGML_GGML_PINNED_COMMIT`). Override the location with `-DKGGML_DIFFUSION_GGML_SOURCE_DIR=<ggml source>`.

### ggml ABI: `GGML_MAX_NAME=128`

stable-diffusion.cpp requires `GGML_MAX_NAME >= 128` (`static_assert` in `src/ggml_extend.hpp`; its tensor names are
longer than ggml's default 64). The value changes `struct ggml_tensor`'s layout, so it cannot be set here alone:
`:kggml-runtime` builds ggml with `GGML_MAX_NAME=128` and exports it as a PUBLIC compile definition of
`kggml::runtime` / `kggml::ggml`, so this library, whisper, llama, every JNI bridge and every test compile with the
same value (see [`../../kggml-runtime/native/README.md`](../../kggml-runtime/native/README.md), "ggml ABI"). This
CMake project refuses to configure against a runtime package whose `KGGML_GGML_MAX_NAME` is below 128 (an old build
tree: rebuild the runtime with `--clean`). The test `DiffusionContext.HundredCharTensorNamesParseThroughTheSharedGgml`
proves the agreement end to end (a 100-character tensor name written and parsed by the shared `libggml-base` reaches
stable-diffusion.cpp's model detection).

## Build

```bash
# Host (Linux / macOS): builds :kggml-runtime into build/native first if needed, then build/native-diffusion + CTest.
kggml-diffusion/native/scripts/build-host.sh
ctest --test-dir build/native-diffusion --output-on-failure
./scripts/ci/verify-diffusion-pin.sh
./scripts/ci/audit-diffusion-linkage.sh host

# Android arm64-v8a + x86_64 (needs ANDROID_NDK_HOME). Stages libkggml_diffusion.so + libkggml_diffusion_jni.so in
# build/native-android-diffusion-jniLibs/<abi>.
kggml-diffusion/native/scripts/build-android.sh
./scripts/ci/audit-diffusion-linkage.sh android

# iOS (macOS + Xcode): KggmlDiffusion.xcframework (device + simulator) linked to GgmlRuntime.framework.
kggml-diffusion/native/scripts/build-ios-xcframework.sh

# Windows (MSVC developer prompt, after building the runtime with its windows-x64 preset):
cmake -S kggml-diffusion/native --preset windows-x64 -B build/native-diffusion
cmake --build build/native-diffusion --config Release
ctest --test-dir build/native-diffusion --build-config Release --output-on-failure
```

With an installed runtime, point CMake at its package: `-Dkggml-runtime_DIR=<prefix>/lib/cmake/kggml-runtime`.
The installed `kggml-diffusion` package exports `kggml::diffusion` (headers `kggml_diffusion.h`,
`stable-diffusion.h`).

The build also produces `kggml_diffusion_jni` (`jni/kggml_diffusion_jni.cpp`, option `KGGML_DIFFUSION_BUILD_JNI`, on
by default for Android and for desktop hosts with JNI headers): the JNI bridge of the Kotlin `DiffusionEngine` on
Android / JVM Desktop. It links `kggml_diffusion` dynamically, exports only `Java_*` entry points (ELF version script
`jni/kggml_diffusion_jni.map`) and is audited by `scripts/ci/audit-diffusion-linkage.sh`. iOS calls the C ABI directly
through Kotlin/Native cinterop. See [`../README.md`](../README.md) for the Kotlin API.

## Backend selection

`kggml_diffusion_context_params` maps `BackendConfig` before `new_sd_ctx()` runs:

| Field | stable-diffusion.cpp | Notes |
| --- | --- | --- |
| `backend = KGGML_BACKEND_CPU` (default) | `sd_ctx_params_t.backend = "CPU"` (the ggml CPU device, for every module) | No GPU backend is ever created; safe for iOS background work. |
| `backend = METAL` / `VULKAN`, `gpu_device` | `backend = <name of the gpu_device-th device of that kind>` (e.g. `MTL0`, `Vulkan0`) | `BACKEND_UNAVAILABLE` if the runtime has no such device. |
| `backend = COREML` | — | `BACKEND_UNAVAILABLE` (stable-diffusion.cpp has no CoreML backend). |
| `n_threads` (`cpuThreads`) | `n_threads` | `<= 0` = performance cores (`kggml_runtime_performance_thread_count`). |
| `use_mmap` (`useMmap`) | `enable_mmap` | |
| `flash_attn` (`useFlashAttention`) | `flash_attn` + `diffusion_flash_attn` | |
| `offload_params_to_cpu` | `offload_params_to_cpu` (GPU backends only) | Closest equivalent of `gpuLayers = 0`; stable-diffusion.cpp has no per-layer offload. |
| `keep_clip_on_cpu`, `keep_vae_on_cpu` | same | |

Weights stay resident between generations (`free_params_immediately = false`).

## Error handling, threading and cancellation

- Every `kggml_status`-returning entry point runs behind `kggml::diffusion::guarded()` (`src/guard.hpp`):
  `std::bad_alloc` → `KGGML_STATUS_OUT_OF_MEMORY`, other `std::exception` → `KGGML_STATUS_NATIVE_FAILURE`, anything
  else → `KGGML_STATUS_UNKNOWN`, with a thread-local message in `kggml_diffusion_last_error()`. No C++ exception
  crosses the FFI boundary. User callbacks are wrapped too (a throwing callback aborts the generation).
- Model loading: every given path is opened first (`MODEL_LOAD_FAILED` naming the missing / empty file);
  `new_sd_ctx()` returning NULL, or throwing a `std::exception` while parsing, is `MODEL_LOAD_FAILED` with the first
  stable-diffusion.cpp `ERROR` log line appended (e.g. `get sd version from file failed`).
- stable-diffusion.cpp's log and progress hooks are **process-wide** and it has **no abort hook** at the pinned
  commit. The wrapper installs one dispatcher for each, once per process (`std::call_once`; never re-registered, so
  contexts cannot clobber each other), and routes every call to the generation running on the **calling thread**
  (a `thread_local` pointer set for the duration of `kggml_diffusion_generate()`). The sampler reports progress on
  the thread that called `generate_image()`, so concurrent generations on different contexts never see each other's
  callbacks. Progress reported while no generation is active (weight loading inside `new_sd_ctx()`) is ignored.
- Abort: `kggml_diffusion_request_abort()` latches an atomic flag (any thread); the per-call `abort_cb` is polled
  too. Both are checked before generation starts and at every progress point (each denoising step, each VAE tile
  with `vae_tiling`). On abort the dispatcher throws a private `AbortSignal` (not a `std::exception`, so upstream
  `catch (const std::exception &)` blocks never swallow it). It unwinds stable-diffusion.cpp's sampler — RAII-only
  C++ frames at the pinned commit — back into `kggml_diffusion_generate()`, which returns `KGGML_STATUS_CANCELLED`.
  **Limitation:** the granularity is one denoising step (the step being computed finishes first); a graph compute
  itself is not interrupted, because stable-diffusion.cpp does not expose its ggml backends for
  `ggml_backend_cpu_set_abort_callback()`. Transient compute buffers of an aborted run stay allocated in the context
  until the next generation reuses them or the context is freed. The flag stays latched until
  `kggml_diffusion_clear_abort()`. On MSVC the library is compiled with `/EHs /EHc-` because the unwind passes
  `generate_image()`, which is declared `extern "C"`.
- Logging: stable-diffusion.cpp is silent by default (`KGGML_DIFFUSION_LOG=1` or `=debug` echoes it to stderr).
  `new_sd_ctx()` resets the process-wide ggml logger to ggml's stderr default; `kggml_diffusion_context_create()`
  saves and restores the previous ggml logger around it (context creation is serialized by a mutex for that).
  Upstream formats log lines in a static buffer, so concurrent contexts may interleave diagnostic text (never
  status codes).
- Contexts are not thread-safe; the Kotlin engine serializes every call on one dedicated single-threaded dispatcher.
  Generation output is the RGB8 buffer allocated by stable-diffusion.cpp, handed over without a copy; free it with
  `kggml_diffusion_image_free()`.

## Tests

`tests/` is a GoogleTest suite registered with CTest (50 cases, labelled `kggml-diffusion`). It needs no model:
real inference with a synthetic diffusion model is not part of #16. It covers:

- **Init error handling** through the real stable-diffusion.cpp loader: missing / empty files, junk bytes with
  `.gguf` / `.safetensors` / `.ckpt` / `.bin` extensions, a safetensors header with an absurd length, a truncated
  GGUF, a well-formed GGUF that is not a diffusion model (all `MODEL_LOAD_FAILED` with a message), out-of-range
  parameters (`INVALID_ARGUMENT`), CoreML / missing GPU devices (`BACKEND_UNAVAILABLE`), concurrent failing creates
  with per-thread messages, and no leaked runtime reference or context after any failure. GGUF fixtures are written
  at test time through the shared runtime's `gguf_*` API (see `NOTICE`).
- **`extern "C"` exception safety** with test-only hooks (`src/kggml_diffusion_testing.h`, compiled in only with
  `KGGML_DIFFUSION_TEST_HOOKS`, never in release builds): faults injected at the guard entry and thrown mid-generation
  through upstream frames (`std::bad_alloc` → `OUT_OF_MEMORY`, `std::runtime_error` / `std::out_of_range` →
  `NATIVE_FAILURE`, `throw 42` → `UNKNOWN`), throwing user callbacks, thread-local errors and faults.
- **Progress / abort plumbing**: `kggml_diffusion_test_simulate_generate()` runs the real generation machinery with
  the sampler replaced by a loop over upstream's own `pretty_progress()`, so callbacks travel through the
  process-wide hook exactly as in a real run: steps `0..N` in order with the right totals and `user_data` on the
  calling thread, abort callback at step k → `CANCELLED` with no later progress, latched `request_abort()` (before
  start and from another thread mid-run), re-entrant generation rejected, six concurrent contexts with different
  step counts / abort points without cross-talk, the context reusable after an abort or fault, and the zero-copy RGB8
  hand-over / `kggml_diffusion_image_free()`.
- **ABI**: `GGML_MAX_NAME == 128` in this module's translation units and the 100-character tensor-name check above.

## Manual verification

Deferred to the owner's acceptance pass (`docs/MANUAL_VERIFICATION.md`): generate one image with a real model through
the C ABI on desktop (CPU, and Metal on macOS / Vulkan where available) and check the output and abort latency.

1. Download a small SD 1.5-class checkpoint in GGUF / safetensors form (for example a q8_0 SD 1.5 GGUF) to
   `$MODEL`. Build the host library: `kggml-diffusion/native/scripts/build-host.sh --no-tests`.
2. Save as `/tmp/sd-smoke.c` and build it against `build/native-diffusion` (`cc /tmp/sd-smoke.c
   -Ikggml-diffusion/native/include -Ikggml-runtime/native/include -Lbuild/native-diffusion/lib -lkggml_diffusion
   -Wl,-rpath,$PWD/build/native-diffusion/lib -Wl,-rpath,$PWD/build/native/lib -o /tmp/sd-smoke`):

   ```c
   #include "kggml_diffusion.h"
   #include <stdio.h>
   #include <stdlib.h>
   static void on_progress(int32_t s, int32_t n, float t, void * u) { (void) u; printf("step %d/%d %.2fs\n", s, n, t); }
   static int32_t stop_at(void * u) { (void) u; return 0; }
   int main(int argc, char ** argv) {
       kggml_diffusion_context_params cp; kggml_diffusion_context_default_params(&cp);
       cp.model_path = argv[1]; cp.backend = argc > 2 ? atoi(argv[2]) : KGGML_BACKEND_CPU;
       kggml_diffusion_context * c = NULL;
       if (kggml_diffusion_context_create(&cp, &c)) { printf("%s\n", kggml_diffusion_last_error()); return 1; }
       kggml_diffusion_generate_params gp; kggml_diffusion_generate_default_params(&gp);
       gp.prompt = "a lighthouse on a cliff at sunset"; gp.width = gp.height = 512; gp.sample_steps = 20; gp.seed = 42;
       kggml_diffusion_image img;
       int s = kggml_diffusion_generate(c, &gp, on_progress, stop_at, NULL, &img);
       printf("status %d %ux%ux%u %s\n", s, img.width, img.height, img.channels, kggml_diffusion_last_error());
       FILE * f = fopen("/tmp/sd-smoke.ppm", "wb");
       fprintf(f, "P6 %u %u 255\n", img.width, img.height); fwrite(img.data, 1, img.width * img.height * 3, f); fclose(f);
       kggml_diffusion_image_free(&img); kggml_diffusion_context_free(c); return s;
   }
   ```
3. Run `/tmp/sd-smoke "$MODEL"` (CPU) and, on macOS, `/tmp/sd-smoke "$MODEL" 1` (Metal) / elsewhere `... 2` (Vulkan):
   steps 0..20 are printed in order, status is 0, and `/tmp/sd-smoke.ppm` shows the prompt.
4. Change `stop_at` to `return 1;` after a few steps (e.g. count calls in a static) and rerun: status is 5
   (`KGGML_STATUS_CANCELLED`) within one step's time of the abort, nothing is written past that step, and a second
   `generate` on the same context (after removing the abort) still succeeds.
