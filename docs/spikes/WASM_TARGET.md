# Spike: a `wasmJs` (browser) target

Issue #94. Question: can Kggml's Kotlin API (`WhisperEngine`, `LlamaEngine`, …) run in a browser from a Kotlin/Wasm
(Compose Multiplatform web) app, and what would it cost? This is a paper spike against the pinned sources in this
repository (ggml v0.9.9, Kotlin 2.2.0); nothing here has been built with Emscripten yet (see "Next step").

## Verdict

**Go, scoped to whisper on the CPU first.** Every native piece needed exists upstream, and the Kotlin API is already
`suspend`-based, which maps onto JavaScript promises. The cost is in the plumbing: a second native build flavour, a
JavaScript bridge running in a Web Worker, browser-specific model storage, and browser tests. Estimate: about 4–6 weeks
for whisper on the CPU, with llama next. WebGPU and diffusion should wait until the CPU path ships.

## What exists already

| Piece | Status in the pinned sources |
| --- | --- |
| ggml under Emscripten | Supported: `if (EMSCRIPTEN)` in ggml's CMake builds static libraries and can embed the wasm in one `.js` (`GGML_WASM_SINGLE_FILE`). The CPU backend compiles with `-msimd128` and has WASM SIMD kernels (`src/ggml-cpu/arch/wasm/quants.c`) |
| whisper.cpp in the browser | Upstream examples: `examples/whisper.wasm`, `stream.wasm`, `command.wasm`, `bench.wasm` (pthreads + SIMD) |
| llama.cpp in the browser | Builds with Emscripten (community projects such as wllama ship it); no upstream example in our pin |
| GPU | `ggml-webgpu` backend (WGSL shaders, `GGML_WEBGPU`), with an Emscripten path through Dawn's `emdawnwebgpu` and JSPI (`GGML_WEBGPU_JSPI`, default ON) |
| Kotlin side | Engines expose `suspend` functions (`WhisperEngine.transcribe`, …) and serialize native work on a `SerialExecutor`, so an async bridge fits without API changes |

## What does not carry over

1. **No C interop in Kotlin/Wasm.** Kotlin/Wasm cannot link C (no cinterop). The native code has to be a separate
   Emscripten module, called through `external` JavaScript declarations (`@JsModule`). Data crosses two heaps: Kotlin's
   GC heap and Emscripten's linear memory. PCM audio and token buffers must be copied through typed arrays (cheap next
   to inference), and every call is asynchronous.
2. **One wasm module, not shared libraries.** Our runtime design (one shared ggml loaded by every engine,
   `GGML_BACKEND_DL` add-ons) does not apply: Emscripten links statically. Each app bundle needs one module containing
   ggml plus the engines it uses (whisper-only, whisper+llama, …). On iOS, ggml is one dynamic framework
   (`GgmlRuntime.framework`) that each engine framework links; in the browser the engines have to be built into the
   same module as ggml. Two separately built engine modules would each carry their own ggml and memory.
3. **Threads need cross-origin isolation.** Multithreaded ggml (pthreads) needs `SharedArrayBuffer`, which needs the
   page served with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Without
   them ggml runs single-threaded (several times slower). The module must run in a Web Worker anyway, so long inference
   does not block the UI thread.
4. **Memory.** wasm32 caps linear memory at 4 GB (browsers often allow less on mobile). Memory64 (wasm64) has shipped
   in Chromium and Firefox but not, as of mid-2026, in Safari, so wasm32 is the baseline. Practical models:
   whisper tiny/base/small (75–466 MB), quantized LLMs up to about 1–3B parameters (Q4: 0.7–2 GB with KV cache).
   SD-Turbo (2 GB+ of weights plus compute buffers) is borderline and out of scope.
5. **Files.** Engines take a `modelPath`. In the browser that path is in Emscripten's file system: MEMFS (the model is
   copied into wasm memory, doubling peak use while loading) or WasmFS backed by OPFS (the Origin Private File System,
   persistent and streamable). `kggml-model-manager`'s `ModelFileSystem` is **synchronous**. OPFS offers synchronous
   access handles only inside Workers, so the model manager needs either a Worker-hosted implementation or an async
   variant of `ModelFileSystem`. That is the one public-API question this target raises.
6. **WebGPU needs JSPI.** ggml-webgpu's browser path uses JavaScript Promise Integration, which has shipped in Chromium
   but not everywhere. Treat WebGPU as an optional add-on later, the way Vulkan and OpenCL are on Android.

## Proposed shape

- A new native flavour, `kggml-web/native`: CMake + Emscripten builds ggml (SIMD128, pthreads) plus whisper (later
  llama) and the `kggml_*` C wrappers into one ES module with a small C API exported via `EXPORTED_FUNCTIONS`. It
  reuses the same pinned submodules and `GGML_MAX_NAME` (128) as every other build.
- A small JavaScript worker (`kggml-worker.js`) owns the module. It takes messages (`init`, `loadModel`,
  `transcribe`, `cancel`, `close`) and posts results and progress events back. It is shipped as a resource of the
  `wasmJs` artifacts.
- Kotlin `wasmJs` actuals: `kggml-core-api` (dispatchers: everything is a single-threaded event loop),
  `kggml-runtime` (backends: CPU, plus WebGPU later; thread count from `navigator.hardwareConcurrency`) and
  `kggml-whisper` (a `WhisperBridge` sending worker messages and awaiting promises). The public API stays the same.
- `kggml-model-manager` for web: `fetch` with progress and OPFS caching, keeping the SHA-256 verification.
- Tests: `wasmJsBrowserTest` in headless Chrome, with a dev server that sends the COOP/COEP headers. Run the
  existing `commonTest` contracts plus a real whisper tiny transcription of the JFK sample.

## Estimate (whisper, CPU)

| Work | Weeks |
| --- | --- |
| Emscripten build + CI (emsdk pinned, size audit like `whisper-only-minimal`) | 1 |
| Worker protocol + JS glue + Kotlin bridge (whisper) | 1.5 |
| core-api / runtime / whisper `wasmJs` actuals, threading and cancellation semantics | 1 |
| Model manager on OPFS (async question above) | 1 |
| Browser test harness (headless Chrome, COOP/COEP), sample web page | 1 |

Then llama: about 2 more weeks (a larger module and the KV-cache memory ceiling). Diffusion and WebGPU are separate
decisions.

## Risks

- **Bundle size.** ggml + whisper with SIMD is roughly 1–2 MB of wasm; llama.cpp adds several MB. A whisper-only
  bundle must stay small (extend `audit-binary-symbols.sh` to the wasm module).
- **Duplicated memory while loading** with MEMFS. Use WasmFS + OPFS, or load models in chunks.
- **Mobile Safari:** tighter memory limits and no Memory64 or JSPI. Whisper tiny/base should still work.
- **Kotlin/Wasm maturity** (Beta in Kotlin 2.2): JS interop details and the test tooling may change between Kotlin
  releases.

## Next step

A one-day hands-on prototype: build `whisper.wasm` from our pinned submodule with emsdk (a pinned emsdk download, ~1 GB,
which the maintainer has to approve first). Then call it from a minimal `wasmJs` test through a worker, and measure
bundle size, load time and the real-time factor on whisper tiny/base in Chrome and Safari. If those numbers hold up,
start with the Emscripten build + CI row above.
