# kggml-runtime native layer

This is the CMake build for the one shared `ggml` runtime used by every Kggml engine. It builds upstream
[`ggml`](https://github.com/ggml-org/ggml), pinned to `v0.9.9` /
`a04eea0761a85d18f3f504d6ab970c5c9dce705f` (see `NOTICE`), into **shared** libraries. It also builds the
`kggml_runtime` C ABI. Engines (`whisper.cpp`, `llama.cpp`, `stable-diffusion.cpp`) link these libraries
dynamically and never carry their own copy of ggml.

## Layout

| Path | Contents |
| --- | --- |
| `third_party/ggml/` | Upstream ggml git submodule (`git submodule update --init --recursive`) |
| `include/kggml_runtime.h` | Public C ABI: init/shutdown (refcounted), backends, P-core count, versions, last error |
| `jni/kggml_runtime_jni.cpp` | JNI bridge (`libkggml_runtime_jni`) for the Kotlin bindings on Android + JVM Desktop; links `kggml_runtime` dynamically (`KGGML_RUNTIME_BUILD_JNI`, ON when JNI headers are found / on Android, never on iOS) |
| `src/` | Implementation. `guard.hpp` holds the FFI exception firewall. `kggml_runtime_testing.h` holds the test-only fault hooks |
| `tests/` | GoogleTest/CTest suite (`-DKGGML_RUNTIME_BUILD_TESTS=ON`; GoogleTest v1.15.2 via FetchContent) |
| `cmake/kggml-runtime-config.cmake.in` | Installed package: `find_package(kggml-runtime)` → `kggml::runtime`, `kggml::ggml` |
| `scripts/` | `build-host.sh`, `build-android.sh`, `build-ios-xcframework.sh` |

## Outputs

| Platform | Artifacts |
| --- | --- |
| Linux / macOS / Windows | `libggml-base`, `libggml-cpu`, `libggml` (+ `libggml-metal`/`-blas` on macOS, `libggml-vulkan` when enabled), `libkggml_runtime`, `libkggml_runtime_jni` |
| Android (`arm64-v8a`, `x86_64`) | Same libraries with unversioned sonames (`libggml.so`, `libkggml_runtime.so`, `libkggml_runtime_jni.so`, …), 16 KB page aligned, `c++_shared` STL |
| iOS (device, simulator arm64) | `GgmlRuntime.xcframework`. Each slice is one dynamic framework that holds ggml, its backends and `kggml_runtime` (`KGGML_RUNTIME_MONOLITHIC=ON`) |

`GGML_BACKEND_DL` is always OFF, so backends are linked directly and nothing is loaded with `dlopen`.
`libkggml_runtime` has `NEEDED` entries on `libggml`, `libggml-base` and `libggml-cpu`, and defines no
`ggml_*` symbols. CI checks this with `readelf`/`nm`/`otool`.

## Presets

| Preset | Notes |
| --- | --- |
| `linux-x64`, `macos-arm64`, `windows-x64` | Host builds with tests into `<repo>/build/native`. Portable CPU flags: AVX2 on x86; the compiler default on Apple arm64 (Metal + Accelerate + BLAS) |
| `android-arm64-v8a` | `-march=armv8.2-a+dotprod+fp16`. Needs `ANDROID_NDK_HOME` |
| `android-x86_64` | SSE4.2 baseline (the x86_64 Android ABI guarantees nothing higher) |
| `ios-device`, `ios-simulator-arm64` | Xcode generator, monolithic `GgmlRuntime.framework`, Metal (embedded library) + Accelerate |

Options: `KGGML_ENABLE_VULKAN` (OFF; it is forced OFF when the Vulkan SDK or `glslc` is missing),
`KGGML_ENABLE_METAL`, `KGGML_ENABLE_BLAS`, `KGGML_ENABLE_COREML` (a compile-time capability flag only),
`KGGML_ARM_ARCH`, `KGGML_ARM_I8MM` (OFF: there is no runtime CPU dispatch without `GGML_BACKEND_DL`, so `+i8mm`
would raise SIGILL on cores such as Cortex-A76 and Apple M1), `KGGML_X86_LEVEL`, and `KGGML_RUNTIME_PORTABLE_CPU`
(OFF means `-march=native`).

## Build and test

```bash
git submodule update --init --recursive

# Host build + tests (from the repo root)
kggml-runtime/native/scripts/build-host.sh            # --vulkan, --no-tests, --skip-ctest, --clean, --install <prefix>
ctest --test-dir build/native --output-on-failure

# The same steps with presets
cmake -S kggml-runtime/native --preset linux-x64
cmake --build build/native -j"$(nproc)"

# Android (stages jniLibs under build/native-android-jniLibs/<abi>/)
ANDROID_NDK_HOME=/path/to/ndk kggml-runtime/native/scripts/build-android.sh

# iOS (macOS + Xcode) → build/native-ios/GgmlRuntime.xcframework
kggml-runtime/native/scripts/build-ios-xcframework.sh   # --coreml to advertise the CoreML capability

# Gradle helpers (opt-in, not part of `check`)
./gradlew :kggml-runtime:buildNativeHost :kggml-runtime:nativeCtest
```

## Kotlin bindings

`com.donfuldauer.kggml.runtime.KggmlRuntime` (in `kggml-runtime/src/`) exposes the C ABI to Kotlin:
`acquire()` (ref-counted `RuntimeHandle`), `backends()`, `availableBackends()`, `resolveBackend()`,
`recommendedThreadCount()`, `hardwareThreadCount()` and `info()`. Native status codes become typed
`InferenceException`s.

| Platform | Binding | Library loading |
| --- | --- | --- |
| Android | JNI (`jniMain`) | `System.loadLibrary("kggml_runtime_jni")` from the AAR's jniLibs |
| JVM Desktop | JNI (`jniMain`) | `-Dkggml.native.path=<dir>` (or `KGGML_NATIVE_PATH`), else extracted from the jar's `/kggml/native/<os>-<arch>/` (listed in `index.txt` in load order), else `java.library.path` |
| iOS | cinterop (`src/nativeInterop/cinterop/kggmlRuntime.def`) | Linked against the dynamic `GgmlRuntime.framework`; the app embeds `GgmlRuntime.xcframework` |

```bash
# JVM tests against build/native/lib, including C++ fault injection across JNI.
# Without -Pkggml.requireNative the native-backed cases are skipped when build/native is missing.
./gradlew :kggml-runtime:jvmTest -Pkggml.requireNative=true

# Package release (no test hooks) natives: host libs into the JVM jar, arm64-v8a + x86_64 into the AAR.
ANDROID_HOME=/path/to/sdk ./gradlew :kggml-runtime:assemble -Pkggml.bundleNative=true

# iOS (macOS): build the xcframework first; iOS test tasks are skipped until it exists.
kggml-runtime/native/scripts/build-ios-xcframework.sh && ./gradlew :kggml-runtime:iosSimulatorArm64Test
```

The JVM jar bundles only the host platform it was built on. A jar for several desktop platforms needs the
staged `kggml/native/<platform>/` directories from each host merged in; that belongs to the publishing work.

## Status codes

Codes `0..6` of `kggml_status` match `com.donfuldauer.kggml.core.NativeErrorCode`, so Kotlin can pass them
straight to `InferenceException.fromNativeError`. Runtime-only codes are `NOT_INITIALIZED = 100`,
`BUFFER_TOO_SMALL = 101` and `UNKNOWN = 102`. Kotlin turns these into `NativeFailureException` and keeps the
raw code. Every `extern "C"` entry point catches C++ exceptions and maps them as follows:

- `std::bad_alloc` → `OUT_OF_MEMORY`
- any other `std::exception` → `NATIVE_FAILURE`
- anything else → `UNKNOWN`

The message is stored thread-locally and is read with `kggml_runtime_last_error()`.
