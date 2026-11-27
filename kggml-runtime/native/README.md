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
| `include/kggml_runtime.h` | Public C ABI: init/shutdown (refcounted), backends, backend modules (`kggml_runtime_load_backend`), P-core count, versions, last error |
| `jni/kggml_runtime_jni.cpp` | JNI bridge (`libkggml_runtime_jni`) for the Kotlin bindings on Android + JVM Desktop; links `kggml_runtime` dynamically (`KGGML_RUNTIME_BUILD_JNI`, ON when JNI headers are found / on Android, never on iOS) |
| `src/` | Implementation. `guard.hpp` holds the FFI exception firewall. `kggml_runtime_testing.h` holds the test-only fault hooks |
| `tests/` | GoogleTest/CTest suite (`-DKGGML_RUNTIME_BUILD_TESTS=ON`; GoogleTest v1.15.2 via FetchContent) |
| `cmake/kggml-version.cmake` | `KGGML_VERSION`, read from `VERSION_NAME` in `gradle.properties` (the one place the version is set); the runtime and every engine use it as their CMake project version. `-DKGGML_VERSION=x.y.z` overrides it outside the repository. |
| `cmake/kggml-runtime-config.cmake.in` | Installed package: `find_package(kggml-runtime)` → `kggml::runtime`, `kggml::ggml` |
| `scripts/` | `build-host.sh`, `build-android.sh`, `build-ios-xcframework.sh`, `build-windows-arm64.cmd` |

## Outputs

| Platform | Artifacts |
| --- | --- |
| Linux / macOS / Windows | `libggml-base`, `libggml-cpu`, `libggml` (+ `libggml-metal`/`-blas` on macOS, `libggml-vulkan` when enabled), `libkggml_runtime`, `libkggml_runtime_jni` |
| Android (`arm64-v8a`, `x86_64`) | Same libraries with unversioned sonames (`libggml.so`, `libkggml_runtime.so`, `libkggml_runtime_jni.so`, …), 16 KB page aligned, `c++_shared` STL |
| iOS (device, simulator arm64), macOS arm64 | `GgmlRuntime.xcframework`. Each slice is one dynamic framework that holds ggml, its backends and `kggml_runtime` (`KGGML_RUNTIME_MONOLITHIC=ON`); the `macos-arm64` slice serves the Kotlin/Native `macosArm64` target |

`GGML_BACKEND_DL` is always OFF here, so the runtime's own backends are linked directly. The only modules loaded
with `dlopen` are the optional GPU add-ons (`:kggml-runtime-vulkan`, `:kggml-runtime-opencl`), which the Kotlin
loaders register through `kggml_runtime_load_backend`.
`libkggml_runtime` has `NEEDED` entries on `libggml`, `libggml-base` and `libggml-cpu`, and defines no
`ggml_*` symbols. CI checks this with `readelf`/`nm`/`otool`.

## ggml ABI: `GGML_MAX_NAME`

ggml is built with **`GGML_MAX_NAME=128`** (upstream default: 64). The value sizes `char name[GGML_MAX_NAME]`
inside `struct ggml_tensor`, so it changes the struct's layout: ggml itself and **every** translation unit that
includes `ggml.h` (whisper.cpp, llama.cpp, stable-diffusion.cpp, the Kggml wrappers, JNI bridges and tests) must
use the same value, or they read each other's tensors at the wrong offsets. 128 is required by
`stable-diffusion.cpp` (`static_assert(GGML_MAX_NAME >= 128)` in `src/ggml_extend.hpp`; its tensor names exceed 64
characters) and is what its upstream build uses; whisper.cpp and llama.cpp work with any value >= 64.

How it is propagated:

- `CMakeLists.txt` adds `GGML_MAX_NAME=128` as a **PUBLIC** compile definition to every ggml target
  (`ggml-base`, `ggml`, `ggml-cpu`, backends) and to `kggml_runtime` (which is `kggml::ggml` in the monolithic iOS
  layout). It is therefore part of the exported targets' `INTERFACE_COMPILE_DEFINITIONS`: everything that links
  `kggml::runtime` / `kggml::ggml` (the whisper, llama and diffusion native builds, their JNI bridges and tests)
  compiles with it automatically.
- `kggml-runtime-config.cmake` exposes `KGGML_GGML_MAX_NAME` and fails if the exported `kggml::runtime` target does
  not carry the definition; `:kggml-diffusion` refuses to configure against a runtime built with less than 128.
- The Kotlin/Native cinterop `.def` files only include the `kggml_*.h` C ABIs, never `ggml.h`, so they are not
  affected. Code that compiles against the installed / framework ggml headers **without** these CMake targets (for
  example an Xcode target including `GgmlRuntime.framework/Headers/ggml.h` directly) must pass
  `-DGGML_MAX_NAME=128` itself.
- `tests/test_abi.cpp` asserts the compile-time value, round-trips a 100-character tensor name through
  `ggml_set_name()` / `ggml_get_name()` of the shared library and checks that the library's `sizeof(ggml_tensor)`
  (`ggml_tensor_overhead()`) matches the consumer's.

## Presets

| Preset | Notes |
| --- | --- |
| `linux-x64`, `macos-arm64`, `windows-x64` | Host builds with tests into `<repo>/build/native`. Portable CPU flags: AVX2 on x86; the compiler default on Apple arm64 (Metal + Accelerate + BLAS) |
| `linux-arm64` | Host build with tests on an aarch64 Linux machine, `-march=armv8.2-a+dotprod+fp16` |
| `linux-arm64-cross` | The same, cross-compiled on x86_64 Linux into `<repo>/build/native-linux-aarch64` (`g++-aarch64-linux-gnu`; CTest runs under `qemu-aarch64` from `qemu-user`). `build-host.sh --target linux-aarch64` picks it |
| `windows-arm64` | Windows on ARM, built with LLVM clang for the MSVC ABI (ggml rejects `cl.exe` for ARM), no tests. Run [`scripts/build-windows-arm64.cmd`](scripts/build-windows-arm64.cmd), which sets up the MSVC arm64 environment |
| `android-arm64-v8a` | `-march=armv8.2-a+dotprod+fp16`. Needs `ANDROID_NDK_HOME` |
| `android-x86_64` | SSE4.2 baseline (the x86_64 Android ABI guarantees nothing higher) |
| `ios-device`, `ios-simulator-arm64`, `macos-framework` | Xcode generator, monolithic `GgmlRuntime.framework`, Metal (embedded library) + Accelerate. `macos-framework` is the macOS arm64 slice |

Options: `KGGML_ENABLE_VULKAN` (OFF in every preset, and forced OFF when the Vulkan SDK or `glslc` is missing;
`build-host.sh --vulkan` turns it on for a host build. Published artifacts get Vulkan from the `kggml-runtime-vulkan`
add-on module instead),
`KGGML_ENABLE_METAL`, `KGGML_ENABLE_BLAS`, `KGGML_ENABLE_COREML` (a compile-time capability flag only),
`KGGML_ARM_ARCH`, `KGGML_ARM_I8MM` (OFF: there is no runtime CPU dispatch without `GGML_BACKEND_DL`, so `+i8mm`
would raise SIGILL on cores such as Cortex-A76 and Apple M1), `KGGML_X86_LEVEL`, and `KGGML_RUNTIME_PORTABLE_CPU`
(OFF means `-march=native`).

### Windows: MSVC C++ runtime compatibility

Every MSVC build here (and in the whisper, llama and diffusion engines) defines
`_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`. MSVC 19.40 (Visual Studio 2022 17.10) and later otherwise emit a `constexpr`
`std::mutex` constructor that needs `msvcp140.dll` 14.40 or newer. A host process that loads an older copy first,
such as a JVM that bundles its own `msvcp140.dll` (OpenJDK 22.0.1 ships 14.36), would then crash with an access
violation in `msvcp140.dll` on the first mutex use, during `KggmlRuntime.acquire()` (#65).

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

# iOS + macOS arm64 (macOS + Xcode) → build/native-ios/GgmlRuntime.xcframework
kggml-runtime/native/scripts/build-ios-xcframework.sh   # --coreml to advertise the CoreML capability

# Gradle helpers (opt-in, not part of `check`)
./gradlew :kggml-runtime:buildNativeHost :kggml-runtime:nativeCtest
```

## Kotlin bindings

`com.donfuldauer.kggml.runtime.KggmlRuntime` (in `kggml-runtime/src/`) exposes the C ABI to Kotlin:
`acquire()` (ref-counted `RuntimeHandle`), `isInitialized()`, `backends()`, `availableBackends()`, `resolveBackend()`,
`recommendedThreadCount()`, `hardwareThreadCount()` and `info()`. Native status codes become typed
`InferenceException`s.

| Platform | Binding | Library loading |
| --- | --- | --- |
| Android | JNI (`jniMain`) | `System.loadLibrary("kggml_runtime_jni")` from the AAR's jniLibs |
| JVM Desktop | JNI (`jniMain`) | `-Dkggml.native.path=<dir>` (or `KGGML_NATIVE_PATH`), else extracted from the jar's `/kggml/native/<os>-<arch>/` (listed in `index.txt` in load order), else `java.library.path` |
| iOS, macOS (`appleMain`) | cinterop (`src/nativeInterop/cinterop/kggmlRuntime.def`) | Linked against the dynamic `GgmlRuntime.framework`; the app embeds `GgmlRuntime.xcframework` |

```bash
# JVM tests against build/native/lib, including C++ fault injection across JNI.
# Without -Pkggml.requireNative the native-backed cases are skipped when build/native is missing.
./gradlew :kggml-runtime:jvmTest -Pkggml.requireNative=true

# Package release (no test hooks) natives: host libs into the JVM jar, arm64-v8a + x86_64 into the AAR.
ANDROID_HOME=/path/to/sdk ./gradlew :kggml-runtime:assemble -Pkggml.bundleNative=true

# iOS (macOS): build the xcframework first; iOS test tasks are skipped until it exists.
kggml-runtime/native/scripts/build-ios-xcframework.sh && ./gradlew :kggml-runtime:iosSimulatorArm64Test
```

By default the JVM jar bundles only the host platform it was built on. `-Pkggml.extraJvmNativeResources=<dir>[,...]`
(with `bundleNative`) adds the `kggml/native/<platform>/` directories staged on other hosts, and
`-Pkggml.nativePlatform=<os>-<arch>` stages a cross-built runtime (`linux-aarch64`, `windows-aarch64`) instead of the
host's. The release workflow uses both to ship `linux-x86_64`, `linux-aarch64`, `windows-x86_64`, `windows-aarch64`
and `macos-aarch64` in one jar.

## Status codes

Codes `0..6` of `kggml_status` match `com.donfuldauer.kggml.core.NativeErrorCode`, so Kotlin can pass them
straight to `InferenceException.fromNativeError`. Runtime-only codes are `NOT_INITIALIZED = 100`,
`BUFFER_TOO_SMALL = 101` and `UNKNOWN = 102`. Kotlin turns these into `NativeFailureException` and keeps the
raw code. Every `extern "C"` entry point catches C++ exceptions and maps them as follows:

- `std::bad_alloc` → `OUT_OF_MEMORY`
- any other `std::exception` → `NATIVE_FAILURE`
- anything else → `UNKNOWN`

The message is stored thread-locally and is read with `kggml_runtime_last_error()`.
