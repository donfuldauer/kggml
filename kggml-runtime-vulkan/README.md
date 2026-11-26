# `:kggml-runtime-vulkan`

Optional **Android GPU add-on** for the shared Kggml runtime: ggml's Vulkan backend, packaged as
`libggml-vulkan.so` for `arm64-v8a` and `x86_64`.

```kotlin
// commonMain (or androidMain) of an app that already uses the BOM
implementation("com.donfuldauer:kggml-runtime-vulkan")
```

Nothing to call. When `:kggml-runtime` loads, it looks for `libggml-vulkan.so` in the app and registers it with ggml's
backend registry (`kggml_runtime_load_backend`), so every engine sees the GPU: `ComputeBackend.GPU_VULKAN` targets it
and `AUTO` prefers it. `KggmlRuntimeVulkan.isRegistered` / `isAvailable` report the outcome.

| Situation | Result |
| --- | --- |
| Android 9+ device with a Vulkan 1.1+ GPU driver | Vulkan available; engines can run on the GPU |
| Android 8.x, or no usable driver | Module not loaded / no devices; a logcat warning (`kggml`), engines on the CPU |
| Emulator (software Vulkan) | Registered with 0 devices (ggml-vulkan ignores CPU Vulkan implementations); CPU |
| JVM desktop, iOS | No-op (desktop links Vulkan with `KGGML_ENABLE_VULKAN`, iOS uses Metal) |

Why an add-on and not part of `kggml-runtime`: the Vulkan backend carries tens of MB of precompiled shaders per ABI and
initialises the device's Vulkan driver, which CPU-only and whisper-only apps should not pay for (the
`whisper-only-minimal` audit forbids it). See the README section "GPU backends: built in or add-on".

## How it is built

`native/` configures the runtime's pinned ggml with `GGML_BACKEND_DL=ON`, `GGML_CPU=OFF`, `GGML_VULKAN=ON`, so it builds
only the Vulkan backend as a loadable module that links `libggml-base.so` (resolved to `:kggml-runtime`'s copy at run
time; the libggml-base built here is never shipped). Vulkan headers come from the pinned Khronos `Vulkan-Headers`
submodule (the NDK has no `vulkan.hpp`), the loader is the device's `libvulkan.so` (linked against the NDK's API 28
stub, the first that exports Vulkan 1.1), and shaders are compiled with the NDK's `glslc`.

```bash
./kggml-runtime-vulkan/native/scripts/build-android.sh arm64-v8a x86_64   # -> build/native-android-vulkan-jniLibs
./scripts/ci/audit-vulkan-addon.sh arm64-v8a x86_64                       # dependencies, exports, 16 KB alignment
./gradlew :kggml-runtime-vulkan:connectedDebugAndroidTest -Pkggml.bundleNative=true -Pkggml.requireNative=true
```
