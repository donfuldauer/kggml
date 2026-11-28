# `:kggml-runtime-vulkan`

Optional **GPU add-on** for the shared Kggml runtime: ggml's Vulkan backend, packaged as `libggml-vulkan.so` for
Android `arm64-v8a` and `x86_64`, and in the `-jvm` jar for desktop Linux `x86_64` (issue #92; Windows is not
packaged yet).

```kotlin
// commonMain (or androidMain) of an app that already uses the BOM
implementation("com.donfuldauer:kggml-runtime-vulkan")
```

Nothing to call. When `:kggml-runtime` loads, it looks for the module (Android: `libggml-vulkan.so` in the app; JVM:
`/kggml/native-addons/<platform>/ggml-vulkan/` in this jar, or the `kggml.native.addons.path` directory) and registers it
with ggml's backend registry (`kggml_runtime_load_backend`), so every engine sees the GPU: `ComputeBackend.GPU_VULKAN` targets it
and `AUTO` prefers it over the CPU (an OpenCL device from `kggml-runtime-opencl` comes first).
`KggmlRuntimeVulkan.isRegistered` / `isAvailable` report the outcome.

| Situation | Result |
| --- | --- |
| Android 9+ device with a Vulkan 1.1+ GPU driver | Vulkan available; engines can run on the GPU |
| Android 8.x, or no usable driver | Module not loaded / no devices; a logcat warning (`kggml`), engines on the CPU |
| Emulator (software Vulkan) | Registered with 0 devices (ggml-vulkan ignores CPU Vulkan implementations); CPU |
| Imagination PowerVR GPU (Pixel 10 and later: Tensor G5+, some MediaTek phones) | Registered with 0 usable devices: the PowerVR driver runs ggml's shaders incorrectly and can hang, so `AUTO` and `GPU_VULKAN` use the CPU. Set the environment variable `KGGML_VULKAN_ALLOW_POWERVR=1` before the runtime loads to test a newer driver |
| JVM desktop Linux x86_64 with a Vulkan driver (Mesa, NVIDIA, AMD) | Vulkan available; engines can run on the GPU |
| JVM desktop Linux without a Vulkan loader (`libvulkan.so.1`) or driver | Module not loaded / no devices; a warning (`System.Logger` `com.donfuldauer.kggml`), CPU |
| JVM desktop with a runtime built with `KGGML_ENABLE_VULKAN=ON` | The add-on is skipped (Vulkan is already linked in) |
| Windows, macOS, iOS | No-op (no Windows module yet; Apple platforms use Metal) |

Why an add-on and not part of `kggml-runtime`: the Vulkan backend carries tens of MB of precompiled shaders per ABI and
initialises the device's Vulkan driver, which CPU-only and whisper-only apps should not pay for (the
`whisper-only-minimal` audit forbids it). See the README section "GPU backends: built in or add-on".

## How it is built

`native/` configures the runtime's pinned ggml with `GGML_BACKEND_DL=ON`, `GGML_CPU=OFF`, `GGML_VULKAN=ON`, so it builds
only the Vulkan backend as a loadable module that links `libggml-base.so` (resolved to `:kggml-runtime`'s copy at run
time; the libggml-base built here is never shipped). Vulkan headers come from the pinned Khronos `Vulkan-Headers`
submodule (the NDK has no `vulkan.hpp`), the loader is the device's `libvulkan.so` (linked against the NDK's API 28
stub, the first that exports Vulkan 1.1), and shaders are compiled with the NDK's `glslc`.

Desktop (Linux) uses the same pinned headers with the machine's Vulkan loader (`libvulkan-dev`) and `glslc` (the
`glslc` package, or the NDK's when `ANDROID_NDK_HOME` is set); the module links the runtime's `libggml-base.so.0` and the
user's `libvulkan.so.1`. Ggml ignores CPU-type Vulkan devices unless `GGML_VK_VISIBLE_DEVICES` selects one, which is how
CI tests the module on Mesa's lavapipe.

```bash
./kggml-runtime-vulkan/native/scripts/build-host.sh                       # Linux -> build/native-vulkan-jvm
./scripts/ci/audit-gpu-addon.sh vulkan --desktop linux-x86_64
./gradlew :kggml-runtime:jvmTest -Pkggml.requireNative=true \
  -Pkggml.test.vulkanAddonDir=build/native-vulkan-jvm/kggml/native-addons/linux-x86_64/ggml-vulkan
./kggml-runtime-vulkan/native/scripts/build-android.sh arm64-v8a x86_64   # -> build/native-android-vulkan-jniLibs
./scripts/ci/audit-gpu-addon.sh vulkan arm64-v8a x86_64                       # dependencies, exports, 16 KB alignment
./gradlew :kggml-runtime-vulkan:connectedAndroidDeviceTest -Pkggml.bundleNative=true -Pkggml.requireNative=true
```
