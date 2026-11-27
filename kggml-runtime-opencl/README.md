# `:kggml-runtime-opencl`

Optional **Android OpenCL add-on** for the shared Kggml runtime: ggml's OpenCL backend, with the matrix-multiply
kernels Qualcomm tuned for Adreno GPUs, packaged as `libggml-opencl.so` for `arm64-v8a` and `x86_64`.

```kotlin
// commonMain (or androidMain) of an app that already uses the BOM
implementation("com.donfuldauer:kggml-runtime-opencl")
```

Nothing to call. When `:kggml-runtime` loads, it finds `libggml-opencl.so` in the app and registers it
(`kggml_runtime_load_backend`). The module links the device vendor's `libOpenCL.so`, which the add-on's manifest
declares (`<uses-native-library android:name="libOpenCL.so" android:required="false"/>`, needed from Android 12 on).
`ComputeBackend.GPU_OPENCL` targets it, and `AUTO` prefers it over Vulkan. `KggmlRuntimeOpenCL.isRegistered` /
`isAvailable` report the outcome.

| Situation | Result |
| --- | --- |
| Phone whose vendor ships OpenCL (Qualcomm Adreno) | OpenCL available; `AUTO` uses it |
| Phone without OpenCL (Pixel, many Mali / PowerVR devices), emulator | Module does not load; logcat warning (`kggml`); Vulkan (with `kggml-runtime-vulkan`) or the CPU |
| Android 8.x | Module not loaded (it needs API 28); CPU |
| JVM desktop, iOS | No-op |

The kernels ship as source (embedded, about 3 MB per ABI) and are compiled by the device's OpenCL driver on first use.
ggml's OpenCL backend is mostly exercised with LLMs; check diffusion on your target devices.

## How it is built

`native/` configures the runtime's pinned ggml with `GGML_BACKEND_DL=ON`, `GGML_CPU=OFF`, `GGML_OPENCL=ON` (embedded
kernels, Adreno kernels), so it builds only the OpenCL backend as a loadable module linking `libggml-base.so` (resolved
to `:kggml-runtime`'s copy at run time). OpenCL headers come from the pinned Khronos `OpenCL-Headers` submodule; the
pinned Khronos `OpenCL-ICD-Loader` is built only as the link-time `libOpenCL.so` and never packaged.

```bash
./kggml-runtime-opencl/native/scripts/build-android.sh arm64-v8a x86_64   # -> build/native-android-opencl-jniLibs
./scripts/ci/audit-gpu-addon.sh opencl arm64-v8a x86_64                   # dependencies, exports, 16 KB alignment
./gradlew :kggml-runtime-opencl:connectedAndroidDeviceTest -Pkggml.bundleNative=true -Pkggml.requireNative=true
```
