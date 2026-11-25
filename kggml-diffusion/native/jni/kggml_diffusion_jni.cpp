// kggml_diffusion_jni.cpp - JNI bridge between com.donfuldauer.kggml.diffusion.KggmlDiffusionJni (Android + JVM
// Desktop) and the kggml_diffusion C ABI.
//
// Rules (AGENTS.md §2.4):
//  * Every JNI entry point is extern "C" and runs its body behind jni_guard(), so no C++ exception ever crosses
//    into the JVM. Failures are reported as kggml_status codes, which Kotlin maps onto the typed
//    InferenceException hierarchy of :kggml-core-api.
//  * Text (prompts, model paths) crosses as raw UTF-8 byte[] rather than jstring, so supplementary characters are
//    never mangled by JNI's modified UTF-8. Nothing is pinned while stable-diffusion.cpp runs.
//  * The generated RGB8 image is copied into a new byte[] and its native buffer is released with
//    kggml_diffusion_image_free() on every path (RAII, including exceptions and failed Java allocations).
//  * Progress is forwarded to a Kotlin DiffusionProgressListener (onProgress(int, int, float), looked up by name) on
//    the generating thread. A Java exception thrown by the listener is cleared and turned into an abort request on
//    the context, so the generation stops at its next step boundary; no exception is ever left pending in native
//    code.
//  * Contexts cross as opaque jlong handles; Kotlin serializes every call on one engine except nativeRequestAbort /
//    nativeAbortRequested, which are lock-free.
//  * This library links kggml_diffusion (and through it the runtime's shared ggml) dynamically.

#include <jni.h>

#include "kggml_diffusion.h"

#if defined(KGGML_DIFFUSION_TEST_HOOKS)
#    include "kggml_diffusion_testing.h"
#endif

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>

namespace {

// Version selectors for nativeVersionString(); mirrored by KggmlDiffusionJni.VERSION_* in Kotlin.
constexpr jint kVersionModule = 0;
constexpr jint kVersionUpstream = 1;
constexpr jint kVersionUpstreamCommit = 2;
constexpr jint kVersionSystemInfo = 3;

// Array layouts; mirrored by KggmlDiffusionJni.*_COUNT in Kotlin.
constexpr jsize kModelPathCount = 8;     // model, diffusion_model, vae, taesd, clip_l, clip_g, t5xxl, llm
constexpr jsize kContextParamCount = 10; // backend, gpu_device, n_threads, use_mmap, flash_attn, offload_params_to_cpu,
                                         // keep_clip_on_cpu, keep_vae_on_cpu, vae_decode_only, weight_type
constexpr jsize kContextInfoCount = 5;   // backend, uses_gpu, n_threads, supports_image_generation,
                                         // default_sample_method
constexpr jsize kGenerateIntCount = 7;   // width, height, sample_steps, sample_method, scheduler, clip_skip, vae_tiling
constexpr jsize kGenerateFloatCount = 2; // cfg_scale, distilled_guidance
constexpr jsize kImageDimsCount = 3;     // width, height, channels

/// Runs `body` (returning jint status) behind the FFI exception firewall.
template <typename Fn>
jint jni_guard(Fn && body) noexcept {
    try {
        return body();
    } catch (const std::bad_alloc &) {
        return KGGML_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception &) {
        return KGGML_STATUS_NATIVE_FAILURE;
    } catch (...) {
        return KGGML_STATUS_UNKNOWN;
    }
}

template <typename Fn>
jobject jni_guard_object(Fn && body) noexcept {
    try {
        return body();
    } catch (...) {
        return nullptr;
    }
}

/// NewStringUTF expects *modified* UTF-8; diagnostics may carry arbitrary bytes, so anything outside printable
/// 7-bit ASCII (plus tab/newline) is replaced with '?'.
jstring to_ascii_jstring(JNIEnv * env, const char * text) {
    std::string ascii;
    if (text != nullptr) {
        for (const char * p = text; *p != '\0'; ++p) {
            const auto c = static_cast<unsigned char>(*p);
            ascii.push_back((c >= 0x20 && c < 0x7F) || c == '\t' || c == '\n' ? static_cast<char>(c) : '?');
        }
    }
    return env->NewStringUTF(ascii.c_str());
}

kggml_diffusion_context * from_handle(jlong handle) {
    return reinterpret_cast<kggml_diffusion_context *>(static_cast<intptr_t>(handle));
}

jlong to_handle(const void * p) {
    return static_cast<jlong>(reinterpret_cast<intptr_t>(p));
}

/// Copies a Java byte[] holding UTF-8 into a NUL-terminated std::string; `present` is false for null.
struct Utf8Bytes {
    std::string value;
    bool present = false;

    Utf8Bytes() = default;
    Utf8Bytes(JNIEnv * env, jbyteArray bytes) {
        if (bytes == nullptr) return;
        present = true;
        const jsize n = env->GetArrayLength(bytes);
        value.resize(static_cast<size_t>(n));
        if (n > 0) env->GetByteArrayRegion(bytes, 0, n, reinterpret_cast<jbyte *>(value.data()));
    }

    const char * c_str() const { return present ? value.c_str() : nullptr; }
};

bool has_length(JNIEnv * env, jarray array, jsize n) {
    return array != nullptr && env->GetArrayLength(array) >= n;
}

/// Owns a kggml_diffusion_image and releases it with kggml_diffusion_image_free() on scope exit.
struct ImageGuard {
    kggml_diffusion_image image{};
    ImageGuard() = default;
    ImageGuard(const ImageGuard &) = delete;
    ImageGuard & operator=(const ImageGuard &) = delete;
    ~ImageGuard() { kggml_diffusion_image_free(&image); }
};

/// user_data of the kggml_diffusion_progress_callback that forwards to a Kotlin DiffusionProgressListener.
struct JavaProgressListener {
    JNIEnv * env;
    jobject listener;
    jmethodID method; // void onProgress(int step, int totalSteps, float secondsPerStep)
    kggml_diffusion_context * context;
    bool failed = false;
};

void forward_progress(int32_t step, int32_t total_steps, float seconds_per_step, void * user_data) {
    auto * l = static_cast<JavaProgressListener *>(user_data);
    if (l == nullptr || l->failed) return;
    l->env->CallVoidMethod(l->listener, l->method, static_cast<jint>(step), static_cast<jint>(total_steps),
                           static_cast<jfloat>(seconds_per_step));
    if (l->env->ExceptionCheck()) {
        // Never leave a Java exception pending inside native code: clear it and stop at the next step boundary.
        l->env->ExceptionClear();
        l->failed = true;
        kggml_diffusion_request_abort(l->context);
    }
}

/// Resolves `listener.onProgress(IIF)V`. Returns false (with no exception pending) when it cannot.
bool resolve_listener(JNIEnv * env, jobject listener, jmethodID * out_method) {
    jclass cls = env->GetObjectClass(listener);
    if (cls == nullptr) {
        env->ExceptionClear();
        return false;
    }
    const jmethodID method = env->GetMethodID(cls, "onProgress", "(IIF)V");
    env->DeleteLocalRef(cls);
    if (method == nullptr) {
        env->ExceptionClear(); // NoSuchMethodError
        return false;
    }
    *out_method = method;
    return true;
}

/// Copies the RGB8 pixels of `image` into a new byte[] stored in `out_pixels[0]` and its dimensions into `out_dims`.
/// The native buffer itself is released by the caller's ImageGuard.
jint hand_over_image(JNIEnv * env, const kggml_diffusion_image & image, jintArray out_dims, jobjectArray out_pixels) {
    const size_t n = static_cast<size_t>(image.width) * static_cast<size_t>(image.height) * static_cast<size_t>(image.channels);
    if (image.data == nullptr || n == 0) return KGGML_STATUS_NATIVE_FAILURE;
    if (n > static_cast<size_t>(INT32_MAX)) return KGGML_STATUS_OUT_OF_MEMORY;
    jbyteArray array = env->NewByteArray(static_cast<jsize>(n));
    if (array == nullptr) {
        env->ExceptionClear(); // OutOfMemoryError
        return KGGML_STATUS_OUT_OF_MEMORY;
    }
    env->SetByteArrayRegion(array, 0, static_cast<jsize>(n), reinterpret_cast<const jbyte *>(image.data));
    env->SetObjectArrayElement(out_pixels, 0, array);
    env->DeleteLocalRef(array);
    const jint dims[kImageDimsCount] = {static_cast<jint>(image.width), static_cast<jint>(image.height),
                                        static_cast<jint>(image.channels)};
    env->SetIntArrayRegion(out_dims, 0, kImageDimsCount, dims);
    return KGGML_STATUS_OK;
}

bool image_out_arrays_valid(JNIEnv * env, jintArray out_dims, jobjectArray out_pixels) {
    return has_length(env, out_dims, kImageDimsCount) && has_length(env, out_pixels, 1);
}

/// Runs `generate(callback, user_data, &image)` with the listener wired (or not) and hands the image over.
template <typename Generate>
jint generate_with_listener(JNIEnv * env, kggml_diffusion_context * context, jobject listener, jintArray out_dims,
                            jobjectArray out_pixels, Generate && generate) {
    ImageGuard guard;
    kggml_status status;
    if (listener == nullptr) {
        status = generate(nullptr, nullptr, &guard.image);
    } else {
        JavaProgressListener forward{env, listener, nullptr, context};
        if (!resolve_listener(env, listener, &forward.method)) return KGGML_STATUS_NATIVE_FAILURE;
        status = generate(forward_progress, &forward, &guard.image);
    }
    if (status != KGGML_STATUS_OK) return static_cast<jint>(status);
    return hand_over_image(env, guard.image, out_dims, out_pixels);
}

} // namespace

#define KGGML_JNI(ret, name) \
    extern "C" JNIEXPORT ret JNICALL Java_com_donfuldauer_kggml_diffusion_KggmlDiffusionJni_##name

// ---------------------------------------------------------------------------------------------------------------
// Versions & errors
// ---------------------------------------------------------------------------------------------------------------

KGGML_JNI(jstring, nativeVersionString)(JNIEnv * env, jclass, jint which) {
    return static_cast<jstring>(jni_guard_object([&]() -> jobject {
        switch (which) {
            case kVersionModule: return to_ascii_jstring(env, kggml_diffusion_version());
            case kVersionUpstream: return to_ascii_jstring(env, kggml_diffusion_upstream_version());
            case kVersionUpstreamCommit: return to_ascii_jstring(env, kggml_diffusion_upstream_commit());
            case kVersionSystemInfo: return to_ascii_jstring(env, kggml_diffusion_system_info());
            default: return nullptr;
        }
    }));
}

KGGML_JNI(jstring, nativeStatusName)(JNIEnv * env, jclass, jint status) {
    return static_cast<jstring>(jni_guard_object(
        [&]() -> jobject { return to_ascii_jstring(env, kggml_status_name(static_cast<kggml_status>(status))); }));
}

KGGML_JNI(jstring, nativeLastError)(JNIEnv * env, jclass) {
    return static_cast<jstring>(
        jni_guard_object([&]() -> jobject { return to_ascii_jstring(env, kggml_diffusion_last_error()); }));
}

// ---------------------------------------------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------------------------------------------

/// `paths`: 8 UTF-8 byte[] (null = unused): model, diffusion_model, vae, taesd, clip_l, clip_g, t5xxl, llm.
/// `ints`: backend, gpu_device, n_threads, use_mmap, flash_attn, offload_params_to_cpu, keep_clip_on_cpu,
/// keep_vae_on_cpu, vae_decode_only, weight_type.
KGGML_JNI(jint, nativeContextCreate)(JNIEnv * env, jclass, jobjectArray paths, jintArray ints, jlongArray out_handle) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, paths, kModelPathCount) || !has_length(env, ints, kContextParamCount) ||
            !has_length(env, out_handle, 1)) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        Utf8Bytes path[kModelPathCount];
        for (jsize i = 0; i < kModelPathCount; ++i) {
            auto element = static_cast<jbyteArray>(env->GetObjectArrayElement(paths, i));
            path[i] = Utf8Bytes(env, element);
            if (element != nullptr) env->DeleteLocalRef(element);
        }
        jint v[kContextParamCount];
        env->GetIntArrayRegion(ints, 0, kContextParamCount, v);

        kggml_diffusion_context_params p{};
        const kggml_status defaults = kggml_diffusion_context_default_params(&p);
        if (defaults != KGGML_STATUS_OK) return static_cast<jint>(defaults);
        p.model_path = path[0].c_str();
        p.diffusion_model_path = path[1].c_str();
        p.vae_path = path[2].c_str();
        p.taesd_path = path[3].c_str();
        p.clip_l_path = path[4].c_str();
        p.clip_g_path = path[5].c_str();
        p.t5xxl_path = path[6].c_str();
        p.llm_path = path[7].c_str();
        p.backend = v[0];
        p.gpu_device = v[1];
        p.n_threads = v[2];
        p.use_mmap = v[3];
        p.flash_attn = v[4];
        p.offload_params_to_cpu = v[5];
        p.keep_clip_on_cpu = v[6];
        p.keep_vae_on_cpu = v[7];
        p.vae_decode_only = v[8];
        p.weight_type = v[9];

        kggml_diffusion_context * context = nullptr;
        const kggml_status status = kggml_diffusion_context_create(&p, &context);
        const jlong handle = status == KGGML_STATUS_OK ? to_handle(context) : 0;
        env->SetLongArrayRegion(out_handle, 0, 1, &handle);
        return static_cast<jint>(status);
    });
}

KGGML_JNI(void, nativeContextFree)(JNIEnv *, jclass, jlong handle) {
    kggml_diffusion_context_free(from_handle(handle));
}

/// `out`: backend, uses_gpu, n_threads, supports_image_generation, default_sample_method.
KGGML_JNI(jint, nativeContextInfo)(JNIEnv * env, jclass, jlong handle, jintArray out) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out, kContextInfoCount)) return KGGML_STATUS_INVALID_ARGUMENT;
        kggml_diffusion_context_info info{};
        const kggml_status status = kggml_diffusion_context_get_info(from_handle(handle), &info);
        if (status != KGGML_STATUS_OK) return static_cast<jint>(status);
        const jint values[kContextInfoCount] = {info.backend, info.uses_gpu, info.n_threads,
                                                info.supports_image_generation, info.default_sample_method};
        env->SetIntArrayRegion(out, 0, kContextInfoCount, values);
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Abort (lock-free, any thread)
// ---------------------------------------------------------------------------------------------------------------

KGGML_JNI(void, nativeRequestAbort)(JNIEnv *, jclass, jlong handle) {
    kggml_diffusion_request_abort(from_handle(handle));
}

KGGML_JNI(void, nativeClearAbort)(JNIEnv *, jclass, jlong handle) {
    kggml_diffusion_clear_abort(from_handle(handle));
}

KGGML_JNI(jint, nativeAbortRequested)(JNIEnv *, jclass, jlong handle) {
    return kggml_diffusion_abort_requested(from_handle(handle));
}

// ---------------------------------------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------------------------------------

/// txt2img. `ints`: width, height, sample_steps, sample_method, scheduler, clip_skip, vae_tiling. `floats`: cfg_scale,
/// distilled_guidance. `listener` (a DiffusionProgressListener, may be null) receives every progress point on this
/// thread. On OK, `out_dims` = width, height, channels and `out_pixels[0]` = a new byte[] with the RGB8 pixels.
KGGML_JNI(jint, nativeGenerate)(JNIEnv * env, jclass, jlong handle, jbyteArray prompt_utf8, jbyteArray negative_utf8,
                                jintArray ints, jfloatArray floats, jlong seed, jobject listener, jintArray out_dims,
                                jobjectArray out_pixels) {
    return jni_guard([&]() -> jint {
        if (prompt_utf8 == nullptr || !has_length(env, ints, kGenerateIntCount) ||
            !has_length(env, floats, kGenerateFloatCount) || !image_out_arrays_valid(env, out_dims, out_pixels)) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        const Utf8Bytes prompt(env, prompt_utf8);
        const Utf8Bytes negative(env, negative_utf8);
        jint v[kGenerateIntCount];
        env->GetIntArrayRegion(ints, 0, kGenerateIntCount, v);
        jfloat f[kGenerateFloatCount];
        env->GetFloatArrayRegion(floats, 0, kGenerateFloatCount, f);

        kggml_diffusion_generate_params p{};
        const kggml_status defaults = kggml_diffusion_generate_default_params(&p);
        if (defaults != KGGML_STATUS_OK) return static_cast<jint>(defaults);
        p.prompt = prompt.c_str();
        p.negative_prompt = negative.c_str();
        p.width = v[0];
        p.height = v[1];
        p.sample_steps = v[2];
        p.sample_method = v[3];
        p.scheduler = v[4];
        p.clip_skip = v[5];
        p.vae_tiling = v[6];
        p.cfg_scale = f[0];
        p.distilled_guidance = f[1];
        p.seed = static_cast<int64_t>(seed);

        kggml_diffusion_context * context = from_handle(handle);
        return generate_with_listener(env, context, listener, out_dims, out_pixels,
                                      [&](kggml_diffusion_progress_callback cb, void * user_data,
                                          kggml_diffusion_image * out) {
                                          return kggml_diffusion_generate(context, &p, cb, nullptr, user_data, out);
                                      });
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Test hooks (KGGML_DIFFUSION_TEST_HOOKS builds only; release libraries report "unavailable")
// ---------------------------------------------------------------------------------------------------------------

KGGML_JNI(jboolean, nativeTestHooksAvailable)(JNIEnv *, jclass) {
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

/// Live kggml_diffusion contexts in this process, or -1 when unavailable.
KGGML_JNI(jlong, nativeTestLiveObjects)(JNIEnv *, jclass) {
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
    return static_cast<jlong>(kggml_diffusion_test_live_objects());
#else
    return -1;
#endif
}

/// Creates a model-less stub context (abort flag + simulated generation only) into `out_handle[0]`.
KGGML_JNI(jint, nativeTestContextCreateStub)(JNIEnv * env, jclass, jlongArray out_handle) {
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
    return jni_guard([&]() -> jint {
        if (!has_length(env, out_handle, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        kggml_diffusion_context * context = nullptr;
        const kggml_status status = kggml_diffusion_test_context_create_stub(&context);
        const jlong handle = status == KGGML_STATUS_OK ? to_handle(context) : 0;
        env->SetLongArrayRegion(out_handle, 0, 1, &handle);
        return static_cast<jint>(status);
    });
#else
    (void) env;
    (void) out_handle;
    return KGGML_STATUS_BACKEND_UNAVAILABLE;
#endif
}

/// kggml_diffusion_test_simulate_generate(): the real generation machinery (progress dispatch, latched abort,
/// image hand-over) with the sampler replaced by `steps` simulated steps and a deterministic width x height gradient.
KGGML_JNI(jint, nativeTestSimulateGenerate)(JNIEnv * env, jclass, jlong handle, jint steps, jint width, jint height,
                                            jobject listener, jintArray out_dims, jobjectArray out_pixels) {
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
    return jni_guard([&]() -> jint {
        if (!image_out_arrays_valid(env, out_dims, out_pixels)) return KGGML_STATUS_INVALID_ARGUMENT;
        kggml_diffusion_context * context = from_handle(handle);
        return generate_with_listener(
            env, context, listener, out_dims, out_pixels,
            [&](kggml_diffusion_progress_callback cb, void * user_data, kggml_diffusion_image * out) {
                return kggml_diffusion_test_simulate_generate(context, steps, width, height, cb, nullptr, user_data, -1,
                                                              KGGML_DIFFUSION_TEST_FAULT_NONE, out);
            });
    });
#else
    (void) env;
    (void) handle;
    (void) steps;
    (void) width;
    (void) height;
    (void) listener;
    (void) out_dims;
    (void) out_pixels;
    return KGGML_STATUS_BACKEND_UNAVAILABLE;
#endif
}
