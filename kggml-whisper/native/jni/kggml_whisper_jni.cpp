// kggml_whisper_jni.cpp - JNI bridge between com.donfuldauer.kggml.whisper.KggmlWhisperJni (Android + JVM
// Desktop) and the kggml_whisper C ABI.
//
// Rules (AGENTS.md §2.4):
//  * Every JNI entry point is extern "C" and runs its body behind jni_guard(), so no C++ exception ever
//    crosses into the JVM. Failures are reported as kggml_status codes, which Kotlin maps onto the typed
//    InferenceException hierarchy of :kggml-core-api.
//  * PCM crosses the boundary zero-copy: float[] / short[] are pinned with GetPrimitiveArrayCritical and read
//    in place by whisper.cpp. CriticalArray releases them on every path (RAII, including exceptions). No
//    other JNI function is called on the same thread while an array is held. Holding the arrays for a whole
//    inference is fine on ART (moving collectors pin the object) and on HotSpot G1 since JDK 22 (JEP 423
//    region pinning); older HotSpot collectors merely defer GC until the call returns.
//  * Contexts cross as opaque jlong handles; Kotlin serializes every call on one context except
//    nativeRequestAbort / nativeAbortRequested, which are lock-free.
//  * This library links kggml_whisper (and through it the runtime's shared ggml) dynamically.

#include <jni.h>

#include "kggml_whisper.h"

#if defined(KGGML_WHISPER_TEST_HOOKS)
#    include "kggml_whisper_testing.h"
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <string>

namespace {

// Version selectors for nativeVersionString(); mirrored by KggmlWhisperJni.VERSION_* in Kotlin.
constexpr jint kVersionModule = 0;
constexpr jint kVersionUpstream = 1;
constexpr jint kVersionUpstreamCommit = 2;
constexpr jint kVersionSystemInfo = 3;

// Layout of the `ints` argument of nativeTranscribe(); mirrored by KggmlWhisperJni.TRANSCRIBE_* in Kotlin.
constexpr jsize kTranscribeIntCount = 6; // n_threads, translate, no_context, single_segment, no_timestamps, max_tokens

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

/// NewStringUTF expects *modified* UTF-8; diagnostics may carry arbitrary bytes, so anything outside
/// printable 7-bit ASCII (plus tab/newline) is replaced with '?'.
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

template <typename Fn>
jobject jni_guard_object(Fn && body) noexcept {
    try {
        return body();
    } catch (...) {
        return nullptr;
    }
}

kggml_whisper_context * from_handle(jlong handle) {
    return reinterpret_cast<kggml_whisper_context *>(static_cast<intptr_t>(handle));
}

/// Pins a primitive Java array with GetPrimitiveArrayCritical and releases it on scope exit.
class CriticalArray {
public:
    CriticalArray(JNIEnv * env, jarray array, jint release_mode)
        : env_(env), array_(array), mode_(release_mode),
          data_(array != nullptr ? env->GetPrimitiveArrayCritical(array, nullptr) : nullptr) {}
    ~CriticalArray() {
        if (data_ != nullptr) env_->ReleasePrimitiveArrayCritical(array_, data_, mode_);
    }
    CriticalArray(const CriticalArray &) = delete;
    CriticalArray & operator=(const CriticalArray &) = delete;

    void * get() const { return data_; }

private:
    JNIEnv * env_;
    jarray array_;
    jint mode_;
    void * data_;
};

/// Copies a Java string into UTF-8 owned storage (std::string); `present` is false for a null reference.
struct JavaUtf8 {
    std::string value;
    bool present = false;
    bool ok = true; // false if the JVM could not provide the chars (pending OutOfMemoryError)

    JavaUtf8(JNIEnv * env, jstring s) {
        if (s == nullptr) return;
        present = true;
        const char * chars = env->GetStringUTFChars(s, nullptr);
        if (chars == nullptr) {
            ok = false;
            return;
        }
        value.assign(chars);
        env->ReleaseStringUTFChars(s, chars);
    }

    const char * c_str() const { return present ? value.c_str() : nullptr; }
};

jint pcm_to_whisper(JNIEnv * env, jarray in, jint format, jint frames, jint channels, jint sample_rate,
                    jint quality, jfloatArray out, jintArray out_frames) {
    if (in == nullptr || out == nullptr || out_frames == nullptr || env->GetArrayLength(out_frames) < 1 ||
        frames < 0 || channels <= 0) {
        return KGGML_STATUS_INVALID_ARGUMENT;
    }
    if (static_cast<int64_t>(env->GetArrayLength(in)) < static_cast<int64_t>(frames) * channels) {
        return KGGML_STATUS_INVALID_ARGUMENT;
    }
    const jsize capacity = env->GetArrayLength(out);
    size_t produced = 0;
    kggml_status status;
    {
        CriticalArray src(env, in, JNI_ABORT);
        CriticalArray dst(env, out, 0);
        if (src.get() == nullptr || dst.get() == nullptr) return KGGML_STATUS_OUT_OF_MEMORY;
        status = kggml_whisper_pcm_to_whisper(src.get(), format, static_cast<size_t>(frames), channels, sample_rate,
                                              quality, static_cast<float *>(dst.get()),
                                              static_cast<size_t>(capacity), &produced);
    }
    const jint jproduced = produced > static_cast<size_t>(INT32_MAX) ? INT32_MAX : static_cast<jint>(produced);
    env->SetIntArrayRegion(out_frames, 0, 1, &jproduced);
    return static_cast<jint>(status);
}

} // namespace

#define KGGML_JNI(name) Java_com_donfuldauer_kggml_whisper_KggmlWhisperJni_##name

extern "C" {

JNIEXPORT jstring JNICALL KGGML_JNI(nativeVersionString)(JNIEnv * env, jclass, jint which) {
    return static_cast<jstring>(jni_guard_object([&]() -> jobject {
        switch (which) {
            case kVersionModule: return to_ascii_jstring(env, kggml_whisper_version());
            case kVersionUpstream: return to_ascii_jstring(env, kggml_whisper_upstream_version());
            case kVersionUpstreamCommit: return to_ascii_jstring(env, kggml_whisper_upstream_commit());
            case kVersionSystemInfo: return to_ascii_jstring(env, kggml_whisper_system_info());
            default: return nullptr;
        }
    }));
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeHasCoreml)(JNIEnv *, jclass) {
    return jni_guard([] { return static_cast<jint>(kggml_whisper_has_coreml()); });
}

JNIEXPORT jstring JNICALL KGGML_JNI(nativeStatusName)(JNIEnv * env, jclass, jint status) {
    return static_cast<jstring>(jni_guard_object([&]() -> jobject {
        return to_ascii_jstring(env, kggml_status_name(static_cast<kggml_status>(status)));
    }));
}

JNIEXPORT jstring JNICALL KGGML_JNI(nativeLastError)(JNIEnv * env, jclass) {
    return static_cast<jstring>(jni_guard_object([&]() -> jobject {
        return to_ascii_jstring(env, kggml_whisper_last_error());
    }));
}

/** Loads a model. `outHandle[0]` receives the context handle on success. Returns a kggml_status. */
JNIEXPORT jint JNICALL KGGML_JNI(nativeInitFromFile)(JNIEnv * env, jclass, jstring modelPath, jint backend,
                                                     jint gpuDevice, jint flashAttn, jlongArray outHandle) {
    return jni_guard([&]() -> jint {
        if (modelPath == nullptr || outHandle == nullptr || env->GetArrayLength(outHandle) < 1) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        const JavaUtf8 path(env, modelPath);
        if (!path.ok) return KGGML_STATUS_OUT_OF_MEMORY;

        kggml_whisper_context_params params;
        if (const auto s = kggml_whisper_context_default_params(&params); s != KGGML_STATUS_OK) {
            return static_cast<jint>(s);
        }
        params.backend = backend;
        params.gpu_device = gpuDevice;
        params.flash_attn = flashAttn;

        kggml_whisper_context * context = nullptr;
        const kggml_status status = kggml_whisper_init_from_file(path.c_str(), &params, &context);
        if (status == KGGML_STATUS_OK) {
            const jlong handle = static_cast<jlong>(reinterpret_cast<intptr_t>(context));
            env->SetLongArrayRegion(outHandle, 0, 1, &handle);
        }
        return static_cast<jint>(status);
    });
}

JNIEXPORT void JNICALL KGGML_JNI(nativeFree)(JNIEnv *, jclass, jlong handle) {
    jni_guard([&] {
        kggml_whisper_free(from_handle(handle));
        return 0;
    });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeGetBackend)(JNIEnv * env, jclass, jlong handle, jintArray out) {
    return jni_guard([&]() -> jint {
        if (out == nullptr || env->GetArrayLength(out) < 1) return KGGML_STATUS_INVALID_ARGUMENT;
        int32_t value = 0;
        const kggml_status status = kggml_whisper_get_backend(from_handle(handle), &value);
        const jint jvalue = value;
        env->SetIntArrayRegion(out, 0, 1, &jvalue);
        return static_cast<jint>(status);
    });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeUsesGpu)(JNIEnv * env, jclass, jlong handle, jintArray out) {
    return jni_guard([&]() -> jint {
        if (out == nullptr || env->GetArrayLength(out) < 1) return KGGML_STATUS_INVALID_ARGUMENT;
        int32_t value = 0;
        const kggml_status status = kggml_whisper_uses_gpu(from_handle(handle), &value);
        const jint jvalue = value;
        env->SetIntArrayRegion(out, 0, 1, &jvalue);
        return static_cast<jint>(status);
    });
}

JNIEXPORT void JNICALL KGGML_JNI(nativeRequestAbort)(JNIEnv *, jclass, jlong handle) {
    jni_guard([&] {
        kggml_whisper_request_abort(from_handle(handle));
        return 0;
    });
}

JNIEXPORT void JNICALL KGGML_JNI(nativeClearAbort)(JNIEnv *, jclass, jlong handle) {
    jni_guard([&] {
        kggml_whisper_clear_abort(from_handle(handle));
        return 0;
    });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeAbortRequested)(JNIEnv *, jclass, jlong handle) {
    return jni_guard([&] { return static_cast<jint>(kggml_whisper_abort_requested(from_handle(handle))); });
}

/**
 * Transcribes `count` 16 kHz mono samples of `samples`, read in place (zero-copy). `ints` holds n_threads,
 * translate, no_context, single_segment, no_timestamps, max_tokens. Returns a kggml_status.
 */
JNIEXPORT jint JNICALL KGGML_JNI(nativeTranscribe)(JNIEnv * env, jclass, jlong handle, jfloatArray samples,
                                                   jint count, jstring language, jstring initialPrompt,
                                                   jintArray ints, jfloat temperature, jfloat temperatureInc) {
    return jni_guard([&]() -> jint {
        if (samples == nullptr || ints == nullptr || count < 0 || env->GetArrayLength(samples) < count ||
            env->GetArrayLength(ints) < kTranscribeIntCount) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        // Every JNI call that is not allowed inside a critical region happens first.
        const JavaUtf8 lang(env, language);
        const JavaUtf8 prompt(env, initialPrompt);
        if (!lang.ok || !prompt.ok) return KGGML_STATUS_OUT_OF_MEMORY;
        jint flags[kTranscribeIntCount] = {};
        env->GetIntArrayRegion(ints, 0, kTranscribeIntCount, flags);

        kggml_whisper_transcribe_params params;
        if (const auto s = kggml_whisper_transcribe_default_params(&params); s != KGGML_STATUS_OK) {
            return static_cast<jint>(s);
        }
        params.language = lang.c_str();
        params.initial_prompt = prompt.c_str();
        params.n_threads = flags[0];
        params.translate = flags[1];
        params.no_context = flags[2];
        params.single_segment = flags[3];
        params.no_timestamps = flags[4];
        params.max_tokens = flags[5];
        params.temperature = temperature;
        params.temperature_inc = temperatureInc;

        CriticalArray pinned(env, samples, JNI_ABORT); // read-only: nothing to copy back
        if (pinned.get() == nullptr) return KGGML_STATUS_OUT_OF_MEMORY;
        return static_cast<jint>(kggml_whisper_transcribe(from_handle(handle), static_cast<const float *>(pinned.get()),
                                                          static_cast<size_t>(count), &params));
    });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeSegmentCount)(JNIEnv * env, jclass, jlong handle, jintArray out) {
    return jni_guard([&]() -> jint {
        if (out == nullptr || env->GetArrayLength(out) < 1) return KGGML_STATUS_INVALID_ARGUMENT;
        int32_t value = 0;
        const kggml_status status = kggml_whisper_segment_count(from_handle(handle), &value);
        const jint jvalue = value;
        env->SetIntArrayRegion(out, 0, 1, &jvalue);
        return static_cast<jint>(status);
    });
}

/**
 * Reads segment `index`: `times` receives t0_ms, t1_ms; `probs[0]` the no-speech probability; `text[0]` the raw
 * UTF-8 bytes (decoded leniently in Kotlin, since whisper.cpp may split multi-byte characters across tokens).
 */
JNIEXPORT jint JNICALL KGGML_JNI(nativeGetSegment)(JNIEnv * env, jclass, jlong handle, jint index, jlongArray times,
                                                   jfloatArray probs, jobjectArray text) {
    return jni_guard([&]() -> jint {
        if (times == nullptr || probs == nullptr || text == nullptr || env->GetArrayLength(times) < 2 ||
            env->GetArrayLength(probs) < 1 || env->GetArrayLength(text) < 1) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        kggml_whisper_segment segment{};
        const kggml_status status = kggml_whisper_get_segment(from_handle(handle), index, &segment);
        if (status != KGGML_STATUS_OK) return static_cast<jint>(status);

        const jlong t[2] = {static_cast<jlong>(segment.t0_ms), static_cast<jlong>(segment.t1_ms)};
        env->SetLongArrayRegion(times, 0, 2, t);
        const jfloat p = segment.no_speech_prob;
        env->SetFloatArrayRegion(probs, 0, 1, &p);

        const size_t len = segment.text != nullptr ? std::strlen(segment.text) : 0;
        if (len > static_cast<size_t>(INT32_MAX)) return KGGML_STATUS_NATIVE_FAILURE;
        jbyteArray bytes = env->NewByteArray(static_cast<jsize>(len));
        if (bytes == nullptr) return KGGML_STATUS_OUT_OF_MEMORY;
        if (len > 0) env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(len), reinterpret_cast<const jbyte *>(segment.text));
        env->SetObjectArrayElement(text, 0, bytes);
        env->DeleteLocalRef(bytes);
        return KGGML_STATUS_OK;
    });
}

/** Exact 16 kHz output length for `frames` input frames at `sampleRate`, or -1 on invalid input. */
JNIEXPORT jlong JNICALL KGGML_JNI(nativeResampledLength)(JNIEnv *, jclass, jlong frames, jint sampleRate) {
    try {
        if (frames < 0) return -1;
        size_t out = 0;
        if (kggml_whisper_resampled_length(static_cast<size_t>(frames), sampleRate, KGGML_WHISPER_SAMPLE_RATE, &out) !=
            KGGML_STATUS_OK) {
            return -1;
        }
        return out > static_cast<size_t>(INT64_MAX) ? -1 : static_cast<jlong>(out);
    } catch (...) {
        return -1;
    }
}

/** Converts interleaved float PCM to 16 kHz mono (both arrays pinned zero-copy). Returns a kggml_status. */
JNIEXPORT jint JNICALL KGGML_JNI(nativePcmF32ToWhisper)(JNIEnv * env, jclass, jfloatArray in, jint frames,
                                                        jint channels, jint sampleRate, jint quality,
                                                        jfloatArray out, jintArray outFrames) {
    return jni_guard([&] {
        return pcm_to_whisper(env, in, KGGML_PCM_F32, frames, channels, sampleRate, quality, out, outFrames);
    });
}

/** Converts interleaved int16 PCM to 16 kHz mono float (both arrays pinned zero-copy). Returns a kggml_status. */
JNIEXPORT jint JNICALL KGGML_JNI(nativePcmS16ToWhisper)(JNIEnv * env, jclass, jshortArray in, jint frames,
                                                        jint channels, jint sampleRate, jint quality,
                                                        jfloatArray out, jintArray outFrames) {
    return jni_guard([&] {
        return pcm_to_whisper(env, in, KGGML_PCM_S16, frames, channels, sampleRate, quality, out, outFrames);
    });
}

/* ---------------------------------------------------------------------------------------------------- */
/* Test hooks (KGGML_WHISPER_TEST_HOOKS builds only; otherwise they report "unsupported")                */
/* ---------------------------------------------------------------------------------------------------- */

JNIEXPORT jboolean JNICALL KGGML_JNI(nativeTestHooksAvailable)(JNIEnv *, jclass) {
#if defined(KGGML_WHISPER_TEST_HOOKS)
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

/** use_gpu (out[0]) and coreml_allowed (out[1]) that init would pass for `backend`. */
JNIEXPORT jint JNICALL KGGML_JNI(nativeTestResolveUseGpu)(JNIEnv * env, jclass, jint backend, jintArray out) {
    return jni_guard([&]() -> jint {
#if defined(KGGML_WHISPER_TEST_HOOKS)
        if (out == nullptr || env->GetArrayLength(out) < 2) return KGGML_STATUS_INVALID_ARGUMENT;
        kggml_whisper_context_params params;
        kggml_whisper_context_default_params(&params);
        params.backend = backend;
        int32_t use_gpu = 0;
        int32_t coreml = 0;
        const kggml_status status = kggml_whisper_test_resolve_use_gpu(&params, &use_gpu, &coreml);
        const jint values[2] = {use_gpu, coreml};
        env->SetIntArrayRegion(out, 0, 2, values);
        return static_cast<jint>(status);
#else
        (void) env;
        (void) backend;
        (void) out;
        return KGGML_STATUS_NATIVE_FAILURE;
#endif
    });
}

/** Number of abort-callback polls whisper.cpp made during the last transcription. */
JNIEXPORT jlong JNICALL KGGML_JNI(nativeTestAbortPollCount)(JNIEnv *, jclass, jlong handle) {
    try {
#if defined(KGGML_WHISPER_TEST_HOOKS)
        int64_t polls = 0;
        if (kggml_whisper_test_abort_poll_count(from_handle(handle), &polls) != KGGML_STATUS_OK) return -1;
        return static_cast<jlong>(polls);
#else
        (void) handle;
        return -1;
#endif
    } catch (...) {
        return -1;
    }
}

} // extern "C"
