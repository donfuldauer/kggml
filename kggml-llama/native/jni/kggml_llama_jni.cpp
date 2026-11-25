// kggml_llama_jni.cpp - JNI bridge between com.donfuldauer.kggml.llama.KggmlLlamaJni (Android + JVM Desktop) and
// the kggml_llama C ABI.
//
// Rules (AGENTS.md §2.4):
//  * Every JNI entry point is extern "C" and runs its body behind jni_guard(), so no C++ exception ever crosses
//    into the JVM. Failures are reported as kggml_status codes, which Kotlin maps onto the typed
//    InferenceException hierarchy of :kggml-core-api.
//  * Token arrays cross the boundary zero-copy: int[] is pinned with GetPrimitiveArrayCritical and read in place
//    by llama_decode(). CriticalArray releases it on every path (RAII, including exceptions). No other JNI
//    function is called on the same thread while an array is held.
//  * Text (prompts, grammars, model paths) crosses as raw UTF-8 byte[] rather than jstring, so supplementary
//    characters and embedded bytes are never mangled by JNI's modified UTF-8.
//  * Models, contexts and samplers cross as opaque jlong handles; Kotlin serializes every call on one engine
//    except nativeRequestAbort / nativeAbortRequested, which are lock-free.
//  * This library links kggml_llama (and through it the runtime's shared ggml) dynamically.

#include <jni.h>

#include "kggml_llama.h"

#if defined(KGGML_LLAMA_TEST_HOOKS)
#    include "kggml_llama_testing.h"
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

namespace {

// Version selectors for nativeVersionString(); mirrored by KggmlLlamaJni.VERSION_* in Kotlin.
constexpr jint kVersionModule = 0;
constexpr jint kVersionUpstream = 1;
constexpr jint kVersionUpstreamCommit = 2;
constexpr jint kVersionSystemInfo = 3;

// Array layouts; mirrored by KggmlLlamaJni.*_COUNT in Kotlin.
constexpr jsize kModelParamCount = 5;   // backend, gpu_layers, gpu_device, use_mmap, use_mlock
constexpr jsize kModelInfoInts = 10;    // n_vocab, n_ctx_train, n_embd, n_layer, has_encoder, has_decoder, backend,
                                        // uses_gpu, bos, eos
constexpr jsize kModelInfoLongs = 2;    // size_bytes, n_params
constexpr jsize kContextParamCount = 9; // n_ctx, n_batch, n_ubatch, n_seq_max, n_threads, n_threads_batch,
                                        // flash_attn, embeddings, pooling
constexpr jsize kSamplerIntCount = 3;   // top_k, repeat_last_n, seed (uint32 bits)
constexpr jsize kSamplerFloatCount = 4; // temperature, top_p, min_p, repeat_penalty

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

template <typename T>
T * from_handle(jlong handle) {
    return reinterpret_cast<T *>(static_cast<intptr_t>(handle));
}

jlong to_handle(const void * p) {
    return static_cast<jlong>(reinterpret_cast<intptr_t>(p));
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

/// Copies a Java byte[] holding UTF-8 into a NUL-terminated std::string; `present` is false for null.
struct Utf8Bytes {
    std::string value;
    bool present = false;

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

void set_int(JNIEnv * env, jintArray out, jint value) {
    env->SetIntArrayRegion(out, 0, 1, &value);
}

void set_long(JNIEnv * env, jlongArray out, jlong value) {
    env->SetLongArrayRegion(out, 0, 1, &value);
}

/// Stores a new byte[] with `n` bytes of `data` into `out[0]`.
jint set_byte_array_element(JNIEnv * env, jobjectArray out, const void * data, size_t n) {
    if (n > static_cast<size_t>(INT32_MAX)) return KGGML_STATUS_OUT_OF_MEMORY;
    jbyteArray array = env->NewByteArray(static_cast<jsize>(n));
    if (array == nullptr) return KGGML_STATUS_OUT_OF_MEMORY;
    if (n > 0) env->SetByteArrayRegion(array, 0, static_cast<jsize>(n), static_cast<const jbyte *>(data));
    env->SetObjectArrayElement(out, 0, array);
    env->DeleteLocalRef(array);
    return KGGML_STATUS_OK;
}

} // namespace

#define KGGML_JNI(ret, name) extern "C" JNIEXPORT ret JNICALL Java_com_donfuldauer_kggml_llama_KggmlLlamaJni_##name

// ---------------------------------------------------------------------------------------------------------------
// Versions & errors
// ---------------------------------------------------------------------------------------------------------------

KGGML_JNI(jstring, nativeVersionString)(JNIEnv * env, jclass, jint which) {
    return static_cast<jstring>(jni_guard_object([&]() -> jobject {
        switch (which) {
            case kVersionModule: return to_ascii_jstring(env, kggml_llama_version());
            case kVersionUpstream: return to_ascii_jstring(env, kggml_llama_upstream_version());
            case kVersionUpstreamCommit: return to_ascii_jstring(env, kggml_llama_upstream_commit());
            case kVersionSystemInfo: return to_ascii_jstring(env, kggml_llama_system_info());
            default: return nullptr;
        }
    }));
}

KGGML_JNI(jstring, nativeStatusName)(JNIEnv * env, jclass, jint status) {
    return static_cast<jstring>(
        jni_guard_object([&]() -> jobject { return to_ascii_jstring(env, kggml_status_name(static_cast<kggml_status>(status))); }));
}

KGGML_JNI(jstring, nativeLastError)(JNIEnv * env, jclass) {
    return static_cast<jstring>(jni_guard_object([&]() -> jobject { return to_ascii_jstring(env, kggml_llama_last_error()); }));
}

// ---------------------------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------------------------

/// `ints`: backend, gpu_layers, gpu_device, use_mmap, use_mlock.
KGGML_JNI(jint, nativeModelLoad)(JNIEnv * env, jclass, jbyteArray path_utf8, jintArray ints, jlongArray out_handle) {
    return jni_guard([&]() -> jint {
        if (path_utf8 == nullptr || !has_length(env, ints, kModelParamCount) || !has_length(env, out_handle, 1)) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        const Utf8Bytes path(env, path_utf8);
        jint v[kModelParamCount] = {};
        env->GetIntArrayRegion(ints, 0, kModelParamCount, v);
        kggml_llama_model_params params{};
        if (const auto s = kggml_llama_model_default_params(&params); s != KGGML_STATUS_OK) return s;
        params.backend = v[0];
        params.gpu_layers = v[1];
        params.gpu_device = v[2];
        params.use_mmap = v[3];
        params.use_mlock = v[4];
        kggml_llama_model * model = nullptr;
        const kggml_status s = kggml_llama_model_load(path.c_str(), &params, &model);
        set_long(env, out_handle, s == KGGML_STATUS_OK ? to_handle(model) : 0);
        return s;
    });
}

KGGML_JNI(void, nativeModelFree)(JNIEnv *, jclass, jlong handle) {
    kggml_llama_model_free(from_handle<kggml_llama_model>(handle));
}

KGGML_JNI(jint, nativeModelInfo)(JNIEnv * env, jclass, jlong handle, jintArray out_ints, jlongArray out_longs) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out_ints, kModelInfoInts) || !has_length(env, out_longs, kModelInfoLongs)) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        kggml_llama_model_info info{};
        const kggml_status s = kggml_llama_model_get_info(from_handle<const kggml_llama_model>(handle), &info);
        if (s != KGGML_STATUS_OK) return s;
        const jint ints[kModelInfoInts] = {info.n_vocab, info.n_ctx_train, info.n_embd,    info.n_layer, info.has_encoder,
                                           info.has_decoder, info.backend, info.uses_gpu, info.bos,     info.eos};
        env->SetIntArrayRegion(out_ints, 0, kModelInfoInts, ints);
        const jlong longs[kModelInfoLongs] = {static_cast<jlong>(info.size_bytes), static_cast<jlong>(info.n_params)};
        env->SetLongArrayRegion(out_longs, 0, kModelInfoLongs, longs);
        return KGGML_STATUS_OK;
    });
}

/// `out[0]` receives llama_model_desc() as UTF-8 bytes.
KGGML_JNI(jint, nativeModelDesc)(JNIEnv * env, jclass, jlong handle, jobjectArray out) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        const auto * model = from_handle<const kggml_llama_model>(handle);
        size_t length = 0;
        kggml_status s = kggml_llama_model_desc(model, nullptr, 0, &length);
        if (s != KGGML_STATUS_OK && s != KGGML_STATUS_BUFFER_TOO_SMALL) return s;
        std::vector<char> buffer(length + 1, '\0');
        s = kggml_llama_model_desc(model, buffer.data(), buffer.size(), &length);
        if (s != KGGML_STATUS_OK) return s;
        return set_byte_array_element(env, out, buffer.data(), std::strlen(buffer.data()));
    });
}

KGGML_JNI(jint, nativeTokenIsEog)(JNIEnv *, jclass, jlong model, jint token) {
    return kggml_llama_token_is_eog(from_handle<const kggml_llama_model>(model), token);
}

/// Tokenizes UTF-8 `text`; `out[0]` receives the token ids.
KGGML_JNI(jint, nativeTokenize)(JNIEnv * env, jclass, jlong model, jbyteArray text, jint add_special, jint parse_special,
                                jobjectArray out) {
    return jni_guard([&]() -> jint {
        if (text == nullptr || !has_length(env, out, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        const Utf8Bytes utf8(env, text);
        const auto * m = from_handle<const kggml_llama_model>(model);
        // A token covers at least one byte, plus BOS/EOS: a generous first guess that usually fits.
        std::vector<kggml_llama_token> tokens(utf8.value.size() + 8);
        int32_t count = 0;
        kggml_status s = kggml_llama_tokenize(m, utf8.value.data(), utf8.value.size(), add_special, parse_special,
                                              tokens.data(), static_cast<int32_t>(tokens.size()), &count);
        if (s == KGGML_STATUS_BUFFER_TOO_SMALL && count > 0) {
            tokens.resize(static_cast<size_t>(count));
            s = kggml_llama_tokenize(m, utf8.value.data(), utf8.value.size(), add_special, parse_special, tokens.data(),
                                     static_cast<int32_t>(tokens.size()), &count);
        }
        if (s != KGGML_STATUS_OK) return s;
        jintArray array = env->NewIntArray(count);
        if (array == nullptr) return KGGML_STATUS_OUT_OF_MEMORY;
        if (count > 0) env->SetIntArrayRegion(array, 0, count, reinterpret_cast<const jint *>(tokens.data()));
        env->SetObjectArrayElement(out, 0, array);
        env->DeleteLocalRef(array);
        return KGGML_STATUS_OK;
    });
}

/// `out[0]` receives the raw UTF-8 bytes of `token` (possibly an incomplete multi-byte sequence).
KGGML_JNI(jint, nativeTokenToPiece)(JNIEnv * env, jclass, jlong model, jint token, jint special, jobjectArray out) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        const auto * m = from_handle<const kggml_llama_model>(model);
        char small[64];
        int32_t length = 0;
        kggml_status s = kggml_llama_token_to_piece(m, token, special, small, sizeof(small), &length);
        if (s == KGGML_STATUS_OK) return set_byte_array_element(env, out, small, static_cast<size_t>(length));
        if (s != KGGML_STATUS_BUFFER_TOO_SMALL || length <= 0) return s;
        std::vector<char> big(static_cast<size_t>(length));
        s = kggml_llama_token_to_piece(m, token, special, big.data(), length, &length);
        if (s != KGGML_STATUS_OK) return s;
        return set_byte_array_element(env, out, big.data(), static_cast<size_t>(length));
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------------------------------------------

/// `ints`: n_ctx, n_batch, n_ubatch, n_seq_max, n_threads, n_threads_batch, flash_attn, embeddings, pooling.
KGGML_JNI(jint, nativeContextCreate)(JNIEnv * env, jclass, jlong model, jintArray ints, jlongArray out_handle) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, ints, kContextParamCount) || !has_length(env, out_handle, 1)) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        jint v[kContextParamCount] = {};
        env->GetIntArrayRegion(ints, 0, kContextParamCount, v);
        for (int i = 0; i < 4; ++i) {
            if (v[i] < 0) return KGGML_STATUS_INVALID_ARGUMENT;
        }
        kggml_llama_context_params params{};
        if (const auto s = kggml_llama_context_default_params(&params); s != KGGML_STATUS_OK) return s;
        params.n_ctx = static_cast<uint32_t>(v[0]);
        params.n_batch = static_cast<uint32_t>(v[1]);
        params.n_ubatch = static_cast<uint32_t>(v[2]);
        params.n_seq_max = static_cast<uint32_t>(v[3]);
        params.n_threads = v[4];
        params.n_threads_batch = v[5];
        params.flash_attn = v[6];
        params.embeddings = v[7];
        params.pooling = v[8];
        kggml_llama_context * context = nullptr;
        const kggml_status s = kggml_llama_context_create(from_handle<kggml_llama_model>(model), &params, &context);
        set_long(env, out_handle, s == KGGML_STATUS_OK ? to_handle(context) : 0);
        return s;
    });
}

KGGML_JNI(void, nativeContextFree)(JNIEnv *, jclass, jlong handle) {
    kggml_llama_context_free(from_handle<kggml_llama_context>(handle));
}

KGGML_JNI(jint, nativeContextNCtx)(JNIEnv * env, jclass, jlong handle, jintArray out) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        uint32_t n_ctx = 0;
        const kggml_status s = kggml_llama_context_n_ctx(from_handle<const kggml_llama_context>(handle), &n_ctx);
        set_int(env, out, n_ctx > static_cast<uint32_t>(INT32_MAX) ? INT32_MAX : static_cast<jint>(n_ctx));
        return s;
    });
}

KGGML_JNI(void, nativeRequestAbort)(JNIEnv *, jclass, jlong handle) {
    kggml_llama_request_abort(from_handle<kggml_llama_context>(handle));
}

KGGML_JNI(void, nativeClearAbort)(JNIEnv *, jclass, jlong handle) {
    kggml_llama_clear_abort(from_handle<kggml_llama_context>(handle));
}

KGGML_JNI(jint, nativeAbortRequested)(JNIEnv *, jclass, jlong handle) {
    return kggml_llama_abort_requested(from_handle<const kggml_llama_context>(handle));
}

/// Decodes tokens[offset, offset + count) zero-copy (the int[] is pinned for the duration of the call).
KGGML_JNI(jint, nativeDecode)(JNIEnv * env, jclass, jlong handle, jintArray tokens, jint offset, jint count, jint pos0,
                              jint seq_id, jint logits_last) {
    return jni_guard([&]() -> jint {
        if (tokens == nullptr || offset < 0 || count <= 0) return KGGML_STATUS_INVALID_ARGUMENT;
        if (static_cast<int64_t>(offset) + count > env->GetArrayLength(tokens)) return KGGML_STATUS_INVALID_ARGUMENT;
        static_assert(sizeof(jint) == sizeof(kggml_llama_token), "token ids are int32");
        CriticalArray pinned(env, tokens, JNI_ABORT);
        if (pinned.get() == nullptr) return KGGML_STATUS_OUT_OF_MEMORY;
        const auto * ids = static_cast<const kggml_llama_token *>(pinned.get()) + offset;
        return kggml_llama_decode(from_handle<kggml_llama_context>(handle), ids, count, pos0, seq_id, logits_last);
    });
}

KGGML_JNI(jint, nativeMemorySeqRm)(JNIEnv *, jclass, jlong handle, jint seq_id, jint p0, jint p1) {
    return jni_guard([&]() -> jint { return kggml_llama_memory_seq_rm(from_handle<kggml_llama_context>(handle), seq_id, p0, p1); });
}

KGGML_JNI(jint, nativeMemoryClear)(JNIEnv *, jclass, jlong handle) {
    return jni_guard([&]() -> jint { return kggml_llama_memory_clear(from_handle<kggml_llama_context>(handle)); });
}

KGGML_JNI(jint, nativeMemorySeqPosMax)(JNIEnv * env, jclass, jlong handle, jint seq_id, jintArray out) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        int32_t pos = -1;
        const kggml_status s = kggml_llama_memory_seq_pos_max(from_handle<kggml_llama_context>(handle), seq_id, &pos);
        set_int(env, out, pos);
        return s;
    });
}

/// `out[0]` receives the (optionally L2-normalized) embedding of `seq_id` as a float[].
KGGML_JNI(jint, nativeGetEmbeddings)(JNIEnv * env, jclass, jlong handle, jint seq_id, jint normalize, jobjectArray out) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        auto * context = from_handle<kggml_llama_context>(handle);
        int32_t count = 0;
        kggml_status s = kggml_llama_get_embeddings(context, seq_id, normalize, nullptr, 0, &count);
        if (s != KGGML_STATUS_BUFFER_TOO_SMALL && s != KGGML_STATUS_OK) return s;
        std::vector<float> values(static_cast<size_t>(count > 0 ? count : 0));
        s = kggml_llama_get_embeddings(context, seq_id, normalize, values.data(), count, &count);
        if (s != KGGML_STATUS_OK) return s;
        jfloatArray array = env->NewFloatArray(count);
        if (array == nullptr) return KGGML_STATUS_OUT_OF_MEMORY;
        if (count > 0) env->SetFloatArrayRegion(array, 0, count, values.data());
        env->SetObjectArrayElement(out, 0, array);
        env->DeleteLocalRef(array);
        return KGGML_STATUS_OK;
    });
}

/// Serializes the state of `seq_id` (< 0 = whole context) into a new byte[] stored in `out[0]`.
KGGML_JNI(jint, nativeStateSave)(JNIEnv * env, jclass, jlong handle, jint seq_id, jobjectArray out) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        auto * context = from_handle<kggml_llama_context>(handle);
        size_t size = 0;
        kggml_status s = kggml_llama_state_size(context, seq_id, &size);
        if (s != KGGML_STATUS_OK) return s;
        if (size > static_cast<size_t>(INT32_MAX)) return KGGML_STATUS_OUT_OF_MEMORY; // Java arrays are int-indexed
        std::vector<uint8_t> buffer(size);
        size_t written = 0;
        s = kggml_llama_state_save(context, seq_id, buffer.data(), buffer.size(), &written);
        if (s != KGGML_STATUS_OK) return s;
        return set_byte_array_element(env, out, buffer.data(), written);
    });
}

/// Restores a state from `data` (copied out of the Java heap first: llama.cpp may allocate while reading it).
KGGML_JNI(jint, nativeStateLoad)(JNIEnv * env, jclass, jlong handle, jint seq_id, jbyteArray data) {
    return jni_guard([&]() -> jint {
        if (data == nullptr) return KGGML_STATUS_INVALID_ARGUMENT;
        const jsize n = env->GetArrayLength(data);
        std::vector<uint8_t> bytes(static_cast<size_t>(n));
        if (n > 0) env->GetByteArrayRegion(data, 0, n, reinterpret_cast<jbyte *>(bytes.data()));
        return kggml_llama_state_load(from_handle<kggml_llama_context>(handle), seq_id, bytes.data(), bytes.size());
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------------------------------------------

/// `ints`: top_k, repeat_last_n, seed; `floats`: temperature, top_p, min_p, repeat_penalty. Grammar strings are UTF-8.
KGGML_JNI(jint, nativeSamplerCreate)(JNIEnv * env, jclass, jlong model, jintArray ints, jfloatArray floats,
                                     jbyteArray grammar_utf8, jbyteArray root_utf8, jlongArray out_handle) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, ints, kSamplerIntCount) || !has_length(env, floats, kSamplerFloatCount) ||
            !has_length(env, out_handle, 1)) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        jint iv[kSamplerIntCount] = {};
        jfloat fv[kSamplerFloatCount] = {};
        env->GetIntArrayRegion(ints, 0, kSamplerIntCount, iv);
        env->GetFloatArrayRegion(floats, 0, kSamplerFloatCount, fv);
        const Utf8Bytes grammar(env, grammar_utf8);
        const Utf8Bytes root(env, root_utf8);
        kggml_llama_sampler_params params{};
        if (const auto s = kggml_llama_sampler_default_params(&params); s != KGGML_STATUS_OK) return s;
        params.top_k = iv[0];
        params.repeat_last_n = iv[1];
        params.seed = static_cast<uint32_t>(iv[2]);
        params.temperature = fv[0];
        params.top_p = fv[1];
        params.min_p = fv[2];
        params.repeat_penalty = fv[3];
        params.grammar = grammar.c_str();
        params.grammar_root = root.c_str();
        kggml_llama_sampler * sampler = nullptr;
        const kggml_status s = kggml_llama_sampler_create(from_handle<const kggml_llama_model>(model), &params, &sampler);
        set_long(env, out_handle, s == KGGML_STATUS_OK ? to_handle(sampler) : 0);
        return s;
    });
}

KGGML_JNI(void, nativeSamplerFree)(JNIEnv *, jclass, jlong handle) {
    kggml_llama_sampler_free(from_handle<kggml_llama_sampler>(handle));
}

KGGML_JNI(jint, nativeSamplerSample)(JNIEnv * env, jclass, jlong sampler, jlong context, jint idx, jintArray out) {
    return jni_guard([&]() -> jint {
        if (!has_length(env, out, 1)) return KGGML_STATUS_INVALID_ARGUMENT;
        kggml_llama_token token = -1;
        const kggml_status s = kggml_llama_sampler_sample(from_handle<kggml_llama_sampler>(sampler),
                                                          from_handle<kggml_llama_context>(context), idx, &token);
        set_int(env, out, token);
        return s;
    });
}

KGGML_JNI(jint, nativeSamplerAccept)(JNIEnv *, jclass, jlong sampler, jint token, jint apply_grammar) {
    return jni_guard([&]() -> jint {
        return kggml_llama_sampler_accept(from_handle<kggml_llama_sampler>(sampler), token, apply_grammar);
    });
}

KGGML_JNI(jint, nativeSamplerReset)(JNIEnv *, jclass, jlong sampler) {
    return jni_guard([&]() -> jint { return kggml_llama_sampler_reset(from_handle<kggml_llama_sampler>(sampler)); });
}

// ---------------------------------------------------------------------------------------------------------------
// Test hooks (KGGML_LLAMA_TEST_HOOKS builds only; release libraries report "unavailable")
// ---------------------------------------------------------------------------------------------------------------

KGGML_JNI(jboolean, nativeTestHooksAvailable)(JNIEnv *, jclass) {
#if defined(KGGML_LLAMA_TEST_HOOKS)
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

/// Live kggml_llama models / contexts / samplers in this process, or -1 when unavailable.
KGGML_JNI(jlong, nativeTestLiveObjects)(JNIEnv *, jclass) {
#if defined(KGGML_LLAMA_TEST_HOOKS)
    return kggml_llama_test_live_objects();
#else
    return -1;
#endif
}

/// Abort-callback polls llama.cpp made on the context since it was created, or -1 when unavailable.
KGGML_JNI(jlong, nativeTestAbortPollCount)(JNIEnv *, jclass, jlong handle) {
#if defined(KGGML_LLAMA_TEST_HOOKS)
    int64_t polls = -1;
    if (kggml_llama_test_abort_poll_count(from_handle<const kggml_llama_context>(handle), &polls) != KGGML_STATUS_OK) {
        return -1;
    }
    return polls;
#else
    (void) handle;
    return -1;
#endif
}

/// Makes the context request abort on itself after `n` further abort polls (-1 disarms). Returns a status.
KGGML_JNI(jint, nativeTestAbortAfterPolls)(JNIEnv *, jclass, jlong handle, jint n) {
#if defined(KGGML_LLAMA_TEST_HOOKS)
    return kggml_llama_test_abort_after_polls(from_handle<kggml_llama_context>(handle), n);
#else
    (void) handle;
    (void) n;
    return KGGML_STATUS_BACKEND_UNAVAILABLE;
#endif
}
