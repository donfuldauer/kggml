// kggml_runtime_jni.cpp - JNI bridge between com.donfuldauer.kggml.runtime.KggmlRuntimeJni (Android + JVM
// Desktop) and the kggml_runtime C ABI.
//
// Rules (AGENTS.md §2.4):
//  * Every JNI entry point is extern "C" and runs its body behind jni_guard(), so no C++ exception ever
//    crosses into the JVM. Failures are reported as kggml_status codes, which Kotlin maps onto the typed
//    InferenceException hierarchy of :kggml-core-api.
//  * This library links kggml_runtime dynamically; it never embeds ggml symbols.
//  * Only primitive arrays are copied (Get/SetIntArrayRegion); no array is held across calls.

#include <jni.h>

#include "kggml_runtime.h"

#if defined(KGGML_RUNTIME_TEST_HOOKS)
#    include "kggml_runtime_testing.h"
#endif

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>

namespace {

constexpr int kBackendIntStride = 4; // kind, compiled, available, device_count

// Version selectors for nativeVersionString(); mirrored by KggmlRuntimeJni.VERSION_* in Kotlin.
constexpr jint kVersionRuntime = 0;
constexpr jint kVersionGgml = 1;
constexpr jint kVersionGgmlCommit = 2;
constexpr jint kVersionGgmlBuildCommit = 3;

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

/// NewStringUTF expects *modified* UTF-8; native diagnostics may carry arbitrary bytes, so anything outside
/// printable 7-bit ASCII (plus tab/newline) is replaced with '?'. Returns NULL (with a pending Java
/// OutOfMemoryError) only if the JVM cannot allocate the string.
jstring to_jstring(JNIEnv * env, const char * text) {
    std::string ascii;
    if (text != nullptr) {
        for (const char * p = text; *p != '\0'; ++p) {
            const auto c = static_cast<unsigned char>(*p);
            ascii.push_back((c >= 0x20 && c < 0x7F) || c == '\t' || c == '\n' ? static_cast<char>(c) : '?');
        }
    }
    return env->NewStringUTF(ascii.c_str());
}

/// Like jni_guard() for object-returning entry points: returns NULL on any C++ exception.
template <typename Fn>
jstring jni_guard_string(Fn && body) noexcept {
    try {
        return body();
    } catch (...) {
        return nullptr;
    }
}

} // namespace

#define KGGML_JNI(name) Java_com_donfuldauer_kggml_runtime_KggmlRuntimeJni_##name

extern "C" {

JNIEXPORT jint JNICALL KGGML_JNI(nativeInit)(JNIEnv *, jclass) {
    return jni_guard([] { return static_cast<jint>(kggml_runtime_init()); });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeShutdown)(JNIEnv *, jclass) {
    return jni_guard([] { return static_cast<jint>(kggml_runtime_shutdown()); });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeIsInitialized)(JNIEnv *, jclass) {
    return jni_guard([] { return static_cast<jint>(kggml_runtime_is_initialized()); });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeRefCount)(JNIEnv *, jclass) {
    return jni_guard([] { return static_cast<jint>(kggml_runtime_ref_count()); });
}

/**
 * Fills `ints` (stride 4: kind, compiled, available, device_count) and `names` with up to names.length
 * backend entries. `outCount[0]` always receives the required entry count. Returns a kggml_status.
 */
JNIEXPORT jint JNICALL KGGML_JNI(nativeGetBackends)(JNIEnv * env, jclass, jintArray ints, jobjectArray names,
                                                    jintArray outCount) {
    return jni_guard([&]() -> jint {
        if (ints == nullptr || names == nullptr || outCount == nullptr || env->GetArrayLength(outCount) < 1) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        const jsize capacity = env->GetArrayLength(names);
        if (env->GetArrayLength(ints) < capacity * kBackendIntStride) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }

        kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT] = {};
        const int32_t usable = capacity < KGGML_BACKEND_KIND_COUNT ? capacity : KGGML_BACKEND_KIND_COUNT;
        int32_t count = 0;
        const kggml_status status = kggml_runtime_get_backends(usable > 0 ? infos : nullptr, usable, &count);

        const jint jcount = count;
        env->SetIntArrayRegion(outCount, 0, 1, &jcount);
        if (status != KGGML_STATUS_OK) {
            return static_cast<jint>(status);
        }

        jint packed[KGGML_BACKEND_KIND_COUNT * kBackendIntStride] = {};
        for (int32_t i = 0; i < count; ++i) {
            packed[i * kBackendIntStride + 0] = infos[i].kind;
            packed[i * kBackendIntStride + 1] = infos[i].compiled;
            packed[i * kBackendIntStride + 2] = infos[i].available;
            packed[i * kBackendIntStride + 3] = infos[i].device_count;

            char name[KGGML_BACKEND_NAME_MAX + 1] = {};
            for (int k = 0; k < KGGML_BACKEND_NAME_MAX && infos[i].name[k] != '\0'; ++k) {
                name[k] = infos[i].name[k];
            }
            jstring jname = to_jstring(env, name);
            if (jname == nullptr) {
                return KGGML_STATUS_OUT_OF_MEMORY; // pending OutOfMemoryError is surfaced by the JVM
            }
            env->SetObjectArrayElement(names, i, jname);
            env->DeleteLocalRef(jname);
        }
        env->SetIntArrayRegion(ints, 0, count * kBackendIntStride, packed);
        return KGGML_STATUS_OK;
    });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativePerformanceThreadCount)(JNIEnv * env, jclass, jintArray out) {
    return jni_guard([&]() -> jint {
        if (out == nullptr || env->GetArrayLength(out) < 1) {
            return KGGML_STATUS_INVALID_ARGUMENT;
        }
        int32_t count = 0;
        const kggml_status status = kggml_runtime_performance_thread_count(&count);
        const jint jcount = count;
        env->SetIntArrayRegion(out, 0, 1, &jcount);
        return static_cast<jint>(status);
    });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeHardwareThreadCount)(JNIEnv *, jclass) {
    return jni_guard([] { return static_cast<jint>(kggml_runtime_hardware_thread_count()); });
}

JNIEXPORT jint JNICALL KGGML_JNI(nativeHasCoreml)(JNIEnv *, jclass) {
    return jni_guard([] { return static_cast<jint>(kggml_runtime_has_coreml()); });
}

JNIEXPORT jstring JNICALL KGGML_JNI(nativeVersionString)(JNIEnv * env, jclass, jint which) {
    return jni_guard_string([&]() -> jstring {
        switch (which) {
            case kVersionRuntime: return to_jstring(env, kggml_runtime_version());
            case kVersionGgml: return to_jstring(env, kggml_runtime_ggml_version());
            case kVersionGgmlCommit: return to_jstring(env, kggml_runtime_ggml_commit());
            case kVersionGgmlBuildCommit: return to_jstring(env, kggml_runtime_ggml_build_commit());
            default: return to_jstring(env, "");
        }
    });
}

JNIEXPORT jstring JNICALL KGGML_JNI(nativeStatusName)(JNIEnv * env, jclass, jint status) {
    return jni_guard_string([&] { return to_jstring(env, kggml_status_name(static_cast<kggml_status>(status))); });
}

JNIEXPORT jstring JNICALL KGGML_JNI(nativeLastError)(JNIEnv * env, jclass) {
    return jni_guard_string([&] { return to_jstring(env, kggml_runtime_last_error()); });
}

JNIEXPORT void JNICALL KGGML_JNI(nativeClearLastError)(JNIEnv *, jclass) {
    jni_guard([] {
        kggml_runtime_clear_last_error();
        return static_cast<jint>(KGGML_STATUS_OK);
    });
}

/**
 * Test-only: arms a one-shot C++ fault inside the next guarded kggml_runtime_* call on this thread.
 * Returns JNI_TRUE only in builds compiled with KGGML_RUNTIME_TEST_HOOKS (never in shipped binaries).
 */
JNIEXPORT jboolean JNICALL KGGML_JNI(nativeTestInjectFault)(JNIEnv *, jclass, jint fault) {
#if defined(KGGML_RUNTIME_TEST_HOOKS)
    try {
        kggml_runtime_test_inject_fault(fault);
        return JNI_TRUE;
    } catch (...) {
        return JNI_FALSE;
    }
#else
    (void) fault;
    return JNI_FALSE;
#endif
}

} // extern "C"
