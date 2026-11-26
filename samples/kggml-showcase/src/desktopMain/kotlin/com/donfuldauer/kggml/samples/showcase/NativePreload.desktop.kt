package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.runtime.KggmlRuntime

private val nativeRuntimePreload: Result<Unit> by lazy { runCatching { KggmlRuntime.info(); Unit } }

/**
 * Loads the kggml native runtime, and with it the system C++ library (`libstdc++.so.6` on Linux), before Compose
 * loads Skiko. Call it before anything touches Compose or Skia.
 *
 * On Linux, Skiko links its own static copy of libstdc++. When Skiko is loaded first, the C++ locale facet ids of the
 * two copies clash and the first `std::ostream` number formatting in llama.cpp (model loading) crashes the JVM with a
 * SIGSEGV in `std::moneypunct::do_grouping`. Loading the system libstdc++ first avoids that. Missing or broken native
 * libraries are ignored here: the screens report them when an engine is used. Idempotent and cheap after the first call.
 */
internal fun preloadNativeRuntime() {
    nativeRuntimePreload
}
