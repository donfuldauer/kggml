package com.donfuldauer.kggml.core

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.newSingleThreadContext

@OptIn(DelicateCoroutinesApi::class, ExperimentalCoroutinesApi::class)
internal actual fun newDedicatedThreadDispatcher(name: String): CoroutineDispatcher =
    newSingleThreadContext(name)
