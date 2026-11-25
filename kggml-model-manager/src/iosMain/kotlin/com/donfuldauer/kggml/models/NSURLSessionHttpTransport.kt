package com.donfuldauer.kggml.models

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.suspendCancellableCoroutine
import platform.Foundation.NSData
import platform.Foundation.NSHTTPURLResponse
import platform.Foundation.NSMutableURLRequest
import platform.Foundation.NSURL
import platform.Foundation.NSURLRequestReloadIgnoringLocalCacheData
import platform.Foundation.NSURLSession
import platform.Foundation.dataTaskWithRequest
import platform.Foundation.setValue
import platform.posix.memcpy
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * [HttpTransport] on `NSURLSession`. Each request (one Range chunk) is buffered in memory, so keep
 * [ModelDownloader]'s chunk size moderate (the default 8 MiB). Cancelling the coroutine cancels the task.
 */
@OptIn(ExperimentalForeignApi::class)
public class NSURLSessionHttpTransport(
    private val session: NSURLSession = NSURLSession.sharedSession,
    private val timeoutSeconds: Double = 30.0,
) : HttpTransport {
    private class Result(val status: Int, val response: NSHTTPURLResponse, val body: ByteArray)

    override suspend fun <T> get(url: String, headers: Map<String, String>, handle: suspend (HttpResponse) -> T): T {
        val nsUrl = NSURL.URLWithString(url) ?: throw DownloadProtocolException("invalid URL $url")
        val request = NSMutableURLRequest(
            uRL = nsUrl,
            // Never serve a Range chunk from the URL cache; GET is the default method.
            cachePolicy = NSURLRequestReloadIgnoringLocalCacheData,
            timeoutInterval = timeoutSeconds,
        ).apply {
            setValue("identity", forHTTPHeaderField = "Accept-Encoding")
            headers.forEach { (k, v) -> setValue(v, forHTTPHeaderField = k) }
        }
        val result = suspendCancellableCoroutine<Result> { cont ->
            val task = session.dataTaskWithRequest(request) { data, response, error ->
                when {
                    error != null -> cont.resumeWithException(
                        DownloadTransportException("request to $url failed: ${error.localizedDescription}"),
                    )
                    response !is NSHTTPURLResponse -> cont.resumeWithException(
                        DownloadTransportException("no HTTP response from $url"),
                    )
                    else -> cont.resume(Result(response.statusCode.toInt(), response, data.toByteArray()))
                }
            }
            cont.invokeOnCancellation { task.cancel() }
            task.resume()
        }
        var offset = 0
        return handle(
            object : HttpResponse {
                override val statusCode: Int = result.status

                override fun header(name: String): String? = result.response.valueForHTTPHeaderField(name)

                override suspend fun read(buffer: ByteArray): Int {
                    if (offset >= result.body.size) return -1
                    val n = minOf(buffer.size, result.body.size - offset)
                    result.body.copyInto(buffer, 0, offset, offset + n)
                    offset += n
                    return n
                }
            },
        )
    }

    private fun NSData?.toByteArray(): ByteArray {
        val data = this ?: return ByteArray(0)
        val length = data.length.toInt()
        if (length == 0) return ByteArray(0)
        return ByteArray(length).also { bytes ->
            bytes.usePinned { memcpy(it.addressOf(0), data.bytes, data.length) }
        }
    }
}
