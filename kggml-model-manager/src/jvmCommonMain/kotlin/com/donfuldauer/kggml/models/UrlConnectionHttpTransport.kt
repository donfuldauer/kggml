package com.donfuldauer.kggml.models

import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URI
import java.net.URISyntaxException

/**
 * [HttpTransport] on `java.net.HttpURLConnection` (no third-party HTTP stack). Blocking; [ModelDownloader] calls it
 * on [modelIoDispatcher]. Redirects are followed within the same protocol.
 */
public class UrlConnectionHttpTransport(
    private val connectTimeoutMillis: Int = 15_000,
    private val readTimeoutMillis: Int = 30_000,
) : HttpTransport {
    override suspend fun <T> get(url: String, headers: Map<String, String>, handle: suspend (HttpResponse) -> T): T {
        val connection = try {
            (URI(url).toURL().openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = connectTimeoutMillis
                readTimeout = readTimeoutMillis
                instanceFollowRedirects = true
                // Byte offsets must refer to the stored representation, not a gzip-decoded stream.
                setRequestProperty("Accept-Encoding", "identity")
                headers.forEach { (k, v) -> setRequestProperty(k, v) }
                connect()
            }
        } catch (e: IOException) {
            throw DownloadTransportException("cannot connect to $url: ${e.message}", e)
        } catch (e: URISyntaxException) {
            throw DownloadProtocolException("invalid URL $url: ${e.message}")
        } catch (e: IllegalArgumentException) {
            throw DownloadProtocolException("invalid URL $url: ${e.message}")
        }
        try {
            val status = try {
                connection.responseCode
            } catch (e: IOException) {
                throw DownloadTransportException("no response from $url: ${e.message}", e)
            }
            val stream: InputStream? = try {
                if (status >= 400) connection.errorStream else connection.inputStream
            } catch (e: IOException) {
                throw DownloadTransportException("cannot read $url: ${e.message}", e)
            }
            return stream.use { input ->
                handle(
                    object : HttpResponse {
                        override val statusCode: Int = status

                        override fun header(name: String): String? = connection.getHeaderField(name)

                        override suspend fun read(buffer: ByteArray): Int = try {
                            input?.read(buffer) ?: -1
                        } catch (e: IOException) {
                            throw DownloadTransportException("read from $url failed: ${e.message}", e)
                        }
                    },
                )
            }
        } finally {
            connection.disconnect()
        }
    }
}
