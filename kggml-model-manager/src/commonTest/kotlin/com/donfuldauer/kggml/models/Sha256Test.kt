package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class Sha256Test {
    @Test
    fun knownVectors() {
        assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Sha256.hashHex(ByteArray(0)))
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", Sha256.hashHex("abc".encodeToByteArray()))
        assertEquals(
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            Sha256.hashHex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".encodeToByteArray()),
        )
    }

    @Test
    fun millionAsInUnevenChunks() {
        val sha = Sha256()
        val chunk = ByteArray(997) { 'a'.code.toByte() }
        var left = 1_000_000
        while (left > 0) {
            val n = minOf(left, chunk.size)
            sha.update(chunk, 0, n)
            left -= n
        }
        assertEquals("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", sha.digestHex())
    }

    @Test
    fun paddingBoundariesMatchOneShot() {
        for (len in listOf(55, 56, 63, 64, 65, 119, 120, 128)) {
            val data = ByteArray(len) { (it * 31 + 7).toByte() }
            val incremental = Sha256()
            data.forEach { incremental.update(byteArrayOf(it)) }
            assertEquals(Sha256.hashHex(data), incremental.digestHex(), "length $len")
        }
        // Length 64 known vector: 64 x 'a'.
        assertEquals(
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            Sha256.hashHex(ByteArray(64) { 'a'.code.toByte() }),
        )
    }

    @Test
    fun misuseIsRejected() {
        val sha = Sha256()
        sha.digest()
        assertFailsWith<IllegalStateException> { sha.digest() }
        assertFailsWith<IllegalStateException> { sha.update(ByteArray(1)) }
        assertFailsWith<IllegalArgumentException> { Sha256().update(ByteArray(2), 1, 2) }
    }

    @Test
    fun hashFileReadsInChunks() {
        val fs = FakeFileSystem()
        val data = ByteArray(10_000) { it.toByte() }
        fs.files["/m.bin"] = data
        assertEquals(Sha256.hashHex(data), Sha256.hashFileHex(fs, "/m.bin", bufferSize = 333))
    }
}
