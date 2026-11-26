package com.donfuldauer.kggml.runtime

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.lang.reflect.InvocationTargetException
import java.net.URL
import java.net.URLConnection
import java.net.URLStreamHandler
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * Exercises [JvmNativeLibraryLoader.load]'s lookup order (explicit directory → libraries bundled in the jar →
 * `java.library.path`) without real native libraries.
 *
 * Each test loads a private copy of the loader class through [IsolatingClassLoader], so it starts from a fresh,
 * not-yet-loaded singleton (the shared one stays untouched for the native-backed tests in this JVM) and sees only
 * the `kggml/native/…` resources the test provides. The "libraries" are plain bytes, so the final `System.load`
 * fails; the JVM reports that failure as `<canonical path>: <reason>`, which tells the test exactly which file
 * the loader tried to load first.
 */
class JvmNativeLibraryLoaderLookupTest {
    private val platform = JvmNativeLibraryLoader.platformId()
    private val resourceDir = "kggml/native/$platform"
    private val tempDir: File = Files.createTempDirectory("kggml-lookup-test").toFile()
    private val extractedDirs = mutableListOf<File>()
    private var savedPathProperty: String? = null

    @BeforeTest
    fun isolateFromConfiguredPaths() {
        assumeTrue(System.getenv(JvmNativeLibraryLoader.PATH_ENV) == null, "KGGML_NATIVE_PATH is set in the environment")
        savedPathProperty = System.getProperty(JvmNativeLibraryLoader.PATH_PROPERTY)
        System.clearProperty(JvmNativeLibraryLoader.PATH_PROPERTY)
    }

    @AfterTest
    fun restore() {
        val saved = savedPathProperty
        if (saved == null) {
            System.clearProperty(JvmNativeLibraryLoader.PATH_PROPERTY)
        } else {
            System.setProperty(JvmNativeLibraryLoader.PATH_PROPERTY, saved)
        }
        tempDir.deleteRecursively()
        extractedDirs.forEach { it.deleteRecursively() }
    }

    // ---- bundled (jar) libraries ------------------------------------------------------------------------

    @Test
    fun bundledLibrariesAreAllExtractedIntoOnePrivateDirectoryAndLoadedInIndexOrder() {
        val first = System.mapLibraryName("ggml-base")
        val second = System.mapLibraryName("kggml_runtime_jni")
        val firstBytes = "not a real library: ggml-base".encodeToByteArray()
        val secondBytes = "not a real library: jni".encodeToByteArray()
        val loader = isolatedLoader(
            "$resourceDir/index.txt" to bytes("# load order\n$first\n\n$second\n".encodeToByteArray()),
            "$resourceDir/$first" to bytes(firstBytes),
            "$resourceDir/$second" to bytes(secondBytes),
        )

        val e = assertFailsWith<UnsatisfiedLinkError> { loader.load() }

        // The JVM names the library it failed to load: the first index entry, inside a fresh kggml-native-* dir.
        val attempted = File(e.message!!.substringBefore(": "))
        val dir = attempted.parentFile.also(extractedDirs::add)
        assertEquals(first, attempted.name, e.message)
        assertTrue(dir.name.startsWith("kggml-native-"), dir.path)
        assertEquals(File(System.getProperty("java.io.tmpdir")).canonicalFile, dir.canonicalFile.parentFile)
        // Every listed library is extracted next to the others before the first one is loaded, so sibling
        // ($ORIGIN / @loader_path) lookups between them resolve.
        assertEquals(setOf(first, second), dir.list()!!.toSet())
        assertContentEquals(firstBytes, File(dir, first).readBytes())
        assertContentEquals(secondBytes, File(dir, second).readBytes())
        assertNull(loader.origin(), "a failed load must not be recorded as loaded")
    }

    @Test
    fun failedLoadIsRetriedOnTheNextCall() {
        val lib = System.mapLibraryName("ggml-base")
        val loader = isolatedLoader(
            "$resourceDir/index.txt" to bytes("$lib\n".encodeToByteArray()),
            "$resourceDir/$lib" to bytes("garbage".encodeToByteArray()),
        )
        val firstAttempt = File(assertFailsWith<UnsatisfiedLinkError> { loader.load() }.message!!.substringBefore(": "))
        val secondAttempt = File(assertFailsWith<UnsatisfiedLinkError> { loader.load() }.message!!.substringBefore(": "))
        extractedDirs += firstAttempt.parentFile
        extractedDirs += secondAttempt.parentFile
        // Each attempt extracts into its own fresh private directory.
        assertTrue(firstAttempt.parentFile != secondAttempt.parentFile, "$firstAttempt vs $secondAttempt")
        assertEquals(lib, secondAttempt.name)
    }

    @Test
    fun indexWithoutEntriesFails() {
        val loader = isolatedLoader("$resourceDir/index.txt" to bytes("# nothing bundled\n\n   \n".encodeToByteArray()))
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.load() }
        assertEquals("Empty Kggml native index for platform '$platform'", e.message)
    }

    @Test
    fun libraryListedInTheIndexButNotBundledFailsBeforeAnythingIsLoaded() {
        val present = System.mapLibraryName("ggml-base")
        val missing = System.mapLibraryName("ggml")
        val loader = isolatedLoader(
            "$resourceDir/index.txt" to bytes("$present\n$missing\n".encodeToByteArray()),
            "$resourceDir/$present" to bytes("garbage".encodeToByteArray()),
        )
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.load() }
        // Not a load failure of `present`: extraction of the whole index happens before any System.load.
        assertEquals("Bundled Kggml native library '$missing' missing for '$platform'", e.message)
    }

    @Test
    fun readFailureWhileExtractingIsReportedWithItsCause() {
        val lib = System.mapLibraryName("ggml-base")
        val ioFailure = IOException("disk full")
        val loader = isolatedLoader(
            "$resourceDir/index.txt" to bytes("$lib\n".encodeToByteArray()),
            "$resourceDir/$lib" to { throwingStream(ioFailure) },
        )
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.load() }
        assertEquals("Failed to extract '$lib': disk full", e.message)
        assertSame(ioFailure, e.cause)
    }

    @Test
    fun indexIsLookedUpUnderTheHostPlatformDirectoryOnly() {
        // A bundle for another platform is ignored, so the loader falls through to java.library.path.
        val other = if (platform == "linux-x86_64") "windows-x86_64" else "linux-x86_64"
        val loader = isolatedLoader("kggml/native/$other/index.txt" to bytes("# empty\n".encodeToByteArray()))
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.load() }
        assertTrue(e.message!!.startsWith("no ${JvmNativeLibraryLoader.JNI_LIBRARY} in java.library.path"), e.message)
    }

    // ---- java.library.path fallback -----------------------------------------------------------------------

    @Test
    fun withoutDirectoryOrBundleTheJniBridgeIsLoadedFromJavaLibraryPath() {
        val loader = isolatedLoader()
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.load() }
        assertTrue(e.message!!.startsWith("no ${JvmNativeLibraryLoader.JNI_LIBRARY} in java.library.path"), e.message)
        assertNull(loader.origin())
    }

    // ---- explicit directory -------------------------------------------------------------------------------

    @Test
    fun explicitDirectoryTakesPrecedenceOverBundledLibraries() {
        val missingDir = File(tempDir, "does-not-exist")
        System.setProperty(JvmNativeLibraryLoader.PATH_PROPERTY, missingDir.path)
        val loader = isolatedLoader("$resourceDir/index.txt" to bytes("# empty\n".encodeToByteArray()))
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.load() }
        assertEquals("kggml.native.path points to '$missingDir', which is not a directory", e.message)
    }

    @Test
    fun blankExplicitDirectoryIsIgnored() {
        System.setProperty(JvmNativeLibraryLoader.PATH_PROPERTY, "   ")
        val loader = isolatedLoader("$resourceDir/index.txt" to bytes("# empty\n".encodeToByteArray()))
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.load() }
        assertEquals("Empty Kggml native index for platform '$platform'", e.message)
    }

    @Test
    fun explicitDirectoryLoadsLibrariesInDependencyOrder() {
        for (name in listOf("kggml_runtime_jni", "kggml_runtime", "ggml", "ggml-cpu", "ggml-base")) {
            File(tempDir, System.mapLibraryName(name)).writeText("not a real library: $name")
        }
        System.setProperty(JvmNativeLibraryLoader.PATH_PROPERTY, tempDir.path)
        val e = assertFailsWith<UnsatisfiedLinkError> { isolatedLoader().load() }
        // ggml-base has no Kggml dependencies, so it is the first library handed to System.load.
        assertEquals(File(tempDir, System.mapLibraryName("ggml-base")).canonicalPath, e.message!!.substringBefore(": "))
    }

    // ---- helpers ------------------------------------------------------------------------------------------

    private fun bytes(content: ByteArray): () -> InputStream = { content.inputStream() }

    private fun throwingStream(failure: IOException): InputStream = object : InputStream() {
        override fun read(): Int = throw failure
        override fun read(b: ByteArray, off: Int, len: Int): Int = throw failure
    }

    private fun isolatedLoader(vararg resources: Pair<String, () -> InputStream>): IsolatedLoader =
        IsolatedLoader(IsolatingClassLoader(resources.toMap()))

    /** Reflective handle on the private copy of [JvmNativeLibraryLoader] defined by [classLoader]. */
    private class IsolatedLoader(classLoader: IsolatingClassLoader) {
        private val type = Class.forName(LOADER_CLASS, true, classLoader)
        private val instance = type.getField("INSTANCE").get(null)

        init {
            check(type.classLoader === classLoader && type != JvmNativeLibraryLoader::class.java)
        }

        fun load() {
            try {
                type.getMethod("load").invoke(instance)
            } catch (e: InvocationTargetException) {
                throw e.targetException
            }
        }

        fun origin(): String? = type.getMethod("getOrigin").invoke(instance) as String?
    }

    /**
     * Defines its own copy of the loader class (child-first) and serves `kggml/native/…` resources only from
     * [resources], never from the test classpath (which may carry real bundled libraries with
     * `-Pkggml.bundleNative=true`).
     */
    private class IsolatingClassLoader(
        private val resources: Map<String, () -> InputStream>,
    ) : ClassLoader(JvmNativeLibraryLoaderLookupTest::class.java.classLoader) {
        override fun loadClass(name: String, resolve: Boolean): Class<*> {
            if (name != LOADER_CLASS && !name.startsWith("$LOADER_CLASS$")) return super.loadClass(name, resolve)
            synchronized(getClassLoadingLock(name)) {
                findLoadedClass(name)?.let { return it }
                val bytes = parent.getResourceAsStream(name.replace('.', '/') + ".class")!!.use { it.readBytes() }
                return defineClass(name, bytes, 0, bytes.size)
            }
        }

        override fun getResource(name: String): URL? {
            if (!name.startsWith("kggml/native/")) return super.getResource(name)
            val open = resources[name] ?: return null
            val handler = object : URLStreamHandler() {
                override fun openConnection(u: URL): URLConnection = object : URLConnection(u) {
                    override fun connect() = Unit
                    override fun getInputStream(): InputStream = open()
                }
            }
            @Suppress("DEPRECATION")
            return URL(null, "kggml-test:/$name", handler)
        }
    }

    private companion object {
        val LOADER_CLASS: String = JvmNativeLibraryLoader::class.java.name
    }
}
