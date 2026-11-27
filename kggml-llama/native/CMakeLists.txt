# Kggml llama.cpp + libmtmd engine (:kggml-llama native layer).
#
# Builds ONE shared library, kggml_llama (KggmlLlama.framework on iOS), from the pinned llama.cpp sources
# (git submodule third_party/llama.cpp: src/ = libllama, tools/mtmd/ = libmtmd) plus the Kggml C ABI wrapper
# (include/kggml_llama.h): model / context lifecycle, backend selection, abort plumbing, KV-cache state
# serialization, GBNF grammar samplers, embeddings, in-memory multimodal (libmtmd) evaluation and FFI error mapping.
#
# ggml is NOT built here. The library links the shared ggml of :kggml-runtime dynamically, found through the
# runtime's CMake package (build tree or install prefix):
#
#   cmake -S kggml-llama/native -B build/native-llama -Dkggml-runtime_DIR=<repo>/build/native
#
# llama.cpp's vendored ggml/ copy, its common/ helpers and its own CMakeLists.txt are never used, so the binary
# cannot contain a second set of ggml symbols (audited by scripts/ci/audit-llama-linkage.sh).

cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../kggml-runtime/native/cmake/kggml-version.cmake") # KGGML_VERSION

project(kggml_llama
    VERSION ${KGGML_VERSION}
    DESCRIPTION "Kggml llama.cpp + libmtmd engine"
    LANGUAGES C CXX)

# Host processes may load msvcp140 < 14.40 (e.g. a JVM's bundled copy); see kggml-runtime/native (issue #65).
if (MSVC)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# Exact upstream llama.cpp pin (release tag b8598, "sync : ggml" = ggml v0.9.9, the :kggml-runtime pin).
# libmtmd is built from the same commit (tools/mtmd). Keep in sync with the submodule gitlink and NOTICE.
set(KGGML_LLAMA_UPSTREAM_COMMIT "9281dd135d6f175ceacca7d3a3ec440012264cfe")
set(KGGML_LLAMA_UPSTREAM_VERSION "b8598")
set(KGGML_LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")

if (NOT EXISTS "${KGGML_LLAMA_SOURCE_DIR}/src/llama.cpp" OR NOT EXISTS "${KGGML_LLAMA_SOURCE_DIR}/tools/mtmd/mtmd.cpp")
    message(FATAL_ERROR
        "llama.cpp sources not found at ${KGGML_LLAMA_SOURCE_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

# ---------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------

include(CMakeDependentOption)

option(KGGML_LLAMA_BUILD_TESTS "kggml-llama: build the GoogleTest/CTest suite" OFF)
cmake_dependent_option(KGGML_LLAMA_TEST_HOOKS
    "kggml-llama: compile test hooks into kggml_llama (never ship)" ON
    "KGGML_LLAMA_BUILD_TESTS" OFF)

# ---------------------------------------------------------------------------------------------------------
# Global build settings (mirror :kggml-runtime)
# ---------------------------------------------------------------------------------------------------------

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

if (ANDROID)
    set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
    add_link_options("LINKER:-z,max-page-size=16384")
endif()

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../${CMAKE_INSTALL_LIBDIR};@loader_path/Frameworks;@executable_path/Frameworks")
elseif (UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

# ---------------------------------------------------------------------------------------------------------
# :kggml-runtime (shared ggml) - consumed, never built here
# ---------------------------------------------------------------------------------------------------------

find_package(Threads REQUIRED)
find_package(kggml-runtime ${KGGML_VERSION} CONFIG REQUIRED)

if (NOT TARGET kggml::runtime OR NOT TARGET kggml::ggml)
    message(FATAL_ERROR "kggml-runtime package does not provide kggml::runtime / kggml::ggml")
endif()
foreach (_t ggml ggml-base ggml-cpu llama mtmd)
    if (TARGET ${_t})
        message(FATAL_ERROR "A local '${_t}' target exists: kggml-llama must link the runtime's shared ggml only "
                            "and compile llama.cpp / libmtmd into kggml_llama itself")
    endif()
endforeach()

if (NOT DEFINED KGGML_RUNTIME_MONOLITHIC)
    set(KGGML_RUNTIME_MONOLITHIC OFF)
endif()
message(STATUS "kggml-llama: using kggml-runtime from ${kggml-runtime_DIR} (monolithic=${KGGML_RUNTIME_MONOLITHIC}, "
               "ggml ${KGGML_GGML_PINNED_VERSION} @ ${KGGML_GGML_PINNED_COMMIT})")

# Warn when the checked-out submodule does not match the pin recorded above / in NOTICE.
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${KGGML_LLAMA_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${KGGML_LLAMA_SOURCE_DIR}"
        OUTPUT_VARIABLE _kggml_llama_head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (_kggml_llama_head AND NOT _kggml_llama_head STREQUAL KGGML_LLAMA_UPSTREAM_COMMIT)
        message(WARNING
            "third_party/llama.cpp is at ${_kggml_llama_head}, expected pinned ${KGGML_LLAMA_UPSTREAM_COMMIT}. "
            "Run: git submodule update --init --recursive")
    endif()
endif()

# ---------------------------------------------------------------------------------------------------------
# kggml_llama shared library = libllama + libmtmd + Kggml C ABI
# ---------------------------------------------------------------------------------------------------------

# libllama: every translation unit of upstream src/CMakeLists.txt (src/*.cpp + src/models/*.cpp; verified equal
# to the upstream list at the pinned commit). libmtmd: the library sources of upstream tools/mtmd/CMakeLists.txt
# (the CLI / debug executables are not built).
file(GLOB _kggml_llama_src CONFIGURE_DEPENDS
    "${KGGML_LLAMA_SOURCE_DIR}/src/*.cpp"
    "${KGGML_LLAMA_SOURCE_DIR}/src/models/*.cpp")
set(_kggml_mtmd_dir "${KGGML_LLAMA_SOURCE_DIR}/tools/mtmd")
file(GLOB _kggml_mtmd_models CONFIGURE_DEPENDS "${_kggml_mtmd_dir}/models/*.cpp")
set(_kggml_mtmd_src
    "${_kggml_mtmd_dir}/mtmd.cpp"
    "${_kggml_mtmd_dir}/mtmd-audio.cpp"
    "${_kggml_mtmd_dir}/mtmd-image.cpp"
    "${_kggml_mtmd_dir}/mtmd-helper.cpp"
    "${_kggml_mtmd_dir}/clip.cpp"
    ${_kggml_mtmd_models})

set(_kggml_own_sources
    src/kggml_llama.cpp
    src/mtmd_bridge.cpp
    src/errors.cpp)

add_library(kggml_llama SHARED ${_kggml_own_sources} ${_kggml_llama_src} ${_kggml_mtmd_src})
add_library(kggml::llama ALIAS kggml_llama)

target_include_directories(kggml_llama
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${KGGML_LLAMA_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${_kggml_mtmd_dir}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${KGGML_LLAMA_SOURCE_DIR}/src
        ${KGGML_LLAMA_SOURCE_DIR}
        ${KGGML_LLAMA_SOURCE_DIR}/vendor)

target_compile_definitions(kggml_llama
    PRIVATE
        KGGML_LLAMA_BUILD
        LLAMA_BUILD
        KGGML_LLAMA_VERSION_STRING="${PROJECT_VERSION}"
        KGGML_LLAMA_UPSTREAM_VERSION="${KGGML_LLAMA_UPSTREAM_VERSION}"
        KGGML_LLAMA_UPSTREAM_COMMIT="${KGGML_LLAMA_UPSTREAM_COMMIT}"
        $<$<BOOL:${WIN32}>:_CRT_SECURE_NO_WARNINGS>
        $<$<BOOL:${KGGML_LLAMA_TEST_HOOKS}>:KGGML_LLAMA_TEST_HOOKS=1>
    PUBLIC
        KGGML_LLAMA_SHARED
        LLAMA_SHARED)

# Exported: the kggml_llama_* C ABI and llama.cpp's own LLAMA_API / MTMD_API. Everything else stays hidden.
set_target_properties(kggml_llama PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    EXPORT_NAME llama)

if (NOT ANDROID)
    set_target_properties(kggml_llama PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

# Warnings for Kggml's own sources only; upstream llama.cpp is compiled with its defaults.
if (MSVC)
    target_compile_options(kggml_llama PRIVATE /EHsc /utf-8 /bigobj)
    set_source_files_properties(${_kggml_own_sources} PROPERTIES COMPILE_OPTIONS "/W4")
else()
    set_source_files_properties(${_kggml_own_sources} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Wpedantic")
    # Upstream sets this for stb_image.h / miniaudio.h (tools/mtmd/CMakeLists.txt).
    set_source_files_properties(${_kggml_mtmd_src} PROPERTIES COMPILE_OPTIONS "-Wno-cast-qual")
endif()

# Dynamic linkage only: NEEDED / LC_LOAD_DYLIB on libggml(+base, cpu, ...) and libkggml_runtime, or on
# GgmlRuntime.framework in the monolithic iOS layout.
target_link_libraries(kggml_llama
    PUBLIC
        kggml::runtime
        kggml::ggml
    PRIVATE
        Threads::Threads)

if (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set_target_properties(kggml_llama PROPERTIES
        OUTPUT_NAME KggmlLlama
        FRAMEWORK TRUE
        MACOSX_FRAMEWORK_IDENTIFIER com.donfuldauer.kggml.llama
        MACOSX_FRAMEWORK_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${PROJECT_VERSION}
        PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/kggml_llama.h"
        XCODE_ATTRIBUTE_SKIP_INSTALL NO
        XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION YES
        XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/Frameworks @loader_path/Frameworks @loader_path/..")
endif()

# ---------------------------------------------------------------------------------------------------------
# Install + exported CMake package (kggml-llama-config.cmake -> kggml::llama)
# ---------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)
set(KGGML_LLAMA_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/kggml-llama")

install(TARGETS kggml_llama
    EXPORT kggmlLlamaTargets
    RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES
    include/kggml_llama.h
    "${KGGML_LLAMA_SOURCE_DIR}/include/llama.h"
    "${_kggml_mtmd_dir}/mtmd.h"
    "${_kggml_mtmd_dir}/mtmd-helper.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT kggmlLlamaTargets
    NAMESPACE kggml::
    FILE kggml-llama-targets.cmake
    DESTINATION ${KGGML_LLAMA_CMAKE_DIR})
configure_package_config_file(
    cmake/kggml-llama-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-llama-config.cmake"
    INSTALL_DESTINATION ${KGGML_LLAMA_CMAKE_DIR})
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-llama-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-llama-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-llama-config-version.cmake"
    DESTINATION ${KGGML_LLAMA_CMAKE_DIR})
export(EXPORT kggmlLlamaTargets
    NAMESPACE kggml::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/kggml-llama-targets.cmake")

# ---------------------------------------------------------------------------------------------------------
# kggml_llama_jni: JNI bridge for the Kotlin LlamaEngine on Android + JVM Desktop (not used on iOS, where
# Kotlin/Native calls the C ABI directly through cinterop). Links kggml_llama dynamically.
# ---------------------------------------------------------------------------------------------------------

set(KGGML_LLAMA_JNI_INCLUDE_DIRS "")
if (ANDROID)
    set(_kggml_llama_jni_default ON) # jni.h ships in the NDK sysroot
elseif (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_kggml_llama_jni_default OFF)
else()
    # Only the headers are needed (the JVM provides the symbols at load time).
    find_package(JNI QUIET)
    if (JAVA_INCLUDE_PATH AND JAVA_INCLUDE_PATH2)
        set(_kggml_llama_jni_default ON)
        set(KGGML_LLAMA_JNI_INCLUDE_DIRS "${JAVA_INCLUDE_PATH}" "${JAVA_INCLUDE_PATH2}")
    else()
        set(_kggml_llama_jni_default OFF)
    endif()
endif()
option(KGGML_LLAMA_BUILD_JNI "kggml-llama: build the kggml_llama_jni JNI bridge (Android / JVM Desktop)"
    ${_kggml_llama_jni_default})

if (KGGML_LLAMA_BUILD_JNI)
    if (NOT ANDROID AND NOT KGGML_LLAMA_JNI_INCLUDE_DIRS)
        message(FATAL_ERROR "KGGML_LLAMA_BUILD_JNI=ON but JNI headers were not found (set JAVA_HOME)")
    endif()
    add_library(kggml_llama_jni SHARED jni/kggml_llama_jni.cpp)
    target_include_directories(kggml_llama_jni PRIVATE ${KGGML_LLAMA_JNI_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(kggml_llama_jni PRIVATE kggml_llama kggml::runtime)
    target_compile_definitions(kggml_llama_jni PRIVATE
        $<$<BOOL:${KGGML_LLAMA_TEST_HOOKS}>:KGGML_LLAMA_TEST_HOOKS=1>)
    # Exports only the Java_* entry points. Unversioned (loaded by file name from Java), like on Android.
    set_target_properties(kggml_llama_jni PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    if (NOT APPLE AND NOT WIN32)
        # ELF: hidden visibility does not cover std:: template instantiations (e.g. std::vector<int>), so an explicit
        # version script keeps every non-JNI symbol local.
        set(_kggml_jni_map "${CMAKE_CURRENT_SOURCE_DIR}/jni/kggml_llama_jni.map")
        target_link_options(kggml_llama_jni PRIVATE "LINKER:--version-script=${_kggml_jni_map}")
        set_property(TARGET kggml_llama_jni APPEND PROPERTY LINK_DEPENDS "${_kggml_jni_map}")
    endif()
    if (MSVC)
        target_compile_options(kggml_llama_jni PRIVATE /W4 /EHsc)
    else()
        target_compile_options(kggml_llama_jni PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    # Installed next to kggml_llama but not part of the exported CMake package (private to the Kotlin bindings).
    install(TARGETS kggml_llama_jni
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# ---------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------

if (KGGML_LLAMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "kggml-llama: ${PROJECT_VERSION}, llama.cpp ${KGGML_LLAMA_UPSTREAM_VERSION} @ ${KGGML_LLAMA_UPSTREAM_COMMIT}")
message(STATUS "kggml-llama: jni=${KGGML_LLAMA_BUILD_JNI} tests=${KGGML_LLAMA_BUILD_TESTS} hooks=${KGGML_LLAMA_TEST_HOOKS}")
