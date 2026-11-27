# Kggml whisper.cpp engine (:kggml-whisper native layer).
#
# Builds ONE shared library, kggml_whisper (KggmlWhisper.framework on iOS), from the pinned whisper.cpp
# sources (git submodule third_party/whisper.cpp) plus the Kggml C ABI wrapper (include/kggml_whisper.h):
# PCM resampler / downmixer, backend toggle (use_gpu), CoreML gate, abort plumbing and FFI error mapping.
#
# ggml is NOT built here. The library links the shared ggml of :kggml-runtime dynamically, found through the
# runtime's CMake package (build tree or install prefix):
#
#   cmake -S kggml-whisper/native -B build/native-whisper -Dkggml-runtime_DIR=<repo>/build/native
#
# whisper.cpp's vendored ggml/ copy and its own CMakeLists.txt are never used, so the binary cannot contain
# a second set of ggml symbols (audited by scripts/ci/audit-whisper-linkage.sh).

cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../kggml-runtime/native/cmake/kggml-version.cmake") # KGGML_VERSION

project(kggml_whisper
    VERSION ${KGGML_VERSION}
    DESCRIPTION "Kggml whisper.cpp engine"
    LANGUAGES C CXX)

# Host processes may load msvcp140 < 14.40 (e.g. a JVM's bundled copy); see kggml-runtime/native (issue #65).
if (MSVC)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# Exact upstream whisper.cpp pin (tag v1.8.4). Keep in sync with the submodule gitlink and NOTICE.
set(KGGML_WHISPER_UPSTREAM_COMMIT "9386f239401074690479731c1e41683fbbeac557")
set(KGGML_WHISPER_UPSTREAM_VERSION "1.8.4")
set(KGGML_WHISPER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp")

if (NOT EXISTS "${KGGML_WHISPER_SOURCE_DIR}/src/whisper.cpp")
    message(FATAL_ERROR
        "whisper.cpp sources not found at ${KGGML_WHISPER_SOURCE_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

# ---------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------

include(CMakeDependentOption)

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_kggml_whisper_coreml_default ON)
else()
    set(_kggml_whisper_coreml_default OFF)
endif()

option(KGGML_WHISPER_BUILD_TESTS "kggml-whisper: build the GoogleTest/CTest suite" OFF)
cmake_dependent_option(KGGML_WHISPER_TEST_HOOKS
    "kggml-whisper: compile test hooks into kggml_whisper (never ship)" ON
    "KGGML_WHISPER_BUILD_TESTS" OFF)
option(KGGML_WHISPER_COREML
    "kggml-whisper: compile the CoreML encoder (WHISPER_COREML, Apple only, runtime-gated per context)"
    ${_kggml_whisper_coreml_default})

if (KGGML_WHISPER_COREML AND NOT APPLE)
    message(WARNING "KGGML_WHISPER_COREML is only supported on Apple platforms; disabling")
    set(KGGML_WHISPER_COREML OFF CACHE BOOL "" FORCE)
endif()
if (KGGML_WHISPER_COREML)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

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
foreach (_t ggml ggml-base ggml-cpu)
    if (TARGET ${_t})
        message(FATAL_ERROR "A local '${_t}' target exists: kggml-whisper must link the runtime's shared ggml only")
    endif()
endforeach()

# Monolithic runtime (iOS GgmlRuntime.framework): kggml::ggml is an alias of kggml::runtime.
if (NOT DEFINED KGGML_RUNTIME_MONOLITHIC)
    set(KGGML_RUNTIME_MONOLITHIC OFF)
endif()
message(STATUS "kggml-whisper: using kggml-runtime from ${kggml-runtime_DIR} (monolithic=${KGGML_RUNTIME_MONOLITHIC}, "
               "ggml ${KGGML_GGML_PINNED_VERSION} @ ${KGGML_GGML_PINNED_COMMIT})")

# Warn when the checked-out submodule does not match the pin recorded above / in NOTICE.
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${KGGML_WHISPER_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${KGGML_WHISPER_SOURCE_DIR}"
        OUTPUT_VARIABLE _kggml_whisper_head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (_kggml_whisper_head AND NOT _kggml_whisper_head STREQUAL KGGML_WHISPER_UPSTREAM_COMMIT)
        message(WARNING
            "third_party/whisper.cpp is at ${_kggml_whisper_head}, expected pinned ${KGGML_WHISPER_UPSTREAM_COMMIT}. "
            "Run: git submodule update --init --recursive")
    endif()
endif()

# ---------------------------------------------------------------------------------------------------------
# kggml_whisper shared library = whisper.cpp + Kggml C ABI
# ---------------------------------------------------------------------------------------------------------

set(KGGML_WHISPER_SOURCES
    src/kggml_whisper.cpp
    src/pcm.cpp
    src/resampler.cpp
    src/errors.cpp
    src/coreml_gate.cpp
    "${KGGML_WHISPER_SOURCE_DIR}/src/whisper.cpp")

if (KGGML_WHISPER_COREML)
    set(_kggml_coreml_dir "${KGGML_WHISPER_SOURCE_DIR}/src/coreml")
    list(APPEND KGGML_WHISPER_SOURCES
        "${_kggml_coreml_dir}/whisper-compat.m"
        "${_kggml_coreml_dir}/whisper-encoder.mm"
        "${_kggml_coreml_dir}/whisper-encoder-impl.m")
    set_source_files_properties(
        "${_kggml_coreml_dir}/whisper-compat.m"
        "${_kggml_coreml_dir}/whisper-encoder.mm"
        "${_kggml_coreml_dir}/whisper-encoder-impl.m"
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    # The upstream loader is renamed; src/coreml_gate.cpp provides whisper_coreml_init() as a runtime gate.
    set_source_files_properties("${_kggml_coreml_dir}/whisper-encoder.mm"
        PROPERTIES COMPILE_DEFINITIONS "whisper_coreml_init=kggml_upstream_whisper_coreml_init")
endif()

add_library(kggml_whisper SHARED ${KGGML_WHISPER_SOURCES})
add_library(kggml::whisper ALIAS kggml_whisper)

target_include_directories(kggml_whisper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${KGGML_WHISPER_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${KGGML_WHISPER_SOURCE_DIR}/src)

target_compile_definitions(kggml_whisper
    PRIVATE
        KGGML_WHISPER_BUILD
        WHISPER_BUILD
        KGGML_WHISPER_VERSION_STRING="${PROJECT_VERSION}"
        KGGML_WHISPER_UPSTREAM_VERSION="${KGGML_WHISPER_UPSTREAM_VERSION}"
        KGGML_WHISPER_UPSTREAM_COMMIT="${KGGML_WHISPER_UPSTREAM_COMMIT}"
        WHISPER_VERSION="${KGGML_WHISPER_UPSTREAM_VERSION}"
        $<$<BOOL:${WIN32}>:_CRT_SECURE_NO_WARNINGS>
        $<$<BOOL:${KGGML_WHISPER_COREML}>:WHISPER_USE_COREML>
        $<$<BOOL:${KGGML_WHISPER_COREML}>:WHISPER_COREML_ALLOW_FALLBACK>
        $<$<BOOL:${KGGML_WHISPER_COREML}>:KGGML_WHISPER_COREML=1>
        $<$<BOOL:${KGGML_WHISPER_TEST_HOOKS}>:KGGML_WHISPER_TEST_HOOKS=1>
    PUBLIC
        KGGML_WHISPER_SHARED
        WHISPER_SHARED)

# Exported: the kggml_whisper_* C ABI and whisper.cpp's own WHISPER_API. Everything else stays hidden.
set_target_properties(kggml_whisper PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    OBJC_VISIBILITY_PRESET hidden
    OBJCXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    EXPORT_NAME whisper)

if (NOT ANDROID)
    set_target_properties(kggml_whisper PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

# Warnings for Kggml's own sources only; upstream whisper.cpp is compiled with its defaults.
set(_kggml_own_sources src/kggml_whisper.cpp src/pcm.cpp src/resampler.cpp src/errors.cpp src/coreml_gate.cpp)
if (MSVC)
    target_compile_options(kggml_whisper PRIVATE /EHsc /utf-8)
    set_source_files_properties(${_kggml_own_sources} PROPERTIES COMPILE_OPTIONS "/W4")
else()
    set_source_files_properties(${_kggml_own_sources} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Wpedantic")
endif()

# Dynamic linkage only: NEEDED / LC_LOAD_DYLIB on libggml(+base, cpu, ...) and libkggml_runtime, or on
# GgmlRuntime.framework in the monolithic iOS layout.
target_link_libraries(kggml_whisper
    PUBLIC
        kggml::runtime
    PRIVATE
        kggml::ggml
        Threads::Threads)

if (KGGML_WHISPER_COREML)
    find_library(KGGML_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(KGGML_COREML_FRAMEWORK CoreML REQUIRED)
    target_link_libraries(kggml_whisper PRIVATE ${KGGML_FOUNDATION_FRAMEWORK} ${KGGML_COREML_FRAMEWORK})
    set_target_properties(kggml_whisper PROPERTIES XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES)
endif()

if (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set_target_properties(kggml_whisper PROPERTIES
        OUTPUT_NAME KggmlWhisper
        FRAMEWORK TRUE
        MACOSX_FRAMEWORK_IDENTIFIER com.donfuldauer.kggml.whisper
        MACOSX_FRAMEWORK_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${PROJECT_VERSION}
        PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/kggml_whisper.h;${KGGML_WHISPER_SOURCE_DIR}/include/whisper.h"
        XCODE_ATTRIBUTE_SKIP_INSTALL NO
        XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION YES
        XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/Frameworks @loader_path/Frameworks @loader_path/..")
endif()

# ---------------------------------------------------------------------------------------------------------
# kggml_whisper_jni: JNI bridge for the Kotlin WhisperEngine on Android + JVM Desktop (not used on iOS, where
# Kotlin/Native calls the C ABI directly through cinterop). Links kggml_whisper dynamically.
# ---------------------------------------------------------------------------------------------------------

set(KGGML_WHISPER_JNI_INCLUDE_DIRS "")
if (ANDROID)
    set(_kggml_whisper_jni_default ON) # jni.h ships in the NDK sysroot
elseif (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_kggml_whisper_jni_default OFF)
else()
    # Only the headers are needed (the JVM provides the symbols at load time).
    find_package(JNI QUIET)
    if (JAVA_INCLUDE_PATH AND JAVA_INCLUDE_PATH2)
        set(_kggml_whisper_jni_default ON)
        set(KGGML_WHISPER_JNI_INCLUDE_DIRS "${JAVA_INCLUDE_PATH}" "${JAVA_INCLUDE_PATH2}")
    else()
        set(_kggml_whisper_jni_default OFF)
    endif()
endif()
option(KGGML_WHISPER_BUILD_JNI "kggml-whisper: build the kggml_whisper_jni JNI bridge (Android / JVM Desktop)"
    ${_kggml_whisper_jni_default})

if (KGGML_WHISPER_BUILD_JNI)
    if (NOT ANDROID AND NOT KGGML_WHISPER_JNI_INCLUDE_DIRS)
        message(FATAL_ERROR "KGGML_WHISPER_BUILD_JNI=ON but JNI headers were not found (set JAVA_HOME)")
    endif()
    add_library(kggml_whisper_jni SHARED jni/kggml_whisper_jni.cpp)
    target_include_directories(kggml_whisper_jni PRIVATE ${KGGML_WHISPER_JNI_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(kggml_whisper_jni PRIVATE kggml_whisper kggml::runtime)
    target_compile_definitions(kggml_whisper_jni PRIVATE
        $<$<BOOL:${KGGML_WHISPER_TEST_HOOKS}>:KGGML_WHISPER_TEST_HOOKS=1>)
    # Exports only the Java_* entry points. Unversioned (loaded by file name from Java), like on Android.
    set_target_properties(kggml_whisper_jni PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    if (NOT APPLE AND NOT WIN32)
        # ELF: hidden visibility does not cover std:: template instantiations (e.g. std::vector<int>), so an explicit
        # version script keeps every non-JNI symbol local.
        set(_kggml_jni_map "${CMAKE_CURRENT_SOURCE_DIR}/jni/kggml_whisper_jni.map")
        target_link_options(kggml_whisper_jni PRIVATE "LINKER:--version-script=${_kggml_jni_map}")
        set_property(TARGET kggml_whisper_jni APPEND PROPERTY LINK_DEPENDS "${_kggml_jni_map}")
    endif()
    if (MSVC)
        target_compile_options(kggml_whisper_jni PRIVATE /W4 /EHsc)
    else()
        target_compile_options(kggml_whisper_jni PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    # Installed next to kggml_whisper but not part of the exported CMake package (private to the Kotlin bindings).
    install(TARGETS kggml_whisper_jni
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# ---------------------------------------------------------------------------------------------------------
# Install + exported CMake package (kggml-whisper-config.cmake -> kggml::whisper)
# ---------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)
set(KGGML_WHISPER_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/kggml-whisper")

install(TARGETS kggml_whisper
    EXPORT kggmlWhisperTargets
    RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES include/kggml_whisper.h "${KGGML_WHISPER_SOURCE_DIR}/include/whisper.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT kggmlWhisperTargets
    NAMESPACE kggml::
    FILE kggml-whisper-targets.cmake
    DESTINATION ${KGGML_WHISPER_CMAKE_DIR})
configure_package_config_file(
    cmake/kggml-whisper-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-whisper-config.cmake"
    INSTALL_DESTINATION ${KGGML_WHISPER_CMAKE_DIR})
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-whisper-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-whisper-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-whisper-config-version.cmake"
    DESTINATION ${KGGML_WHISPER_CMAKE_DIR})
export(EXPORT kggmlWhisperTargets
    NAMESPACE kggml::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/kggml-whisper-targets.cmake")

# ---------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------

if (KGGML_WHISPER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "kggml-whisper: ${PROJECT_VERSION}, whisper.cpp ${KGGML_WHISPER_UPSTREAM_VERSION} @ ${KGGML_WHISPER_UPSTREAM_COMMIT}")
message(STATUS "kggml-whisper: coreml=${KGGML_WHISPER_COREML} jni=${KGGML_WHISPER_BUILD_JNI} tests=${KGGML_WHISPER_BUILD_TESTS} hooks=${KGGML_WHISPER_TEST_HOOKS}")
