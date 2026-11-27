# Kggml stable-diffusion.cpp engine (:kggml-diffusion native layer).
#
# Builds ONE shared library, kggml_diffusion (KggmlDiffusion.framework on iOS), from the pinned stable-diffusion.cpp
# sources (git submodule third_party/stable-diffusion.cpp: src/, include/ and the vendored thirdparty/zip.c) plus the
# Kggml C ABI wrapper (include/kggml_diffusion.h): context lifecycle, backend selection, txt2img generation with a
# per-step progress callback and a mid-generation abort, and FFI error mapping.
#
# ggml is NOT built here. The library links the shared ggml of :kggml-runtime dynamically, found through the
# runtime's CMake package (build tree or install prefix):
#
#   cmake -S kggml-diffusion/native -B build/native-diffusion -Dkggml-runtime_DIR=<repo>/build/native
#
# stable-diffusion.cpp's own CMakeLists.txt, its vendored ggml/ submodule (never initialised), libwebp / libwebm and
# the examples / server are never used, so the binary cannot contain a second set of ggml symbols (audited by
# scripts/ci/audit-diffusion-linkage.sh).
#
# ggml ABI: stable-diffusion.cpp requires GGML_MAX_NAME >= 128 (static_assert in src/ggml_extend.hpp). The value is
# NOT set here: it is a PUBLIC compile definition (GGML_MAX_NAME=128) of the kggml::runtime / kggml::ggml targets,
# so this library, whisper, llama and the runtime's ggml all agree on struct ggml_tensor's layout. See README.md.

cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../kggml-runtime/native/cmake/kggml-version.cmake") # KGGML_VERSION

project(kggml_diffusion
    VERSION ${KGGML_VERSION}
    DESCRIPTION "Kggml stable-diffusion.cpp engine"
    LANGUAGES C CXX)

# Host processes may load msvcp140 < 14.40 (e.g. a JVM's bundled copy); see kggml-runtime/native (issue #65).
if (MSVC)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# Exact upstream stable-diffusion.cpp pin (no release tag; `git describe` = master-612-d7ecbe1-2-g3b4d26f). Its
# ggml submodule is upstream ggml 404fcb9d, two commits behind the :kggml-runtime pin (ggml v0.9.9, a04eea07),
# which is what makes the shared runtime ggml a drop-in. Keep in sync with the submodule gitlink and NOTICE.
set(KGGML_DIFFUSION_UPSTREAM_COMMIT "3b4d26f3d92c8db0199d7326f66260f4c18f4b41")
set(KGGML_DIFFUSION_UPSTREAM_VERSION "master-612-d7ecbe1-2-g3b4d26f")
set(KGGML_DIFFUSION_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp")

if (NOT EXISTS "${KGGML_DIFFUSION_SOURCE_DIR}/src/stable-diffusion.cpp"
        OR NOT EXISTS "${KGGML_DIFFUSION_SOURCE_DIR}/include/stable-diffusion.h")
    message(FATAL_ERROR
        "stable-diffusion.cpp sources not found at ${KGGML_DIFFUSION_SOURCE_DIR}.\n"
        "Run: git submodule update --init kggml-diffusion/native/third_party/stable-diffusion.cpp")
endif()

# ---------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------

include(CMakeDependentOption)

option(KGGML_DIFFUSION_BUILD_TESTS "kggml-diffusion: build the GoogleTest/CTest suite" OFF)
cmake_dependent_option(KGGML_DIFFUSION_TEST_HOOKS
    "kggml-diffusion: compile test hooks into kggml_diffusion (never ship)" ON
    "KGGML_DIFFUSION_BUILD_TESTS" OFF)

# ---------------------------------------------------------------------------------------------------------
# Global build settings (mirror :kggml-runtime)
# ---------------------------------------------------------------------------------------------------------

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
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
foreach (_t ggml ggml-base ggml-cpu stable-diffusion zip)
    if (TARGET ${_t})
        message(FATAL_ERROR "A local '${_t}' target exists: kggml-diffusion must link the runtime's shared ggml only "
                            "and compile stable-diffusion.cpp into kggml_diffusion itself")
    endif()
endforeach()

# stable-diffusion.cpp static_asserts GGML_MAX_NAME >= 128; the runtime must export a matching ABI value.
if (NOT DEFINED KGGML_GGML_MAX_NAME OR KGGML_GGML_MAX_NAME LESS 128)
    message(FATAL_ERROR
        "kggml-runtime at ${kggml-runtime_DIR} was built with GGML_MAX_NAME=${KGGML_GGML_MAX_NAME} (< 128 or unknown); "
        "stable-diffusion.cpp needs 128. Rebuild :kggml-runtime (kggml-runtime/native/scripts/build-host.sh --clean).")
endif()

if (NOT DEFINED KGGML_RUNTIME_MONOLITHIC)
    set(KGGML_RUNTIME_MONOLITHIC OFF)
endif()
message(STATUS "kggml-diffusion: using kggml-runtime from ${kggml-runtime_DIR} (monolithic=${KGGML_RUNTIME_MONOLITHIC}, "
               "ggml ${KGGML_GGML_PINNED_VERSION} @ ${KGGML_GGML_PINNED_COMMIT}, GGML_MAX_NAME=${KGGML_GGML_MAX_NAME})")

# Warn when the checked-out submodule does not match the pin recorded above / in NOTICE.
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${KGGML_DIFFUSION_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${KGGML_DIFFUSION_SOURCE_DIR}"
        OUTPUT_VARIABLE _kggml_sd_head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (_kggml_sd_head AND NOT _kggml_sd_head STREQUAL KGGML_DIFFUSION_UPSTREAM_COMMIT)
        message(WARNING
            "third_party/stable-diffusion.cpp is at ${_kggml_sd_head}, expected pinned ${KGGML_DIFFUSION_UPSTREAM_COMMIT}. "
            "Run: git submodule update --init kggml-diffusion/native/third_party/stable-diffusion.cpp")
    endif()
endif()

# ---------------------------------------------------------------------------------------------------------
# kggml_diffusion shared library = stable-diffusion.cpp + Kggml C ABI
# ---------------------------------------------------------------------------------------------------------

# Every translation unit of upstream's `stable-diffusion` library target (same GLOB as its CMakeLists.txt at the
# pinned commit) plus its `zip` object library (thirdparty/zip.c, which includes miniz.h). The vendored json.hpp
# and darts.h are header-only and compiled as part of these sources; stb_image* / httplib.h are only used by the
# upstream examples and are not compiled.
file(GLOB _kggml_sd_src CONFIGURE_DEPENDS
    "${KGGML_DIFFUSION_SOURCE_DIR}/src/*.cpp"
    "${KGGML_DIFFUSION_SOURCE_DIR}/src/model_io/*.cpp"
    "${KGGML_DIFFUSION_SOURCE_DIR}/src/tokenizers/*.cpp"
    "${KGGML_DIFFUSION_SOURCE_DIR}/src/tokenizers/vocab/*.cpp")
set(_kggml_sd_zip_src "${KGGML_DIFFUSION_SOURCE_DIR}/thirdparty/zip.c")

set(_kggml_own_sources
    src/kggml_diffusion.cpp
    src/errors.cpp)

# src/ggml_graph_cut.cpp does `#include "../ggml/src/ggml-impl.h"` (the vendored ggml submodule, never initialised
# here) for struct ggml_cgraph's internals. Those internals must be the ones of the ggml we actually link, so a
# generated shim resolves that include to :kggml-runtime's pinned ggml (same repository, same commit as the shared
# libggml-base). An include directory <shim>/include makes "../ggml/src/ggml-impl.h" -> <shim>/ggml/src/ggml-impl.h.
set(KGGML_DIFFUSION_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../kggml-runtime/native/third_party/ggml"
    CACHE PATH "kggml-diffusion: source tree of the :kggml-runtime ggml (for ggml-impl.h)")
get_filename_component(KGGML_DIFFUSION_GGML_SOURCE_DIR "${KGGML_DIFFUSION_GGML_SOURCE_DIR}" ABSOLUTE)
if (NOT EXISTS "${KGGML_DIFFUSION_GGML_SOURCE_DIR}/src/ggml-impl.h")
    message(FATAL_ERROR "ggml-impl.h not found under ${KGGML_DIFFUSION_GGML_SOURCE_DIR}/src (the :kggml-runtime ggml "
                        "submodule). Run: git submodule update --init kggml-runtime/native/third_party/ggml")
endif()
if (GIT_FOUND AND EXISTS "${KGGML_DIFFUSION_GGML_SOURCE_DIR}/.git" AND KGGML_GGML_PINNED_COMMIT)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${KGGML_DIFFUSION_GGML_SOURCE_DIR}"
        OUTPUT_VARIABLE _kggml_ggml_src_head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (_kggml_ggml_src_head AND NOT _kggml_ggml_src_head STREQUAL KGGML_GGML_PINNED_COMMIT)
        message(FATAL_ERROR "${KGGML_DIFFUSION_GGML_SOURCE_DIR} is at ${_kggml_ggml_src_head} but the linked "
                            "kggml-runtime was built from ${KGGML_GGML_PINNED_COMMIT} (ggml-impl.h must match)")
    endif()
endif()
set(_kggml_ggml_shim "${CMAKE_CURRENT_BINARY_DIR}/sd-ggml-shim")
file(MAKE_DIRECTORY "${_kggml_ggml_shim}/include")
file(CONFIGURE OUTPUT "${_kggml_ggml_shim}/ggml/src/ggml-impl.h" CONTENT
"// Generated by kggml-diffusion/native/CMakeLists.txt: stable-diffusion.cpp's \"../ggml/src/ggml-impl.h\" -> the
// :kggml-runtime ggml it links against.
#pragma once
#include \"${KGGML_DIFFUSION_GGML_SOURCE_DIR}/src/ggml-impl.h\"
")

add_library(kggml_diffusion SHARED ${_kggml_own_sources} ${_kggml_sd_src} ${_kggml_sd_zip_src})
add_library(kggml::diffusion ALIAS kggml_diffusion)

target_include_directories(kggml_diffusion
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${KGGML_DIFFUSION_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${KGGML_DIFFUSION_SOURCE_DIR}/src
        ${KGGML_DIFFUSION_SOURCE_DIR}
        ${KGGML_DIFFUSION_SOURCE_DIR}/thirdparty
        ${_kggml_ggml_shim}/include)

target_compile_definitions(kggml_diffusion
    PRIVATE
        KGGML_DIFFUSION_BUILD
        SD_BUILD_DLL
        KGGML_DIFFUSION_VERSION_STRING="${PROJECT_VERSION}"
        KGGML_DIFFUSION_UPSTREAM_VERSION="${KGGML_DIFFUSION_UPSTREAM_VERSION}"
        KGGML_DIFFUSION_UPSTREAM_COMMIT="${KGGML_DIFFUSION_UPSTREAM_COMMIT}"
        $<$<BOOL:${WIN32}>:_CRT_SECURE_NO_WARNINGS>
        $<$<BOOL:${WIN32}>:_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING>
        $<$<BOOL:${KGGML_DIFFUSION_TEST_HOOKS}>:KGGML_DIFFUSION_TEST_HOOKS=1>
    PUBLIC
        KGGML_DIFFUSION_SHARED
        SD_BUILD_SHARED_LIB) # SD_API = dllexport here (SD_BUILD_DLL), dllimport for consumers on Windows

# Upstream bakes `git describe` / short SHA into sd_version() / sd_commit(); use the recorded pin instead so the
# result does not depend on the checkout (shallow CI clones, source tarballs).
string(SUBSTRING "${KGGML_DIFFUSION_UPSTREAM_COMMIT}" 0 7 _kggml_sd_short_commit)
set_property(SOURCE "${KGGML_DIFFUSION_SOURCE_DIR}/src/version.cpp" APPEND PROPERTY COMPILE_DEFINITIONS
    SDCPP_BUILD_COMMIT=${_kggml_sd_short_commit}
    SDCPP_BUILD_VERSION=${KGGML_DIFFUSION_UPSTREAM_VERSION})

# Exported: the kggml_diffusion_* C ABI and stable-diffusion.cpp's own SD_API. Everything else stays hidden.
set_target_properties(kggml_diffusion PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    EXPORT_NAME diffusion)

if (NOT ANDROID)
    set_target_properties(kggml_diffusion PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

# Warnings for Kggml's own sources only; upstream stable-diffusion.cpp is compiled with its defaults.
if (MSVC)
    # Abort unwinds a private C++ exception from the progress callback through stable-diffusion.cpp's C++ frames up
    # to kggml_diffusion_generate(), passing generate_image(), which is declared extern "C". /EHsc lets MSVC assume
    # extern "C" functions never throw (and drop the handler), so use /EHs /EHc- for the whole library.
    target_compile_options(kggml_diffusion PRIVATE /EHs /EHc- /utf-8 /bigobj)
    set_source_files_properties(${_kggml_own_sources} PROPERTIES COMPILE_OPTIONS "/W4")
else()
    set_source_files_properties(${_kggml_own_sources} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Wpedantic")
endif()

# Dynamic linkage only: NEEDED / LC_LOAD_DYLIB on libggml(+base, cpu, ...) and libkggml_runtime, or on
# GgmlRuntime.framework in the monolithic iOS layout. kggml::runtime also carries GGML_MAX_NAME=128.
target_link_libraries(kggml_diffusion
    PUBLIC
        kggml::runtime
        kggml::ggml
    PRIVATE
        Threads::Threads)

if (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set_target_properties(kggml_diffusion PROPERTIES
        OUTPUT_NAME KggmlDiffusion
        FRAMEWORK TRUE
        MACOSX_FRAMEWORK_IDENTIFIER com.donfuldauer.kggml.diffusion
        MACOSX_FRAMEWORK_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${PROJECT_VERSION}
        PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/kggml_diffusion.h"
        XCODE_ATTRIBUTE_SKIP_INSTALL NO
        XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION YES
        XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/Frameworks @loader_path/Frameworks @loader_path/..")
endif()

# ---------------------------------------------------------------------------------------------------------
# Install + exported CMake package (kggml-diffusion-config.cmake -> kggml::diffusion)
# ---------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)
set(KGGML_DIFFUSION_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/kggml-diffusion")

install(TARGETS kggml_diffusion
    EXPORT kggmlDiffusionTargets
    RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES
    include/kggml_diffusion.h
    "${KGGML_DIFFUSION_SOURCE_DIR}/include/stable-diffusion.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT kggmlDiffusionTargets
    NAMESPACE kggml::
    FILE kggml-diffusion-targets.cmake
    DESTINATION ${KGGML_DIFFUSION_CMAKE_DIR})
configure_package_config_file(
    cmake/kggml-diffusion-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-diffusion-config.cmake"
    INSTALL_DESTINATION ${KGGML_DIFFUSION_CMAKE_DIR})
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-diffusion-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-diffusion-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-diffusion-config-version.cmake"
    DESTINATION ${KGGML_DIFFUSION_CMAKE_DIR})
export(EXPORT kggmlDiffusionTargets
    NAMESPACE kggml::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/kggml-diffusion-targets.cmake")

# ---------------------------------------------------------------------------------------------------------
# kggml_diffusion_jni: JNI bridge for the Kotlin DiffusionEngine on Android + JVM Desktop (not used on iOS, where
# Kotlin/Native calls the C ABI directly through cinterop). Links kggml_diffusion dynamically.
# ---------------------------------------------------------------------------------------------------------

set(KGGML_DIFFUSION_JNI_INCLUDE_DIRS "")
if (ANDROID)
    set(_kggml_diffusion_jni_default ON) # jni.h ships in the NDK sysroot
elseif (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_kggml_diffusion_jni_default OFF)
else()
    # Only the headers are needed (the JVM provides the symbols at load time).
    find_package(JNI QUIET)
    if (JAVA_INCLUDE_PATH AND JAVA_INCLUDE_PATH2)
        set(_kggml_diffusion_jni_default ON)
        set(KGGML_DIFFUSION_JNI_INCLUDE_DIRS "${JAVA_INCLUDE_PATH}" "${JAVA_INCLUDE_PATH2}")
    else()
        set(_kggml_diffusion_jni_default OFF)
    endif()
endif()
option(KGGML_DIFFUSION_BUILD_JNI "kggml-diffusion: build the kggml_diffusion_jni JNI bridge (Android / JVM Desktop)"
    ${_kggml_diffusion_jni_default})

if (KGGML_DIFFUSION_BUILD_JNI)
    if (NOT ANDROID AND NOT KGGML_DIFFUSION_JNI_INCLUDE_DIRS)
        message(FATAL_ERROR "KGGML_DIFFUSION_BUILD_JNI=ON but JNI headers were not found (set JAVA_HOME)")
    endif()
    add_library(kggml_diffusion_jni SHARED jni/kggml_diffusion_jni.cpp)
    target_include_directories(kggml_diffusion_jni PRIVATE ${KGGML_DIFFUSION_JNI_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(kggml_diffusion_jni PRIVATE kggml_diffusion kggml::runtime)
    target_compile_definitions(kggml_diffusion_jni PRIVATE
        $<$<BOOL:${KGGML_DIFFUSION_TEST_HOOKS}>:KGGML_DIFFUSION_TEST_HOOKS=1>)
    # Exports only the Java_* entry points. Unversioned (loaded by file name from Java), like on Android.
    set_target_properties(kggml_diffusion_jni PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    if (NOT APPLE AND NOT WIN32)
        # ELF: hidden visibility does not cover std:: template instantiations, so an explicit version script keeps
        # every non-JNI symbol local.
        set(_kggml_jni_map "${CMAKE_CURRENT_SOURCE_DIR}/jni/kggml_diffusion_jni.map")
        target_link_options(kggml_diffusion_jni PRIVATE "LINKER:--version-script=${_kggml_jni_map}")
        set_property(TARGET kggml_diffusion_jni APPEND PROPERTY LINK_DEPENDS "${_kggml_jni_map}")
    endif()
    if (MSVC)
        target_compile_options(kggml_diffusion_jni PRIVATE /W4 /EHsc)
    else()
        target_compile_options(kggml_diffusion_jni PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    # Installed next to kggml_diffusion but not part of the exported CMake package (private to the Kotlin bindings).
    install(TARGETS kggml_diffusion_jni
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# ---------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------

if (KGGML_DIFFUSION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "kggml-diffusion: ${PROJECT_VERSION}, stable-diffusion.cpp ${KGGML_DIFFUSION_UPSTREAM_VERSION} @ ${KGGML_DIFFUSION_UPSTREAM_COMMIT}")
message(STATUS "kggml-diffusion: jni=${KGGML_DIFFUSION_BUILD_JNI} tests=${KGGML_DIFFUSION_BUILD_TESTS} hooks=${KGGML_DIFFUSION_TEST_HOOKS}")
