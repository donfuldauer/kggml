# Kggml shared ggml runtime (:kggml-runtime native layer).
#
# Builds upstream ggml (pinned git submodule under third_party/ggml) into SHARED dynamic libraries
# (ggml-base, ggml-cpu, ggml-metal, ggml-vulkan, ggml) plus the `kggml_runtime` C ABI library that every
# Kggml engine (whisper / llama / diffusion) links against dynamically. Engines must never embed a static
# copy of ggml: they consume `kggml::ggml` / `kggml::runtime` from this build tree or from the installed
# `kggml-runtime` CMake package.
#
# On iOS the same sources are linked into one dynamic framework (`GgmlRuntime.framework`, see
# KGGML_RUNTIME_MONOLITHIC) because an XCFramework slice carries a single binary.

cmake_minimum_required(VERSION 3.24) # $<LINK_LIBRARY:WHOLE_ARCHIVE,...> for the monolithic iOS framework

project(kggml_runtime
    VERSION 1.0.0
    DESCRIPTION "Kggml shared ggml runtime"
    LANGUAGES C CXX)

# MSVC >= 19.40 (VS 2022 17.10) made std::mutex's constructor constexpr; code built that way crashes in
# msvcp140.dll when the host process (e.g. a JVM that bundles its own, older copy) loads msvcp140 < 14.40.
# Keep the runtime-compatible constructor for every target in this tree, ggml included (issue #65).
# clang targeting the MSVC ABI (the windows-arm64 preset) uses the same STL, so it needs it too.
if (MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# Exact upstream ggml pin (tag v0.9.9). Keep in sync with the submodule gitlink and NOTICE.
set(KGGML_GGML_PINNED_COMMIT "a04eea0761a85d18f3f504d6ab970c5c9dce705f")
set(KGGML_GGML_PINNED_VERSION "0.9.9")
set(KGGML_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")

if (NOT EXISTS "${KGGML_GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml sources not found at ${KGGML_GGML_SOURCE_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

# ---------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------

include(CMakeDependentOption)

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_kggml_monolithic_default ON)
else()
    set(_kggml_monolithic_default OFF)
endif()

option(KGGML_RUNTIME_BUILD_TESTS  "kggml: build the GoogleTest/CTest suite"                         OFF)
cmake_dependent_option(KGGML_RUNTIME_TEST_HOOKS
    "kggml: compile fault-injection test hooks into kggml_runtime (never ship)" ON
    "KGGML_RUNTIME_BUILD_TESTS" OFF)
option(KGGML_RUNTIME_MONOLITHIC
    "kggml: link ggml statically into ONE shared kggml_runtime binary (GgmlRuntime framework on Apple)"
    ${_kggml_monolithic_default})
option(KGGML_RUNTIME_PORTABLE_CPU "kggml: portable CPU flags instead of -march=native"              ON)
option(KGGML_ENABLE_METAL         "kggml: build ggml-metal (Apple only, embedded .metallib)"         ${APPLE})
option(KGGML_ENABLE_VULKAN        "kggml: build ggml-vulkan (needs Vulkan headers/loader + glslc)"   OFF)
option(KGGML_ENABLE_BLAS          "kggml: build ggml-blas (Accelerate BLAS on Apple)"                ${APPLE})
option(KGGML_ENABLE_COREML        "kggml: advertise the CoreML capability flag (Apple only)"         OFF)
option(KGGML_ARM_I8MM             "kggml: add +i8mm to the arm64 -march (SIGILL on cores without it)" OFF)
set(KGGML_ARM_ARCH "" CACHE STRING
    "kggml: arm64 -march baseline for portable builds (empty = compiler default, e.g. apple-m1 on macOS)")
set(KGGML_X86_LEVEL "AVX2" CACHE STRING "kggml: x86 portable ISA level (AVX2 or SSE42)")
set_property(CACHE KGGML_X86_LEVEL PROPERTY STRINGS AVX2 SSE42)

if (KGGML_ENABLE_METAL AND NOT APPLE)
    message(WARNING "KGGML_ENABLE_METAL is only supported on Apple platforms; disabling")
    set(KGGML_ENABLE_METAL OFF CACHE BOOL "" FORCE)
endif()
if (KGGML_ENABLE_COREML AND NOT APPLE)
    message(WARNING "KGGML_ENABLE_COREML is only meaningful on Apple platforms; disabling")
    set(KGGML_ENABLE_COREML OFF CACHE BOOL "" FORCE)
endif()

if (KGGML_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    find_program(KGGML_GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
    if (NOT Vulkan_FOUND OR NOT KGGML_GLSLC_EXECUTABLE)
        message(WARNING "KGGML_ENABLE_VULKAN=ON but Vulkan SDK/glslc not found; building without ggml-vulkan")
        set(KGGML_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------------------------------------
# Global build settings
# ---------------------------------------------------------------------------------------------------------

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

# Android packages jniLibs by plain file name, so sonames must never be versioned (libggml.so, not .so.0).
# Android 15+ devices may use 16 KB pages; NDK r27 still defaults to 4 KB ELF segment alignment.
if (ANDROID)
    set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
    add_link_options("LINKER:-z,max-page-size=16384")
endif()

# Installed / relocated shared libraries find their siblings next to themselves.
if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif (UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

# Put every runtime artifact in one directory so tests (and Windows DLL lookup) find them without RPATH games.
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
# Upstream ggml configuration
# ---------------------------------------------------------------------------------------------------------

if (KGGML_RUNTIME_MONOLITHIC)
    # ggml objects are whole-archived into the single kggml_runtime shared binary below.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "ggml: build shared libraries" FORCE)
else()
    set(BUILD_SHARED_LIBS ON CACHE BOOL "ggml: build shared libraries" FORCE)
endif()

set(GGML_BACKEND_DL       OFF CACHE BOOL "" FORCE) # backends are linked directly, no dlopen() plugin loading
set(GGML_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(GGML_CPU              ON  CACHE BOOL "" FORCE)
set(GGML_OPENMP           OFF CACHE BOOL "" FORCE) # ggml's own threadpool; avoids shipping libomp on Android
set(GGML_CCACHE           OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE       ${APPLE}               CACHE BOOL "" FORCE)
set(GGML_BLAS             ${KGGML_ENABLE_BLAS}   CACHE BOOL "" FORCE)
set(GGML_METAL            ${KGGML_ENABLE_METAL}  CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ${KGGML_ENABLE_METAL} CACHE BOOL "" FORCE) # .metallib embedded in ggml-metal
set(GGML_VULKAN           ${KGGML_ENABLE_VULKAN} CACHE BOOL "" FORCE)

if (KGGML_RUNTIME_PORTABLE_CPU)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|i[3-6]86)$"
            OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64"
            OR CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
        if (KGGML_X86_LEVEL STREQUAL "AVX2")
            set(_kggml_avx2 ON)
        else()
            set(_kggml_avx2 OFF) # x86_64 Android ABI only guarantees SSE4.2 + POPCNT
        endif()
        set(GGML_SSE42 ON            CACHE BOOL "" FORCE)
        set(GGML_AVX   ${_kggml_avx2} CACHE BOOL "" FORCE)
        set(GGML_AVX2  ${_kggml_avx2} CACHE BOOL "" FORCE)
        set(GGML_FMA   ${_kggml_avx2} CACHE BOOL "" FORCE)
        set(GGML_F16C  ${_kggml_avx2} CACHE BOOL "" FORCE)
        set(GGML_BMI2  ${_kggml_avx2} CACHE BOOL "" FORCE)
    else()
        set(_kggml_arm_arch "${KGGML_ARM_ARCH}")
        if (KGGML_ARM_I8MM)
            if (_kggml_arm_arch STREQUAL "")
                set(_kggml_arm_arch "armv8.6-a+dotprod+fp16+i8mm")
            elseif (NOT _kggml_arm_arch MATCHES "\\+i8mm")
                string(APPEND _kggml_arm_arch "+i8mm")
            endif()
        endif()
        set(GGML_CPU_ARM_ARCH "${_kggml_arm_arch}" CACHE STRING "" FORCE)
    endif()
else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()

# Split layout: keep ggml's own install rules too, so its upstream `ggml-config.cmake` (ggml::ggml) is
# installed next to ours for engines whose upstream CMake expects find_package(ggml) (e.g. whisper.cpp's
# WHISPER_USE_SYSTEM_GGML). Monolithic layout: ggml archives are internal and must not be installed.
if (KGGML_RUNTIME_MONOLITHIC)
    set(_kggml_ggml_exclude EXCLUDE_FROM_ALL)
else()
    set(_kggml_ggml_exclude "")
endif()
add_subdirectory("${KGGML_GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml" ${_kggml_ggml_exclude})

# Every ggml library target that actually got configured (backend list is maintained by ggml itself).
set(KGGML_GGML_TARGETS ggml-base ggml)
foreach (_backend IN LISTS GGML_AVAILABLE_BACKENDS)
    if (TARGET ${_backend})
        list(APPEND KGGML_GGML_TARGETS ${_backend})
    endif()
endforeach()
list(REMOVE_DUPLICATES KGGML_GGML_TARGETS)
message(STATUS "kggml: ggml targets: ${KGGML_GGML_TARGETS}")

# ABI: GGML_MAX_NAME sizes `char name[GGML_MAX_NAME]` inside `struct ggml_tensor`, so ggml itself and EVERY
# translation unit that includes ggml.h (whisper.cpp, llama.cpp, stable-diffusion.cpp, the Kggml wrappers, JNI
# bridges, tests) must agree on it. stable-diffusion.cpp needs >= 128 (static_assert in ggml_extend.hpp; its
# tensor names exceed 64 chars), so the shared runtime is built with 128 and exports it as a PUBLIC usage
# requirement of every ggml target (and of kggml_runtime below, which covers the monolithic iOS layout).
# Consumers that link kggml::ggml / kggml::runtime inherit it automatically; anything compiling against ggml.h
# without these targets must pass -DGGML_MAX_NAME=128 itself (see README.md, "ggml ABI: GGML_MAX_NAME").
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/kggml-ggml-abi.cmake") # KGGML_GGML_MAX_NAME, shared with the GPU add-ons
foreach (_t IN LISTS KGGML_GGML_TARGETS)
    target_compile_definitions(${_t} PUBLIC GGML_MAX_NAME=${KGGML_GGML_MAX_NAME})
endforeach()

# Warn when the checked-out submodule does not match the pin recorded above / in NOTICE.
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${KGGML_GGML_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${KGGML_GGML_SOURCE_DIR}"
        OUTPUT_VARIABLE _kggml_ggml_head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (_kggml_ggml_head AND NOT _kggml_ggml_head STREQUAL KGGML_GGML_PINNED_COMMIT)
        message(WARNING
            "third_party/ggml is at ${_kggml_ggml_head}, expected pinned ${KGGML_GGML_PINNED_COMMIT}. "
            "Run: git submodule update --init --recursive")
    endif()
endif()

# ---------------------------------------------------------------------------------------------------------
# kggml_runtime C ABI library
# ---------------------------------------------------------------------------------------------------------

add_library(kggml_runtime SHARED
    src/kggml_runtime.cpp
    src/cpu_topology.cpp
    src/device_policy.cpp)
add_library(kggml::runtime ALIAS kggml_runtime)

target_include_directories(kggml_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(kggml_runtime
    PRIVATE
        KGGML_RUNTIME_BUILD
        KGGML_RUNTIME_VERSION_STRING="${PROJECT_VERSION}"
        KGGML_GGML_PINNED_COMMIT="${KGGML_GGML_PINNED_COMMIT}"
        KGGML_GGML_PINNED_VERSION="${KGGML_GGML_PINNED_VERSION}"
        $<$<BOOL:${KGGML_ENABLE_COREML}>:KGGML_HAS_COREML=1>
        $<$<BOOL:${KGGML_RUNTIME_TEST_HOOKS}>:KGGML_RUNTIME_TEST_HOOKS=1>
    PUBLIC
        KGGML_RUNTIME_SHARED
        GGML_MAX_NAME=${KGGML_GGML_MAX_NAME}) # ggml ABI, see KGGML_GGML_MAX_NAME above

set_target_properties(kggml_runtime PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    EXPORT_NAME runtime)

if (NOT ANDROID)
    set_target_properties(kggml_runtime PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

if (MSVC)
    target_compile_options(kggml_runtime PRIVATE /W4 /EHsc)
else()
    target_compile_options(kggml_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (KGGML_RUNTIME_MONOLITHIC)
    # One binary carries ggml + kggml_runtime. ggml's public API stays exported (default visibility, not
    # affected by kggml_runtime's hidden preset because it is compiled in ggml's own targets).
    # Every ggml archive (including the ones pulled in transitively through `ggml`) is linked whole, so
    # the full ggml API is exported and no backend object is dropped by the linker. The regular link
    # item still carries the transitive system deps (Accelerate, Metal, Foundation, Threads, ...).
    target_link_libraries(kggml_runtime PRIVATE "$<BUILD_INTERFACE:ggml>")
    if (APPLE)
        # ld64 -force_load works with every generator, including Xcode (used for the iOS presets).
        foreach (_t IN LISTS KGGML_GGML_TARGETS)
            target_link_options(kggml_runtime PRIVATE "LINKER:-force_load,$<TARGET_FILE:${_t}>")
        endforeach()
    else()
        string(REPLACE ";" "," _kggml_whole_archive "${KGGML_GGML_TARGETS}")
        set_property(TARGET kggml_runtime PROPERTY LINK_LIBRARY_OVERRIDE "WHOLE_ARCHIVE,${_kggml_whole_archive}")
    endif()
    target_include_directories(kggml_runtime PUBLIC
        $<BUILD_INTERFACE:${KGGML_GGML_SOURCE_DIR}/include>)
    set_target_properties(kggml_runtime PROPERTIES OUTPUT_NAME GgmlRuntime)
    if (APPLE)
        # Framework Headers/ carry the kggml C ABI plus ggml's public headers, so engine frameworks
        # (whisper / llama / diffusion) compile against exactly the ggml they link to.
        file(GLOB _kggml_ggml_public_headers "${KGGML_GGML_SOURCE_DIR}/include/*.h")
        set_target_properties(kggml_runtime PROPERTIES
            FRAMEWORK TRUE
            MACOSX_FRAMEWORK_IDENTIFIER com.donfuldauer.kggml.runtime
            MACOSX_FRAMEWORK_BUNDLE_VERSION ${PROJECT_VERSION}
            MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${PROJECT_VERSION}
            PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/kggml_runtime.h;${_kggml_ggml_public_headers}"
            XCODE_ATTRIBUTE_SKIP_INSTALL NO
            XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION YES)
    endif()

    # Downstream consumers see the same logical targets in both layouts.
    add_library(kggml::ggml ALIAS kggml_runtime)
else()
    target_link_libraries(kggml_runtime PUBLIC ggml) # dynamic NEEDED on libggml / libggml-base

    foreach (_t IN LISTS KGGML_GGML_TARGETS)
        set_target_properties(${_t} PROPERTIES EXPORT_NAME ${_t})
    endforeach()
    # ggml lists its PUBLIC_HEADER files relative to its own source dir; make them absolute so the
    # install(TARGETS ...) below (issued from this directory) can find them.
    get_target_property(_kggml_ggml_headers ggml PUBLIC_HEADER)
    if (_kggml_ggml_headers)
        list(TRANSFORM _kggml_ggml_headers PREPEND "${KGGML_GGML_SOURCE_DIR}/" REGEX "^[^/]")
        set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${_kggml_ggml_headers}")
    endif()
    add_library(kggml::ggml      ALIAS ggml)
    add_library(kggml::ggml-base ALIAS ggml-base)
endif()

# ---------------------------------------------------------------------------------------------------------
# kggml_runtime_jni: JNI bridge for the Kotlin bindings on Android + JVM Desktop (not used on iOS, where
# Kotlin/Native calls the C ABI directly through cinterop). Links kggml_runtime dynamically.
# ---------------------------------------------------------------------------------------------------------

set(KGGML_JNI_INCLUDE_DIRS "")
if (ANDROID)
    set(_kggml_jni_default ON) # jni.h ships in the NDK sysroot
elseif (KGGML_RUNTIME_MONOLITHIC OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_kggml_jni_default OFF)
else()
    # Only the headers are needed (the JVM provides the symbols at load time), so do not require JNI_FOUND
    # (which also demands libjawt / libjvm on some CMake versions).
    find_package(JNI QUIET)
    if (JAVA_INCLUDE_PATH AND JAVA_INCLUDE_PATH2)
        set(_kggml_jni_default ON)
        set(KGGML_JNI_INCLUDE_DIRS "${JAVA_INCLUDE_PATH}" "${JAVA_INCLUDE_PATH2}")
    else()
        set(_kggml_jni_default OFF)
    endif()
endif()
option(KGGML_RUNTIME_BUILD_JNI "kggml: build the kggml_runtime_jni JNI bridge (Android / JVM Desktop)"
    ${_kggml_jni_default})

if (KGGML_RUNTIME_BUILD_JNI)
    if (NOT ANDROID AND NOT KGGML_JNI_INCLUDE_DIRS)
        message(FATAL_ERROR "KGGML_RUNTIME_BUILD_JNI=ON but JNI headers were not found (set JAVA_HOME)")
    endif()
    add_library(kggml_runtime_jni SHARED jni/kggml_runtime_jni.cpp)
    target_include_directories(kggml_runtime_jni PRIVATE ${KGGML_JNI_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(kggml_runtime_jni PRIVATE kggml_runtime)
    target_compile_definitions(kggml_runtime_jni PRIVATE
        $<$<BOOL:${KGGML_RUNTIME_TEST_HOOKS}>:KGGML_RUNTIME_TEST_HOOKS=1>)
    # Exports only the Java_* entry points. Unversioned (loaded by file name from Java), like on Android.
    set_target_properties(kggml_runtime_jni PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    if (MSVC)
        target_compile_options(kggml_runtime_jni PRIVATE /W4 /EHsc)
    else()
        target_compile_options(kggml_runtime_jni PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    # Installed next to kggml_runtime but deliberately not part of the exported CMake package: it is a
    # private implementation detail of the Kotlin bindings.
    install(TARGETS kggml_runtime_jni
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# ---------------------------------------------------------------------------------------------------------
# Install + exported CMake package (kggml-runtime-config.cmake → kggml::runtime, kggml::ggml)
# ---------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

set(KGGML_RUNTIME_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/kggml-runtime")

if (KGGML_RUNTIME_MONOLITHIC)
    set(_kggml_install_targets kggml_runtime)
else()
    set(_kggml_install_targets kggml_runtime ${KGGML_GGML_TARGETS})
endif()

install(TARGETS ${_kggml_install_targets}
    EXPORT kggmlRuntimeTargets
    RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES include/kggml_runtime.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if (KGGML_RUNTIME_MONOLITHIC)
    # Split layout installs ggml headers through ggml's PUBLIC_HEADER property above.
    install(DIRECTORY "${KGGML_GGML_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
endif()

install(EXPORT kggmlRuntimeTargets
    NAMESPACE kggml::
    FILE kggml-runtime-targets.cmake
    DESTINATION ${KGGML_RUNTIME_CMAKE_DIR})

configure_package_config_file(
    cmake/kggml-runtime-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-runtime-config.cmake"
    INSTALL_DESTINATION ${KGGML_RUNTIME_CMAKE_DIR})
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-runtime-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-runtime-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kggml-runtime-config-version.cmake"
    DESTINATION ${KGGML_RUNTIME_CMAKE_DIR})

# Build-tree export so sibling builds can find_package(kggml-runtime) without installing.
export(EXPORT kggmlRuntimeTargets
    NAMESPACE kggml::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/kggml-runtime-targets.cmake")

# ---------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------

if (KGGML_RUNTIME_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "kggml: runtime ${PROJECT_VERSION}, ggml ${KGGML_GGML_PINNED_VERSION} @ ${KGGML_GGML_PINNED_COMMIT}")
message(STATUS "kggml: monolithic=${KGGML_RUNTIME_MONOLITHIC} metal=${GGML_METAL} vulkan=${GGML_VULKAN} "
               "blas=${GGML_BLAS} coreml=${KGGML_ENABLE_COREML} jni=${KGGML_RUNTIME_BUILD_JNI} native=${GGML_NATIVE} "
               "arm_arch='${GGML_CPU_ARM_ARCH}' tests=${KGGML_RUNTIME_BUILD_TESTS}")
