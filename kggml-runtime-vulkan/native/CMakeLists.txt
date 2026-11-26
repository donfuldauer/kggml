# kggml-runtime-vulkan/native - ggml's Vulkan backend as a dynamically loadable ggml backend module
# (libggml-vulkan.so) for the Android GPU add-on.
#
# Built from :kggml-runtime's pinned ggml with GGML_BACKEND_DL: the module exports ggml_backend_init and links
# libggml-base.so dynamically, which at run time resolves to the libggml-base.so shipped by :kggml-runtime (same ggml
# commit). The runtime registers it with kggml_runtime_load_backend when the add-on is packaged in the app; the CPU
# backend and the backend registry stay in :kggml-runtime. Only libggml-vulkan.so is packaged; the libggml-base.so /
# libggml.so built here exist to link against and are never shipped.
#
# Vulkan headers: the pinned Khronos Vulkan-Headers submodule (vulkan.h + vulkan.hpp of one version; the NDK has no
# vulkan.hpp). Loader: the NDK's libvulkan.so stub (the device's loader at run time) of API level 28, the first that
# exports the Vulkan 1.1 entry points ggml-vulkan calls: on Android 8.x the module fails to load and the runtime stays
# on the CPU. Shader compiler: the NDK's glslc.
cmake_minimum_required(VERSION 3.21)
project(kggml_runtime_vulkan LANGUAGES C CXX)

if (NOT ANDROID)
    message(FATAL_ERROR "kggml-runtime-vulkan builds the Android GPU add-on only (desktop builds link Vulkan into the "
                        "runtime with KGGML_ENABLE_VULKAN=ON)")
endif()

set(KGGML_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../kggml-runtime/native/third_party/ggml")
set(KGGML_VULKAN_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-Headers")
foreach(_required IN ITEMS "${KGGML_GGML_SOURCE_DIR}/CMakeLists.txt" "${KGGML_VULKAN_HEADERS_DIR}/include/vulkan/vulkan.hpp")
    if (NOT EXISTS "${_required}")
        message(FATAL_ERROR "missing ${_required}: run `git submodule update --init` (ggml and Vulkan-Headers)")
    endif()
endforeach()

# Same Android conventions as :kggml-runtime: unversioned sonames (jniLibs are packaged by file name) and 16 KB pages.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
add_link_options("LINKER:-z,max-page-size=16384")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Vulkan for the ggml-vulkan CMake (find_package(Vulkan COMPONENTS glslc)): pinned headers, NDK loader stub and glslc.
string(REGEX REPLACE "^android-" "" _api "${ANDROID_PLATFORM}")
if (ANDROID_ABI STREQUAL "arm64-v8a")
    set(_triple "aarch64-linux-android")
elseif (ANDROID_ABI STREQUAL "x86_64")
    set(_triple "x86_64-linux-android")
else()
    message(FATAL_ERROR "kggml-runtime-vulkan: unsupported ANDROID_ABI ${ANDROID_ABI} (arm64-v8a, x86_64)")
endif()
set(_ndk_sysroot_lib "${ANDROID_TOOLCHAIN_ROOT}/sysroot/usr/lib/${_triple}/${_api}")
file(GLOB _ndk_glslc "${ANDROID_NDK}/shader-tools/*/glslc")
if (NOT EXISTS "${_ndk_sysroot_lib}/libvulkan.so" OR NOT _ndk_glslc)
    message(FATAL_ERROR "NDK at ${ANDROID_NDK} lacks libvulkan.so (${_ndk_sysroot_lib}) or shader-tools/*/glslc")
endif()
list(GET _ndk_glslc 0 _ndk_glslc)
set(Vulkan_INCLUDE_DIR "${KGGML_VULKAN_HEADERS_DIR}/include" CACHE PATH "" FORCE)
set(Vulkan_LIBRARY "${_ndk_sysroot_lib}/libvulkan.so" CACHE FILEPATH "" FORCE)
set(Vulkan_GLSLC_EXECUTABLE "${_ndk_glslc}" CACHE FILEPATH "" FORCE)

# ggml: only the Vulkan backend, as a GGML_BACKEND_DL module.
set(BUILD_SHARED_LIBS     ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL       ON  CACHE BOOL "" FORCE)
set(GGML_CPU              OFF CACHE BOOL "" FORCE) # the CPU backend lives in :kggml-runtime
set(GGML_VULKAN           ON  CACHE BOOL "" FORCE)
set(GGML_METAL            OFF CACHE BOOL "" FORCE)
set(GGML_BLAS             OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP           OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE           OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
add_subdirectory("${KGGML_GGML_SOURCE_DIR}" ggml EXCLUDE_FROM_ALL)

if (NOT TARGET ggml-vulkan)
    message(FATAL_ERROR "ggml did not define the ggml-vulkan target")
endif()
add_custom_target(kggml_runtime_vulkan ALL DEPENDS ggml-vulkan)
