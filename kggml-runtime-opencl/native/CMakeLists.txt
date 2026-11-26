# kggml-runtime-opencl/native - ggml's OpenCL backend as a dynamically loadable ggml backend module (libggml-opencl.so)
# for the Android OpenCL add-on (Qualcomm Adreno: ggml's OpenCL kernels, incl. the Adreno-optimized matmuls).
#
# Built from :kggml-runtime's pinned ggml with GGML_BACKEND_DL: the module exports ggml_backend_init and links
# libggml-base.so dynamically, which at run time resolves to the libggml-base.so shipped by :kggml-runtime (same ggml
# commit). Only libggml-opencl.so is packaged.
#
# OpenCL: headers from the pinned Khronos OpenCL-Headers submodule (compiled in); the module links "libOpenCL.so",
# built here from the pinned Khronos OpenCL-ICD-Loader only to link against and never shipped: at run time the device's
# vendor libOpenCL.so is used (the AAR manifest declares it with <uses-native-library android:required="false">). On
# devices without OpenCL the module fails to load and the runtime stays on the CPU. Kernels are embedded at build time
# (Python 3).
cmake_minimum_required(VERSION 3.21)
project(kggml_runtime_opencl LANGUAGES C CXX)

if (NOT ANDROID)
    message(FATAL_ERROR "kggml-runtime-opencl builds the Android OpenCL add-on only")
endif()

set(KGGML_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../kggml-runtime/native/third_party/ggml")
set(KGGML_OPENCL_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL-Headers")
set(KGGML_OPENCL_LOADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL-ICD-Loader")
foreach(_required IN ITEMS "${KGGML_GGML_SOURCE_DIR}/CMakeLists.txt" "${KGGML_OPENCL_HEADERS_DIR}/CL/cl.h"
                           "${KGGML_OPENCL_LOADER_DIR}/CMakeLists.txt")
    if (NOT EXISTS "${_required}")
        message(FATAL_ERROR "missing ${_required}: run `git submodule update --init` (ggml, OpenCL-Headers, OpenCL-ICD-Loader)")
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

# Link-time libOpenCL.so (soname libOpenCL.so, unversioned): the Khronos ICD loader. Never packaged.
set(OPENCL_ICD_LOADER_HEADERS_DIR "${KGGML_OPENCL_HEADERS_DIR}" CACHE PATH "" FORCE)
set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(OPENCL_ICD_LOADER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENCL_LAYERS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("${KGGML_OPENCL_LOADER_DIR}" opencl-icd-loader EXCLUDE_FROM_ALL)

# FindOpenCL (ggml-opencl's find_package(OpenCL)): the pinned headers and the loader target (a target name, so the
# module links it with a build dependency).
set(OpenCL_INCLUDE_DIR "${KGGML_OPENCL_HEADERS_DIR}" CACHE PATH "" FORCE)
set(OpenCL_LIBRARY OpenCL CACHE STRING "" FORCE)

# ggml: only the OpenCL backend, as a GGML_BACKEND_DL module.
set(BUILD_SHARED_LIBS     ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL       ON  CACHE BOOL "" FORCE)
set(GGML_CPU              OFF CACHE BOOL "" FORCE) # the CPU backend lives in :kggml-runtime
set(GGML_OPENCL           ON  CACHE BOOL "" FORCE)
set(GGML_OPENCL_EMBED_KERNELS      ON CACHE BOOL "" FORCE)
set(GGML_OPENCL_USE_ADRENO_KERNELS ON CACHE BOOL "" FORCE)
set(GGML_METAL            OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN           OFF CACHE BOOL "" FORCE)
set(GGML_BLAS             OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP           OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE           OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
add_subdirectory("${KGGML_GGML_SOURCE_DIR}" ggml EXCLUDE_FROM_ALL)

if (NOT TARGET ggml-opencl)
    message(FATAL_ERROR "ggml did not define the ggml-opencl target")
endif()
add_custom_target(kggml_runtime_opencl ALL DEPENDS ggml-opencl)
