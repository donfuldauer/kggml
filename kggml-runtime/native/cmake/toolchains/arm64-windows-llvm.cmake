# Cross-compiles for Windows arm64 (Windows on ARM, e.g. Snapdragon X) with LLVM clang targeting the MSVC ABI
# (issue #91). ggml's CPU backend rejects MSVC's cl.exe for ARM, so cl.exe cannot be used here.
#
# Run from an MSVC developer environment for arm64 libraries: `vcvarsall.bat amd64_arm64` on an x64 host
# (`vcvarsall.bat arm64` on an arm64 host), so link.exe / lld-link find the arm64 CRT, STL and SDK libraries.
# clang comes from LLVM (C:\Program Files\LLVM) or Visual Studio's "C++ Clang tools" component.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)

set(_kggml_target arm64-pc-windows-msvc)
set(CMAKE_C_COMPILER   clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET   ${_kggml_target})
set(CMAKE_CXX_COMPILER_TARGET ${_kggml_target})
set(CMAKE_ASM_COMPILER_TARGET ${_kggml_target})
