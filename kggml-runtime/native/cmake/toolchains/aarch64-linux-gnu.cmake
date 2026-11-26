# Cross-compiles for Linux aarch64 from an x86_64 Linux host with the distribution's GNU cross toolchain
# (Debian/Ubuntu: g++-aarch64-linux-gnu). Used by the `linux-arm64-cross` preset (issue #91).
#
# When qemu-aarch64 is installed (Debian/Ubuntu: qemu-user), test binaries run under it, so CTest and
# gtest_discover_tests work on the x86_64 host.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(KGGML_CROSS_TRIPLE "aarch64-linux-gnu" CACHE STRING "kggml: GNU cross triple")
set(CMAKE_C_COMPILER   "${KGGML_CROSS_TRIPLE}-gcc")
set(CMAKE_CXX_COMPILER "${KGGML_CROSS_TRIPLE}-g++")

set(KGGML_CROSS_SYSROOT "/usr/${KGGML_CROSS_TRIPLE}" CACHE PATH "kggml: target root for libraries and the emulator")
set(CMAKE_FIND_ROOT_PATH "${KGGML_CROSS_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
# BOTH: the JNI headers come from the host JDK (jni.h / jni_md.h are the same for every Linux architecture).
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)

find_program(KGGML_QEMU_AARCH64 NAMES qemu-aarch64 qemu-aarch64-static)
if (KGGML_QEMU_AARCH64)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${KGGML_QEMU_AARCH64};-L;${KGGML_CROSS_SYSROOT}")
endif()
