@echo off
rem Builds and installs the release runtime for Windows arm64 (windows-aarch64 in the JVM jar; issue #91).
rem
rem Usage (from the repository root, in cmd): kggml-runtime\native\scripts\build-windows-arm64.cmd <install-prefix>
rem
rem ggml's CPU backend rejects cl.exe for ARM, so this compiles with LLVM clang for the MSVC ABI and links with the
rem MSVC arm64 libraries. Needs Visual Studio 2022 or later (or Build Tools) with the MSVC ARM64 build tools and the
rem "C++ Clang tools for Windows" component. Visual Studio's own clang is preferred: its STL headers check the clang
rem version (STL1000), so a standalone LLVM (C:\Program Files\LLVM, also on GitHub's Windows images) that is older
rem than the Visual Studio release fails to compile. A standalone LLVM is used only when Visual Studio has no clang.
rem On an x64 host the result cannot run there; scripts/ci/check_native_arch.py checks it after staging.
setlocal EnableExtensions
set "PREFIX=%~1"
if "%PREFIX%"=="" (echo usage: %~nx0 ^<install-prefix^> & exit /b 2)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo vswhere.exe not found: install Visual Studio 2022 or its Build Tools & exit /b 1)
set "VSINSTALL="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath`) do set "VSINSTALL=%%i"
if "%VSINSTALL%"=="" (echo No Visual Studio with the MSVC ARM64 build tools ^(Microsoft.VisualStudio.Component.VC.Tools.ARM64^) & exit /b 1)

rem Start from a clean MSVC environment: a job may already have an x64 one (LIB / INCLUDE / LIBPATH).
set "LIB=" & set "INCLUDE=" & set "LIBPATH=" & set "VSCMD_ARG_TGT_ARCH=" & set "VSCMD_ARG_HOST_ARCH="
set "VCVARS_ARCH=amd64_arm64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "VCVARS_ARCH=arm64"
call "%VSINSTALL%\VC\Auxiliary\Build\vcvarsall.bat" %VCVARS_ARCH% >nul || (echo vcvarsall %VCVARS_ARCH% failed & exit /b 1)

if exist "%VSINSTALL%\VC\Tools\Llvm\x64\bin\clang.exe" set "PATH=%VSINSTALL%\VC\Tools\Llvm\x64\bin;%PATH%"
where clang >nul 2>&1 || if exist "%ProgramFiles%\LLVM\bin\clang.exe" set "PATH=%ProgramFiles%\LLVM\bin;%PATH%"
where clang >nul 2>&1 || (echo clang not found: install LLVM or the Visual Studio "C++ Clang tools for Windows" component & exit /b 1)
set "CLANG_EXE="
for /f "delims=" %%c in ('where clang') do if not defined CLANG_EXE set "CLANG_EXE=%%c"
for %%d in ("%CLANG_EXE%") do set "CLANG_DIR=%%~dpd"
echo clang: %CLANG_EXE%
"%CLANG_EXE%" --version | findstr /i "version"
where ninja >nul 2>&1 || (echo ninja not found & exit /b 1)

rem Pass the chosen clang by full path: a build directory configured earlier with another clang keeps that one in
rem its CMake cache otherwise. When the path differs, CMake discards the cache and configures again.
cmake -S kggml-runtime/native --preset windows-arm64 -B build/native-windows-arm64 ^
  "-DCMAKE_C_COMPILER=%CLANG_DIR%clang.exe" "-DCMAKE_CXX_COMPILER=%CLANG_DIR%clang++.exe" || exit /b 1
cmake --build build/native-windows-arm64 --config Release -j 4 || exit /b 1
cmake --install build/native-windows-arm64 --config Release --prefix "%PREFIX%" || exit /b 1
echo Installed the windows-arm64 runtime to %PREFIX%
