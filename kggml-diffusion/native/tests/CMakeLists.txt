# kggml_diffusion GoogleTest / CTest suite (enabled with -DKGGML_DIFFUSION_BUILD_TESTS=ON).

include(FetchContent)

option(KGGML_DIFFUSION_USE_SYSTEM_GTEST "kggml-diffusion: use an installed GTest package instead of FetchContent" OFF)

if (KGGML_DIFFUSION_USE_SYSTEM_GTEST)
    find_package(GTest 1.15 REQUIRED)
else()
    # googletest v1.15.2, pinned by commit SHA (test-only dependency, never shipped). Same pin as kggml-runtime.
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        b514bdc898e2951020cbdca1304b75f5950d1f59 # v1.15.2
        EXCLUDE_FROM_ALL)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

if (NOT KGGML_DIFFUSION_TEST_HOOKS)
    message(FATAL_ERROR "the kggml-diffusion tests need KGGML_DIFFUSION_TEST_HOOKS=ON")
endif()

add_executable(kggml_diffusion_tests
    test_errors.cpp
    test_context.cpp
    test_progress_abort.cpp
    fixtures.cpp)

target_include_directories(kggml_diffusion_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
# kggml::diffusion carries kggml::runtime / kggml::ggml (and GGML_MAX_NAME=128) for the gguf_* fixture writer.
target_link_libraries(kggml_diffusion_tests PRIVATE kggml::diffusion GTest::gtest_main Threads::Threads)
target_compile_definitions(kggml_diffusion_tests PRIVATE
    KGGML_TEST_EXPECT_DIFFUSION_VERSION="${PROJECT_VERSION}"
    KGGML_TEST_EXPECT_UPSTREAM_VERSION="${KGGML_DIFFUSION_UPSTREAM_VERSION}"
    KGGML_TEST_EXPECT_UPSTREAM_COMMIT="${KGGML_DIFFUSION_UPSTREAM_COMMIT}"
    KGGML_DIFFUSION_TEST_HOOKS=1)
if (MSVC)
    target_compile_options(kggml_diffusion_tests PRIVATE /W4 /EHsc /utf-8)
else()
    target_compile_options(kggml_diffusion_tests PRIVATE -Wall -Wextra)
endif()

# Windows has no RPATH: put the runtime / ggml DLLs (imported from the kggml-runtime build) next to the tests.
if (WIN32)
    add_custom_command(TARGET kggml_diffusion_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:kggml_diffusion_tests> $<TARGET_FILE_DIR:kggml_diffusion_tests>
        COMMAND_EXPAND_LISTS)
endif()

gtest_discover_tests(kggml_diffusion_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "kggml-diffusion")
