# kggml_llama GoogleTest / CTest suite (enabled with -DKGGML_LLAMA_BUILD_TESTS=ON).

include(FetchContent)

option(KGGML_LLAMA_USE_SYSTEM_GTEST "kggml-llama: use an installed GTest package instead of FetchContent" OFF)

if (KGGML_LLAMA_USE_SYSTEM_GTEST)
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

add_executable(kggml_llama_tests
    test_errors.cpp
    test_model.cpp
    test_context.cpp
    test_abort.cpp
    test_sampler.cpp
    test_mtmd.cpp
    tiny_model.cpp)

target_include_directories(kggml_llama_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(kggml_llama_tests PRIVATE kggml::llama GTest::gtest_main Threads::Threads)
target_compile_definitions(kggml_llama_tests PRIVATE
    KGGML_TEST_EXPECT_LLAMA_VERSION="${PROJECT_VERSION}"
    KGGML_TEST_EXPECT_UPSTREAM_VERSION="${KGGML_LLAMA_UPSTREAM_VERSION}"
    KGGML_TEST_EXPECT_UPSTREAM_COMMIT="${KGGML_LLAMA_UPSTREAM_COMMIT}"
    $<$<BOOL:${KGGML_LLAMA_TEST_HOOKS}>:KGGML_LLAMA_TEST_HOOKS=1>)
if (MSVC)
    target_compile_options(kggml_llama_tests PRIVATE /W4 /EHsc /utf-8)
else()
    target_compile_options(kggml_llama_tests PRIVATE -Wall -Wextra)
endif()

# Windows has no RPATH: put the runtime / ggml DLLs (imported from the kggml-runtime build) next to the tests.
if (WIN32)
    add_custom_command(TARGET kggml_llama_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:kggml_llama_tests> $<TARGET_FILE_DIR:kggml_llama_tests>
        COMMAND_EXPAND_LISTS)
endif()

gtest_discover_tests(kggml_llama_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "kggml-llama")
