# kggml_whisper GoogleTest / CTest suite (enabled with -DKGGML_WHISPER_BUILD_TESTS=ON).

include(FetchContent)

option(KGGML_WHISPER_USE_SYSTEM_GTEST "kggml-whisper: use an installed GTest package instead of FetchContent" OFF)

if (KGGML_WHISPER_USE_SYSTEM_GTEST)
    find_package(GTest 1.15 REQUIRED)
else()
    # googletest v1.18.0, pinned by commit SHA (test-only dependency, never shipped). Same pin as kggml-runtime.
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        063de7e9578f82b369302001269680b4b1553359 # v1.18.0
        EXCLUDE_FROM_ALL)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

add_executable(kggml_whisper_tests
    test_pcm.cpp
    test_resampler.cpp
    test_errors.cpp
    test_context.cpp
    test_abort.cpp
    test_stream.cpp
    tiny_model.cpp
    # Internal (hidden-visibility) helpers under test are compiled straight into the test binary.
    ${PROJECT_SOURCE_DIR}/src/resampler.cpp)

target_include_directories(kggml_whisper_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(kggml_whisper_tests PRIVATE kggml::whisper GTest::gtest_main Threads::Threads)
target_compile_definitions(kggml_whisper_tests PRIVATE
    KGGML_TEST_EXPECT_WHISPER_VERSION="${PROJECT_VERSION}"
    KGGML_TEST_EXPECT_UPSTREAM_VERSION="${KGGML_WHISPER_UPSTREAM_VERSION}"
    KGGML_TEST_EXPECT_UPSTREAM_COMMIT="${KGGML_WHISPER_UPSTREAM_COMMIT}"
    KGGML_TEST_EXPECT_COREML=$<BOOL:${KGGML_WHISPER_COREML}>
    $<$<BOOL:${KGGML_WHISPER_TEST_HOOKS}>:KGGML_WHISPER_TEST_HOOKS=1>)
if (MSVC)
    target_compile_options(kggml_whisper_tests PRIVATE /W4 /EHsc /utf-8)
else()
    target_compile_options(kggml_whisper_tests PRIVATE -Wall -Wextra)
endif()

# Windows has no RPATH: put the runtime / ggml DLLs (imported from the kggml-runtime build) next to the tests.
if (WIN32)
    add_custom_command(TARGET kggml_whisper_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:kggml_whisper_tests> $<TARGET_FILE_DIR:kggml_whisper_tests>
        COMMAND_EXPAND_LISTS)
endif()

gtest_discover_tests(kggml_whisper_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "kggml-whisper")
