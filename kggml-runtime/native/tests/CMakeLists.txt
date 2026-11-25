# kggml_runtime GoogleTest / CTest suite (enabled with -DKGGML_RUNTIME_BUILD_TESTS=ON).

include(FetchContent)

option(KGGML_RUNTIME_USE_SYSTEM_GTEST "kggml: use an installed GTest package instead of FetchContent" OFF)

if (KGGML_RUNTIME_USE_SYSTEM_GTEST)
    find_package(GTest 1.15 REQUIRED)
else()
    # googletest v1.15.2, pinned by commit SHA (test-only dependency, never shipped).
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        b514bdc898e2951020cbdca1304b75f5950d1f59 # v1.15.2
        EXCLUDE_FROM_ALL)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # MSVC: match the runtime's /MD CRT
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    # googletest must stay a static test-only library even though BUILD_SHARED_LIBS is ON for ggml.
    set(_kggml_saved_shared ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(googletest)
    set(BUILD_SHARED_LIBS ${_kggml_saved_shared})
endif()

include(GoogleTest)

add_executable(kggml_runtime_tests
    test_lifecycle.cpp
    test_backends.cpp
    test_cpu_topology.cpp
    test_versions.cpp
    test_errors.cpp
    test_abi.cpp
    # Internal (hidden-visibility) code under test is compiled straight into the test binary.
    ${PROJECT_SOURCE_DIR}/src/cpu_topology.cpp)

target_include_directories(kggml_runtime_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(kggml_runtime_tests PRIVATE kggml::runtime GTest::gtest_main)

if (KGGML_RUNTIME_MONOLITHIC)
    target_include_directories(kggml_runtime_tests PRIVATE ${KGGML_GGML_SOURCE_DIR}/include)
else()
    # Tests compare runtime answers against the linked ggml directly (dynamic, same shared libggml).
    target_link_libraries(kggml_runtime_tests PRIVATE kggml::ggml)
endif()

target_compile_definitions(kggml_runtime_tests PRIVATE
    KGGML_TEST_EXPECT_RUNTIME_VERSION="${PROJECT_VERSION}"
    KGGML_TEST_EXPECT_GGML_VERSION="${KGGML_GGML_PINNED_VERSION}"
    KGGML_TEST_EXPECT_GGML_COMMIT="${KGGML_GGML_PINNED_COMMIT}"
    KGGML_TEST_EXPECT_METAL=$<BOOL:${GGML_METAL}>
    KGGML_TEST_EXPECT_VULKAN=$<BOOL:${GGML_VULKAN}>
    KGGML_TEST_EXPECT_COREML=$<BOOL:${KGGML_ENABLE_COREML}>
    $<$<BOOL:${KGGML_RUNTIME_TEST_HOOKS}>:KGGML_RUNTIME_TEST_HOOKS=1>)

gtest_discover_tests(kggml_runtime_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "kggml-runtime")
