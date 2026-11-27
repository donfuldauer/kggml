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
    ${PROJECT_SOURCE_DIR}/src/cpu_topology.cpp
    ${PROJECT_SOURCE_DIR}/src/device_policy.cpp
    test_device_policy.cpp)

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

# kggml_runtime_load_backend against fake ggml backend modules (GGML_BACKEND_DL entry point, no devices). Only with a
# shared libggml: backend modules (the Android GPU add-ons) link it, and the test queries its registry directly.
if (NOT KGGML_RUNTIME_MONOLITHIC)
target_sources(kggml_runtime_tests PRIVATE test_backend_modules.cpp)
foreach(_fake IN ITEMS good bad)
    add_library(kggml_fake_backend_${_fake} MODULE fake_backend_module.cpp)
    target_include_directories(kggml_fake_backend_${_fake} PRIVATE
        ${KGGML_GGML_SOURCE_DIR}/include ${KGGML_GGML_SOURCE_DIR}/src)
    target_compile_definitions(kggml_fake_backend_${_fake} PRIVATE GGML_BACKEND_DL GGML_BACKEND_SHARED GGML_BACKEND_BUILD)
    add_dependencies(kggml_runtime_tests kggml_fake_backend_${_fake})
    # Test-only modules stay out of lib/ and bin/, which hold the runtime's shipped libraries (binary audits read them).
    set_target_properties(kggml_fake_backend_${_fake} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test-modules"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test-modules")
endforeach()
target_compile_definitions(kggml_fake_backend_good PRIVATE KGGML_FAKE_BACKEND_NAME="KggmlFake")
target_compile_definitions(kggml_fake_backend_bad PRIVATE KGGML_FAKE_BACKEND_NAME="KggmlFakeBad" KGGML_FAKE_BAD_API_VERSION=1)
target_compile_definitions(kggml_runtime_tests PRIVATE
    KGGML_TEST_BACKEND_MODULE="$<TARGET_FILE:kggml_fake_backend_good>"
    KGGML_TEST_BAD_BACKEND_MODULE="$<TARGET_FILE:kggml_fake_backend_bad>")
endif()

gtest_discover_tests(kggml_runtime_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "kggml-runtime")
