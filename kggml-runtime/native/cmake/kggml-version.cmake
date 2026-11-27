# The Kggml version has one source: VERSION_NAME in the repository's gradle.properties, which Gradle publishes with.
# Every native project includes this file before project() and uses KGGML_VERSION as its VERSION, so the native
# libraries (runtimeVersion / moduleVersion, SOVERSION, framework bundle versions, the CMake package versions) always
# match the Maven artifacts.
#
# Sets KGGML_VERSION to the numeric MAJOR.MINOR.PATCH part (a "-SNAPSHOT" suffix is dropped, since CMake versions are
# numeric). A build outside the repository can pass -DKGGML_VERSION=<x.y.z> instead.
if (NOT DEFINED KGGML_VERSION)
    get_filename_component(_kggml_properties "${CMAKE_CURRENT_LIST_DIR}/../../../gradle.properties" ABSOLUTE)
    if (NOT EXISTS "${_kggml_properties}")
        message(FATAL_ERROR "kggml-version.cmake: ${_kggml_properties} not found; pass -DKGGML_VERSION=<x.y.z>")
    endif()
    file(STRINGS "${_kggml_properties}" _kggml_version_line REGEX "^VERSION_NAME=")
    if (NOT _kggml_version_line MATCHES "^VERSION_NAME=([0-9]+\\.[0-9]+\\.[0-9]+)")
        message(FATAL_ERROR "kggml-version.cmake: no VERSION_NAME=<x.y.z> line in ${_kggml_properties}")
    endif()
    set(KGGML_VERSION "${CMAKE_MATCH_1}")
    # Re-configure when the version changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_kggml_properties}")
    unset(_kggml_properties)
    unset(_kggml_version_line)
endif()
