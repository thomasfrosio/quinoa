cmake_minimum_required(VERSION 3.21)

# Project options.
option(QN_ENABLE_WARNINGS "Enable compiler warnings" ON) # FIXME this is currently not used
option(QN_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(QN_ENABLE_CUDA "Enable CUDA support" ON)

# Global settings.
# Note: This project is meant to be the root project and shouldn't be included as a subdirectory.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Prevent in-source builds.
get_filename_component(_srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_bindir "${CMAKE_BINARY_DIR}" REALPATH)
if ("${_srcdir}" STREQUAL "${_bindir}")
    message(FATAL_ERROR "In-source builds are disabled, please create a separate build directory (which can be inside the source directory) and run cmake from there")
endif ()

# Start the project.
project(quinoa
    LANGUAGES CXX
    VERSION 0.1.0
    HOMEPAGE_URL https://github.com/thomasfrosio/quinoa
)
if (QN_ENABLE_CUDA)
    enable_language(CUDA)
endif ()

# Get the sources.
include(${PROJECT_SOURCE_DIR}/src/quinoa/Append.cmake)
if (QN_ENABLE_CUDA)
    set_source_files_properties(${QUINOA_SOURCES_UNIFIED} PROPERTIES LANGUAGE CUDA)
endif ()

# Create the executable.
add_executable(quinoa ${QUINOA_HEADERS} ${QUINOA_SOURCES_CXX} ${QUINOA_SOURCES_UNIFIED})
target_include_directories(quinoa PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(quinoa PRIVATE "$<$<CONFIG:DEBUG>:QN_DEBUG>")

# Gather and link dependencies.
include(${PROJECT_SOURCE_DIR}/cmake/ext/noa.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/yaml-cpp.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/nlopt.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/cxxopts.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/tomlplusplus.cmake)
target_link_libraries(quinoa
    PRIVATE
    noa::noa
    noa::prj_compiler_warnings
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    nlopt
    cxxopts::cxxopts
    tomlplusplus::tomlplusplus
    Eigen3::Eigen
)

# In CUDA, allow extended lambdas. We use this feature in a few places.
target_compile_options(quinoa PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)

# TODO install
