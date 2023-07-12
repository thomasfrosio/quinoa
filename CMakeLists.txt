cmake_minimum_required(VERSION 3.21)

project(quinoa
        LANGUAGES CXX CUDA
        VERSION 0.1.0
        HOMEPAGE_URL https://github.com/thomasfrosio/quinoa
        )

set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------------------
# Project options
include(${PROJECT_SOURCE_DIR}/cmake/utils/ProjectOptions.cmake)
#include(CheckIPOSupported)
#check_ipo_supported(RESULT result OUTPUT output)
#if(result)
#    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
#else()
#    message(SEND_ERROR "IPO is not supported: ${output}")
#endif()

# Set up the C++ compiler.
add_library(quinoa_compiler_option INTERFACE)
include(${PROJECT_SOURCE_DIR}/cmake/utils/CompilerWarnings.cmake)
set_cxx_compiler_warnings(quinoa_compiler_option)
set_cuda_compiler_warnings(quinoa_compiler_option)
target_compile_features(quinoa_compiler_option INTERFACE cxx_std_17)

# FIXME
include(${PROJECT_SOURCE_DIR}/cmake/utils/CUDASelectArch.cmake)
set(CMAKE_CUDA_ARCHITECTURES ${CUDASelectArch_RESULTS})
target_compile_options(quinoa_compiler_option INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)

# Gather dependencies.
add_library(quinoa_libraries INTERFACE)
include(${PROJECT_SOURCE_DIR}/cmake/ext/noa.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/yaml-cpp.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/nlopt.cmake)
target_link_libraries(quinoa_libraries
    INTERFACE
    noa::noa
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    nlopt
    )

# ---------------------------------------------------------------------------------------
# Project target
# ---------------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/src/quinoa/Append.cmake)
add_executable(quinoa ${QUINOA_SOURCES})

target_include_directories(quinoa
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        )

target_link_libraries(quinoa PRIVATE
        quinoa_compiler_option
        quinoa_libraries)

# Set definitions:
target_compile_definitions(quinoa
        PRIVATE
        "$<$<CONFIG:DEBUG>:QN_DEBUG>"
        "$<$<BOOL:${QN_ENABLE_CHECKS_RELEASE}>:QN_ENABLE_CHECKS_RELEASE>"
        )

set_target_properties(quinoa
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        )
