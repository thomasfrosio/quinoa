cmake_minimum_required(VERSION 3.18)

project(quinoa
        LANGUAGES CXX
        VERSION 0.1.0
        HOMEPAGE_URL https://github.com/ffyr2w/quinoa
        )

# ---------------------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------------------
# Project options
include(${PROJECT_SOURCE_DIR}/cmake/utils/ProjectOptions.cmake)

# Set up the C++ compiler.
add_library(quinoa_compiler_option INTERFACE)
include(${PROJECT_SOURCE_DIR}/cmake/utils/CompilerWarnings.cmake)
set_cxx_compiler_warnings(quinoa_compiler_option)
target_compile_features(quinoa_compiler_option INTERFACE cxx_std_17)

# Gather dependencies.
add_library(quinoa_libraries INTERFACE)
include(${PROJECT_SOURCE_DIR}/cmake/ext/noa.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/yaml-cpp.cmake)
target_link_libraries(quinoa_libraries INTERFACE
        noa::noa
        yaml-cpp::yaml-cpp
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
