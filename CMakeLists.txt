cmake_minimum_required(VERSION 3.23)

# Project options.
option(QN_ENABLE_CUDA "Enable CUDA support" ON)
option(QN_CTF_ONLY "Only build the CTF estimation" ON)
if (NOT QN_CTF_ONLY)
    message(ERROR "The tilt-series alignment is a work in progress and still in development. Except an update end of August 2026")
endif()

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

include(GNUInstallDirs)
if(PROJECT_IS_TOP_LEVEL)
    # Prevent in-source builds.
    get_filename_component(_srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
    get_filename_component(_bindir "${CMAKE_BINARY_DIR}" REALPATH)
    if ("${_srcdir}" STREQUAL "${_bindir}")
        message("In-source builds are a bad idea")
        message("Please create a separate build directory (which can be inside the project directory) and run cmake from there")
        message(FATAL_ERROR "Quitting configuration")
    endif ()

    # Set default build type if not specified.
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
    endif()

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # generate compile_commands.json for clang-based tools
    set(CMAKE_DEBUG_POSTFIX "d") # make sure debug and release builds don't collide
endif()

# Get the sources.
include(${PROJECT_SOURCE_DIR}/src/quinoa/Append.cmake)
if (QN_ENABLE_CUDA)
    enable_language(CUDA)
    set_source_files_properties(${QUINOA_SOURCES_UNIFIED} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${QUINOA_SOURCES_UNIFIED} PROPERTIES COMPILE_FLAGS -Wno-useless-cast)
endif ()

# Create the executable.
add_executable(quinoa)
target_sources(quinoa
    PRIVATE
    ${QUINOA_SOURCES_CXX}
    ${QUINOA_SOURCES_UNIFIED}
    PRIVATE
    FILE_SET quinoa_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES ${QUINOA_HEADERS}
)
target_include_directories(quinoa PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(quinoa PRIVATE
    "$<$<CONFIG:DEBUG>:QN_DEBUG>"
    "$<$<BOOL:${QN_CTF_ONLY}>:QN_CTF_ONLY>")

# Gather and link dependencies.
include(${PROJECT_SOURCE_DIR}/cmake/ext/noa.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/nlopt.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/cxxopts.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/tomlplusplus.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/eigen.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ext/glob-cpp.cmake)

target_link_libraries(quinoa
    PRIVATE
    noa::noa
    noa::prj_compiler_warnings
    spdlog::spdlog
    nlopt
    cxxopts::cxxopts
    tomlplusplus::tomlplusplus
    Eigen3::Eigen
    glob::glob
)

# In CUDA, allow extended lambdas. We use this feature in a few places.
target_compile_options(quinoa PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)

install(TARGETS quinoa
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    ${PROJECT_SOURCE_DIR}/share/settings_ctf.toml
    ${PROJECT_SOURCE_DIR}/share/settings_all.toml
    ${PROJECT_SOURCE_DIR}/share/plot_diagnostics.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/quinoa
)
