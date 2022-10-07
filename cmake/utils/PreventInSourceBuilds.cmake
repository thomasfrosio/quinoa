# This aims to prevent in-source builds.
# REALPATH: make sure the user doesn't play dirty with symlinks
get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)

if ("${srcdir}" STREQUAL "${bindir}")
    message("In-source builds are disabled")
    message("Please create a separate build directory (which can be inside the source directory) and run cmake from there")
    message(FATAL_ERROR "Quitting configuration")
endif ()
