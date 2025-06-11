message(STATUS "nlopt: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET nlopt::nlopt)
    message(STATUS "Target already exists: nlopt")
else ()
    set(nlopt_REPOSITORY https://github.com/stevengj/nlopt)
    set(nlopt_TAG v2.10.0)

    include(FetchContent)
    FetchContent_Declare(
        nlopt
        GIT_REPOSITORY ${nlopt_REPOSITORY}
        GIT_TAG ${nlopt_TAG}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )

    option(NLOPT_CXX "" ON)
    option(NLOPT_FORTRAN "" OFF)
    option(BUILD_SHARED_LIBS "" OFF)
    option(NLOPT_PYTHON "" OFF)
    option(NLOPT_OCTAVE "" OFF)
    option(NLOPT_MATLAB "" OFF)
    option(NLOPT_GUILE "" OFF)
    option(NLOPT_SWIG "" OFF)
    option(NLOPT_JAVA "" OFF)
    option(NLOPT_LUKSAN "" ON)
    option(NLOPT_TESTS "" OFF)
    FetchContent_MakeAvailable(nlopt)

    message(STATUS "New imported target available: nlopt")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "nlopt: fetching static dependency... done")
