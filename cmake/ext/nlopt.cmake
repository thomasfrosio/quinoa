set(nlopt_REPOSITORY https://github.com/stevengj/nlopt)
set(nlopt_TAG master)

include(FetchContent)

FetchContent_Declare(
        nlopt
        GIT_REPOSITORY ${nlopt_REPOSITORY}
        GIT_TAG ${nlopt_TAG}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)

option(NLOPT_CXX "" ON)
option(NLOPT_FORTRAN "" OFF)
option(BUILD_SHARED_LIBS "" OFF)
option(NLOPT_PYTHON "" OFF)
option(NLOPT_OCTAVE "" OFF)
option(NLOPT_MATLAB "" OFF)
option(NLOPT_GUILE "" OFF)
option(NLOPT_SWIG "" OFF)
option (NLOPT_TESTS "" OFF)
FetchContent_MakeAvailable(nlopt)
