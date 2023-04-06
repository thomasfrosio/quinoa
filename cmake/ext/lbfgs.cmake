set(LBFGS_REPOSITORY https://github.com/chokkan/liblbfgs)
set(LBFGS_TAG master)

include(FetchContent)

FetchContent_Declare(
        LBFGS
        GIT_REPOSITORY ${LBFGS_REPOSITORY}
        GIT_TAG ${LBFGS_TAG}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)

option(LBFGS_INSTALL_STATIC_LIBS "" ON)
option(LBFGS_USE_DOUBLE "" ON)
option(LBFGS_USE_SSE "" OFF)
option(LBFGS_USE_IEEE754 "" ON)
FetchContent_MakeAvailable(LBFGS)
