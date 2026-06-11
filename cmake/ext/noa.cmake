message(STATUS "noa: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET noa::noa)
    message(STATUS "Target already exists: noa::noa")
else ()
    set(noa_REPOSITORY https://github.com/thomasfrosio/noa)
    set(noa_TAG quinoa)

    message(STATUS "Repository: ${noa_REPOSITORY}")
    message(STATUS "Git tag: ${noa_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        noa
        GIT_REPOSITORY ${noa_REPOSITORY}
        GIT_TAG ${noa_TAG}
    )

    set(NOA_ENABLE_CPU ON)
    set(NOA_CPU_OPENMP ON)
    set(NOA_CPU_FFTW3_MULTITHREADED ON)
    set(NOA_CPU_FFTW3_STATIC ON)
    set(NOA_ENABLE_CUDA ${QN_ENABLE_CUDA})
    set(NOA_CUDA_STATIC OFF)
    set(NOA_ERROR_POLICY 2)
    set(NOA_ENABLE_WARNINGS ON)
    set(NOA_ENABLE_WARNINGS_AS_ERRORS OFF)
    set(NOA_ENABLE_TIFF ON)
    set(NOA_BUILD_TESTS OFF)
    set(NOA_BUILD_BENCHMARKS OFF)
    FetchContent_MakeAvailable(noa)

    message(STATUS "New imported target available: noa::noa")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "noa: fetching static dependency... done")
