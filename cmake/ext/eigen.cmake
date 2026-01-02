message(STATUS "Eigen3: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET Eigen3::Eigen)
    message(STATUS "Target already exists: Eigen3::Eigen")
else ()
    set(Eigen3_REPOSITORY https://gitlab.com/libeigen/eigen.git)
    set(Eigen3_TAG e67c494cba7180066e73b9f6234d0b2129f1cdf5)

    message(STATUS "Repository: ${Eigen3_REPOSITORY}")
    message(STATUS "Git tag: ${Eigen3_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY ${Eigen3_REPOSITORY}
        GIT_TAG ${Eigen3_TAG}
        GIT_PROGRESS TRUE
        EXCLUDE_FROM_ALL # no install
    )

    # Configure Eigen build options
    set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)

    message(STATUS "New imported target available: Eigen3::Eigen")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "Eigen3: fetching static dependency... done")
