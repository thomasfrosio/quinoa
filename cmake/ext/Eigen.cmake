set(EIGEN_REPOSITORY https://gitlab.com/libeigen/eigen.git)
set(EIGEN_TAG 3.4)

message(STATUS "Repository: ${EIGEN_REPOSITORY}")
message(STATUS "Git tag: ${EIGEN_TAG}")

include(FetchContent)
FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY ${EIGEN_REPOSITORY}
        GIT_TAG ${EIGEN_TAG}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(Eigen3)
