message(STATUS "glob: fetching static dependency...")
set(glob_REPOSITORY https://github.com/thomasfrosio/glob-cpp)
set(glob_TAG master)

message(STATUS "Repository: ${glob_REPOSITORY}")
message(STATUS "Git tag: ${glob_TAG}")

include(FetchContent)
FetchContent_Declare(
    glob
    GIT_REPOSITORY ${glob_REPOSITORY}
    GIT_TAG ${glob_TAG}
)
FetchContent_MakeAvailable(glob)
message(STATUS "glob: fetching static dependency... done")
