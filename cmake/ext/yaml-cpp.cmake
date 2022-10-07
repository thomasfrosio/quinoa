set(yaml-cpp_REPOSITORY https://github.com/ffyr2w/yaml-cpp)
set(yaml-cpp_TAG master)

message(STATUS "Repository: ${yaml-cpp_REPOSITORY}")
message(STATUS "Git tag: ${yaml-cpp_TAG}")

include(FetchContent)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY ${yaml-cpp_REPOSITORY}
        GIT_TAG ${yaml-cpp_TAG}
)
FetchContent_MakeAvailable(yaml-cpp)
