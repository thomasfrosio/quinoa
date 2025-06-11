set(cxxopts_REPOSITORY https://github.com/jarro2783/cxxopts)
set(cxxopts_TAG 52e8f52)

message(STATUS "Repository: ${cxxopts_REPOSITORY}")
message(STATUS "Git tag: ${cxxopts_TAG}")

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY ${cxxopts_REPOSITORY}
    GIT_TAG ${cxxopts_TAG}
)
FetchContent_MakeAvailable(cxxopts)
