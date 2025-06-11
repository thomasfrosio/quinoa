set(tomlplusplus_REPOSITORY https://github.com/marzer/tomlplusplus)
set(tomlplusplus_TAG 2f35c28)

message(STATUS "Repository: ${tomlplusplus_REPOSITORY}")
message(STATUS "Git tag: ${tomlplusplus_TAG}")

include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY ${tomlplusplus_REPOSITORY}
    GIT_TAG ${tomlpp_TAG}
)
FetchContent_MakeAvailable(tomlplusplus)
