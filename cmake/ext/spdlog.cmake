message(STATUS "spdlog: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET spdlog::spdlog)
    message(STATUS "Target already exists: spdlog::spdlog")
else ()
    # spdlog made it difficult to change the log level names from CMake, so use own fork.
    set(spdlog_REPOSITORY https://github.com/thomasfrosio/spdlog)
    set(spdlog_TAG v1.15.2-name)

    message(STATUS "Repository: ${spdlog_REPOSITORY}")
    message(STATUS "Git tag: ${spdlog_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY ${spdlog_REPOSITORY}
        GIT_TAG ${spdlog_TAG}
    )
    set(SPDLOG_INSTALL ON)
    set(SPDLOG_FMT_EXTERNAL ON)
    FetchContent_MakeAvailable(spdlog)

    message(STATUS "New imported target available: spdlog::spdlog")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "spdlog: fetching static dependency... done")
