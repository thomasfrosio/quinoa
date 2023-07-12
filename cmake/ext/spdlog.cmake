message(STATUS "spdlog: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET spdlog::spdlog)
    message(STATUS "Target already exists: spdlog::spdlog")
else ()
    set(spdlog_REPOSITORY https://github.com/gabime/spdlog)
    set(spdlog_TAG v1.11.0)

    message(STATUS "Repository: ${spdlog_REPOSITORY}")
    message(STATUS "Git tag: ${spdlog_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY ${spdlog_REPOSITORY}
        GIT_TAG ${spdlog_TAG}
    )
    option(SPDLOG_INSTALL "Enable installation for the spdlog project." ON)
    option(SPDLOG_FMT_EXTERNAL "Use the external {fmt} library" ON)
    FetchContent_MakeAvailable(spdlog)

    message(STATUS "New imported target available: spdlog::spdlog")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "spdlog: fetching static dependency... done")
