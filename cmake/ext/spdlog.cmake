message(STATUS "spdlog: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET spdlog::spdlog)
    message(STATUS "Target already exists: spdlog::spdlog")
else ()
    set(spdlog_REPOSITORY https://github.com/gabime/spdlog)
    set(spdlog_TAG v1.14.1)

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

    # original: "trace","debug","info","warn","error","critical","off"
    target_compile_definitions(spdlog
        PRIVATE
        SPDLOG_LEVEL_NAMES={"debug","trace","info","status","warn","error","off"})

    message(STATUS "New imported target available: spdlog::spdlog")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "spdlog: fetching static dependency... done")
