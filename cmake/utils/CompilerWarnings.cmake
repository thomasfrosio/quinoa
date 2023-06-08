# Adds the C++ compiler warning (GCC or Clang) to the interface.
# Use:
#   - QN_ENABLE_WARNINGS
#   - QN_ENABLE_WARNINGS_AS_ERRORS
function(set_cxx_compiler_warnings interface)
    if (NOT QN_ENABLE_WARNINGS)
        return()
    endif ()

    set(PRJ_CLANG_WARNINGS
            -Wall
            -Wextra # reasonable and standard
            # -Wunused # warn on anything being unused
            -Wshadow # warn the user if a variable declaration shadows one from a parent context
            -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor.
            # This helps catch hard to track down memory errors
            # -Wold-style-cast # warn for c-style casts
            -Wcast-align # warn for potential performance problem casts
            -Woverloaded-virtual # warn if you overload (not override) a virtual function
            -Wpedantic # warn if non-standard C++ is used
            -Wconversion # warn on type conversions that may lose data
            -Wsign-conversion # warn on sign conversions
            -Wnull-dereference # warn if a null dereference is detected
            -Wdouble-promotion # warn if float is implicit promoted to double
            -Wformat=2 # warn on security issues around functions that format output (ie printf)
            -Wimplicit-fallthrough # warn on statements that fallthrough without an explicit annotation
            )

    if (QN_ENABLE_WARNINGS_AS_ERRORS)
        set(PRJ_CLANG_WARNINGS ${PRJ_CLANG_WARNINGS} -Werror)
    endif ()

    set(PRJ_GCC_WARNINGS
            ${PRJ_CLANG_WARNINGS}
            -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
            -Wduplicated-cond # warn if if / else chain has duplicated conditions
            -Wduplicated-branches # warn if if / else branches have duplicated code
            -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
            -Wuseless-cast # warn if you perform a cast to the same type
            )

    if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(PRJ_WARNINGS ${PRJ_CLANG_WARNINGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(PRJ_WARNINGS ${PRJ_GCC_WARNINGS})
    else ()
        # TODO Add MSVC
        message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif ()

    # Only add the warnings for the C++ language, e.g. nvcc doesn't support these warnings.
    target_compile_options(${interface} INTERFACE $<$<COMPILE_LANGUAGE:CXX>: ${PRJ_WARNINGS}>)
endfunction()

function(set_cuda_compiler_warnings interface)
    if (NOT QN_ENABLE_WARNINGS)
        set(PRJ_CUDA_WARNINGS --disable-warnings)
    else ()
        set(PRJ_CUDA_WARNINGS
                --Wreorder # Generate warnings when member initializers are reordered.
                --Wdefault-stream-launch # Generate warning when an explicit stream argument is not provided in the <<<...>>> kernel launch syntax.
                --Wext-lambda-captures-this # Generate warning when an extended lambda implicitly captures 'this'.
                )
    endif ()

    if (QN_ENABLE_WARNINGS_AS_ERRORS)
        set(PRJ_CUDA_WARNINGS ${PRJ_CUDA_WARNINGS} -Werror all-warnings)
    endif ()

    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(PRJ_WARNINGS ${PRJ_CUDA_WARNINGS})
    else ()
        message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CUDA_COMPILER_ID}' compiler.")
    endif ()

    # Only add the warnings for the CUDA language.
    target_compile_options(${interface} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: ${PRJ_WARNINGS}>)
endfunction()
