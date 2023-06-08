# Find the local CUDA architecture for all current devices.
# Results are written in variable CUDASelectArch_RESULTS and can be directly passed to CUDA_ARCHITECTURES.

set(IFILE ${PROJECT_SOURCE_DIR}/cmake/utils/CUDASelectArch.cu)
set(OFILE ${CMAKE_CURRENT_BINARY_DIR}/CUDASelectArch.o)
execute_process(COMMAND ${CMAKE_CUDA_COMPILER} ${IFILE} -o ${OFILE})
execute_process(COMMAND ${OFILE}
        RESULT_VARIABLE CUDASelectArch_FAILED # Returns -1 if failed, 0 otherwise
        OUTPUT_VARIABLE CUDASelectArch_RESULTS)

if (NOT ${CUDASelectArch_FAILED} EQUAL 0)
    message(FATAL_ERROR "Could not detect the CUDA architecture on this system. CUDA_ARCHITECTURES have to be set manually")
endif()
