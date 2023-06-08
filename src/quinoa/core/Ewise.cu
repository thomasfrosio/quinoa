#include <noa/gpu/cuda/utils/EwiseBinary.cuh>
#include "quinoa/core/Ewise.hpp"

NOA_CUDA_EWISE_BINARY_GENERATE_API

NOA_CUDA_EWISE_BINARY_INSTANTIATE_API(c32, f32, c32, qn::divide_max_one_t)
NOA_CUDA_EWISE_BINARY_INSTANTIATE_API(c32, f32, c32, qn::multiply_min_one_t)
