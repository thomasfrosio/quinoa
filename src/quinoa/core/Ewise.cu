#include <noa/gpu/cuda/utils/EwiseBinary.cuh>
#include <noa/gpu/cuda/utils/EwiseTrinary.cuh>
#include "quinoa/core/Ewise.hpp"

NOA_CUDA_EWISE_BINARY_GENERATE_API
NOA_CUDA_EWISE_TRINARY_GENERATE_API

NOA_CUDA_EWISE_BINARY_INSTANTIATE_API(c32, f32, c32, qn::correct_multiplicity_t)
NOA_CUDA_EWISE_BINARY_INSTANTIATE_API(c32, f32, c32, qn::correct_multiplicity_rasterize_t)
NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(f32, f32, f32, f32, qn::subtract_within_mask_t)
