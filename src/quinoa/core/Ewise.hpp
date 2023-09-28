#pragma once

#include <quinoa/Types.h>

// Add our own types for noa::ewise_* functions.
// Note that this is only required for the CUDA backend.
#include <noa/gpu/cuda/Ewise.hpp>

namespace qn {
    struct correct_multiplicity_t {
        template<typename T0>
        NOA_FHD constexpr auto operator()(T0 value, f32 multiplicity) const noexcept {
            if (noa::math::abs(multiplicity) > 1.f)
                value /= multiplicity;
            return value;
        }
    };

    struct correct_multiplicity_rasterize_t {
        template<typename T0>
        NOA_FHD constexpr auto operator()(T0 value, f32 multiplicity) const noexcept {
            return value / (multiplicity + 1e-3f);
        }
    };

    struct subtract_within_mask_t {
        NOA_FHD constexpr auto operator()(f32 lhs, f32 rhs, f32 mask) const noexcept {
            return (lhs - rhs) * mask;
        }
    };
}

namespace noa::cuda {
    template<> struct proclaim_is_user_ewise_binary<c32, f32, c32, ::qn::correct_multiplicity_t> : std::true_type {};
    template<> struct proclaim_is_user_ewise_binary<c32, f32, c32, ::qn::correct_multiplicity_rasterize_t> : std::true_type {};
    template<> struct proclaim_is_user_ewise_trinary<f32, f32, f32, f32, ::qn::subtract_within_mask_t> : std::true_type {};
}
