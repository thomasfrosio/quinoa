#include <quinoa/Types.h>

// Add our own types for noa::ewise_* functions.
// Note that this is only required for the CUDA backend.
#include <noa/gpu/cuda/Ewise.hpp>

namespace qn {
    // Frequencies with a multiplicity less than one are below 1 not divided by the multiplicity, with low
    // confidence or without any signal in the projected-reference, but everything above 1 is left unchanged.
    struct correct_multiplicity_t {
        template<typename T0>
        NOA_FHD constexpr auto operator()(T0 value, f32 multiplicity) const noexcept {
            if (noa::math::abs(multiplicity) > 1.f)
                value /= multiplicity;
            return value;
        }
    };

    struct correct_multiplicity_and_multiply_t {
        template<typename T0, typename T1>
        NOA_FHD constexpr auto operator()(T0 value, f32 multiplicity, T1 scale) const noexcept {
            if (noa::math::abs(multiplicity) > 1.f)
                value /= multiplicity;
            return value * scale;
        }
    };
}

namespace noa::cuda {
    template<> struct proclaim_is_user_ewise_binary<c32, f32, c32, ::qn::correct_multiplicity_t> : std::true_type {};
    template<> struct proclaim_is_user_ewise_binary<f32, f32, f32, ::qn::correct_multiplicity_t> : std::true_type {};
    template<> struct proclaim_is_user_ewise_trinary<c32, f32, f32, c32, ::qn::correct_multiplicity_and_multiply_t> : std::true_type {};
    template<> struct proclaim_is_user_ewise_trinary<f32, f32, c32, c32, ::qn::correct_multiplicity_and_multiply_t> : std::true_type {};
}
