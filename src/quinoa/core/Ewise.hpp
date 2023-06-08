#include <quinoa/Types.h>

// Add our own types for noa::ewise_* functions.
// Note that this is only required for the CUDA backend.
#include <noa/gpu/cuda/Ewise.hpp>

namespace qn {
    struct divide_max_one_t {
        NOA_FHD constexpr auto operator()(const c32& value, const f32& weight) const noexcept {
            return value / ::noa::math::max(1.f, weight);
        }
    };

    struct multiply_min_one_t {
        NOA_FHD constexpr auto operator()(const c32& value, const f32& weight) const noexcept {
            return value * ::noa::math::min(1.f, weight); // assume weight is >0
        }
    };
}

namespace noa::cuda {
    template<> struct proclaim_is_user_ewise_binary<c32, f32, c32, ::qn::divide_max_one_t> : std::true_type {};
    template<> struct proclaim_is_user_ewise_binary<c32, f32, c32, ::qn::multiply_min_one_t> : std::true_type {};
}
