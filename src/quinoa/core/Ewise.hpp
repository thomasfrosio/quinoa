#include <quinoa/Types.h>

// Add our own types for noa::ewise_* functions.
// Note that this is only required for the CUDA backend.
#include <noa/gpu/cuda/Ewise.hpp>

namespace qn {

    struct subtract_within_mask_t {
        NOA_FHD constexpr auto operator()(f32 lhs, f32 rhs, f32 mask) const noexcept {
            if (mask > 0)
                lhs -= rhs;
            return lhs;
        }
    };
}

namespace noa::cuda {
    template<> struct proclaim_is_user_ewise_trinary<f32, f32, f32, f32, ::qn::subtract_within_mask_t> : std::true_type {};
}
