#pragma once

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/io/Logging.h"


// Rotation-offset alignment using cosine-stretching.
// ------------------------------------
//
// Concept:
//  Select the lowest tilt as reference. Then cosine-stretch the other higher-tilt views perpendicular to various
//  tilt-axes. Cross-correlate the stretched views with the reference and find the tilt-axis, i.e., the rotation offset,
//  that gives the best cross-correlation. Indeed, the cosine-stretching should be best where the actual tilt-axis is.
//
// Strength:
//  While a good enough estimate of the shifts is required*, it is the only requisite we need to run this alignment.
//  It is also very efficient, so can be used iteratively with the PairwiseShift alignment for instance.
//  TODO *we use the cross-correlation coefficient, but we could use the cross-correlation peak
//       and make the optimization cost shift-invariant.
//
// Issues:
//  - This method will be more accurate if high tilts are included, simply because the cosine-stretching is more
//    "visible" as the tilt increases, so even a small rotation change will end up making a big change in the
//    high-tilts.
//  - This method cannot distinguish between x and x+pi. Indeed, the cosine-stretching is exactly the same for these
//    two rotation offsets. However, the CTF fitting is aware of that and will be able to select the correct rotation.

namespace qn {
    struct GlobalRotationParameters {
        Vec2<f32> highpass_filter{0.1f, 0.08f};
        Vec2<f32> lowpass_filter{0.4f, 0.05f};
        f64 absolute_max_tilt_difference{40};
        bool solve_using_estimated_gradient{false};
        noa::InterpMode interpolation_mode = noa::InterpMode::LINEAR_FAST;
        Path debug_directory;
    };

    class GlobalRotation {
    public:
        GlobalRotation() = default;

        GlobalRotation(
                const Array<f32>& stack,
                const MetadataStack& metadata,
                const GlobalRotationParameters& parameters,
                Device compute_device,
                Allocator allocator = Allocator::DEFAULT_ASYNC
        );

        void initialize(
                MetadataStack& metadata,
                const GlobalRotationParameters& parameters
        );

        void update(
                MetadataStack& metadata,
                const GlobalRotationParameters& parameters,
                f64 range_degrees
        );

    private:
        noa::Texture<f32> m_targets;
        noa::Array<c32> m_targets_stretched_rfft;
        noa::Array<c32> m_reference_rfft;

        noa::Array<Float33> m_inv_stretching_matrices;
        noa::Array<f32> m_xcorr_coefficients;

        std::vector<i32> m_target_indexes;
        i32 m_reference_index{};
    };
}
