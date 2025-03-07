#pragma once

#include <noa/Array.hpp>
#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>
#include <noa/IO.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/CommonArea.hpp"

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
    struct RotationOffsetParameters {
        bool reset_rotation{false};
        noa::Interp interp{noa::Interp::LINEAR};
        ns::Bandpass bandpass{
            .highpass_cutoff = 0.1,
            .highpass_width = 0.08,
            .lowpass_cutoff = 0.4,
            .lowpass_width = 0.05,
        };

        f64 grid_search_range{90.};
        f64 grid_search_step{1.};
        f64 grid_search_line_range{1.2};
        f64 grid_search_line_delta{0.1};

        f64 local_search_range{5.};
        f64 local_search_line_range{1.2};
        f64 local_search_line_delta{0.1};
        bool local_search_using_estimated_gradient{false};

        Path output_directory;
        Path debug_directory;
    };

    class RotationOffset {
    public:
        RotationOffset() = default;

        RotationOffset(
            const View<f32>& stack,
            const MetadataStack& metadata,
            f64 absolute_max_tilt_difference = 70
        );

        void search(
            const View<f32>& input,
            MetadataStack& metadata,
            const RotationOffsetParameters& parameters
        );

    private:
        Array<f32> m_slices{};
        Array<c32> m_slices_rfft{};
        MetadataStack m_metadata_sorted{};
    };
}

