#pragma once

#include <numeric>

#include <noa/Array.hpp>
#include <noa/Signal.hpp>
#include <noa/Session.hpp>

#include "quinoa/CommonArea.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Types.hpp"

/// Pairwise cosine stretching alignment
/// ------------------------------------
///
/// Concept:
///  Go through the stack ordered by tilt angles, each view is aligned to a lower tilt neighbor (so by definition,
///  the lowest tilt view is the global reference and is not aligned). The known shift and rotation difference between
///  the target and the reference is corrected. The tilt and pitch, which are 3d transformations, cannot be
///  corrected. Instead, a 2d cosine stretching is applied to estimate the tilt and pitch difference. Note that
///  the pitch difference is almost always 0. Once these corrections are applied, the shift between the stretched
///  target and the reference is computed using conventional cross-correlation.
///
/// Strength:
///  This method is quite robust to large shifts, so is perfect to use as initial alignment. If the rotation is not
///  known, cosine stretching can be turned off and still be robust enough for a first estimate of the shifts.
///  It is also very efficient, so can be used iteratively, each iteration building on the next one (because the
///  difference between the target and the reference can be better estimated).
///  Overall, it gives us a good starting point.
///
/// Issues:
///  Since the neighboring views are aligned together, to get the global shift (the shift relative to the global
///  reference, i.e., the lowest tilt), we need to add the relative shifts (by an inclusive sum operation),
///  effectively accumulating the errors to the higher tilts.
///  The tilt (and pitch) difference cannot be correctly accounted for, and the cosine stretching is only a
///  2d approximation of what these 3d geometric differences would look like in the projections. While it mostly holds
///  for thin samples and at low tilt, it quickly becomes imprecise at high tilt. This implementation tries to limit
///  the drift that these errors can cause, mostly by restricting and enforcing a common area for the cross-correlation,
///  excluding (mostly at high tilt) the regions perpendicular to the tilt axis that move a lot from one tilt to the
///  next.

namespace qn {
    struct PairwiseShiftParameters {
        ns::Bandpass bandpass{};
        noa::Interp interp{noa::Interp::LINEAR_FAST};
        Path debug_directory;
    };

    struct PairwiseShiftUpdateParameters {
        bool cosine_stretch;
        bool area_match;
        f64 smooth_edge_percent;
        f64 max_shift_percent{1};
    };

    /// Shift alignment by pairwise alignment.
    /// Cosine stretching can be used to approximate for the tilt angles for the images.
    class PairwiseShift {
    public:
        PairwiseShift() = default;

        /// Initial setup. Allocates for the internal buffers.
        PairwiseShift(
            const Shape4<i64>& shape,
            Device compute_device,
            Allocator allocator = Allocator::DEFAULT_ASYNC
        );

        /// Updates the shifts in the metadata using pairwise cross-correlation.
        ///  - Starts with the view at the lowest tilt angle. To align a neighboring view
        ///    (which has a higher tilt angle), stretch that view by a factor of cos(x)
        ///    perpendicular to the tilt-axis, where x is the tilt angle of the neighboring
        ///    view, in radians. Then use conventional correlation to find the XY shift between images.
        void update(
            const View<f32>& stack,
            MetadataStack& metadata,
            const PairwiseShiftParameters& parameters,
            const PairwiseShiftUpdateParameters& update_parameters
        );

    private:
        // Compute the conventional-cross-correlation between the reference and target, accounting for the known
        // difference in rotation and shift between the two. The tilt and pitch cannot be corrected (they are
        // 3d transformations). To mitigate this difference, a scaling is applied onto the target, usually resulting
        // in a stretch perpendicular to the tilt and pitch axes (usually the pitch increment is 0).
        //
        // Importantly, the common area can be used to restrict and improve the alignment. It is computed
        // using the common geometry, which can be far off. The better the current geometry is, the better the
        // estimation of the FOV.
        //
        // As explained below, the output shift is the shift of the targets relative to their reference in the
        // "microscope" reference frame (i.e. 0-degree tilt and pitch). This simplifies computation later
        // when the global shifts and centering needs to be computed.
        [[nodiscard]] auto find_relative_shifts_(
            const View<f32>& stack,
            const MetadataSlice& reference_slice,
            const MetadataSlice& target_slice,
            const PairwiseShiftParameters& parameters,
            const PairwiseShiftUpdateParameters& update_parameters
        ) const -> Vec2<f64>;

        // Compute the global shifts, i.e., the shifts to apply to a slice so that it becomes aligned with the
        // global reference slice (i.e., the lowest tilt). At this point, we have the relative (i.e., slice-to-slice)
        // shifts in the 0deg reference frame, so we need to accumulate the shifts of the lower degree slices.
        // At the same time, we can center the global shifts to minimize the overall movement of the slices, a step
        // we referred to as "centering".
        static auto relative2global_shifts_(
            const std::vector<Vec2<f64>>& relative_shifts,
            const MetadataStack& metadata,
            i64 index_lowest_tilt,
            bool cosine_stretch
        ) -> std::vector<Vec2<f64>>;

    private:
        noa::Array<c32> m_buffer_rfft;
        noa::Array<f32> m_xmap;
        CommonArea m_common_area;
    };
}
