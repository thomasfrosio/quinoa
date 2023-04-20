#pragma once

#include <numeric>

#include <noa/Session.hpp>
#include <noa/Array.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/io/Logging.h"

// Pairwise cosine stretching alignment
// ------------------------------------
//
// Concept:
//  Find the per-view shift of the
//  Go through the stack ordered by tilt angles, where each view is aligned to lower tilt neighbour (so by definition
//  the lowest tilt view is the global reference and is not aligned). The known shift and rotation difference between
//  the target and the reference is corrected. The tilt and elevation, which are 3d transformations, cannot be
//  corrected. Instead, a 2d cosine stretching is applied to estimate the tilt and elevation difference. Note that
//  the elevation difference is almost always 0. Once these correction are applied, the shift between the stretched
//  target and the reference is computed using the conventional cross-correlation.
//
// Strength:
//  This method is quite robust to large shifts so is perfect as first step of the alignment. If the rotation is not
//  known, cosine stretching can be turned off and still be robust enough for a first estimate of the shifts.
//  It is also very efficient, so can be used iteratively, each iteration building on the next one (because the
//  difference between the target and the reference can be better estimated).
//  Overall, it gives us a good starting point.
//
// Issues:
//  Since the neighbouring views are aligned together, to get the global shift (the shift relative to the global
//  reference), we need to add the relative shifts (inclusive sum operation), effectively accumulating the errors
//  to the higher tilts.
//  The tilt (and elevation) difference cannot be correctly accounted for and the cosine stretching is only an
//  approximation. While it mostly holds for thin samples and at low tilt, it quickly becomes imprecise at high tilt.
//  This implementation tries to limit the drift that these errors can cause, mostly by restricting and enforcing
//  a common area for the cross-correlation, excluding the regions perpendicular to the tilt axis that move a lot
//  from one tilt to the next.

namespace qn {
    struct PairwiseShiftParameters {
        Vec2<f32> max_shift{};
        f32 pairwise_fov_taper{}; // percent of image size
        f32 area_match_taper{}; // percent of image size

        Vec2<f32> highpass_filter{};
        Vec2<f32> lowpass_filter{};

        bool center_shifts{};
        noa::InterpMode interpolation_mode = noa::InterpMode::LINEAR_FAST;
        Path debug_directory;
    };

    /// Shift alignment using cosine stretching on the higher tilt images.
    class PairwiseShift {
    public:
        PairwiseShift() = default;

        /// Allocates for the internal buffers.
        /// \param shape                (BD)HW shape of the slices. The BD dimensions are ignored.
        /// \param compute_device       Device where to perform the alignment.
        /// \param smooth_edge_percent  Size, in percent of the maximum-sized dimension, of the zero-edge taper.
        /// \param interpolation_mode   Interpolation mode used for the cosine-stretching.
        /// \param allocator            Allocator to use for \p compute_device.
        PairwiseShift(const Shape4<i64>& shape, noa::Device compute_device,
                       noa::Allocator allocator = noa::Allocator::DEFAULT_ASYNC);

        /// 2D in-place shifts alignment using cosine-stretching.
        /// \details Starts with the view at the lowest tilt angle. To align a neighbouring view
        ///          (which has a higher tilt angle), stretch that view by a factor of cos(x)
        ///          perpendicular to the tilt-axis, where x is the tilt angle of the neighbouring
        ///          view, in radians. Then use conventional correlation to find the XY shift between images.
        /// \param[in] stack        Input stack.
        /// \param[in,out] metadata Metadata of \p stack. This is updated with the new shifts.
        /// \param max_shift        Maximum YX shift a slice is allowed to have. If <= 0, it is ignored.
        /// \param center           Whether the average shift (in the microscope reference frame) should be centered.
        void update(const Array<f32>& stack,
                    MetadataStack& metadata,
                    const PairwiseShiftParameters& parameters,
                    bool cosine_stretch = true,
                    bool area_match = true);

    private:
        // Compute the conventional-cross-correlation between the reference and target, accounting for the known
        // difference in rotation and shift between the two. The tilt and elevation cannot be corrected (they are
        // 3d transformations). To mitigate this difference, a scaling is applied onto the target, usually resulting
        // in a stretch perpendicular to the tilt and elevation axes (usually the elevation increment is 0).
        //
        // Importantly, the common field-of-view (FOV) can be aligned, and edges are smoothed out. This FOV is computed
        // using the common geometry, which can be far off. The better the current geometry is, the better the
        // estimation of the FOV.
        //
        // As explained below, the output shift is the shift of the targets relative to their reference in the
        // "microscope" reference frame (i.e. 0-degree tilt and pitch). This simplifies computation later
        // when the global shifts and centering needs to be computed.
        Vec2<f64> find_relative_shifts_(const Array<f32>& stack,
                                        const MetadataSlice& reference_slice,
                                        const MetadataSlice& target_slice,
                                        const PairwiseShiftParameters& parameters,
                                        bool cosine_stretch,
                                        bool area_match);

        // Enforce a common area across the tilt series.
        // This is more restrictive and removes regions from the higher tilts that aren't in the 0deg view.
        static void apply_area_match_(
                const View<f32>& input,
                const View<f32>& output,
                const MetadataSlice& metadata,
                const PairwiseShiftParameters& parameters);

        // Compute the global shifts, i.e. the shifts to apply to a slice so that it becomes aligned with the
        // global reference slice (i.e. the lowest tilt). At this point, we have the relative (i.e. slice-to-slice)
        // shifts in the 0deg reference frame, so we need to accumulate the shifts of the lower degree slices.
        // At the same time, we can center the global shifts to minimize the overall movement of the slices, a step
        // referred to as "centering".
        static std::vector<Vec2<f64>> relative2global_shifts_(
                const std::vector<Vec2<f64>>& relative_shifts,
                const MetadataStack& metadata,
                i64 index_lowest_tilt,
                bool center_tilt_axis);

    private:
        noa::Array<c32> m_buffer_rfft;
        noa::Array<f32> m_xmap;
    };
}
