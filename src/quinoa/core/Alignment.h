#pragma once

#include <numeric>

#include <noa/Session.h>
#include <noa/Array.h>
#include <noa/Memory.h>
#include <noa/Math.h>
#include <noa/Geometry.h>
#include <noa/FFT.h>
#include <noa/Signal.h>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"

namespace qn::align {
    /// 2D XY shifts alignment using cosine stretching on the reference images.
    /// \details Starts with the view with the lowest tiltY angle. To align a neighbouring view
    ///          (which has a higher tiltY angle), stretch that view by a factor of cosine(tiltY)
    ///          perpendicular to the tilt-axis, where tiltY is the tilt angle of the neighbouring
    ///          view, in radians. Then use phase correlation to find the XY shift between images.
    ///          The returned shifts are unstretched and the average shift is centered.
    /// \param[in] tilt_series      Input tilt series.
    /// \param[in,out] stack_meta   Metadata of \p tilt_series.
    ///                             It is sorted by "tilt" when the function returns.
    /// \param compute_device       Device where to do the computation.
    /// \param max_shift            Maximum YX shift an slice is allowed to have. If <= 0, it is ignored.
    /// \returns YX shifts, in the "tilt" order.
    std::vector<float2_t>
    shiftPairwiseCosine(const Array<float>& tilt_series,
                        MetadataStack& stack_meta,
                        Device compute_device,
                        float2_t max_shift) {
        const size4_t slice_shape{1, 1, tilt_series.shape()[2], tilt_series.shape()[3]};
        const float2_t slice_center = float2_t(slice_shape.get(2)) / 2;
        const ArrayOption options{compute_device, Allocator::DEFAULT_ASYNC};

        // The "target" is the higher (absolute) tilt image that we want to align onto the "reference".
        auto [reference, reference_fft] = fft::empty<float>(slice_shape, options);
        auto [target, target_fft] = fft::empty<float>(slice_shape, options);

        // Since we need to transform the target to "overlap" with the reference, use a GPU texture
        // if possible since we need to copy the target to a new array anyway.
        Texture<float> texture(slice_shape, compute_device, INTERP_LINEAR_FAST, BORDER_ZERO);

        // The cross-correlation map. This is reused for every iteration.
        Array<float> xmap = memory::empty<float>(slice_shape, options);

        // Returns the reference-to-target YX shift, i.e. the shift of the target relative to the reference.
        // Reference: this is the lower abs(tilt), and it should be loaded in "reference".
        // Target: this is the higher abs(tilt), and it should be loaded in "texture".
        auto find_shift_yx = [&](const MetadataSlice& reference_slice,
                                 const MetadataSlice& target_slice) mutable -> float2_t {
            const float3_t target_angles = math::deg2rad(target_slice.angles);
            const float3_t reference_angles = math::deg2rad(reference_slice.angles);

            // Compute the affine matrix to transform the target "onto" the reference.
            // The tilt and pitch angle difference between the target and the reference should be >= 0,
            // resulting in a cos factor >= 1. Otherwise, we'll shrink the target and expose its
            // edges, which is an issue for the alignment since we don't taper the edges.
            // For the tilt, it should be guaranteed since the target is at a higher absolute tilt.
            // For the pitch, the scaling should be 1 anyway since this function is meant for the initial
            // alignment, when the pitch is 0.
            const float2_t cos_factor{math::cos(reference_angles[1]) / math::cos(target_angles[1]),
                                      math::cos(reference_angles[1]) / math::cos(target_angles[1])};
            // FIXME If both axis are < 1, flip target and reference.
            //       If only one is < 1, taper the edge?

            // Apply the cos stretch on the target, perpendicular to its pitch (X') and tilt (Y') axis.
            // Take into account any eventual shifts.
            const float33_t cos_matrix{
                    geometry::translate(slice_center + target_slice.shifts) *
                    float33_t{geometry::rotate(target_angles[0])} *
                    float33_t{geometry::scale(cos_factor)} *
                    float33_t{geometry::rotate(-target_angles[0])} *
                    geometry::translate(-slice_center - target_slice.shifts)
            };

            // In practice, since this function is used for the initial alignment, this matrix
            // should not be necessary. However, if the reference and target have a shift and/or
            // in-place rotation (yaw), we need to apply it.
            const float33_t diff_matrix{
                    geometry::translate(slice_center + reference_slice.shifts) *
                    float33_t{geometry::rotate(-target_angles[0] + reference_angles[0])} *
                    geometry::translate(-slice_center - target_slice.shifts)
            };

            // After this point, the target should "overlap" with the reference.
            geometry::transform2D(texture, target, math::inverse(diff_matrix * cos_matrix));

            // Find and apply shift:
            fft::r2c(reference, reference_fft);
            fft::r2c(target, target_fft);
            signal::fft::xmap<fft::H2F>(reference_fft, target_fft, xmap);

            // Computes the YX shift of the target. To shift-align the target onto the reference,
            // we would then need to subtract this shift. Note however that the target is stretched
            // at this point, so as the shift. Thus, we need to apply the inverse of the stretch
            // to the shift for it to become the shift of the original target.
            const float2_t peak = signal::fft::xpeak2D<fft::F2F>(xmap, max_shift);
            const float2_t stretched_shift = peak - slice_center;
            const float3_t shift = math::inverse(cos_matrix) * float3_t{1, stretched_shift[0], stretched_shift[1]};

            return {shift[1], shift[2]};
        };

        // We'll need the slices sorted from by tilt angles from this point.
        stack_meta.sort("tilt");
        const size_t idx_lowest_tilt = stack_meta.lowestTilt();

        // The main processing loop. From the lowest to the highest tilt, find the per-view shifts.
        // These shifts are the slice-to-slice shifts, i.e. the shift to apply to the target to align
        // it onto its neighbour reference.
        std::vector<float2_t> slice_to_slice_shifts(stack_meta.size());
        for (size_t idx_target = 0; idx_target < stack_meta.size(); ++idx_target) {
            if (idx_lowest_tilt == idx_target) {
                slice_to_slice_shifts.emplace_back(0);
                continue;
            }

            // Save the target in the texture/buffer.
            texture.update(tilt_series.subregion(stack_meta[idx_target].index));

            // If ith target has:
            //  - negative tilt angle, then reference is i + 1.
            //  - positive tilt angle, then reference is i - 1.
            const bool is_negative = idx_target < idx_lowest_tilt;
            const size_t idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;

            // Point to the reference slice.
            memory::copy(tilt_series.subregion(stack_meta[idx_reference].index), reference);

            const float2_t shift = find_shift_yx(stack_meta[idx_reference], stack_meta[idx_target]);
            slice_to_slice_shifts.emplace_back(shift);
        }

        // Compute the global shifts (i.e. shifts relative to the lowest tilt).
        // This is the main drawback of this method, since the high tilt slices accumulate
        // the errors of the lower tilts. Projection matching should be more robust against this...
        std::vector<float2_t> global_shifts(slice_to_slice_shifts.size());
        const auto idx_pivot = static_cast<int64_t>(idx_lowest_tilt);
        for (size_t i = 0; i < stack_meta.size(); ++i) {
            if (idx_lowest_tilt == i)
                continue;

            const bool is_negative = i < idx_lowest_tilt;
            const int64_t direction = 1 * is_negative - 1 * !is_negative;

            // Accumulate the shifts up to the lowest absolute tilt.
            float2_t global_shift{0};
            auto idx_current = static_cast<int64_t>(i);
            while (idx_current != idx_pivot) {
                global_shift += global_shifts[static_cast<size_t>(idx_current)];
                idx_current += direction;
            }
        }

        // The shifts are not centered, and ultimately the lowest tilt has no shift since it is not aligned.
        return global_shifts;
    }
}
