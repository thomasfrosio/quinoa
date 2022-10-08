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

const path_t OUTPUT_PATH = "/home/thomas/Projects/quinoa/tests/qn_tilt1/";

namespace qn::align {
    /// 2D XY shifts alignment using cosine stretching on the reference images.
    /// \details Starts with the view with the lowest tiltY angle. To align a neighbouring view
    ///          (which has a higher tiltY angle), stretch that view by a factor of cosine(tiltY)
    ///          perpendicular to the tilt-axis, where tiltY is the tilt angle of the neighbouring
    ///          view, in radians. Then use phase correlation to find the XY shift between images.
    ///          The returned shifts are unstretched and the average shift is centered.
    /// \param[in] stack            Input stack.
    /// \param[in,out] stack_meta   Metadata of \p stack. The slice indexes should correspond to the batch indexes
    ///                             is the \p stack. It can contain excluded slices, these will simply be ignored.
    ///                             When the function returns, this metadata is updated with the new shifts.
    ///                             The order of the slices is unchanged and excluded slices are preserved.
    /// \param compute_device       Device where to do the computation. If it is a GPU, \p stack can be on any device,
    ///                             including the CPU. If it is a CPU, \p stack must be on the CPU as well.
    /// \param max_shift            Maximum YX shift a slice is allowed to have. If <= 0, it is ignored.
    void shiftPairwiseCosine(const Array<float>& stack,
                             MetadataStack& stack_meta,
                             Device compute_device,
                             float2_t max_shift) {
        const size4_t slice_shape{1, 1, stack.shape()[2], stack.shape()[3]};
        const float2_t slice_center = float2_t(slice_shape.get(2)) / 2;
        const ArrayOption options{compute_device, Allocator::DEFAULT_ASYNC};

        // The "target" is the higher (absolute) tilt image that we want to align onto the "reference".
        auto [reference, reference_fft] = fft::empty<float>(slice_shape, options);
        auto [target, target_fft] = fft::empty<float>(slice_shape, options);

        // Since we need to transform the target to "overlap" with the reference, use a GPU texture
        // if possible since we need to copy the target to a new array anyway.
        Texture<float> target_texture(slice_shape, compute_device, INTERP_LINEAR_FAST, BORDER_MIRROR);

        // The cross-correlation map. This is reused for every iteration.
        Array<float> xmap = memory::empty<float>(slice_shape, options);

        // Reference: this is the lower abs(tilt), and it should be loaded in "reference".
        // Target: this is the higher abs(tilt), and it should be loaded in "texture".
        // This lambda computes the phase-correlation between the reference and target, accounting for the
        // known difference in rotation and shift between the reference and target. The yaw and shift difference
        // can be directly corrected, but the tilt and pitch cannot (these are 3D transformations for 2D slices).
        // To still mitigated this difference, a scaling is applied onto the target, usually resulting in a stretch
        // perpendicular to the tilt and pitch axes.
        // As explained below, the output shift is the shift of the target relative to the reference, but this
        // shift is stretched to the 0deg reference frame. This simplifies computation later when the global shifts
        // and centering needs to be computed.
        auto find_shift_yx = [&](const MetadataSlice& reference_slice,
                                 const MetadataSlice& target_slice) mutable -> double2_t {
            const float3_t target_angles = math::deg2rad(target_slice.angles);
            const float3_t reference_angles = math::deg2rad(reference_slice.angles);

            // Compute the affine matrix to transform the target "onto" the reference.
            // We expected the (absolute value of the) tilt and pitch of the target to be greater than the
            // reference's, resulting in a stretching of the target. Otherwise, we'll end up shrinking the target,
            // exposing its edges, which is an issue for the alignment since we don't taper the edges.
            // For the tilt, it should be guaranteed since the target is at a higher absolute tilt.
            // For the pitch, the scaling should be 1 anyway since this function is meant for the initial
            // alignment, when the pitch is 0.
            // These angles are flipped, since the scaling is perpendicular to the axis of rotation.
            const float2_t cos_factor{math::cos(reference_angles[2]) / math::cos(target_angles[2]),
                                      math::cos(reference_angles[1]) / math::cos(target_angles[1])};
            // FIXME If both axis are < 1, flip target and reference.
            //       If only one is < 1, taper the edge?

            // Apply the scaling for the tilt and pitch difference,
            // and cancel the difference (if any) in yaw and shift as well.
            const float33_t stretch_target_to_reference{
                    geometry::translate(slice_center + reference_slice.shifts) *
                    float33_t{geometry::rotate(reference_angles[0])} *
                    float33_t{geometry::scale(cos_factor)} *
                    float33_t{geometry::rotate(-target_angles[0])} *
                    geometry::translate(-slice_center - target_slice.shifts)
            };

            // After this point, the target should "overlap" with the reference.
            geometry::transform2D(target_texture, target, math::inverse(stretch_target_to_reference));
//            io::save(target, OUTPUT_PATH / "target_after.mrc");
//            io::save(reference, OUTPUT_PATH / "reference.mrc");

            // Find and apply shift:
            fft::r2c(reference, reference_fft);
            fft::r2c(target, target_fft);
            signal::fft::xmap<fft::H2F>(target_fft, reference_fft, xmap);
//            io::save(xmap, OUTPUT_PATH / "xmap.mrc");

            // Computes the YX shift of the target. To shift-align the stretched target onto the reference,
            // we would then need to subtract this shift to the stretched target.
            const float2_t peak = signal::fft::xpeak2D<fft::F2F>(xmap, max_shift);
            const double2_t shift_stretched_reference(peak - slice_center);

            // We could destretch the shift to bring it back to the original target. However, we'll need
            // to compute the global shifts later on, as well as "centering the tilt-axis". This implies
            // to accumulate the shifts of the lower view while accounting for their own scaling.
            // It seems that it is simpler to scale all the slice-to-slice shifts to the same reference
            // frame, process everything there, and then go back to whatever higher tilt reference at
            // the end. Here, this common reference frame is the untilted and unpitched specimen.
            const float2_t reference_pivot_tilt{reference_angles[2], reference_angles[1]};
            const double22_t stretch_to_0deg{
                    geometry::rotate(target_angles[0]) *
                    geometry::scale(1 / math::cos(reference_pivot_tilt)) * // 1 = cos(0deg)
                    geometry::rotate(-target_angles[0])
            };
            const double2_t shift_stretched_to_0deg = stretch_to_0deg * shift_stretched_reference;
            return shift_stretched_to_0deg;
        };

        // We'll need the slices sorted by tilt angles, with the lowest absolute tilt being the pivot point.
        // The metadata is allowed to contain excluded views, so for simplicity here, squeeze them out.
        MetadataStack metadata = stack_meta;
        metadata.squeeze().sort("tilt");
        const size_t idx_lowest_tilt = metadata.lowestTilt();
        const size_t slice_count = metadata.size();

        // The main processing loop. From the lowest to the highest tilt, find the per-view shifts.
        // These shifts are the slice-to-slice shifts, i.e. the shift to apply to the target to align
        // it onto its neighbour reference.
        std::vector<double2_t> slice_to_slice_shifts;
        slice_to_slice_shifts.reserve(slice_count);
        for (size_t idx_target = 0; idx_target < slice_count; ++idx_target) {
            if (idx_lowest_tilt == idx_target) {
                slice_to_slice_shifts.emplace_back(0);
                continue;
            }

            // If ith target has:
            //  - negative tilt angle, then reference is at i + 1.
            //  - positive tilt angle, then reference is at i - 1.
            const bool is_negative = idx_target < idx_lowest_tilt;
            const size_t idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;

//            io::save(stack.subregion(metadata[idx_target].index), OUTPUT_PATH / "target_before.mrc");

            // Compute the shifts.
            target_texture.update(stack.subregion(metadata[idx_target].index));
            memory::copy(stack.subregion(metadata[idx_reference].index), reference);
            slice_to_slice_shifts.emplace_back(find_shift_yx(metadata[idx_reference], metadata[idx_target]));
        }

        // Compute the global shifts, i.e. the shifts to apply to a slice so that it becomes aligned with the
        // lowest slice. At this point, we have the slice-to-slice shifts at the 0deg reference frame, so we
        // need to accumulate the shifts of the lower slices. At the same time, we can center the average
        // global shift to minimize the overall movement of the slices.
        // This is the main drawback of this method; the high tilt slices accumulate the errors of the lower tilts.
        std::vector<double2_t> global_shifts(slice_count);
        const auto idx_pivot = static_cast<int64_t>(idx_lowest_tilt);
        const auto idx_rpivot = static_cast<int64_t>(slice_count) - 1 - idx_pivot;
        const auto mean_scale = 1 / static_cast<double>(slice_count);
        double2_t mean{0};
        auto scan_op = [mean_scale, &mean](auto& current, auto& next) {
            const auto out = current + next;
            mean += out * mean_scale;
            return out;
        };
        std::inclusive_scan(slice_to_slice_shifts.begin() + idx_pivot,
                            slice_to_slice_shifts.end(),
                            global_shifts.begin() + idx_pivot,
                            scan_op);
        std::inclusive_scan(slice_to_slice_shifts.rbegin() + idx_rpivot,
                            slice_to_slice_shifts.rend(),
                            global_shifts.rbegin() + idx_rpivot,
                            scan_op);

        // We have the global shifts, so center them and scale them back to the original reference frame
        // of the slices, i.e. shrink the shifts to account for the slice's tilt and pitch.
        for (size_t i = 0; i < slice_count; ++i) {
            const double3_t angles(math::deg2rad(metadata[i].angles));
            const double22_t shrink_matrix{
                    geometry::rotate(angles[0]) *
                    geometry::scale(math::cos(double2_t(angles[2], angles[1]))) *
                    geometry::rotate(-angles[0])
            };
            global_shifts[i] = shrink_matrix * (global_shifts[i] - mean);
        }

        // Update the metadata. Use the slice index to find the math between slices.
        for (size_t i = 0; i < slice_count; ++i) {
            for (auto& original_slice: stack_meta.slices()) {
                if (original_slice.index == metadata[i].index)
                    original_slice.shifts += static_cast<float2_t>(global_shifts[i]);
            }
        }
    }
}
