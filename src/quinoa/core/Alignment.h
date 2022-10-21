#pragma once

#include <numeric>

#include <noa/Session.h>
#include <noa/Array.h>
#include <noa/Memory.h>
#include <noa/Math.h>
#include <noa/Geometry.h>
#include <noa/FFT.h>
#include <noa/Signal.h>
#include <noa/Utils.h>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Geometry.h"
#include "quinoa/io/Logging.h"

const path_t OUTPUT_PATH = "/home/thomas/Projects/quinoa/tests/debug_data/";

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
    /// \param center               Whether the average shift (in the microscope reference frame) should be centered.
    void shiftPairwiseCosine(const Array<float>& stack,
                             MetadataStack& stack_meta,
                             Device compute_device,
                             float2_t max_shift = {},
                             bool center = true) {
        qn::Logger::trace("Pairwise alignment using cosine-stretching...");
        Timer timer0;
        timer0.start();

        const size4_t slice_shape{1, 1, stack.shape()[2], stack.shape()[3]};
        const float2_t slice_center = float2_t(size2_t(slice_shape.get(2)) / 2);
        const ArrayOption options{compute_device, Allocator::DEFAULT_ASYNC};

        // The "target" is the higher (absolute) tilt image that we want to align onto the "reference".
        auto [reference, reference_fft] = fft::empty<float>(slice_shape, options);
        auto [target, target_fft] = fft::empty<float>(slice_shape, options);

        // Since we need to transform the target to "overlap" with the reference, use a GPU texture
        // if possible since we need to copy the target to a new array anyway. Use BORDER_MIRROR assuming
        // the edges of the stack are not tapered. Stretching/shrinking can introduce very small edges, which is
        // completely mitigated by this border mode.
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

            // Apply the scaling for the tilt and pitch difference,
            // and cancel the difference (if any) in yaw and shift as well.
            const float33_t stretch_target_to_reference{
                    noa::geometry::translate(slice_center + reference_slice.shifts) *
                    float33_t{noa::geometry::rotate(reference_angles[0])} *
                    float33_t{noa::geometry::scale(cos_factor)} *
                    float33_t{noa::geometry::rotate(-target_angles[0])} *
                    noa::geometry::translate(-slice_center - target_slice.shifts)
            };

            // After this point, the target should "overlap" with the reference.
            noa::geometry::transform2D(target_texture, target, math::inverse(stretch_target_to_reference));

            // Phase cross-correlation:
            fft::r2c(reference, reference_fft);
            fft::r2c(target, target_fft);
            signal::fft::xmap<fft::H2F>(target_fft, reference_fft, xmap);

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
                    noa::geometry::rotate(target_angles[0]) *
                    noa::geometry::scale(1 / math::cos(reference_pivot_tilt)) * // 1 = cos(0deg)
                    noa::geometry::rotate(-target_angles[0])
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
            Timer timer1;
            timer1.start();

            // If ith target has:
            //  - negative tilt angle, then reference is at i + 1.
            //  - positive tilt angle, then reference is at i - 1.
            const bool is_negative = idx_target < idx_lowest_tilt;
            const size_t idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;

            // Compute the shifts.
            target_texture.update(stack.subregion(metadata[idx_target].index));
            memory::copy(stack.subregion(metadata[idx_reference].index), reference);
            slice_to_slice_shifts.emplace_back(find_shift_yx(metadata[idx_reference], metadata[idx_target]));

            qn::Logger::trace("Slice {:0>2} took {}ms", metadata[idx_target].index, timer1.elapsed());
        }

        // Compute the global shifts, i.e. the shifts to apply to a slice so that it becomes aligned with the
        // lowest slice. At this point, we have the slice-to-slice shifts at the 0deg reference frame, so we
        // need to accumulate the shifts of the lower slices. At the same time, we can center the global shifts
        // to minimize the overall movement of the slices.
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
        if (!center)
            mean = 0;

        // We have the global shifts, so center them and scale them back to the original reference frame
        // of the slices, i.e. shrink the shifts to account for the slice's tilt and pitch.
        for (size_t i = 0; i < slice_count; ++i) {
            const double3_t angles(math::deg2rad(metadata[i].angles));
            const double22_t shrink_matrix{
                    noa::geometry::rotate(angles[0]) *
                    noa::geometry::scale(math::cos(double2_t(angles[2], angles[1]))) *
                    noa::geometry::rotate(-angles[0])
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

        qn::Logger::trace("Pairwise alignment using cosine-stretching... took {}ms", timer0.elapsed());
    }

    /// Preprocess the stack for project matching.
    /// \details Projection matching needs the slices to be tapered to 0, highpass filtered
    ///          (preferably with a very smooth edge) and standardize.
    /// \param[in,out] stack    Stack to preprocess.
    /// \param[in] stack_meta   Metadata of \p stack. This is just to keep track of the excluded slices.
    /// \param compute_device   Device where to do the computation.
    /// \param highpass_cutoff  Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                         At this frequency, the pass is fully recovered.
    /// \param highpass_edge    Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \param smooth_edge_size Size of the (real space) taper, in pixels.
    ///                         If negative, it becomes the size, in percent of the largest dimension.
    /// \param standardize      Whether the slices should be centered and standardized. Note that this
    ///                         is applied before the taper, so the output mean and variance might be slightly off.
    void preprocessProjectionMatching(const Array<float>& stack,
                                      MetadataStack& stack_meta,
                                      Device compute_device,
                                      float highpass_cutoff = 0.05f,
                                      float highpass_edge = 0.05f,
                                      float smooth_edge_size = -0.05f,
                                      bool standardize = true) {
        const ArrayOption options{compute_device, Allocator::DEFAULT_ASYNC};
        const size4_t slice_shape{1, 1, stack.shape()[2], stack.shape()[3]};
        const float2_t slice_center = float2_t(size2_t(slice_shape.get(2)) / 2);

        highpass_cutoff = std::max(highpass_cutoff, 0.f);
        if (smooth_edge_size < 0)
            smooth_edge_size = static_cast<float>(std::max(stack.shape()[2], stack.shape()[3])) * -smooth_edge_size;

        const bool copy_to_compute_device = compute_device != stack.device();
        Array slice = copy_to_compute_device ? Array<float>(slice_shape, options) : Array<float>();
        Array slice_fft = noa::memory::empty<cfloat_t>(slice_shape.fft(), options);

        for (dim_t i = 0; i < stack_meta.size(); ++i) {
            if (stack_meta[i].excluded)
                continue;

            if (copy_to_compute_device)
                noa::memory::copy(stack.subregion(stack_meta[i].index), slice);
            else
                slice = stack.subregion(stack_meta[i].index);

            noa::fft::r2c(slice, slice_fft);
            noa::signal::fft::highpass<fft::H2H>(slice_fft, slice_fft, slice_shape, highpass_cutoff, highpass_edge);
            if (standardize)
                noa::signal::fft::standardize<fft::H2H>(slice_fft, slice_fft, slice_shape);
            noa::fft::c2r(slice_fft, slice);

            noa::signal::rectangle(slice, slice, slice_center, slice_center - smooth_edge_size, smooth_edge_size);

            if (copy_to_compute_device)
                noa::memory::copy(slice, stack.subregion(stack_meta[i].index));
        }
    }

    void shiftProjectionMatching(const Array<float>& stack,
                                 MetadataStack& stack_meta,
                                 Device compute_device,
                                 float2_t max_shift = {}) {
        const dim_t size_z_pad = std::min(stack.shape()[2], stack.shape()[3]) * 2;
        const dim_t size_y_pad = stack.shape()[2] * 2;
        const dim_t size_x_pad = stack.shape()[3] * 2;

        // The projector needs the following: 1) The target shape is the shape of the 3D Fourier volume.
        // Here, we'll use a
        const dim4_t slice_shape{1, 1, stack.shape()[2], stack.shape()[3]};
        const dim4_t slice_shape_padded{1, 1, size_y_pad, size_x_pad};
        const float2_t slice_center = float2_t(dim2_t(slice_shape.get(2)) / 2);
        const dim4_t target_shape_padded{1, 512, size_y_pad, size_x_pad};
        const dim4_t grid_shape_padded{1,
                                       static_cast<float>(size_z_pad) * 0.15f,
                                       size_y_pad,
                                       static_cast<float>(size_x_pad) * 0.2f};

        //
        MetadataStack metadata = stack_meta;
        metadata.squeeze().sort("absolute_tilt");

        // Allocating buffers.
        const ArrayOption options{compute_device, Allocator::DEFAULT_ASYNC};
        auto [reference_pad, reference_pad_fft] = noa::fft::empty<float>(slice_shape_padded, options);
        auto [target_pad, target_pad_fft] = noa::fft::empty<float>(slice_shape_padded, options);
        auto [reference, reference_fft] = noa::fft::empty<float>(slice_shape, options);
        Array target = noa::memory::empty<float>(slice_shape, options);
        Array target_fft = noa::memory::empty<cfloat_t>(slice_shape.fft(), options);
        Array xmap = noa::memory::empty<float>(slice_shape, options);
        qn::geometry::Projector projector(grid_shape_padded, slice_shape_padded, target_shape_padded, options);
        Array<float> tmp(slice_shape.fft(), options);

        // The yaw is the CCW angle where the tilt-axis is in the slice. We want to transform this axis back to
        // the Y axis of the 3D Fourier volume of the projector, so take negative. Then, apply the tilt and pitch.
        // For backward projection, we'll also need to cancel any remaining shift before the insertion.
        auto projectReference = [&](size_t reference_index) {
            // For every slice:
            // 1) get the weighting relative to the reference.
            // 2) weight the slice, zero-pad it, take the fft, and backward project using the slice angles/shifts.
            // Once every aligned slice is inserted, forward project at the reference angles/shifts.
            // Then, ifft and crop.

            projector.reset();
            const float3_t& reference_angles = noa::math::deg2rad(metadata[reference_index].angles);
            for (size_t i = 0; i < metadata.size(); ++i) {
                const float3_t target_angles = noa::math::deg2rad(metadata[i].angles);

                const float cos_weight = math::cos(reference_angles[1] - target_angles[1]);
                if (i == reference_index || cos_weight < 0)
                    continue;
                memory::fill(projector.m_weights_ones_fft, 1 / cos_weight);

                noa::memory::copy(stack.subregion(metadata[i].index), target);
                noa::memory::resize(target, target_pad);
                noa::fft::r2c(target_pad, target_pad_fft);

                const float33_t rotation = noa::geometry::euler2matrix(
                        float3_t{-target_angles[0], target_angles[1], target_angles[2]}, "ZYX", false);
                projector.backward(target_pad_fft,
                                   rotation,
                                   -metadata[i].shifts);
            }

//            io::save(projector.m_grid_weights_fft, OUTPUT_PATH / "weights.mrc");

            const float33_t rotation = noa::geometry::euler2matrix(
                    float3_t{-reference_angles[0], reference_angles[1], reference_angles[2]}, "ZYX", false);
            projector.forward(reference_pad_fft, rotation, metadata[reference_index].shifts);

            noa::fft::c2r(reference_pad_fft, reference_pad);
            noa::memory::resize(reference_pad, reference);
            // FIXME For phase-correlation, I think it makes sense to keep the zero-padding.
            //       So instead, apply rectangular mask. Maybe enforce the common field of view?
            io::save(reference, OUTPUT_PATH / "reference.mrc");
            noa::fft::r2c(reference, reference_fft);
            {
//                math::ewise(reference_fft, tmp, math::abs_one_log_t{});
//                io::save(tmp, OUTPUT_PATH / "reference_fft.mrc");
            }
        };

        // Projection matching:
        std::vector<float2_t> shift_output;
        shift_output.emplace_back(0);
        for (size_t i = 1; i < metadata.size(); ++i) {
            MetadataSlice& slice = metadata[i];

            // Get the target:
            noa::memory::copy(stack.subregion(slice.index), target);
            noa::fft::r2c(target, target_fft);
            {
                io::save(target, OUTPUT_PATH / "target.mrc");
//                math::ewise(target_fft, tmp, math::abs_one_log_t{});
//                io::save(tmp, OUTPUT_PATH / "target_fft.mrc");
            }

            // Forward project with the transformation of the target.
            projectReference(i); // FIXME Check the reference to see if it looks OK.

            // Find and apply shift:
            noa::signal::fft::xmap<fft::H2FC>(target_fft, reference_fft, xmap); // TODO Check with zero padding

            float33_t xmap_fwd_transform(
                    noa::geometry::translate(slice_center) *
                    float33_t(noa::geometry::rotate(math::deg2rad(-slice.angles[0]))) *
                    noa::geometry::translate(-slice_center)
            );
            noa::geometry::transform2D(xmap, target, math::inverse(xmap_fwd_transform)); // TODO Output small region in max_shift

            io::save(target, OUTPUT_PATH / string::format("xmap{:0>2}.mrc", i));
            const float2_t peak_rotated = noa::signal::fft::xpeak2D<fft::FC2FC>(target, max_shift);
            const float2_t shift_rotated = peak_rotated - slice_center;
            const float2_t shift = noa::geometry::rotate(math::deg2rad(slice.angles[0])) * shift_rotated;
            qn::Logger::trace("peak {}: {}", i, shift);

            // Update the shift of this slice. This will be used to compute the reference of the next iteration.
            slice.shifts += shift;
            shift_output.emplace_back(shift);
        }

        // Update the metadata with the new centered shifts.
        // Use the slice index to find the math between slices.
        // FIXME The mean should be computed and subtracted in a common reference frame...
//        const auto mean_scale = static_cast<float>(metadata.size());
//        const float2_t mean = std::accumulate(
//                metadata.slices().begin(), metadata.slices().end(), float2_t{0},
//                [mean_scale](const float2_t& init, const MetadataSlice& slice) {
//                    return (init + slice.shifts) / mean_scale;
//                });

        for (size_t i = 0; i < metadata.size(); ++i) {
            for (auto& original_slice: stack_meta.slices()) {
                if (original_slice.index == metadata[i].index)
                    original_slice.shifts = metadata[i].shifts;
            }
        }
    }
}
