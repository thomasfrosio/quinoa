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
        const float2_t slice_center = MetadataSlice::center(slice_shape);

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
                                 float2_t max_shift = {},
                                 bool center = true) {
        const dim_t size_z_pad = std::min(stack.shape()[2], stack.shape()[3]) * 2;
        const dim_t size_y_pad = stack.shape()[2] * 2;
        const dim_t size_x_pad = stack.shape()[3] * 2;

        // The projector needs the following: 1) The target shape is the shape of the 3D Fourier volume.
        // TODO Compute min grid shape!
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

        qn::geometry::Projector projector(slice_shape_padded, slice_center,
                                          grid_shape_padded, target_shape_padded,
                                          options);

        Array xmap = noa::memory::empty<float>(slice_shape, options);
        Array<float> tmp_buffer_io(xmap.shape().fft(), options);

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
            for (size_t i = 0; i < reference_index; ++i) {
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
            noa::signal::rectangle(reference, reference, slice_center, slice_center - 50, 50);

            io::save(reference, OUTPUT_PATH / string::format("reference_{:0>2}.mrc", reference_index));
            noa::fft::r2c(reference, reference_fft);
        };

        // Projection matching:
        std::vector<float2_t> shift_output;
        shift_output.emplace_back(0);
        for (size_t i = 1; i < metadata.size(); ++i) {
            MetadataSlice& slice = metadata[i];

            // Get the target:
            noa::memory::copy(stack.subregion(slice.index), target);
//            noa::memory::resize(target, target_pad);
            noa::io::save(target, OUTPUT_PATH / string::format("target_{:0>2}.mrc", i));
            noa::fft::r2c(target, target_fft);

            // Forward project with the transformation of the target.
            projectReference(i); // FIXME Check the reference to see if it looks OK.

            // Find and apply shift:
//            {
//                noa::math::ewise(target_pad_fft, tmp_buffer_io, math::abs_one_log_t{});
//                noa::io::save(tmp_buffer_io, OUTPUT_PATH / string::format("target_fft_{:0>2}.mrc", i));
//                math::ewise(reference_pad_fft, tmp_buffer_io, math::abs_one_log_t{});
//                io::save(tmp_buffer_io, OUTPUT_PATH / string::format("reference_fft_{:0>2}.mrc", i));
//            }

            noa::signal::fft::xmap<fft::H2FC>(target_fft, reference_fft, xmap, true); // TODO Check with zero padding
            noa::io::save(xmap, OUTPUT_PATH / string::format("xmap_{:0>2}.mrc", i));

            float33_t xmap_fwd_transform(
                    noa::geometry::translate(slice_center) *
                    float33_t(noa::geometry::rotate(math::deg2rad(-slice.angles[0]))) *
                    noa::geometry::translate(-slice_center)
            );
            noa::geometry::transform2D(xmap, target, math::inverse(xmap_fwd_transform)); // TODO Output small region in max_shift
            noa::io::save(target, OUTPUT_PATH / string::format("xmap_rotated_{:0>2}.mrc", i));

            const float2_t peak_rotated = noa::signal::fft::xpeak2D<fft::FC2FC>(target, max_shift, {2, 15}); // TODO COM?
            const float2_t shift_rotated = peak_rotated - slice_center;
            const float2_t shift = noa::geometry::rotate(math::deg2rad(slice.angles[0])) * shift_rotated;
            qn::Logger::trace("peak {}: {}", i, shift);

            // Update the shift of this slice. This will be used to compute the reference of the next iteration.
            slice.shifts += shift;
            shift_output.emplace_back(shift);
        }

        if (center) {
            // Center the shifts. The mean should be computed and subtracted using a common reference frame.
            // Here, stretch the shifts to the 0deg reference frame and compute the mean there. Then transform
            // the mean to the slice tilt and pivot angles before subtraction.
            double2_t mean{0};
            auto mean_scale = 1 / static_cast<double>(metadata.size());
            for (size_t i = 0; i < metadata.size(); ++i) {
                const double3_t angles(math::deg2rad(metadata[i].angles));
                const double2_t pivot_tilt{angles[2], angles[1]};
                const double22_t stretch_to_0deg{
                        noa::geometry::rotate(angles[0]) *
                        noa::geometry::scale(1 / math::cos(pivot_tilt)) * // 1 = cos(0deg)
                        noa::geometry::rotate(-angles[0])
                };
                const double2_t shift_at_0deg = stretch_to_0deg * double2_t(metadata[i].shifts);
                mean += shift_at_0deg * mean_scale;
            }
            for (size_t i = 0; i < metadata.size(); ++i) {
                const double3_t angles(math::deg2rad(metadata[i].angles));
                const double22_t shrink_matrix{
                        noa::geometry::rotate(angles[0]) *
                        noa::geometry::scale(math::cos(double2_t(angles[2], angles[1]))) *
                        noa::geometry::rotate(-angles[0])
                };
                const float2_t shrank_mean(shrink_matrix * mean);
                metadata[i].shifts -= shrank_mean;
            }
        }

        // Update the metadata.
        for (size_t i = 0; i < metadata.size(); ++i) {
            for (auto& original_slice: stack_meta.slices()) {
                if (original_slice.index == metadata[i].index)
                    original_slice.shifts = metadata[i].shifts;
            }
        }
    }
}
