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

namespace qn::alignment {
    class ProjectionMatching {
    public:
        ProjectionMatching(dim4_t shape,
                           Device compute_device,
                           float smooth_edge_size = 0.05f,
                           float slice_z_radius = 0.0005f,
                           float cutoff = 0.5f,
                           Allocator allocator = Allocator::DEFAULT_ASYNC)
                : m_slice_center(MetadataSlice::center(shape)),
                  m_slice_z_radius(slice_z_radius),
                  m_cutoff(cutoff) {
            // Zero padding:
            const dim_t max_size = std::max(shape[2], shape[3]);
            m_size_pad = max_size * 2;
            m_slice_shape = {1, 1, shape[2], shape[3]};
            m_slice_shape_padded = {1, 1, m_size_pad, m_size_pad};
            m_slice_center_padded = static_cast<float>(m_size_pad / 2);
            m_zero_taper_size = static_cast<float>(max_size) * smooth_edge_size;

            // For the buffers, keep them as separated entities (no alias for in-place FFT).
            // This could be improved and when/if textures are added this is likely to change.
            const ArrayOption options(compute_device, allocator);

            m_input_slice = memory::empty<float>(m_slice_shape, options);
            m_output_slice = memory::empty<float>(m_slice_shape, options);
            m_input_slice_fft = memory::empty<cfloat_t>(m_slice_shape.fft(), options);
            m_output_slice_fft = memory::empty<cfloat_t>(m_slice_shape.fft(), options);

            m_input_slice_padded = memory::empty<float>(m_slice_shape_padded, options);
            m_output_slice_padded = memory::empty<float>(m_slice_shape_padded, options);
            m_input_slice_padded_fft = memory::empty<cfloat_t>(m_slice_shape_padded.fft(), options);
            m_output_slice_padded_fft = memory::empty<cfloat_t>(m_slice_shape_padded.fft(), options);

            m_input_weight_padded = memory::empty<float>(m_slice_shape_padded.fft(), options);
            m_output_weight_padded = memory::empty<float>(m_slice_shape_padded.fft(), options);
        }

        void updateShift(const Array<float>& stack,
                         MetadataStack& stack_meta,
                         float2_t max_shift = {},
                         float max_angle_difference = 30.f,
                         bool center = true) {
            qn::Logger::trace("Projection matching shift alignment...");
            Timer timer;
            timer.start();

            max_angle_difference = math::deg2rad(max_angle_difference);

            // Projection matching:
            MetadataStack metadata = stack_meta;
            metadata.squeeze().sort("absolute_tilt");
            for (size_t i = 1; i < metadata.size(); ++i) {
                MetadataSlice& slice = metadata[i];

                // The target view is saved in m_input_slice.
                // The projected reference is saved in m_output_slice.
                computeTargetAndReference_(stack, metadata, i, max_angle_difference);
                {
                    Array<float> debug_target_reference({2, 1, m_slice_shape[2], m_slice_shape[3]});
                    noa::memory::copy(m_input_slice, debug_target_reference.subregion(0));
                    noa::memory::copy(m_output_slice, debug_target_reference.subregion(1));
                    noa::io::save(debug_target_reference, string::format("/home/thomas/Projects/quinoa/tests/debug_pm_shift/target_reference_{:>02}.mrc", i));
                }

                // Cross-correlation to find the shift between target and reference:
                noa::fft::r2c(m_input_slice, m_input_slice_fft);
                noa::fft::r2c(m_output_slice, m_output_slice_fft);
                noa::signal::fft::xmap<fft::H2FC>(m_input_slice_fft, m_output_slice_fft, m_input_slice, true);
//                    noa::io::save(m_input_slice, string::format("/home/thomas/Projects/quinoa/tests/debug_data1/xmap_{:>02}.mrc", i));
                const float2_t shift = extractShiftFromXmap_(slice, max_shift);

                // Update the metadata.
                slice.shifts += shift;
            }

            if (center)
                centerShifts_(metadata);

            // Update the metadata.
            for (size_t i = 0; i < metadata.size(); ++i) {
                for (auto& original_slice: stack_meta.slices()) {
                    if (original_slice.index == metadata[i].index)
                        original_slice.shifts = metadata[i].shifts;
                }
            }

            qn::Logger::trace("Projection matching shift alignment... took {}ms", timer.elapsed());
        }

    private:
        // Computes the reference by backward projecting the stack (without the target view).
        // Computes the target by simply extracting the reference view from the stack.
        //  - Proximity weighting: when backward projecting the neighbour views, weight these views using
        //    the angle difference. At the max angle difference, the weight is 0.
        //  - Oversampling: the projection is done using zero-padded views.
        //  - Field of view: the current geometry is used to ensure a common field of view (FOV) between the target
        //    slice and the projected-reference slice. 1) The reference slices are masked out by the FOV of
        //    the target slice, ensuring that the final projected-reference doesn't have any contribution that
        //    are not included in the target slice. 2) The target slice is also masked out, but using the
        //    "cumulative" FOV of every projected slice, removing any eventual regions that would be unique to
        //    the target slice and therefore not present in the project-reference slice. It also locally weights
        //    the FOV of the target slice based on how much a region contributes to the projected-reference slice.
        void computeTargetAndReference_(const Array<float>& stack,
                                        const MetadataStack& metadata,
                                        size_t target_index,
                                        float max_angle_difference) {
            // Signal from backward projected neighbouring slices are about to be
            // iteratively collected, so reset buffer slices.
            noa::memory::fill(m_output_slice_padded_fft, cfloat_t{0});
            noa::memory::fill(m_output_weight_padded, float{0});
            noa::memory::fill(m_output_slice, float{0});

            // The yaw is the CCW angle of the tilt-axis in the slices. For the projection, we want to align
            // the tilt-axis along the Y axis, so subtract this angle and then apply the tilt and pitch.
            const float3_t target_angles = noa::math::deg2rad(metadata[target_index].angles);
            const float2_t& target_shifts = metadata[target_index].shifts;
            const float33_t fwd_target_rotation = noa::geometry::euler2matrix(
                    float3_t{-target_angles[0], target_angles[1], target_angles[2]}, "ZYX", false);

            // Go through the stack and backward project the reference slices. Usually only a subset of the
            // total slices in the stack are backward projected, and slices are weighted based on how "close"
            // they are to the target slice.
            float total_weight{0};
            for (size_t reference_index = 0; reference_index < metadata.size(); ++reference_index) {
                const float3_t reference_angles = noa::math::deg2rad(metadata[reference_index].angles);
                const float2_t& reference_shifts = metadata[reference_index].shifts;

                const float tilt_difference = math::abs(target_angles[1] - reference_angles[1]);
                if (reference_index == target_index || tilt_difference > max_angle_difference)
                    continue;

                // TODO Weighting based on the order of collection? Or is exposure weighting enough?
                const float weight = math::sinc(tilt_difference * math::Constants<float>::PI / max_angle_difference);
                noa::memory::fill(m_input_weight_padded, 1 / weight);

                // Collect the FOV of this reference slice.
                // This cumulative FOV will be used to mask the target view.
                addFOVReference(m_output_slice, weight,
                                target_angles, target_shifts,
                                reference_angles, reference_shifts);
                total_weight += weight;

                // Get the reference slice ready for back-projection.
                noa::memory::copy(stack.subregion(metadata[reference_index].index), m_input_slice);
                applyFOVTarget_(m_input_slice,
                                target_angles, target_shifts,
                                reference_angles, reference_shifts);
//                noa::io::save(m_input_slice, string::format("/home/thomas/Projects/quinoa/tests/debug_data1/fov_{:>02}.mrc", reference_index));
                noa::memory::resize(m_input_slice, m_input_slice_padded);
                noa::fft::r2c(m_input_slice_padded, m_input_slice_padded_fft);

                // The shift of the reference slice should be removed to have the rotation center at the origin.
                // shift2D can do the remap, but not in-place, so use remap to center the slice.
                noa::signal::fft::shift2D<fft::H2H>(
                        m_input_slice_padded_fft, m_input_slice_padded_fft,
                        m_slice_shape_padded, -m_slice_center_padded - reference_shifts);
                noa::fft::remap(fft::H2HC, m_input_slice_padded_fft,
                                m_input_slice_padded_fft, m_slice_shape_padded);

                // For backward projection, noa needs the inverse rotation matrix, hence the transpose.
                // For forward projection, it needs the forward matrices, so all good.
                const float33_t inv_reference_rotation = noa::geometry::euler2matrix(
                        float3_t{-reference_angles[0], reference_angles[1], reference_angles[2]},
                        "ZYX", false).transpose();
                noa::geometry::fft::extract3D<fft::HC2H>(
                        m_input_slice_padded_fft, m_slice_shape_padded,
                        m_output_slice_padded_fft, m_slice_shape_padded,
                        float22_t{}, inv_reference_rotation, float22_t{},
                        fwd_target_rotation, m_slice_z_radius, m_cutoff);
                noa::geometry::fft::extract3D<fft::HC2H>(
                        m_input_weight_padded, m_slice_shape_padded,
                        m_output_weight_padded, m_slice_shape_padded,
                        float22_t{}, inv_reference_rotation, float22_t{},
                        fwd_target_rotation, m_slice_z_radius, m_cutoff);
            }

            // For the target view, simply extract it from the stack and apply the cumulative FOV.
            Array target_view = stack.subregion(metadata[target_index].index);
            if (stack.device() != m_output_slice.device()) {
                noa::memory::copy(target_view, m_input_slice);
                target_view = m_input_slice;
            }
            // TODO Apply the cumulative FOV to the projected reference?
            noa::io::save(m_output_slice,
                          string::format("/home/thomas/Projects/quinoa/tests/debug_pm_shift/cumulative_fov{:>02}.mrc", target_index));
            math::ewise(m_output_slice, 1 / total_weight, target_view,
                        m_input_slice, math::multiply_t{});

            // For the reference view, center the output projected slice
            // onto the target and apply the projected-weight/multiplicity.
            signal::fft::shift2D<fft::H2H>(
                    m_output_slice_padded_fft, m_output_slice_padded_fft,
                    m_slice_shape_padded, m_slice_center_padded + target_shifts);
            math::ewise(m_output_slice_padded_fft, m_output_weight_padded, 1e-3f,
                        m_output_slice_padded_fft, math::divide_epsilon_t{});
            io::save(m_output_weight_padded,
                     string::format("/home/thomas/Projects/quinoa/tests/debug_pm_shift/reference_fft_weights_{:>02}.mrc", target_index));

            noa::fft::c2r(m_output_slice_padded_fft, m_output_slice_padded);
            noa::memory::resize(m_output_slice_padded, m_output_slice);
        }

        // Mask out the regions that are not in the target view to not include them in the projected views.
        // To do so, transform a smooth rectangular mask from the target view onto the current view that
        // is about to be backward projected.
        void applyFOVTarget_(const Array<float>& reference,
                             float3_t target_angles, float2_t target_shifts,
                             float3_t reference_angles, float2_t reference_shifts) {

            const float2_t cos_factor{math::cos(reference_angles[2]) / math::cos(target_angles[2]),
                                      math::cos(reference_angles[1]) / math::cos(target_angles[1])};
            const float33_t affine_matrix = math::inverse( // TODO Compute inverse transformation directly
                    noa::geometry::translate(m_slice_center + reference_shifts) *
                    float33_t{noa::geometry::rotate(reference_angles[0])} *
                    float33_t{noa::geometry::scale(cos_factor)} *
                    float33_t{noa::geometry::rotate(-target_angles[0])} *
                    noa::geometry::translate(-m_slice_center - target_shifts)
            );

            noa::signal::rectangle(reference, reference,
                                   m_slice_center, m_slice_center - m_zero_taper_size,
                                   m_zero_taper_size, affine_matrix);
        }

        // Mask out the regions that are not in the reference view to not include
        // To do so, transform a smooth rectangular mask from the reference view onto the target view.
        void addFOVReference(const Array<float>& target_fov, float weight,
                             float3_t target_angles, float2_t target_shifts,
                             float3_t reference_angles, float2_t reference_shifts) {
            const float2_t cos_factor{math::cos(target_angles[2]) / math::cos(reference_angles[2]),
                                      math::cos(target_angles[1]) / math::cos(reference_angles[1])};
            const float33_t affine_matrix = math::inverse( // TODO Compute inverse transformation directly
                    noa::geometry::translate(m_slice_center + target_shifts) *
                    float33_t{noa::geometry::rotate(target_angles[0])} *
                    float33_t{noa::geometry::scale(cos_factor)} *
                    float33_t{noa::geometry::rotate(-reference_angles[0])} *
                    noa::geometry::translate(-m_slice_center - reference_shifts)
            );

            noa::signal::rectangle(target_fov, target_fov,
                                   m_slice_center, m_slice_center - m_zero_taper_size,
                                   m_zero_taper_size, affine_matrix, math::plus_t{}, weight);
        }

        float2_t extractShiftFromXmap_(const MetadataSlice& slice, float2_t max_shift) {
            // TODO Only render small region around max_shift
            // The peak is distorted/stretched perpendicular to the tilt axis.
            // To help for the picking, align the tilt axis onto the vertical axis.
            float33_t xmap_inv_transform(
                    noa::geometry::translate(m_slice_center) *
                    float33_t(noa::geometry::rotate(math::deg2rad(slice.angles[0]))) *
                    noa::geometry::translate(-m_slice_center)
            );
            noa::geometry::transform2D(m_input_slice, m_output_slice, xmap_inv_transform);

            const float2_t peak_rotated = noa::signal::fft::xpeak2D<fft::FC2FC>(m_output_slice, max_shift, {2, 2});
            float2_t shift_rotated = peak_rotated - m_slice_center;
            // The shift perpendicular to the tilt axis is unreliable...
            // For now, only use the shift parallel to the tilt axis.
            shift_rotated[1] = 0;
            const float2_t shift = noa::geometry::rotate(math::deg2rad(slice.angles[0])) * shift_rotated;
            return shift;
        }

        static void centerShifts_(MetadataStack& metadata) {
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

    private:
        Array<float> m_input_slice;
        Array<float> m_output_slice;
        Array<cfloat_t> m_input_slice_fft;
        Array<cfloat_t> m_output_slice_fft;

        Array<float> m_input_slice_padded;
        Array<float> m_output_slice_padded;
        Array<cfloat_t> m_input_slice_padded_fft;
        Array<cfloat_t> m_output_slice_padded_fft;

        Array<float> m_input_weight_padded;
        Array<float> m_output_weight_padded;

        dim_t m_size_pad;
        dim4_t m_slice_shape;
        dim4_t m_slice_shape_padded;
        float2_t m_slice_center;
        float2_t m_slice_center_padded;
        float m_zero_taper_size;
        float m_slice_z_radius;
        float m_cutoff;
    };
}
