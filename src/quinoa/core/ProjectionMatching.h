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
#include "quinoa/io/Logging.h"

namespace qn {
    struct ProjectionMatchingParameters {
        float2_t max_shift = {};
        float smooth_edge_percent = 0.5f;

        float backward_slice_z_radius = 0.0005f;
        float backward_tilt_angle_difference = 30.f;
        bool backward_use_aligned_only = false;

        float forward_cutoff = 0.5f;

        bool center_tilt_axis = true;
        path_t debug_directory;
    };

    class ProjectionMatching {
    public:
        ProjectionMatching(dim4_t shape,
                           noa::Device compute_device,
                           noa::Allocator allocator = noa::Allocator::DEFAULT_ASYNC)
                : m_slice_center(MetadataSlice::center(shape)) {
            // Zero padding:
            m_max_size = std::max(shape[2], shape[3]);
            const dim_t size_pad = m_max_size * 2;
            m_slice_shape = {1, 1, shape[2], shape[3]};
            m_slice_shape_padded = {1, 1, size_pad, size_pad};
            m_slice_center_padded = static_cast<float>(size_pad / 2);

            // For the buffers, keep them as separated entities (no alias for in-place FFT).
            // This could be improved and when/if textures are added this is likely to change.
            const auto options = ArrayOption(compute_device, allocator);

            // TODO Some buffers could be in-place for FFT.
            m_slices = noa::memory::empty<float>({2, 1, shape[2], shape[3]}, options);
            m_slices_padded = noa::memory::empty<float>({2, 1, size_pad, size_pad}, options);
            m_slices_fft = noa::memory::empty<cfloat_t>(m_slices.shape().fft(), options);
            m_slices_padded_fft = noa::memory::empty<cfloat_t>(m_slices_padded.shape().fft(), options);
            m_slice_weight_padded_fft = noa::memory::empty<float>(m_slice_shape_padded.fft(), options);
            m_cumulative_fov = noa::memory::empty<float>(m_slice_shape, options);

            // Alias to individual buffers:
            m_slice0 = m_slices.subregion(0);
            m_slice1 = m_slices.subregion(1);
            m_slice0_fft = m_slices_fft.subregion(0);
            m_slice1_fft = m_slices_fft.subregion(1);

            m_slice0_padded = m_slices_padded.subregion(0);
            m_slice1_padded = m_slices_padded.subregion(1);
            m_slice0_padded_fft = m_slices_padded_fft.subregion(0);
            m_slice1_padded_fft = m_slices_padded_fft.subregion(1);
        }

        void align(const Array<float>& stack,
                   MetadataStack& metadata,
                   const ProjectionMatchingParameters& parameters) {
            qn::Logger::trace("Projection matching alignment...");
            noa::Timer timer;
            timer.start();

            // For every slice (excluding the lowest tilt which defines the reference-frame), find the best
            // geometry parameters using the previously aligned slice as reference for alignment. The geometry
            // parameters are, for each slice, the 3D rotation (yaw, tilt, pitch) and the (y,x) shifts.

            MetadataStack metadata_ = metadata;
            metadata_.sort("exposure");

            const bool save_debug = !parameters.debug_directory.empty();

            for (size_t target_index = 1; target_index < 2; ++target_index) { // metadata_.size()

                // Get the indexes of the reference views.
                const std::vector<size_t> reference_indexes = referenceIndexes(
                        target_index, metadata_.size(), metadata_, parameters);
//                metadata_[0].angles[0] = 175;

//                MetadataSlice& slice = metadata_[target_index];

                // TODO Grid search - optimizer.
                //      Search 3D rotation. Each rotation outputs a shift and a normalized CC-score.
                //      The score is fed to the optimizer, which then returns the next 3D rotation to measure.
                //      The optimizer should also return when the alignment converges and output a convergence score.
                float3_t best_angles_offset{};

                std::array<float, 15> yaw_offsets{
                    -3, -2, -1.5, -1, -0.5, 0, 0.5, 0.67f, 0.7f, 0.73f, 0.9f, 1, 2, 2.81f, 3};
                std::array<float, 15> cc_scores{};
                for (size_t i = 0; i < yaw_offsets.size(); ++i) {
                    const float3_t angle_offsets = {yaw_offsets[i], 0, 0};
                    computeTargetAndReference_(stack, metadata_, target_index, angle_offsets,
                                               reference_indexes, parameters);

                    // Normalize
                    const auto mean = noa::memory::empty<float>(dim4_t{1}, m_slices.options());
                    const auto stddev = noa::memory::empty<float>(dim4_t{1}, m_slices.options());
                    noa::math::mean(m_slices, mean);
                    noa::math::std(m_slices, stddev);
                    noa::math::ewise(m_slices, mean, stddev, m_slices, noa::math::minus_divide_t{});

                    noa::fft::r2c(m_slices, m_slices_fft);
                    noa::signal::fft::bandpass<noa::fft::H2H>(
                            m_slices_fft, m_slices_fft, m_slices.shape(), 0.15f, 0.40f, 0.05f, 0.05f);
                    noa::fft::c2r(m_slices_fft, m_slices);

                    if (save_debug) {
                        noa::io::save(m_slices,
                                      parameters.debug_directory /
                                      noa::string::format("target_reference_{:0>2}_{:0>2}.mrc", target_index, i));
                    }

                    noa::fft::r2c(m_slices, m_slices_fft);
                    cc_scores[i] = noa::signal::fft::xcorr<noa::fft::H2H>(m_slice0_fft, m_slice1_fft, m_slice_shape);
                    noa::signal::fft::xmap<noa::fft::H2FC>(m_slice0_fft, m_slice1_fft, m_slice0); // TODO Double phase

                    if (save_debug) {
                        noa::io::save(m_slice0,
                                      parameters.debug_directory /
                                      noa::string::format("xmap_{:0>2}_{:0>2}.mrc", target_index, i));
                    }
                }

                for (size_t j = 0; j < cc_scores.size(); ++j) {
                    const auto yaw_angle = metadata_[target_index].angles[0] + yaw_offsets[j];
                    qn::Logger::trace("CC scores {}: yaw={}, CC={}", target_index, yaw_angle, cc_scores[j]);
                }
                qn::Logger::trace("done");

//                // Find the shifts:
//                computeTargetAndReference_(stack, metadata_, target_index, best_angles_offset, parameters);
//                noa::io::save(m_slices, string::format("/home/thomas/Projects/quinoa/tests/debug_pm_shift/target_reference_{:>02}.mrc", slice.index));
//                noa::fft::r2c(m_slices, m_slices_fft);
//                noa::signal::fft::xmap<fft::H2FC>(m_slice0_fft, m_slice1_fft, m_slice0); // TODO Double phase
//                const float2_t best_shift = extractShiftFromXmap_(slice, max_shift);
//                noa::io::save(m_slice0, string::format("/home/thomas/Projects/quinoa/tests/debug_pm_shift/xmap_{:>02}.mrc", slice.index));
//
//                // Update the metadata.
//                slice.shifts += best_shift;

//                qn::Logger::trace("Projection matching shift alignment... iter took {}ms", timer0.elapsed());
            }

            if (parameters.center_tilt_axis)
                centerTiltAxis_(metadata_);

            // Update the metadata.
            for (size_t i = 0; i < metadata_.size(); ++i) {
                for (auto& original_slice: metadata.slices()) {
                    if (original_slice.index == metadata_[i].index)
                        original_slice.shifts = metadata_[i].shifts;
                }
            }

            qn::Logger::trace("Projection matching alignment... took {}ms", timer.elapsed());
        }

    private:
        [[nodiscard]] static auto referenceIndexes(
                size_t target_index,
                size_t reference_count,
                const MetadataStack& metadata,
                const ProjectionMatchingParameters& parameters)
        -> std::vector<size_t> {
            const float max_tilt_difference = noa::math::deg2rad(parameters.backward_tilt_angle_difference);
            const size_t max_index = parameters.backward_use_aligned_only ? target_index : reference_count;

            std::vector<size_t> reference_indexes;
            reference_indexes.reserve(reference_count);

            for (size_t reference_index = 0; reference_index < max_index; ++reference_index) {
                const float target_tilt_angle = noa::math::deg2rad(metadata[target_index].angles[1]);
                const float reference_tilt_angle = noa::math::deg2rad(metadata[reference_index].angles[1]);
                const float tilt_difference = noa::math::abs(target_tilt_angle - reference_tilt_angle);

                if (reference_index != target_index && tilt_difference <= max_tilt_difference)
                    reference_indexes.emplace_back(reference_index);
            }

            return reference_indexes;
        }

        /// Compute the target and reference slices.
        /// \details Proximity weighting: the backward projected views are weighted based on their
        ///          tilt angle difference with the target. At the max angle difference, the weight is 0.
        /// \details Common field-of-view (FOV): the current geometry is used to ensure a common FOV between
        ///          the target slice and the projected-reference slice. 1) The reference slices are masked
        ///          out by the FOV of the target slice, ensuring that the final projected-reference doesn't
        ///          have any contribution that are not included in the target slice. 2) The target slice is
        ///          also masked out, but using the "cumulative" FOV of every projected slice, removing any
        ///          eventual regions that would be unique to the target slice and therefore not present in
        ///          the project-reference slice. It also locally weights the FOV of the target slice based
        ///          on how much a region contributes to the projected-reference slice.
        /// \details Oversampling: the projection is done on 2x zero-padded central slices.
        ///
        /// \param[in] stack                Input stack to align.
        /// \param metadata                 Metadata corresponding to \p stack.
        /// \param target_index             Index of the slice to forward-project.
        /// \param target_angle_offset      Angle offsets (in degrees) to add to the target angles.
        ///                                 This is used for the optimization function.
        /// \param projection_parameters    Parameters to use for the projection.
        void computeTargetAndReference_(const noa::Array<float>& stack,
                                        const MetadataStack& metadata,
                                        size_t target_index,
                                        float3_t target_angle_offset,
                                        const std::vector<size_t>& reference_indexes,
                                        ProjectionMatchingParameters parameters) {
            // The yaw is the CCW angle of the tilt-axis in the slices. For the projection, we want to align
            // the tilt-axis along the Y axis, so subtract this angle and then apply the tilt and pitch.
            const float2_t target_shifts = metadata[target_index].shifts;
            const float3_t target_angles = noa::math::deg2rad(metadata[target_index].angles + target_angle_offset);
            const float33_t fwd_target_rotation = noa::geometry::euler2matrix(
                    float3_t{-target_angles[0], target_angles[1], target_angles[2]}, "ZYX", false);

            // Go through the stack and backward project the reference slices.
            // Reset the buffers for backward projection.
            noa::memory::fill(m_slice1_padded_fft, cfloat_t{0});
            noa::memory::fill(m_slice_weight_padded_fft, float{0});
            noa::memory::fill(m_cumulative_fov, float{0});

            const auto zero_taper_size = static_cast<float>(m_max_size) * parameters.smooth_edge_percent;

            float total_weight{0};
            for (size_t reference_index: reference_indexes) {
                const float3_t reference_angles = noa::math::deg2rad(metadata[reference_index].angles);
                const float2_t reference_shifts = metadata[reference_index].shifts;

                // TODO Weighting based on the order of collection? Or is exposure weighting enough?
                // How much the slice should contribute to the final projected-reference.
//                constexpr auto PI = noa::math::Constants<float>::PI;
//                const float tilt_difference = std::abs(target_angles[1] - reference_angles[1]);
                const float weight = 1;//noa::math::sinc(tilt_difference * PI / max_tilt_difference);

                // Collect the FOV of this reference slice.
                addFOVReference(m_cumulative_fov, weight,
                                target_angles, target_shifts,
                                reference_angles, reference_shifts,
                                zero_taper_size);
                total_weight += weight;

                // Get the reference slice ready for back-projection.
                noa::memory::copy(stack.subregion(metadata[reference_index].index), m_slice0); // FIXME
                applyFOVTarget_(m_slice0,
                                target_angles, target_shifts,
                                reference_angles, reference_shifts,
                                zero_taper_size);
                noa::memory::resize(m_slice0, m_slice0_padded);
                noa::fft::r2c(m_slice0_padded, m_slice0_padded_fft);

                // The shift of the reference slice should be removed to have the rotation center at the origin.
                // shift2D can do the remap, but not in-place, so use remap to center the slice.
                noa::signal::fft::shift2D<noa::fft::H2H>(
                        m_slice0_padded_fft, m_slice0_padded_fft,
                        m_slice_shape_padded, -m_slice_center_padded - reference_shifts);
                noa::fft::remap(noa::fft::H2HC, m_slice0_padded_fft,
                                m_slice0_padded_fft, m_slice_shape_padded);

                // For backward projection, noa needs the inverse rotation matrix, hence the transpose.
                // For forward projection, it needs the forward matrices, so all good.
                const float33_t inv_reference_rotation = noa::geometry::euler2matrix(
                        float3_t{-reference_angles[0], reference_angles[1], reference_angles[2]},
                        "ZYX", false).transpose();
                noa::geometry::fft::extract3D<noa::fft::HC2H>(
                        m_slice0_padded_fft, m_slice_shape_padded,
                        m_slice1_padded_fft, m_slice_shape_padded,
                        float22_t{}, inv_reference_rotation,
                        float22_t{}, fwd_target_rotation,
                        parameters.backward_slice_z_radius,
                        parameters.forward_cutoff);
                noa::geometry::fft::extract3D<noa::fft::HC2H>(
                        1 / weight, m_slice_shape_padded,
                        m_slice_weight_padded_fft, m_slice_shape_padded,
                        float22_t{}, inv_reference_rotation,
                        float22_t{}, fwd_target_rotation,
                        parameters.backward_slice_z_radius,
                        parameters.forward_cutoff);
            }

            // For the target view, simply extract it from the stack and apply the cumulative FOV.
            noa::Array target_view = stack.subregion(metadata[target_index].index);
            if (stack.device() != m_slice1.device()) {
                noa::memory::copy(target_view, m_slice0);
                target_view = m_slice0;
            }
            target_view.to(m_slice0);
//            noa::math::ewise(m_cumulative_fov, 1 / total_weight, target_view,
//                             m_slice0, noa::math::multiply_t{});

            // For the reference view, center the output projected slice onto the target,
            // apply the projected-weight/multiplicity, and apply the cumulative FOV.
            noa::signal::fft::shift2D<noa::fft::H2H>(
                    m_slice1_padded_fft, m_slice1_padded_fft,
                    m_slice_shape_padded, m_slice_center_padded + target_shifts);
            noa::math::ewise(m_slice1_padded_fft, m_slice_weight_padded_fft, 1e-3f,
                             m_slice1_padded_fft, noa::math::divide_epsilon_t{});

            noa::fft::c2r(m_slice1_padded_fft, m_slice1_padded);
            noa::memory::resize(m_slice1_padded, m_slice1);
            noa::math::ewise(m_cumulative_fov, 1 / total_weight, m_slice1,
                             m_slice1, noa::math::multiply_t{});
        }

        // Mask out the regions that are not in the target view to not include them in the projected views.
        // To do so, transform a smooth rectangular mask from the target view onto the current view that
        // is about to be backward projected.
        void applyFOVTarget_(const noa::Array<float>& reference,
                             float3_t target_angles, float2_t target_shifts,
                             float3_t reference_angles, float2_t reference_shifts,
                             float zero_taper_size) {

            const float2_t cos_factor{noa::math::cos(reference_angles[2]) / noa::math::cos(target_angles[2]),
                                      noa::math::cos(reference_angles[1]) / noa::math::cos(target_angles[1])};
            const float33_t inv_matrix_target_to_reference = noa::math::inverse( // TODO Compute inverse transformation directly?
                    noa::geometry::translate(m_slice_center + reference_shifts) *
                    float33_t{noa::geometry::rotate(reference_angles[0])} *
                    float33_t{noa::geometry::scale(cos_factor)} *
                    float33_t{noa::geometry::rotate(-target_angles[0])} *
                    noa::geometry::translate(-m_slice_center - target_shifts)
            );

            noa::signal::rectangle(reference, reference,
                                   m_slice_center, m_slice_center - zero_taper_size,
                                   zero_taper_size, inv_matrix_target_to_reference);
        }

        // Mask out the regions that are not in the reference view to not include
        // To do so, transform a smooth rectangular mask from the reference view onto the target view.
        void addFOVReference(const Array<float>& cumulative_fov, float weight,
                             float3_t target_angles, float2_t target_shifts,
                             float3_t reference_angles, float2_t reference_shifts,
                             float zero_taper_size) {
            const float2_t cos_factor{noa::math::cos(target_angles[2]) / noa::math::cos(reference_angles[2]),
                                      noa::math::cos(target_angles[1]) / noa::math::cos(reference_angles[1])};
            const float33_t affine_matrix = noa::math::inverse( // TODO Compute inverse transformation directly
                    noa::geometry::translate(m_slice_center + target_shifts) *
                    float33_t{noa::geometry::rotate(target_angles[0])} *
                    float33_t{noa::geometry::scale(cos_factor)} *
                    float33_t{noa::geometry::rotate(-reference_angles[0])} *
                    noa::geometry::translate(-m_slice_center - reference_shifts)
            );

            noa::signal::rectangle(cumulative_fov, cumulative_fov,
                                   m_slice_center, m_slice_center - zero_taper_size,
                                   zero_taper_size, affine_matrix, math::plus_t{}, weight);
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
            noa::geometry::transform2D(m_slice0, m_slice1, xmap_inv_transform);
//            io::save(m_slice1, string::format(
//                    "/home/thomas/Projects/quinoa/tests/debug_pm_shift/xmap_rotated_{:>02}.mrc", slice.index));

            // TODO Better fitting of the peak. 2D parabola?
            const float2_t peak_rotated = noa::signal::fft::xpeak2D<noa::fft::FC2FC>(m_slice1, max_shift);
            float2_t shift_rotated = peak_rotated - m_slice_center;
            const float2_t shift = noa::geometry::rotate(noa::math::deg2rad(slice.angles[0])) * shift_rotated;
            return shift;
        }

        static void centerTiltAxis_(MetadataStack& metadata) {
            // Center the shifts. The mean should be computed and subtracted using a common reference frame.
            // Here, stretch the shifts to the 0deg reference frame and compute the mean there. Then transform
            // the mean to the slice tilt and pivot angles before subtraction.
            double2_t mean{0};
            auto mean_scale = 1 / static_cast<double>(metadata.size());
            for (size_t i = 0; i < metadata.size(); ++i) {
                const double3_t angles(noa::math::deg2rad(metadata[i].angles));
                const double2_t pivot_tilt{angles[2], angles[1]};
                const double22_t stretch_to_0deg{
                        noa::geometry::rotate(angles[0]) *
                        noa::geometry::scale(1 / noa::math::cos(pivot_tilt)) * // 1 = cos(0deg)
                        noa::geometry::rotate(-angles[0])
                };
                const double2_t shift_at_0deg = stretch_to_0deg * double2_t(metadata[i].shifts);
                mean += shift_at_0deg * mean_scale;
            }
            for (size_t i = 0; i < metadata.size(); ++i) {
                const double3_t angles(noa::math::deg2rad(metadata[i].angles));
                const double22_t shrink_matrix{
                        noa::geometry::rotate(angles[0]) *
                        noa::geometry::scale(noa::math::cos(double2_t(angles[2], angles[1]))) *
                        noa::geometry::rotate(-angles[0])
                };
                const float2_t shrank_mean(shrink_matrix * mean);
                metadata[i].shifts -= shrank_mean;
            }
        }

    private:
        // Main buffers.
        noa::Array<float> m_slices;
        noa::Array<float> m_slices_padded;
        noa::Array<cfloat_t> m_slices_fft;
        noa::Array<cfloat_t> m_slices_padded_fft;

        noa::Array<float> m_slice_weight_padded_fft;
        noa::Array<float> m_cumulative_fov;

        // Alias of the main buffers.
        noa::Array<float> m_slice0;
        noa::Array<float> m_slice1;
        noa::Array<cfloat_t> m_slice0_fft;
        noa::Array<cfloat_t> m_slice1_fft;
        noa::Array<float> m_slice0_padded;
        noa::Array<float> m_slice1_padded;
        noa::Array<cfloat_t> m_slice0_padded_fft;
        noa::Array<cfloat_t> m_slice1_padded_fft;

        dim_t m_max_size;
        dim4_t m_slice_shape;
        dim4_t m_slice_shape_padded;
        float2_t m_slice_center;
        float2_t m_slice_center_padded;
    };
}
