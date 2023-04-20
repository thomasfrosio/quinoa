#pragma once

#include <numeric>

#include <noa/Array.hpp>
#include <noa/Memory.hpp>
#include <noa/Math.hpp>
#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/io/Logging.h"

namespace qn {
    struct ProjectionMatchingParameters {
        Vec2<f32> max_shift = {};
        f32 smooth_edge_percent = 0.5f;

        f32 backward_slice_z_radius = 0.0005f;
        f32 backward_tilt_angle_difference = 30.f;
        bool backward_use_aligned_only = false;

        f32 forward_cutoff = 0.5f;

        bool center_tilt_axis = true;
        Path debug_directory;
    };

    class ProjectionMatching {
    public:
        ProjectionMatching(const noa::Shape4<i64>& shape,
                           noa::Device compute_device,
                           const MetadataStack& metadata,
                           const ProjectionMatchingParameters& parameters,
                           noa::Allocator allocator = noa::Allocator::DEFAULT_ASYNC);

        void update_geometry(const Array<float>& stack,
                             MetadataStack& metadata,
                             const ProjectionMatchingParameters& parameters);

    private:
        [[nodiscard]]
        auto project_and_correlate_(
                const View<f32>& stack,
                const MetadataStack& metadata,
                i64 target_index,
                const std::vector<i64>& reference_indexes,
                const ProjectionMatchingParameters& parameters,
                const Vec3<f32>& angle_offsets
        ) -> f32;

        [[nodiscard]]
        auto extract_final_shift_(
                const MetadataStack& metadata,
                i64 target_index,
                const ProjectionMatchingParameters& parameters,
                const View<f32>& peak_window,
                const Vec2<f32>& peak_window_center
        ) -> Vec2<f32>;

        /// Compute the target and reference slices.
        /// \details Proximity weighting: the backward projected views are weighted based on their
        ///          tilt angle difference with the target. At the max angle difference, the weight is 0.
        /// \details Common field-of-view (FOV): the current geometry is used to ensure a common FOV between
        ///          the target slice and the projected-reference slice(s). 1) The reference slices are masked
        ///          out by the FOV of the target slice, ensuring that the final projected-reference doesn't
        ///          have any contribution that is not included in the target slice. 2) The target slice is
        ///          also masked out, but using the "cumulative" FOV of every projected slice, removing any
        ///          eventual regions that would be unique to the target slice and therefore not present in
        ///          any of the project-reference slices. It also locally weights the FOV of the target slice
        ///          based on how much a region contributes to the projected-reference slices.
        /// \details Oversampling: the projection is done on 2x zero-padded central slices.
        ///
        /// \param[in] stack                Input stack to align.
        /// \param metadata                 Metadata corresponding to \p stack.
        /// \param target_index             Index of the slice to forward-project.
        /// \param target_angles_offset     Angle offsets (in degrees) to add to the target angles.
        ///                                 This is used for the optimization function.
        /// \param projection_parameters    Parameters to use for the projection.
        /// \return
        void compute_target_and_reference_(
                const View<f32>& stack,
                const MetadataStack& metadata,
                i64 target_index,
                const Vec3<f32>& target_angles_offset,
                const std::vector<i64>& reference_indexes,
                const ProjectionMatchingParameters& parameters);

        [[nodiscard]] auto extract_peak_window_(const Vec2<f32>& max_shift) -> View<f32>;

        // Set the indexes of the slices contributing to the projected reference.
        static void set_reference_indexes_(
                i64 target_index,
                const MetadataStack& metadata,
                const ProjectionMatchingParameters& parameters,
                std::vector<i64>& output_reference_indexes);

        // Get the maximum number of reference slices used for projection.
        [[nodiscard]] static i64 max_references_count_(
                const MetadataStack& metadata,
                const ProjectionMatchingParameters& parameters);

        // Extract the peak from the cross-correlation map.
        // The cross-correlation map should be fft-centered.
        // The peak is likely distorted/stretched perpendicular to the tilt axis.
        // To help for the picking, align the tilt axis onto the vertical axis.
        // Since the peak window is likely to be restricted, i.e. max_shift,
        // make sure to only render the _small_ ellipse at the center.
        [[nodiscard]] static auto extract_peak_from_xmap_(
                const View<f32>& xmap,
                const View<f32>& peak_window,
                Vec2<f32> xmap_center,
                Vec2<f32> peak_window_center,
                const MetadataSlice& slice
        ) -> std::pair<Vec2<f32>, f32>;

        // Center the shifts. The mean should be computed and subtracted using a common reference frame.
        // Here, stretch the shifts to the 0deg reference frame and compute the mean there. Then transform
        // the mean to the slice tilt and pitch angles before subtraction.
        static void center_tilt_axis_(MetadataStack& metadata);

    private:
        // Device buffers.
        noa::Array<f32> m_references; // n slices
        noa::Array<c32> m_references_padded_fft; // n slices
        noa::Array<c32> m_target_reference_fft; // 2 slices
        noa::Array<c32> m_target_reference_padded_fft; // 2 slices
        noa::Array<f32> m_multiplicity_padded_fft; // 1 slice
        noa::Array<f32> m_peak_window;

        // Managed buffers (all with n elements each)
        noa::Array<f32> m_reference_weights;
        noa::Array<Vec4<i32>> m_reference_batch_indexes;
        noa::Array<Float23> m_fov_inv_reference2target;
        noa::Array<Float23> m_fov_inv_target2reference;
        noa::Array<Float33> m_insert_inv_references_rotation;
        noa::Array<Vec2<f32>> m_reference_shifts_center2origin;

        i64 m_max_size;
        Shape4<i64> m_slice_padded_shape;
        Shape4<i64> m_slices_padded_shape;
        Vec2<f32> m_slice_center;
        Vec2<f32> m_slice_center_padded;
    };
}
