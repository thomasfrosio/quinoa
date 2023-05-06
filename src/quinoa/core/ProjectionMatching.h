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

// Projection matching
// -------------------
//
// Concept:
//
//
// Strength:
//
//
// Issues:
//

namespace qn {
    struct ProjectionMatchingParameters {
        Vec2<f32> max_shift = {};
        f32 area_match_taper = 0.1f;

        f32 projection_slice_z_radius = 0.0005f;
        f32 projection_cutoff = 0.5f;
        f64 projection_max_tilt_angle = 50;

        Vec2<f32> highpass_filter{};
        Vec2<f32> lowpass_filter{};

        i64 max_iterations = 5;
        bool center_tilt_axis = true;
        Path debug_directory;
    };

    struct ThirdDegreePolynomial {
        f64 a, b, c, d;
        constexpr f64 operator()(f64 x) const noexcept {
            return a * x * x * x + b * x * x + c * x + d;
        }
    };

    class ProjectionMatching {
    public:
        ProjectionMatching(const noa::Shape4<i64>& shape,
                           noa::Device compute_device,
                           const MetadataStack& metadata,
                           const ProjectionMatchingParameters& parameters,
                           noa::Allocator allocator = noa::Allocator::DEFAULT_ASYNC);

        void update(const Array<float>& stack,
                    MetadataStack& metadata,
                    const ProjectionMatchingParameters& parameters,
                    bool shift_only);

    private:
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

        static void apply_area_mask_(
                const View<f32>& input,
                const View<f32>& output,
                const MetadataSlice& metadata,
                const ProjectionMatchingParameters& parameters,
                bool apply_weight
        );

        static void apply_area_mask_(
                const View<f32>& input,
                const View<f32>& output,
                const MetadataStack& metadata,
                const std::vector<i64>& indexes,
                const ProjectionMatchingParameters& parameters,
                bool apply_weight
        );

        // Prepare the references for back-projection.
        void prepare_for_insertion_(
                const View<f32>& stack,
                const MetadataStack& metadata,
                i64 target_index,
                const std::vector<i64>& reference_indexes,
                const ProjectionMatchingParameters& parameters
        );

        // Compute the projected reference.
        void compute_target_reference_(
                const MetadataStack& metadata,
                i64 target_index,
                f64 target_rotation_offset,
                const std::vector<i64>& reference_indexes,
                const ProjectionMatchingParameters& parameters);

        [[nodiscard]]
        auto project_and_correlate_(
                const MetadataStack& metadata,
                i64 target_index,
                const std::vector<i64>& reference_indexes,
                const ProjectionMatchingParameters& parameters,
                noa::signal::CorrelationMode correlation_mode,
                f64 angle_offset
        ) -> std::pair<Vec2<f64>, f64>;

        // Fit a 3rd degree polynomial on the rotation.
        [[nodiscard]] static ThirdDegreePolynomial poly_fit_rotation(const MetadataStack& metadata);

        // Center the shifts. The mean should be computed and subtracted using a common reference frame.
        // Here, stretch the shifts to the 0deg reference frame and compute the mean there. Then transform
        // the mean to the slice tilt and pitch angles before subtraction.
        static void center_tilt_axis_(MetadataStack& metadata);

    private:
        // Device buffers.
        noa::Array<f32> m_slices; // n+1 slices
        noa::Array<c32> m_slices_padded_fft; // n+1 slices
        noa::Array<c32> m_target_reference_fft; // 2 slices
        noa::Array<c32> m_target_reference_padded_fft; // 2 slices
        noa::Array<f32> m_multiplicity_padded_fft; // 1 slice
        noa::Array<f32> m_peak_window;

        // Managed buffers (all with n elements each)
        noa::Array<f32> m_reference_weights;
        noa::Array<Vec4<i32>> m_reference_batch_indexes;
        noa::Array<Float33> m_insert_inv_references_rotation;
        noa::Array<Vec2<f32>> m_reference_shifts_center2origin;

        [[nodiscard]] constexpr i64 height() const noexcept {
            return m_slices.shape()[2];
        }

        [[nodiscard]] constexpr i64 width() const noexcept {
            return m_slices.shape()[3];
        }

        [[nodiscard]] constexpr i64 size_padded() const noexcept {
            // Assumes the padded references are squares.
            return m_slices_padded_fft.shape()[2];
        }

        [[nodiscard]] constexpr Vec2<f32> slice_center() const noexcept {
            return MetadataSlice::center(m_slices.shape());
        }

        [[nodiscard]] constexpr f32 slice_padded_center() const noexcept {
            return MetadataSlice::center(size_padded(), size_padded())[0];
        }
    };
}
