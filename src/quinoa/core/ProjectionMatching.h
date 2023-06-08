#pragma once

#include <numeric>
#include <optional>

#include <noa/Array.hpp>
#include <noa/Memory.hpp>
#include <noa/Math.hpp>
#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/CommonArea.hpp"
#include "quinoa/core/Utilities.h"
#include "quinoa/io/Logging.h"
#include "quinoa/core/CubicGrid.hpp"

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
        bool zero_pad_to_fast_fft_size{true};

        f64 rotation_tolerance_abs{0.005};
        f64 rotation_range{0.5};
        f64 rotation_initial_step{0.15};

        f64 smooth_edge_percent{0.1};

        f32 projection_slice_z_radius = 0.01f;
        f32 projection_cutoff = 0.5f;
        f64 projection_max_tilt_angle_difference = 50;

        Vec2<f32> highpass_filter{};
        Vec2<f32> lowpass_filter{};

        f64 allowed_shift_percent{0.005};

        Path debug_directory;
    };

    class ProjectionMatching {
    public:
        [[nodiscard]]
        static u64 predict_memory_usage(
                const noa::Shape2<i64>& shape,
                const MetadataStack& metadata,
                const ProjectionMatchingParameters& parameters) noexcept;

    public:
        ProjectionMatching() = default;

        ProjectionMatching(const noa::Shape2<i64>& shape,
                           noa::Device compute_device,
                           const MetadataStack& metadata,
                           const ProjectionMatchingParameters& parameters,
                           noa::Allocator allocator = noa::Allocator::DEFAULT_ASYNC);

        void update(const View<f32>& stack,
                    MetadataStack& metadata,
                    const CommonArea& common_area,
                    const CubicSplineGrid<f64, 1>& rotation_model,
                    const ProjectionMatchingParameters& parameters);

    private:
        [[nodiscard]] auto extract_peak_window_(const Vec2<f64>& max_shift) const -> View<f32>;

        static void set_reference_indexes_(
                i64 target_index,
                const MetadataStack& metadata,
                const ProjectionMatchingParameters& parameters,
                std::vector<i64>& output_reference_indexes);

        [[nodiscard]] static i64 max_references_count_(
                const MetadataStack& metadata,
                const ProjectionMatchingParameters& parameters);

        [[nodiscard]] static i64 find_tilt_neighbour_(
                const MetadataStack& metadata,
                i64 target_index,
                const std::vector<i64>& reference_indexes);

        void prepare_for_insertion_(
                const View<f32>& stack,
                const MetadataStack& metadata,
                i64 target_index,
                const std::vector<i64>& reference_indexes,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters) const;

        void compute_target_reference_(
                const MetadataStack& metadata,
                i64 target_index,
                f64 target_rotation_offset,
                const std::vector<i64>& reference_indexes,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters) const;

        [[nodiscard]]
        auto select_peak_(
                const View<f32>& xmap,
                const MetadataStack& metadata,
                i64 target_index,
                f64 target_angle_offset,
                const std::vector<i64>& reference_indexes,
                const std::vector<Vec2<f64>>& reference_shift_offsets,
                const ProjectionMatchingParameters& parameters
        ) const -> std::pair<Vec2<f64>, f64>;

        [[nodiscard]]
        auto project_and_correlate_(
                const MetadataStack& metadata,
                i64 target_index,
                f64 target_rotation_offset,
                const std::vector<i64>& reference_indexes,
                const std::vector<Vec2<f64>>& reference_shift_offsets,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) const -> std::pair<Vec2<f64>, f64>;

        [[nodiscard]]
        f64 run_projection_matching_(
                const View<f32>& stack,
                MetadataStack& metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters,
                const CubicSplineGrid<f64, 1>& rotation_target) const;

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
