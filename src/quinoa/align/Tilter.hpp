#pragma once

#include <noa/Types.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Types.hpp"
#include "quinoa/align/CommonFOV.hpp"

namespace qn {
    struct FindImageShiftsOptions {
        bool cosine_stretch{};
        i32 update_count{1}; // negative means until convergence

        bool fov_mask{};
        f64 smooth_edge_percent{};
        f64 max_shift_percent{1};
    };

    struct FindImageRotationOptions {
        bool accurate_fov{true}; // false triggers faster algorithm with inaccurate FOV
        f64 angle_range{-1}; // negative means full rotation (+-90 degrees)
        f64 angle_step{1}; // only used if accurate_fov=false
        const Path* output_directory{};
    };

    struct FindSpecimenLevelOptions {
        f64 tilt_search_range{};
        f64 pitch_search_range{};
        i32 n_global_search_evaluations{};

        bool fov_mask{};
        f64 smooth_edge_percent{};
        f64 max_shift_percent{1};
    };

    struct FindAccurateImageRotationOptions {
        f64 angle_range{};
        const Path* output_directory{};
    };

    class Tilter {
    public:
        static void find_accurate_image_rotation(
            const View<const f32>& stack,
            Metadata::Stack& metadata,
            Vec<f64, 3>& angle_offsets,
            const FindAccurateImageRotationOptions& parameters
        );

    public:
        Tilter() = default;
        Tilter(const Shape4& shape, Device device);

        void find_image_rotation(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            Vec<f64, 3>& angle_offsets,
            const FindImageRotationOptions& options
        );

        void find_image_shifts(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            const FindImageShiftsOptions& options
        );

        void find_specimen_level(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            Vec<f64, 3>& angle_offsets,
            const FindSpecimenLevelOptions& options
        );

    private:
        [[nodiscard]] auto eval_(
            const View<f32>& stack,
            const Metadata::Stack& metadata,
            bool fov_mask,
            f64 smooth_edge_percent,
            f64 max_shift_percent,
            bool cosine_stretch,
            bool need_score
        ) -> Pair<Vec<f64, 2>, f64>;

        [[nodiscard]] auto buffer(nt::integer auto start, nt::integer auto end) const {
            auto n_targets = m_xmap_centered.shape()[0];
            return m_buffer.subregion(Slice{start * n_targets, end * n_targets});
        }
        [[nodiscard]] auto buffer(nt::integer auto i) const {
            return buffer(i, i + 1);
        }
        [[nodiscard]] auto buffer_rfft(nt::integer auto start, nt::integer auto end) const {
            auto n_targets = m_xmap_centered.shape()[0];
            return m_buffer_rfft.view().subregion(Slice{start * n_targets, end * n_targets});
        }
        [[nodiscard]] auto buffer_rfft(nt::integer auto i) const {
            return buffer_rfft(i, i + 1);
        }

    private:
        Array<c32> m_buffer_rfft;
        View<f32> m_buffer;

        Array<Vec<f32, 4>> m_plane_coefficients;
        Array<Mat<f32, 2, 4>> m_projection_matrices;
        Array<ParallelogramMask> m_fov_masks;
        Array<Mat<f32, 2, 3>> m_shift_matrices;

        Array<f32> m_xmap_centered;
        Array<Vec<f32, 2>> m_peak_shifts;
        Array<f32> m_peak_values;
        Array<Vec<f32, 5>> m_peak_stats;

        std::vector<Vec<f64, 2>> m_relative_shifts;
        std::vector<Vec<f64, 2>> m_global_shifts;
    };
}
