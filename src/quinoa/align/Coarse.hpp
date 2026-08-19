#pragma once

#include <noa/Types.hpp>

#include "quinoa/CommonFOV.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Types.hpp"

namespace qn {
    struct AlignShiftsOptions {
        bool cosine_stretch{};
        i32 update_count{1}; // negative means until convergence

        bool fov_mask{};
        f64 smooth_edge_percent{};
        f64 max_shift_percent{1};
    };

    struct LevelStageOptions {
        f64 tilt_search_range{};
        f64 pitch_search_range{};
        i32 n_global_search_evaluations{};

        bool fov_mask{};
        f64 smooth_edge_percent{};
        f64 max_shift_percent{1};
    };

    class AlignmentCoarse {
    public:
        AlignmentCoarse() = default;
        AlignmentCoarse(
            const Shape4& shape,
            Device device
        );

        void align_shifts(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            const AlignShiftsOptions& options
        );

        void level_stage(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            Vec<f64, 3>& angle_offsets,
            const LevelStageOptions& options
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

        Array<f32> m_xmap_centered;
        Array<Vec<f32, 2>> m_peak_shifts;
        Array<f32> m_peak_values;
        Array<Vec<f32, 5>> m_peak_stats;

        std::vector<Vec<f64, 2>> m_relative_shifts;
        std::vector<Vec<f64, 2>> m_global_shifts;
    };
}
