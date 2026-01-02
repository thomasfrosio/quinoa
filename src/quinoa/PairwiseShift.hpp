#pragma once

#include <noa/Array.hpp>
#include <noa/Signal.hpp>

#include "quinoa/CommonFOV.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Types.hpp"

namespace qn {
    struct PairwiseShiftParameters {
        ns::Bandpass bandpass{0, 0, 0.5, 0};
        noa::Interp interp{noa::Interp::LINEAR_FAST};
        const Path* output_directory{};

        bool cosine_stretch{};
        bool area_match{};
        f64 smooth_edge_percent{};
        f64 max_shift_percent{1};
        i32 update_count{1};

        bool compute_peaks{false};
    };

    class PairwiseShift {
    public:
        PairwiseShift() = default;
        PairwiseShift(
            const Shape4<i64>& shape,
            Device compute_device
        );

        void update(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            const PairwiseShiftParameters& parameters
        );

    private:
        [[nodiscard]] auto find_relative_shifts_(
            const View<f32>& stack,
            Metadata::Image reference_image,
            const Metadata::Image& target_image,
            const PairwiseShiftParameters& parameters
        ) const -> Vec2<f64>;

    private:
        Array<f32> m_buffer;
        Array<c32> m_buffer_rfft; // (3,1,h,w/2+1)
        Array<f32> m_xmap; // (1,1,h,w)
        Array<f32> m_xmap_centered; // (1,1,64,64)
        CommonFOV m_common_fov;
    };

    class PairwiseShift2 {
    public:
        PairwiseShift2() = default;
        PairwiseShift2(
            const Shape4<i64>& shape,
            Device compute_device
        );

        void update(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            const PairwiseShiftParameters& parameters
        );

    public:
        [[nodiscard]] auto find_relative_shifts_(
            View<f32> stack,
            const Metadata::Stack& metadata,
            const PairwiseShiftParameters& parameters
        ) -> Pair<Vec2<f64>, f64>;

        [[nodiscard]] auto buffer(std::integral auto start, std::integral auto end) {
            auto n_targets = m_xmap_centered.shape()[0];
            return m_buffer.subregion(ni::Slice{start * n_targets, end * n_targets});
        }
        [[nodiscard]] auto buffer(std::integral auto i) {
            return buffer(i, i + 1);
        }
        [[nodiscard]] auto buffer_rfft(std::integral auto start, std::integral auto end) {
            auto n_targets = m_xmap_centered.shape()[0];
            return m_buffer_rfft.view().subregion(ni::Slice{start * n_targets, end * n_targets});
        }
        [[nodiscard]] auto buffer_rfft(std::integral auto i) {
            return buffer_rfft(i, i + 1);
        }

    private:
        Array<c32> m_buffer_rfft; // (n*3,1,h,w/2+1)
        View<f32> m_buffer; // (n,1,h,w)

        Array<f32> m_xmap_centered; // (n,1,64,64)

        Array<Vec<f32, 4>> m_plane_coefficients;
        Array<Mat<f32, 2, 4>> m_projection_matrices;
        Array<ParallelogramMask> m_fov_masks;

        Array<Vec<f32, 2>> m_peak_shifts;
        Array<f32> m_peak_values;
        Array<Vec<f32, 5>> m_peak_stats;

        std::vector<Vec<f64, 2>> m_relative_shifts;
        std::vector<Vec<f64, 2>> m_global_shifts;
    };
}
