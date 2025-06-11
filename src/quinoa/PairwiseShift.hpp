#pragma once

#include <numeric>

#include <noa/Array.hpp>
#include <noa/Signal.hpp>
#include <noa/Session.hpp>

#include "quinoa/CommonArea.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Types.hpp"

namespace qn {
    struct PairwiseShiftParameters {
        ns::Bandpass bandpass{0, 0, 0.5, 0};
        noa::Interp interp{noa::Interp::LINEAR_FAST};
        Path output_directory;
        Path debug_directory;

        bool cosine_stretch{};
        bool area_match{};
        f64 smooth_edge_percent{};
        f64 max_shift_percent{1};
        i32 update_count{1};
    };

    class PairwiseShift {
    public:
        PairwiseShift() = default;
        PairwiseShift(
            const Shape4<i64>& shape,
            Device compute_device,
            Allocator allocator = Allocator::DEFAULT_ASYNC
        );

        void update(
            const View<f32>& stack,
            MetadataStack& metadata,
            const PairwiseShiftParameters& parameters
        );

    private:
        [[nodiscard]] auto find_relative_shifts_(
            const View<f32>& stack,
            const MetadataSlice& reference_slice,
            const MetadataSlice& target_slice,
            const PairwiseShiftParameters& parameters
        ) const -> Vec2<f64>;

        static auto relative2global_shifts_(
            const std::vector<Vec2<f64>>& relative_shifts,
            const MetadataStack& metadata,
            i64 index_lowest_tilt,
            bool cosine_stretch
        ) -> Pair<std::vector<Vec<f64, 2>>, Vec<f64, 2>>;

    private:
        noa::Array<c32> m_buffer_rfft;
        noa::Array<f32> m_xmap;
        CommonArea m_common_area;
    };
}
