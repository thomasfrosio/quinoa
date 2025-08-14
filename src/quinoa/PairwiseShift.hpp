#pragma once

#include <noa/Array.hpp>
#include <noa/Signal.hpp>

#include "quinoa/CommonArea.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Types.hpp"

namespace qn {
    struct PairwiseShiftParameters {
        ns::Bandpass bandpass{0, 0, 0.5, 0};
        noa::Interp interp{noa::Interp::LINEAR_FAST};
        Path output_directory;

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

    private:
        Array<c32> m_buffer_rfft; // (3,1,h,w/2+1)
        Array<f32> m_xmap; // (1,1,h,w)
        Array<f32> m_xmap_centered; // (1,1,64,64)
        CommonArea m_common_area;
    };
}
