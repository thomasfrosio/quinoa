#pragma once

#include <noa/Array.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct RotationOffsetParameters {
        bool reset_rotation{false};
        noa::Interp interp{noa::Interp::LINEAR};
        ns::Bandpass bandpass{0, 0, 0.5, 0};
        f64 angle_range{};
        f64 angle_step{};
        f64 line_range{0};
        f64 line_step{1};
        Path output_directory;
    };

    class RotationOffset {
    public:
        void search(
            const View<const f32>& stack,
            MetadataStack& metadata,
            const RotationOffsetParameters& parameters
        );

    private:
        Array<c32> m_stack_padded_rfft{};
    };
}

