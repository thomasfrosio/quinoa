#pragma once

#include <noa/Array.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct RotationOffsetParameters {
        bool reset_rotation{false};
        ns::Bandpass bandpass{0, 0, 0.5, 0};
        f64 angle_range{};
        f64 angle_step{};
        Path output_directory;
    };

    void find_rotation_offset(
        const View<const f32>& stack,
        MetadataStack& metadata,
        const RotationOffsetParameters& parameters
    );
}

