#pragma once

#include <noa/Array.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct RotationOffsetParameters {
        bool check_rotation{false};
        ns::Bandpass bandpass{0, 0, 0.5, 0};
        f64 angle_range{};
        const Path* output_directory{};
    };

    void find_rotation_offset(
        const View<const f32>& stack,
        Metadata::Stack& metadata,
        Vec<f64, 3>& angle_offsets,
        const RotationOffsetParameters& parameters
    );

    inline void find_rotation_offset(
        const View<const f32>& stack,
        Metadata::Stack& metadata,
        const RotationOffsetParameters& parameters
    ) {
        auto angle_offsets = Vec<f64, 3>{};
        find_rotation_offset(stack, metadata, angle_offsets, parameters);
    }
}

