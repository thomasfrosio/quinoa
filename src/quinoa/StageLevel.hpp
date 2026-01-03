#pragma once

#include <noa/Runtime.hpp>

#include "quinoa/Metadata.hpp"

namespace qn {
    struct StageLevelingParameters {
        f64 tilt_search_range{};
        f64 pitch_search_range{};
    };

    void coarse_stage_leveling(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        Vec<f64, 2>& tilt_pitch_offset,
        const StageLevelingParameters& options
    );
}
