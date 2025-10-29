#pragma once

#include <noa/Array.hpp>

#include "quinoa/Metadata.hpp"

namespace qn {
    struct StageLevelingParameters {
        f64 tilt_search_range{};
        f64 tilt_search_step{1};
        f64 pitch_search_range{};
        f64 pitch_search_step{1};
        Path output_directory{};
    };

    void coarse_stage_leveling(
        const View<f32>& stack,
        MetadataStack& metadata,
        Vec<f64, 2>& tilt_pitch_offset,
        const StageLevelingParameters& options
    );
}
