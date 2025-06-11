#pragma once

#include <noa/Array.hpp>

#include "quinoa/Metadata.hpp"

namespace qn {
    struct PairwiseTiltOptions {
        f64 grid_search_range;
        f64 grid_search_step;
        Path output_directory;
    };

    void coarse_fit_tilt(
        const View<f32>& stack,
        MetadataStack& metadata,
        f64& tilt_offset,
        const PairwiseTiltOptions& options
    );
}
