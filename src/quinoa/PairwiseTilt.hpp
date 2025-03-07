#pragma once

#include <noa/Array.hpp>

#include "quinoa/Metadata.hpp"

namespace qn {
    struct PairwiseTiltOptions {
        f64 grid_search_range;
        f64 grid_search_step;
    };

    void coarse_fit_tilt(const View<f32>& stack, MetadataStack& metadata, const PairwiseTiltOptions& options);
}
