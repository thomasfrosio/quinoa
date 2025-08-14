#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Stack.hpp"

namespace qn {
    struct EstimateSampleThicknessParameters {
        f64 resolution;
        Device compute_device;
        Allocator allocator;
        Path output_directory;
    };

    auto estimate_sample_thickness(
        const Path& stack_filename,
        MetadataStack& metadata, // updated: .shifts
        const EstimateSampleThicknessParameters& parameters
    ) -> f64; // nm
}
