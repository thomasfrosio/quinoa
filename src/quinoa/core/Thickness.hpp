#pragma once

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Stack.hpp"

namespace qn {
    struct EstimateSampleThicknessParameters {
        f64 resolution;
        f64 initial_thickness_nm;
        f64 maximum_thickness_nm;
        bool adjust_com;
        Device compute_device;
        Allocator allocator;
        Path debug_directory;
    };

    auto estimate_sample_thickness(
            const Path& stack_filename,
            MetadataStack& metadata, // updated: .shifts
            const EstimateSampleThicknessParameters& parameters
    ) -> f64;
}
