#pragma once

#include "quinoa/core/Metadata.h"
#include "quinoa/core/PairwiseShift.hpp"
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Stack.hpp"
#include "quinoa/core/GlobalRotation.hpp"
#include "quinoa/core/GridSearch1D.hpp"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/CTF.hpp"

namespace qn {
    struct PairwiseAlignmentParameters {
        Device compute_device;
        f64 maximum_resolution;

        bool search_rotation_offset{true};
        bool search_tilt_offset{true};

        Path output_directory;
        Path debug_directory;
    };

    // Initial global alignment.
    // Updates the tilt-series geometry:
    //  - The rotation offset can be either measured or refined if a value is already set.
    //  - The shifts are measured using pairwise comparisons.
    void pairwise_alignment(
            const Path& tilt_series_filename,
            MetadataStack& tilt_series_metadata,
            const PairwiseAlignmentParameters& parameters);


    void ctf_alignment(
            const Path& tilt_series_filename,
            MetadataStack& tilt_series_metadata
    );

    struct ProjectionMatchingAlignmentParameters {
        Device compute_device;
        f64 maximum_resolution;

        bool search_rotation{true};

        Path output_directory;
        Path debug_directory;
    };

    MetadataStack projection_matching_alignment(
            const Path& tilt_series_filename,
            const MetadataStack& tilt_series_metadata,
            const ProjectionMatchingAlignmentParameters& parameters);
}
