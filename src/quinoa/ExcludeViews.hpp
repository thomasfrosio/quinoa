#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Stack.hpp"

namespace qn {
    struct DetectAndExcludeBlankViewsParameters {
        Device compute_device;
        Allocator allocator{Allocator::DEFAULT_ASYNC};
        i64 removable_edges{5}; // TODO
        Path output_directory;
    };

    void detect_and_exclude_blank_views(
        const Path& stack_filename,
        MetadataStack& metadata,
        const DetectAndExcludeBlankViewsParameters& parameters
    );
}
