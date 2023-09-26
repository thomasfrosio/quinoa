#pragma once

#include "quinoa/Types.h"
#include "quinoa/io/Options.h"
#include "quinoa/core/Metadata.h"

namespace qn {
    struct AlignmentOutputs {
        MetadataStack aligned_metadata;
        CTFAnisotropic64 average_ctf;
        f64 sample_thickness_nm{-1};
        f64 alignment_resolution{-1};
    };

    auto tilt_series_alignment(
            const Options& options,
            const MetadataStack& metadata
    ) -> AlignmentOutputs;
}
