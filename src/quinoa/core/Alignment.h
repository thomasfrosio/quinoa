#pragma once

#include "quinoa/Types.h"
#include "quinoa/io/Options.h"
#include "quinoa/core/Metadata.h"

namespace qn {
    auto tilt_series_alignment(
            const Options& options,
            const MetadataStack& metadata
    ) -> std::tuple<MetadataStack, CTFAnisotropic64>;
}
