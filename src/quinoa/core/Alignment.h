#pragma once

#include "quinoa/Types.h"
#include "quinoa/io/Options.h"
#include "quinoa/core/Metadata.h"

namespace qn {
    auto align(
            const Options& options,
            const MetadataStack& metadata
    ) -> std::tuple<MetadataStack, CTFAnisotropic64, std::vector<f64>>;
}
