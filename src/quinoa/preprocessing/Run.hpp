#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    void preprocess(
        const Path& stack_filename,
        Metadata& metadata,
        Device device,
        const Settings::Preprocessing& settings,
        const Path& diagnostics_directory
    );
}
