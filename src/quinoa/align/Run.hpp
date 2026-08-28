#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    void coarse_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        Device device,
        const Settings::Alignment::Coarse& settings,
        const Path& output_directory
    );

    void refine_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        Device device,
        const Settings::Alignment::Refine& settings,
        const Path& output_directory
    );
}
