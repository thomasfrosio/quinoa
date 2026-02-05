#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct CoarseAlignmentSettings {
        Device device;
        bool check_rotation;
        bool fit_rotation_offset;
        bool fit_tilt_offset;
        bool fit_pitch_offset;
        Path output_directory;
    };
    void coarse_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        const CoarseAlignmentSettings& settings
    );

    struct RefineAlignmentSettings {
        Device compute_device;
        bool correct_ctf;
        f64 phase_flip_strength;
        bool fit_thickness;
        bool fit_rotation_offset;
        bool fit_tilt_offset;
        bool fit_pitch_offset;
        Path output_directory;
    };
    void refine_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        const RefineAlignmentSettings& settings
    );
}
