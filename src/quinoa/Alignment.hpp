#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct CoarseAlignmentParameters {
        Device compute_device;
        f64 maximum_resolution;
        bool check_rotation;
        bool fit_rotation_offset;
        bool fit_tilt_offset;
        bool fit_pitch_offset;
        Path output_directory;
    };

    struct CTFAlignmentParameters {
        Device compute_device;
        Path output_directory;

        f64 patch_size_ang;
        i64 n_images_in_initial_average;
        Vec<f64, 2> resolution_range;
        bool fit_phase_shift;
        bool fit_astigmatism;
        bool fit_thickness;
        bool check_rotation;

        // Refine:
        bool fit_rotation;
        bool fit_tilt;
        bool fit_pitch;
    };

    struct RefineAlignmentParameters {
        Device compute_device;
        f64 maximum_resolution;
        bool fit_rotation_offset;
        bool fit_tilt_offset;
        Path output_directory;
    };

    void coarse_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        const CoarseAlignmentParameters& parameters
    );

    void ctf_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        const CTFAlignmentParameters& parameters
    );

    void refine_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        const RefineAlignmentParameters& parameters
    );
}
