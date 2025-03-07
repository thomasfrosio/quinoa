#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Options.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct CoarseAlignmentParameters {
        Device compute_device;
        f64 maximum_resolution;
        bool fit_rotation_offset;
        bool fit_tilt_offset;
        bool has_user_rotation;
        Path output_directory;
        Path debug_directory;
    };

    struct CTFAlignmentParameters {
        Device compute_device;
        Path output_directory;
        Path debug_directory;

        f64 voltage;
        f64 cs;
        f64 amplitude;
        f64 astigmatism_value;
        f64 astigmatism_angle;
        f64 phase_shift;

        i64 patch_size;
        Vec<f64, 2> resolution_range;
        bool fit_envelope;
        bool fit_phase_shift;
        bool fit_astigmatism;

        // Coarse:
        Vec<f64, 2> delta_z_range_nanometers;
        f64 delta_z_shift_nanometers;
        f64 max_tilt_for_average;
        bool has_user_rotation;

        // Refine:
        bool fit_rotation;
        bool fit_tilt;
        bool fit_pitch;
    };

    auto coarse_alignment(
        const Path& stack_filename,
        MetadataStack& metadata,
        const CoarseAlignmentParameters& parameters
    ) -> f64;

    auto ctf_alignment(
        const Path& stack_filename,
        MetadataStack& metadata,
        const CTFAlignmentParameters& parameters
    ) -> f64;
}
