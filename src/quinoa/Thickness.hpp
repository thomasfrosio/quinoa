#pragma once

#include <noa/Runtime.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct EstimateSampleThicknessOptions {
        bool apply_fov{};
        Path output_directory;
    };

    auto estimate_sample_thickness(
        const View<f32>& stack,
        Metadata& metadata, // updated: stack.shifts, sample.thickness
        const EstimateSampleThicknessOptions& options
    ) -> f64; // nm

    struct EstimateSampleThicknessFromFileOptions {
        bool apply_fov{};
        Device device;
        Allocator allocator;
        f64 resolution; // A
        Path output_directory;
    };

    auto estimate_sample_thickness(
        const Path& stack_filename,
        Metadata& metadata, // updated: .shifts
        const EstimateSampleThicknessFromFileOptions& options
    ) -> f64; // nm
}
