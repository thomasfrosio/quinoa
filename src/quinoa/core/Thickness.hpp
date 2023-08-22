#pragma once

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Stack.hpp"

namespace qn {
    auto estimate_sample_thickness(
            const View<f32>& stack,
            const MetadataStack& metadata,
            f64 spacing,
            const Path& debug_directory
    ) -> Vec2<f64>;

    struct EstimateSampleThicknessParameters {
        f64 resolution;
        Device compute_device;
        Allocator allocator;
        Path debug_directory;
    };

    inline auto estimate_sample_thickness(
            const Path& stack_filename,
            MetadataStack metadata,
            const EstimateSampleThicknessParameters& parameters
    ) -> Vec2<f64> {
        const auto debug = !parameters.debug_directory.empty();
        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ Allocator::DEFAULT_ASYNC,
                /*precise_cutoff=*/ true,
                /*rescale_target_resolution=*/ std::max(parameters.resolution, 12.),
                /*rescale_min_size=*/ 1024,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.02, 0.02},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.03f,
                /*zero_pad_to_fast_fft_shape=*/ false,
        };
        auto [stack, stack_spacing, file_spacing] = load_stack(stack_filename, metadata, loading_parameters);
        metadata.rescale_shifts(file_spacing, stack_spacing);
        return estimate_sample_thickness(
                stack.view(), metadata, noa::math::sum(stack_spacing) / 2, parameters.debug_directory);
    }
}
