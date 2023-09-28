#pragma once

#include <noa/core/utils/Timer.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/io/Logging.h"
#include "quinoa/core/Stack.hpp"

namespace qn {
    struct DetectAndExcludeBlankViewsParameters {
        Device compute_device;
        Allocator allocator;
        f64 resolution{};
    };

    void detect_and_exclude_blank_views(
            const Path& stack_filename,
            MetadataStack& metadata,
            const DetectAndExcludeBlankViewsParameters& parameters
    ) {
        noa::Timer timer;
        timer.start();
        qn::Logger::status("Blank view detection...");

        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ Allocator::DEFAULT_ASYNC,
                /*precise_cutoff=*/ false,
                /*rescale_target_resolution=*/ std::max(parameters.resolution, 20.),
                /*rescale_min_size=*/ 512,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.01, 0.01},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ false,
                /*smooth_edge_percent=*/ 0,
                /*zero_pad_to_fast_fft_shape=*/ false,
        };

        // Load the stack at very low resolution, without any normalization/padding/taper,
        // other than setting the means at zero (which is not required for the next steps).
        const auto [tilt_series, stack_spacing, file_spacing] =
                load_stack(stack_filename, metadata, loading_parameters);

        // If a view has much less variance than the rest of the stack, it's probably a bad view...
        auto profile = noa::math::var(tilt_series, Vec4<bool>{0, 1, 1, 1});
        if (profile.device().is_gpu())
            profile = profile.to_cpu();
        const auto span = profile.span();
        auto threshold = noa::math::median(profile);
        threshold *= 0.25f;
        qn::Logger::trace("threshold={:.3f}, variances=[\n{:+.2f}\n]", threshold, fmt::join(span, ",\n"));

        // Remove blank view(s) from the metadata.
        const auto original_size = metadata.size();
        metadata.exclude([&span, threshold](const MetadataSlice& slice) {
            if (span[slice.index] < threshold) {
                qn::Logger::status("Excluded blank view: index={:> 2} (tilt={:+.2f})", slice.index, slice.angles[1]);
                return true;
            }
            return false;
        });
        if (metadata.size() == original_size)
            qn::Logger::status("Excluded blank view: None");

        qn::Logger::status("Blank view detection... done. Took {:.2f}s", timer.elapsed() * 1e-3);
    }
}
