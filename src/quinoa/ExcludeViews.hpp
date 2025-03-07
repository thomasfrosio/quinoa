#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Stack.hpp"

namespace qn {
    struct DetectAndExcludeBlankViewsParameters {
        Device compute_device;
        Allocator allocator{Allocator::DEFAULT_ASYNC};
        f64 resolution{};
    };

    inline void detect_and_exclude_blank_views(
        const Path& stack_filename,
        MetadataStack& metadata,
        const DetectAndExcludeBlankViewsParameters& parameters
    ) {
        auto timer = Logger::info_scope_time("Blank view detection");

        // Load the stack at very low resolution, without any normalization/padding/taper,
        // other than setting the mean to zero (which is not required for the next steps).
        const auto [tilt_series, stack_spacing, file_spacing] = load_stack(stack_filename, metadata, {
            .compute_device = parameters.compute_device,
            .allocator = parameters.allocator,
            .precise_cutoff = false,
            .rescale_target_resolution = std::max(parameters.resolution, 20.),
            .rescale_min_size = 512,
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.01,
                .highpass_width = 0.01,
                .lowpass_cutoff = 0.5,
                .lowpass_width = 0.05,
            },
            .normalize_and_standardize = false,
            .smooth_edge_percent = 0.,
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });

        // If a view has much less variance than the rest of the stack, it's probably a blank view...
        auto profile = noa::variance(tilt_series, noa::ReduceAxes::all_but(0));
        profile = profile.is_dereferenceable() ?
            std::move(profile).reinterpret_as_cpu() :
            std::move(profile).to_cpu();

        const auto threshold = noa::median(profile) * 0.25f;
        const auto span = profile.span_1d_contiguous();
        Logger::trace("threshold={:.3f}, variances=[\n{:+.2f}\n]", threshold, fmt::join(span, ",\n"));

        // Remove blank view(s) from the metadata.
        const auto original_size = metadata.size();
        metadata.exclude([&span, threshold](const MetadataSlice& slice) {
            if (span[slice.index] < threshold) {
                Logger::info("Excluded blank view: index={:> 2} (tilt={:+.2f})", slice.index, slice.angles[1]);
                return true;
            }
            return false;
        });
        if (metadata.size() == original_size)
            Logger::info("Excluded blank view: None");
    }
}
