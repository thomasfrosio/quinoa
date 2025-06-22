#include "quinoa/ExcludeViews.hpp"

#include <noa/Array.hpp>
#include <noa/Geometry.hpp>

namespace {
    using namespace qn;

    void find_bad_images(
        std::vector<i64>& output,
        SpanContiguous<const f32> variances,
        const MetadataStack& metadata,
        bool edge_only,
        i64 removable_edges
    ) {
        if (metadata.ssize() < 2)
            return;

        // Collect the metrics.
        std::vector<i64> indices;
        std::vector<f64> points, gradients;
        for (i64 i{1}; i < metadata.ssize(); ++i) {
            const auto current = metadata[i].index;
            const auto previous = metadata[i - 1].index;
            indices.push_back(current);
            points.push_back(variances[current]);
            gradients.push_back(std::abs(variances[current] - variances[previous]));
        }

        // Quality metrics.
        const auto median_stddev = noa::median(View(points.data(), std::ssize(points)));
        const auto median_gradient = noa::median(View(gradients.data(), std::ssize(gradients)));
        const auto threshold_stddev_low_first = median_stddev * 0.5;
        const auto threshold_stddev_low_second = median_stddev * 0.8;
        const auto threshold_stddev_high = median_stddev * 2;
        const auto threshold_gradient = median_gradient * 4;

        // Flag the bad images.
        for (size_t i{}; i < indices.size(); i++) {
            if (points[i] > threshold_stddev_high or
                points[i] < threshold_stddev_low_first or
                (gradients[i] >= threshold_gradient and points[i] < threshold_stddev_low_second)) {
                indices[i] += 1000;
            }
        }

        // Add indices of images to remove.
        if (edge_only) {
            for (i64 i{}; i64 index: stdv::reverse(indices)) {
                if (index < 1000 or i++ == removable_edges)
                    break; // stop at the first good image
                output.push_back(index - 1000);
            }
        } else {
            for (i64 i: indices)
                if (i >= 1000)
                    output.push_back(i - 1000);
        }
    }
}

namespace qn {
    void detect_and_exclude_blank_views(
        const Path& stack_filename,
        MetadataStack& metadata,
        const DetectAndExcludeBlankViewsParameters& parameters
    ) {
        auto timer = Logger::info_scope_time("Blank view detection");

        // Load the stack at very low resolution, without any normalization/padding/taper,
        // other than setting the mean to zero (which is not required for the next steps).
        const auto [tilt_series, stack_spacing, file_spacing, _] = load_stack(stack_filename, metadata, {
            .compute_device = parameters.compute_device,
            .allocator = parameters.allocator,
            .precise_cutoff = false,
            .rescale_target_resolution = 20.,
            .rescale_min_size = 512,
            .rescale_max_size = 1024,
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
        auto span = profile.span_1d();

        save_plot_xy(
            metadata | stdv::transform([](auto& s) { return s.angles[1]; }), span,
            parameters.output_directory / "exclude_blank_views.txt", {
                .x_name = "Tilt (in degrees)",
                .y_name = "Variance",
                .label = "all",
            });

        std::vector<i64> indices;
        auto metadata_tmp = metadata;

        metadata_tmp.exclude_if([](auto& s) { return s.angles[1] < 20; });
        metadata_tmp.sort("tilt", true);
        find_bad_images(indices, span, metadata_tmp, true, parameters.removable_edges);

        metadata_tmp = metadata;
        metadata_tmp.exclude_if([](auto& s) { return s.angles[1] > -20; });
        metadata_tmp.sort("tilt", false);
        find_bad_images(indices, span, metadata_tmp, true, parameters.removable_edges);

        metadata_tmp = metadata;
        metadata_tmp.exclude_if([](auto& s) { return std::abs(s.angles[1]) > 25; });
        find_bad_images(indices, span, metadata_tmp, false, 0);

        // Remove blank view(s) from the metadata.
        const auto original_size = metadata.size();
        metadata.exclude_if([&](const MetadataSlice& slice) {
            if (stdr::find(indices, slice.index) != indices.end()) {
                Logger::info("Excluding view: index={} (tilt={:+.2f})", slice.index, slice.angles[1]);
                return true;
            }
            return false;
        });
        if (metadata.size() == original_size)
            Logger::info("Excluding view: None");
    }
}
