#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>

#include "quinoa/ExcludeViews.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/Stack.hpp"

namespace {
    using namespace qn;

    void find_bad_images(
        std::vector<i64>& output,
        SpanContiguous<const f32> variances,
        const Metadata::Stack& metadata,
        bool edge_only,
        i64 removable_edges
    ) {
        if (metadata.ssize() < 2)
            return;

        // Collect the data.
        std::vector<i64> indices;
        std::vector<f64> points, gradients;
        for (i64 i{1}; i < metadata.ssize(); ++i) {
            const auto current = metadata[i].index;
            const auto previous = metadata[i - 1].index;
            indices.push_back(current);
            points.push_back(static_cast<f64>(variances[current]));
            gradients.push_back(static_cast<f64>(std::abs(variances[current] - variances[previous])));
        }

        // Quality metrics.
        const auto median_stddev = noa::median(View(points.data(), std::ssize(points)));
        const auto median_gradient = noa::median(View(gradients.data(), std::ssize(gradients)));
        const auto threshold_stddev_low_first = median_stddev * 0.5;
        const auto threshold_stddev_low_second = median_stddev * 0.8;
        const auto threshold_stddev_high = median_stddev * 2;
        const auto threshold_gradient = median_gradient * 4;

        // const auto med_var = noa::median(View(points.data(), std::ssize(points)));
        // std::vector<f64> points_med;
        // for (auto var: points)
        //     points_med.push_back(std::abs(var - med_var));
        // const auto mad_var = noa::median(View(points_med.data(), std::ssize(points)));
        //
        // const auto med_grad = noa::median(View(gradients.data(), std::ssize(gradients)));
        // std::vector<f64> gradients_med;
        // for (auto grad: gradients)
        //     gradients_med.push_back(std::abs(grad - med_grad));
        // const auto mad_grad = noa::median(View(gradients_med.data(), std::ssize(gradients)));

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
        Metadata::Stack& metadata,
        const DetectAndExcludeBlankViewsParameters& parameters
    ) {
        auto timer = Logger::info_scope_time("Blank view detection");
        timer.set_newline(false);

        // Load the stack at very low resolution, without any normalization/padding/taper,
        // other than setting the mean to zero (which is not required for the next steps).
        const auto tilt_series = load_stack(stack_filename, metadata, {
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
        }).stack;

        // Compute the variance of each image.
        auto profile = noa::variance(tilt_series, ReduceAxes::all_but(0));
        profile = profile.is_dereferenceable() ?
            std::move(profile).reinterpret_as_cpu() :
            std::move(profile).to_cpu();

        const auto span = profile.span_1d();
        save_plot_xy(
            metadata | stdv::transform([](auto& s) { return s.angles[1]; }), span,
            parameters.output_directory / "exclude_blank_views.txt", {
                .x_name = "Tilt (in degrees)",
                .y_name = "Variance",
                .label = "all",
            });

        auto indices = std::vector<i64>{};

        auto meta = metadata;
        meta.exclude_if([](auto& s) { return s.angles[1] < 20; });
        meta.sort("tilt", true);
        find_bad_images(indices, span, meta, true, parameters.removable_edges);

        meta = metadata;
        meta.exclude_if([](auto& s) { return s.angles[1] > -20; });
        meta.sort("tilt", false);
        find_bad_images(indices, span, meta, true, parameters.removable_edges);

        meta = metadata;
        meta.exclude_if([](auto& s) { return std::abs(s.angles[1]) > 25; });
        find_bad_images(indices, span, meta, false, 0);

        // Remove blank view(s) from the metadata.
        const auto original_size = metadata.size();
        metadata.exclude_if([&](const auto& image) {
            if (stdr::find(indices, image.index) != indices.end()) {
                Logger::info("Excluding view: index={} (tilt={:+.2f})", image.index, image.angles[1]);
                return true;
            }
            return false;
        });
        if (metadata.size() == original_size)
            Logger::info("Excluding view: None");
    }
}
