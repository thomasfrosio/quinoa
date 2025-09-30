#include <noa/Array.hpp>
#include <noa/Geometry.hpp>

#include "quinoa/Thickness.hpp"
#include "quinoa/Plot.hpp"

namespace {
    using namespace qn;

    /// Index-wise reduction operator sampling the backprojected tomogram.
    struct TomogramVariance {
    public:
        static constexpr auto INTERP = noa::Interp::LINEAR;
        static constexpr auto BORDER = noa::Border::ZERO;
        using input_span_t = SpanContiguous<const f32, 3>;
        using interpolator_t = noa::Interpolator<2, INTERP, BORDER, input_span_t>;
        using matrices_span_t = SpanContiguous<const Mat<f32, 2, 4>>;

    public:
        interpolator_t images{}; // (n,h,w)
        matrices_span_t projection_matrices{}; // (n)
        f64 n_elements_per_image{};
        // SpanContiguous<f32, 3> tomogram{};

    public:
        [[nodiscard]] constexpr auto backproject(const Vec<i32, 3>& indices) const -> f32 {
            const auto volume_coordinates = indices.as<f32>().push_back(1);
            f32 value{};
            for (i64 i{}; i < projection_matrices.ssize(); ++i) {
                const auto image_coordinates = projection_matrices[i] * volume_coordinates;
                value += images.interpolate_at(image_coordinates, i);
            }
            // tomogram(indices) = value;
            return value;
        }

        constexpr void init(const Vec<i32, 3>& indices, f32& sum, f32& sum_sqd) const {
            const auto v = backproject(indices);
            sum += v;
            sum_sqd += v * v;
        }

        static constexpr void join(const f32& isum, const f32& isum_sqd, f32& sum, f32& sum_sqd) {
            sum += isum;
            sum_sqd += isum_sqd;
        }

        using remove_default_final = bool;
        constexpr void final(const f32& sum, const f32& sum_sqd, f64& variance) const {
            const auto mean = static_cast<f64>(sum) / n_elements_per_image;
            variance = static_cast<f64>(sum_sqd) / n_elements_per_image - noa::abs_squared(mean);
        }
    };
}

namespace qn {
    auto estimate_sample_thickness(
        const Path& stack_filename,
        MetadataStack& metadata,
        const EstimateSampleThicknessParameters& parameters
    ) -> f64 {
        auto timer = Logger::info_scope_time("Thickness estimation");

        auto stack_loader = StackLoader(stack_filename, {
            .compute_device = parameters.compute_device,
            .allocator = parameters.allocator,
            .precise_cutoff = true, // enforce isotropic spacing
            .rescale_target_resolution = parameters.resolution,
            .rescale_min_size = 512,
            .rescale_max_size = 1024,
            .bandpass{
                .highpass_cutoff = 0.02,
                .highpass_width = 0.02,
                .lowpass_cutoff = 0.49,
                .lowpass_width = 0.01,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });

        const f64 stack_spacing_nm = 1e-1 * noa::mean(stack_loader.stack_spacing());
        const f64 file_spacing_nm = 1e-1 * noa::mean(stack_loader.file_spacing());
        auto rescaled_metadata = metadata;
        rescaled_metadata.rescale_shifts(stack_loader.file_spacing(), stack_loader.stack_spacing());

        const auto input_images = stack_loader.read_stack(metadata);
        const auto n_images = input_images.shape()[0];
        const auto image_shape = input_images.shape().filter(2, 3);

        // Compute the volume depth.
        // 1. The backward projection can only reconstruct within a sphere of image_min_size diameter.
        //    While the specimen is likely much thinner than this, this is our ultimate limit.
        // 2. The actual limit is 500 nm (technically the algorithm can go above this), but we reconstruct
        //    at least twice as much to include the background from the backward-projection so that it can
        //    be detected more easily (see below). This is also necessary in case the specimen is offset in Z.
        const auto image_min_size = static_cast<f64>(noa::min(image_shape));
        const auto maximum_specimen_thickness = std::min(500. / stack_spacing_nm, image_min_size);
        const auto volume_depth = static_cast<i64>(std::round(maximum_specimen_thickness * 3));

        const auto volume_shape = Shape{volume_depth, image_shape[0], image_shape[1]};
        const auto image_center = (image_shape.vec / 2).as<f64>();
        const auto volume_center = (volume_shape.vec / 2).as<f64>();
        const auto options = ArrayOption{.device = input_images.device(), .allocator = Allocator::MANAGED};

        // TODO exclude edges? Like 10% crop.

        // Compute the projection matrices.
        auto matrices = Array<Mat<f32, 2, 4>>(n_images, options);
        for (auto&& [slice, matrix]: noa::zip(rescaled_metadata, matrices.span_1d())) {
            auto angles = noa::deg2rad(slice.angles);
            matrix = ( // (image->volume).inverse()
                ng::translate(volume_center) * //  + Vec{0., 0., 40.}
                ng::rotate_z<true>(+angles[0]) *
                ng::rotate_x<true>(-angles[2]) *
                ng::rotate_y<true>(-angles[1]) *
                ng::rotate_z<true>(-angles[0]) *
                ng::translate(-(image_center + slice.shifts).push_front(0))
            ).inverse().filter_rows(1, 2).as<f32>(); // (y, x)
        }

        Logger::trace("Computing the variance each z-slice in the tomogram");
        auto variances = noa::Array<f64>(volume_depth, options);
        auto tomogram = noa::Array<f32>(volume_shape.push_front(1), options);
        noa::reduce_axes_iwise( // (d,h,w) -> (d)
            volume_shape.as<i32>(), input_images.device(), noa::wrap(f32{0}, f32{0}), variances.flat(1),
            TomogramVariance{
                .images = TomogramVariance::interpolator_t(input_images.span().filter(0, 2, 3).as_contiguous(), image_shape),
                .projection_matrices = matrices.span_1d(),
                .n_elements_per_image = static_cast<f64>(image_shape.n_elements()),
                // .tomogram = tomogram.span().filter(1, 2, 3).as_contiguous(), // FIXME
            });
        variances = variances.reinterpret_as_cpu();
        noa::normalize(variances, variances, {.mode = noa::Norm::MIN_MAX});
        save_plot_xy({}, variances.eval(), parameters.output_directory / "thickness_profile.txt", {
            .title = "Variance of each z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "variance",
            .label = "variance",
        });

        // noa::write(tomogram, parameters.output_directory / "tomogram.mrc"); // FIXME

        // Compute the baseline. As we get closer to the sample, the variance should progressively increase.
        auto baseline = noa::like<f64>(variances);
        constexpr auto SMOOTHING = GaussianSlider{
            .peak_coordinate = 0.5,
            .peak_value = 70'000,
            .base_width = 0.25,
            .base_value = 10'000,
        };
        asymmetric_least_squares_smoothing(variances.span_1d().as_const(), baseline.span_1d(), {
            .smoothing = SMOOTHING, .asymmetric_penalty = 0.0001, .relaxation = 0.8
        });
        save_plot_xy({}, baseline, parameters.output_directory / "thickness_profile.txt", {.label = "baseline"});

        // Subtract the baseline.
        for (auto&& [v, b]: noa::zip(variances.span_1d(), baseline.span_1d()))
            v -= b;
        noa::normalize(variances, variances, {.mode = noa::Norm::MIN_MAX});
        save_plot_xy({}, variances, parameters.output_directory / "thickness_profile_bs.txt", {
            .title = "Baseline-subtracted variance of each z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "variance - baseline",
        });

        // Find the threshold between background noise and signal.
        const auto threshold = [&] {
            f64 median = noa::median(variances);
            f64 sum{}, sum_squares{};
            i64 count{};
            for (const auto& e: variances.span_1d()) {
                if (e < median) {
                    sum += e;
                    sum_squares += e * e;
                    ++count;
                }
            }
            const f64 background_mean = sum / static_cast<f64>(count);
            const f64 background_variance = sum_squares / static_cast<f64>(count) - (background_mean * background_mean);
            const f64 background_stddev = std::sqrt(background_variance);

            f64 signal_threshold = std::min(0.5, background_mean + 6 * background_stddev);
            Logger::trace("signal_threshold={:.4f} (bg_mean={:.4f}, bg_stddev={:.4f}, signal_scale=6.)",
                          signal_threshold, background_mean, background_stddev);

            // Values are within [0,1], so if we reconstructed a large enough z-section and if the baseline subtraction
            // worked well, the background mean and variance should be close to zero. If not, we may want to add a
            // recovery loop to increase the smoothing of the baseline. However, I have never seen it fail, so for
            // now just give a warning.
            if (background_mean > 0.1 and background_stddev > 0.1) {
                Logger::warn(
                    "Thickness background estimate is likely wrong. Please check and/or report this issue!\n"
                    "As a temporary solution, specify an estimated thickness (using the generated thickness profile, if possible) "
                    "and rerun the program with the thickness estimate turned off"
                );
            }
            return signal_threshold;
        }();

        // Find the specimen window.
        const auto specimen_window = [&] {
            const i64 smallest_window_size = static_cast<i64>(30 / stack_spacing_nm); // 0.03um
            const i64 maximum_distance_between_windows = static_cast<i64>(50 / stack_spacing_nm);
            const i64 biggest_window_size = static_cast<i64>(550 / stack_spacing_nm); // 0.03um
            const i64 maximum_distance_from_center = static_cast<i64>(100 / stack_spacing_nm); // 0.1um

            // First, collect the regions above the threshold.
            bool is_within_window{};
            auto possible_windows = std::vector<Vec<i64, 2>>{};
            for (i64 i{}, start{}; const auto& e: variances.span_1d()) {
                if (not is_within_window and e >= threshold) {
                    is_within_window = true;
                    start = i;
                } else if (is_within_window and (e < threshold or i == volume_depth - 1)) {
                    is_within_window = false;
                    const auto window_size = i - start;
                    if (window_size >= smallest_window_size)
                        possible_windows.push_back({start, i});
                }
                ++i;
            }
            Logger::trace("possible_windows={}", possible_windows);
            check(not possible_windows.empty(), "No possible windows found. Please report this issue");

            // Then, fuse windows that are close to each other.
            for (size_t i{}; i < possible_windows.size() - 1; ++i) {
                const i64 distance = possible_windows[i + 1][0] - possible_windows[i][1];
                if (distance <= maximum_distance_between_windows) {
                    possible_windows[i + 1][0] = possible_windows[i][0];
                    possible_windows[i][0] = -1;
                }
            }
            std::erase_if(possible_windows, [](const auto& window) { return window[0] == -1; });
            Logger::trace("possible_windows={} (after fuse)", possible_windows);

            // Sanitize based on size and distance from the center.
            const i64 center = variances.ssize() / 2;
            i32 n_excluded_windows{};
            for (auto& window: possible_windows) {
                const i64 window_size = window[1] - window[0];
                const i64 window_edge = window[1] < center ? window[1] : window[0] > center ? window[0] : center;
                const i64 distance_from_center = std::abs(window_edge - center);
                if (window_size > biggest_window_size or distance_from_center > maximum_distance_from_center) {
                    window *= -1;
                    ++n_excluded_windows;
                }
            }
            check(
                n_excluded_windows < std::ssize(possible_windows),
                "All windows are either too big or too far away from the center. "
                "Since we can't really tell what is going on, it is best to stop here"
            );

            // Get the largest window.
            auto best_window = Vec<i64, 2>{};
            for (const auto& window: possible_windows) {
                const auto window_size = window[1] - window[0];
                const auto best_size = best_window[1] - best_window[0];
                if (window_size > best_size)
                    best_window = window;
            }

            Logger::trace("best_window={}", best_window);
            return best_window;
        }();

        // Center on the specimen window.
        // TODO For the CTF correction, it may be better to center on the COM.
        const i64 specimen_window_size = specimen_window[1] - specimen_window[0];
        const f64 specimen_window_size_nm = static_cast<f64>(specimen_window_size) * stack_spacing_nm;
        const i64 specimen_window_center = specimen_window[0] + specimen_window_size / 2;
        const i64 specimen_offset_from_center = variances.ssize() / 2 - specimen_window_center; // FIXME
        const f64 specimen_offset_from_center_nm = static_cast<f64>(specimen_offset_from_center) * stack_spacing_nm;
        Logger::info(
            "specimen_window_size={} ({:.2f}nm)\n"
            "specimen_offset_from_center={} ({:.2f}nm)",
            specimen_window_size, specimen_window_size_nm,
            specimen_offset_from_center, specimen_offset_from_center_nm
        );

        // Adjust the shifts to move the specimen to the tomogram center.
        const f64 z_offset = specimen_offset_from_center_nm / file_spacing_nm;
        metadata.add_volume_shift({-z_offset, 0., 0.});

        return specimen_window_size_nm;
    }
}
