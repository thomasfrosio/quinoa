#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>

#include "quinoa/Thickness.hpp"

#include "CommonFOV.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/Stack.hpp"

namespace {
    using namespace qn;

    /// Index-wise reduction operator sampling the backprojected tomogram.
    struct TomogramVariance {
    public:
        static constexpr f32 CYLINDER_FRACTION = 0.49f;
        static constexpr auto INTERP = nx::Interp::LINEAR;
        static constexpr auto BORDER = noa::Border::ZERO;
        using input_span_t = SpanContiguous<const f32, 3>;
        using interpolator_t = nx::Interpolator<2, INTERP, BORDER, input_span_t>;
        using matrices_span_t = SpanContiguous<const Mat<f32, 2, 4>>;

    public:
        interpolator_t images{}; // (n,h,w)
        matrices_span_t projection_matrices{}; // (n)
        f64 n_elements_per_image{};
        SpanContiguous<f32, 3> tomo{}; // FIXME

        Vec<f32, 2> offset;
        // Vec<f32, 2> center;
        // Vec<f32, 2> norm;

    public:
        [[nodiscard]] constexpr auto backproject(const Vec<i32, 3>& indices) const -> f32 {
            auto volume_coordinates = indices.as<f32>().push_back(1);
            volume_coordinates[1] += offset[0];
            volume_coordinates[2] += offset[1];
            f32 value{};
            for (i64 i{}; i < projection_matrices.ssize(); ++i) {
                const auto image_coordinates = projection_matrices[i] * volume_coordinates;
                value += images.interpolate_at(image_coordinates, i);
            }
            return value;
        }

        // TODO On top of masking, crop the compute grid.
        // [[nodiscard]] constexpr auto is_within_cylinder_mask(const Vec<i32, 3>& indices) const -> bool {
        //     const auto centered_indices_2d = (indices.pop_front().as<f32>() - center) * norm;
        //     const auto distance_from_center = dot(centered_indices_2d, centered_indices_2d);
        //     return distance_from_center <= (CYLINDER_FRACTION * CYLINDER_FRACTION);
        // }

        constexpr void init(const Vec<i32, 3>& indices, f32& sum, f32& sum_sqd) const {
            f32 value{};
            // if (is_within_cylinder_mask(indices))
                value = backproject(indices);

            tomo(indices) = value; // FIXME

            sum += value;
            sum_sqd += value * value;
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

    void subtract_background(
        const View<const f64>& input,
        const View<f64>& output,
        const Path& output_directory
    ) {
        check(not noa::are_overlapped(input, output));

        // Compute the baseline.
        constexpr auto SMOOTHING = GaussianSlider{
            .peak_coordinate = 0.5,
            .peak_value = 70'000,
            .base_width = 0.25,
            .base_value = 20'000,
        };
        asymmetric_least_squares_smoothing(input.span_1d(), output.span_1d(), {
            .smoothing = SMOOTHING, .asymmetric_penalty = 0.0001, .relaxation = 0.8
        });
        save_plot_xy({}, output, output_directory / "thickness_profile.txt", {.label = "baseline"});

        // Subtract the baseline.
        for (auto&& [in, out]: noa::zip(input.span_1d(), output.span_1d()))
            out = in - out;
        noa::normalize(output, output, {.mode = noa::Norm::MIN_MAX});
        save_plot_xy({}, output, output_directory / "thickness_profile_bs.txt", {
            .title = "Baseline-subtracted variance of each z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "variance - baseline",
        });
    }

    auto estimate(
        const View<f32>& input_images,
        const Metadata::Stack& metadata,
        f64 spacing_nm,
        const Path& output_directory
    ) {
        const auto n_images = input_images.shape()[0];
        const auto image_shape = input_images.shape().filter(2, 3);

        // Compute the volume depth.
        // 1. The backward projection can only reconstruct within a sphere of image_min_size diameter.
        //    While the specimen is likely much thinner than this, this is our theoretical thickness limit.
        // 2. The actual limit is 500 nm (technically the algorithm can go above this), but we reconstruct
        //    at least twice as much to include the background from the backward-projection so that it can
        //    be detected more easily (see baseline fitting below). This is also necessary in case the
        //    specimen is offset in Z.
        const auto image_min_size = static_cast<f64>(noa::min(image_shape));
        const auto maximum_specimen_thickness = std::min(500. / spacing_nm, image_min_size);
        const auto volume_depth = static_cast<i64>(std::round(maximum_specimen_thickness * 3));

        const auto volume_shape = Shape{volume_depth, image_shape[0], image_shape[1]};
        const auto image_center = (image_shape.vec / 2).as<f64>();
        const auto volume_center = (volume_shape.vec / 2).as<f64>();
        const auto options = ArrayOption{.device = input_images.device(), .allocator = Allocator::MANAGED};

        // Compute the projection matrices.
        const auto matrices = Array<Mat<f32, 2, 4>>(n_images, options);
        for (auto&& [image, matrix]: noa::zip(metadata, matrices.span_1d())) {
            const auto angles = noa::deg2rad(image.angles);
            matrix = ( // (image->volume).inverse()
                nx::translate(volume_center) *
                nx::rotate_z<true>(+angles[0]) *
                nx::rotate_x<true>(-angles[2]) *
                nx::rotate_y<true>(-angles[1]) *
                nx::rotate_z<true>(-angles[0]) *
                nx::translate(-(image_center + image.shifts).push_front(0))
            ).inverse().filter_rows(1, 2).as<f32>(); // (y, x)
        }

        Logger::trace("Computing the variance each z-slice in the (virtual) tomogram");

        auto variances = noa::Array<f64>(volume_depth, options);
        auto variances_bs = noa::like(variances);
        auto debug_tomogram = noa::Array<f32>(volume_shape.push_front(1), options); // FIXME

        noa::reduce_axes_iwise( // (d,h,w) -> (d)
            volume_shape.as<i32>(), input_images.device(), noa::wrap(f32{0}, f32{0}), variances.flat(1),
                TomogramVariance{
                    .images = TomogramVariance::interpolator_t(input_images.span().filter(0, 2, 3).as_contiguous(), image_shape),
                    .projection_matrices = matrices.span_1d(),
                    .n_elements_per_image = static_cast<f64>(image_shape.n_elements()),
                    .tomo = debug_tomogram.span_contiguous<f32, 3>(),  // FIXME
                });

        auto tmp = noa::like(debug_tomogram);
        auto kernel = ns::window_gaussian<f32>(11, 2, {.normalize = true}).to(tmp.options());
        Logger::trace("kernel={::.3f}", kernel.span_1d());
        ns::median_filter_2d(debug_tomogram, tmp, {.window_size = 11});
        ns::convolve_separable(tmp, debug_tomogram, kernel, kernel, kernel, {}, {.border = noa::Border::ZERO});

        noa::write_image(debug_tomogram, output_directory / "tomogram.mrc", {.dtype = "f16"}); // FIXME
        noa::write_image(variances, output_directory / "variances.mrc"); // FIXME
        panic();

        // variances = noa::read_image<f64>(output_directory / "variances.mrc").data;

        variances = variances.reinterpret_as_cpu();
        noa::normalize(variances, variances, {.mode = noa::Norm::MIN_MAX});
        save_plot_xy({}, variances.eval(), output_directory / "thickness_profile.txt", {
            .title = "Variance of each z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "variance",
            .label = "variance",
        });

        // panic();
        subtract_background(variances.view(), variances_bs.view(), output_directory);

        // Find the threshold between background noise and signal.
        const auto threshold = [&] {
            const f64 median = noa::median(variances);
            f64 sum{};
            f64 sum_squares{};
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
            // TODO If the background isn't close to zero, it could mean the baseline was too rigid and didn't follow
            //      the data enough. So redo with a stronger smoothing. detect for threshold at 0.5 even.
            if (background_mean > 0.1 and background_stddev > 0.1) {
                Logger::warn(
                    "Thickness background estimate is likely wrong. Please check and/or report this issue!\n"
                    "As a temporary solution, specify an estimated thickness (using the generated thickness profile, if possible) "
                    "and rerun the program with the thickness estimate turned off"
                );
            }
            return signal_threshold;
        }();
        panic();

        // Find the specimen window.
        const auto specimen_window = [&] {
            const i64 smallest_window_size = static_cast<i64>(30 / spacing_nm); // 0.03um
            const i64 maximum_distance_between_windows = static_cast<i64>(50 / spacing_nm);
            const i64 biggest_window_size = static_cast<i64>(550 / spacing_nm); // 0.03um
            const i64 maximum_distance_from_center = static_cast<i64>(100 / spacing_nm); // 0.1um

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
            // TODO
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

            // TODO If sizes are within 25% of each other, select based on highest average variance?
            //      and/or select based on how close from the current center we are! windows far from the center
            //      are likely to be dust

            // Get the largest and most centered window.
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
        const f64 specimen_window_size_nm = static_cast<f64>(specimen_window_size) * spacing_nm;
        const i64 specimen_window_center = specimen_window[0] + specimen_window_size / 2;
        const i64 specimen_offset_from_center = variances.ssize() / 2 - specimen_window_center; // FIXME
        const f64 specimen_offset_from_center_nm = static_cast<f64>(specimen_offset_from_center) * spacing_nm;
        Logger::info(
            "specimen_window_size={} ({:.2f}nm)\n"
            "specimen_offset_from_center={} ({:.2f}nm)",
            specimen_window_size, specimen_window_size_nm,
            specimen_offset_from_center, specimen_offset_from_center_nm
        );

        return Pair{specimen_window_size_nm, specimen_offset_from_center_nm};
    }
}

namespace qn {
    auto estimate_sample_thickness(
        const View<f32>& stack,
        Metadata& metadata,
        const EstimateSampleThicknessOptions& options
    ) -> f64 {
        auto timer = Logger::info_scope_time("Thickness estimation");

        const auto spacing_nm = mean(metadata.spacing) * 1e-1;
        const auto [specimen_window_size_nm, specimen_offset_from_center_nm] = estimate(
            stack, metadata.stack, spacing_nm, options.output_directory
        );

        // Adjust the shifts to move the specimen to the tomogram center.
        metadata.stack.add_volume_shift({-specimen_offset_from_center_nm * 1e1, 0., 0.});

        metadata.sample.thickness = specimen_window_size_nm;
        return specimen_window_size_nm;
    }

    auto estimate_sample_thickness(
        const Path& stack_filename,
        Metadata& metadata,
        const EstimateSampleThicknessFromFileOptions& options
    ) -> f64 {
        auto timer = Logger::info_scope_time("Thickness estimation");

        auto stack_loader = StackLoader(stack_filename, {
            .compute_device = options.device,
            .allocator = options.allocator,
            .precise_cutoff = true, // enforce isotropic spacing
            .rescale_target_resolution = options.resolution,
            .rescale_min_size = 512,
            .rescale_max_size = 1024,
            .bandpass{
                .highpass_cutoff = 0.1,
                .highpass_width = 0.1,
                .lowpass_cutoff = 0.49,
                .lowpass_width = 0.01,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.2,
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });
        //
        const auto input_images = stack_loader.read_stack(metadata.stack);
        const auto stack_spacing_nm = 1e-1 * noa::mean(stack_loader.stack_spacing());
        //
        const auto original_spacing = metadata.spacing;
        metadata.set_spacing(stack_loader.stack_spacing());

        save_stack(input_images.view(), metadata.spacing, metadata.stack, options.output_directory / "input_stack.mrc");

        // auto input_images = noa::read_image<f32>(options.output_directory / "input_stack.mrc").data;
        // auto stack_spacing_nm = 1.2;
        // metadata.set_spacing(12.);
        // const auto original_spacing = metadata.spacing;
        const auto [specimen_window_size_nm, specimen_offset_from_center_nm] = estimate(
            input_images.view(), metadata.stack, stack_spacing_nm, options.output_directory
        );

        // Adjust the shifts to move the specimen to the tomogram center.
        metadata.stack.add_volume_shift({-specimen_offset_from_center_nm * 1e1, 0., 0.});

        metadata.set_spacing(original_spacing);
        return specimen_window_size_nm;
    }

    void ThicknessModulation::sample(
        SpanContiguous<f32> spectrum,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
        for (i64 i{}; i < spectrum.ssize(); ++i) {
            const auto fftfreq = static_cast<f64>(i) * fftfreq_step + fftfreq_range[0];
            spectrum[i] = static_cast<f32>(sample_at(fftfreq));
        }
    }

    void ThicknessModulation::sample(
        const View<f32>& spectrum,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        auto [b, d, h, w] = spectrum.shape();
        check(b == 1 and d == 1 and h == 1);
        sample(spectrum.reinterpret_as_cpu().span_1d(), fftfreq_range);
    }
}
