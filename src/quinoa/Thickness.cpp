#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>

#include "quinoa/Thickness.hpp"

#include "CommonFOV.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/Stack.hpp"

namespace {
    using namespace qn;

    struct BackwardProjection {
    public:
        using input_span_t = SpanContiguous<const f32, 3, i32>;
        using interpolator_t = nx::Interpolator<2, nx::Interp::LINEAR, noa::Border::ZERO, input_span_t>;

    public:
        interpolator_t images{};
        SpanContiguous<const Mat<f32, 2, 4>, 1, i32> projection_matrices{};
        SpanContiguous<f32, 3, i32> tomogram{};

    public:
        NOA_HD void operator()(const Vec<i32, 3>& indices) const {
            auto volume_coordinates = indices.as<f32>().push_back(1);
            f32 value{};
            for (i64 i{}; i < projection_matrices.ssize(); ++i) {
                const auto image_coordinates = projection_matrices[i] * volume_coordinates;
                value += images.interpolate_at(image_coordinates, i);
            }
            tomogram(indices) = value;
        }
    };

    auto compute_mad_profile(
        const View<f32>& images,
        const Metadata::Stack& metadata,
        f64 spacing_nm,
        const Path& output_directory
    ) -> Array<f64> {
        const auto n_images = images.shape()[0];
        const auto image_shape = images.shape().filter(2, 3);

        // Compute the volume depth.
        // 1. The backward projection can only reconstruct within a sphere of image_min_size diameter.
        //    While the specimen is likely much thinner than this, this is our theoretical thickness limit.
        // 2. The actual limit is 500 nm (technically the algorithm can go above this), but we reconstruct
        //    at least twice as much to include the background from the backward-projection so that it can
        //    be detected more easily (see baseline fitting below). This is also necessary in case the
        //    specimen is offset in Z.
        const auto image_min_size = static_cast<f64>(noa::min(image_shape));
        const auto maximum_specimen_thickness = std::min(500. / spacing_nm, image_min_size);
        const auto volume_depth = static_cast<isize>(std::round(maximum_specimen_thickness * 3));

        const auto volume_shape = Shape{volume_depth, image_shape[0], image_shape[1]};
        const auto image_center = (image_shape.vec / 2).as<f64>();
        const auto volume_center = (volume_shape.vec / 2).as<f64>();
        const auto options = ArrayOption{.device = images.device(), .allocator = Allocator::MANAGED};

        // Compute the projection matrices.
        auto matrices = Array<Mat<f32, 2, 4>>(n_images);
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
        if (options.device.is_gpu())
            matrices = std::move(matrices).to({options.device, Allocator::ASYNC});

        Logger::trace("Computing the low-resolution tomogram");

        // Compute the entire tomogram.
        // This should be low-resolution, so memory shouldn't be an issue. Note that if we were to compute the
        // per-slice variance, we wouldn't need to store the tomogram at all and could compute the variance
        // on-the-fly. For the MAD, this is not possible.
        auto tomogram = noa::Array<f32>(volume_shape.push_front(1), options);

        // TODO smooth 2d ellipse mask?
        tomogram.eval();
        using interp_t = BackwardProjection::interpolator_t;
        noa::iwise(volume_shape.as<i32>(), images.device(), BackwardProjection{
            .images = interp_t(images.span_contiguous<const f32, 3, i32>(), image_shape.as<i32>()),
            .projection_matrices = matrices.span_contiguous<const Mat<f32, 2, 4>, 1, i32>(),
            .tomogram = tomogram.span_contiguous<f32, 3, i32>(),
        });
        tomogram.eval();
        // noa::write_image(tomogram, output_directory / "tomogram.mrc", {.dtype = "f16"}); // FIXME

        // Compute the MAD.
        auto profile = Array<f64>(volume_depth);
        tomogram = tomogram.to_cpu(); // TODO eval
        // FIXME noa batched histogram
        {
            // TODO TMP
            auto buffer = Array<f32>(image_shape.push_front<2>(1));
            for (auto z: noa::irange(volume_depth)) {
                auto img = tomogram.subregion(0, z);
                f32 m = noa::median(img, {.overwrite = false});
                noa::ewise(img, buffer, [m]NOA_HD(f32 i, f32& o) { o = noa::abs(i - m); });
                m = noa::median(buffer, {.overwrite = true});
                profile.span_1d()[z] = static_cast<f64>(m * m);
            }
        }

        // Small Gaussian blur and normalize between [0, 1].
        auto kernel = ns::window_gaussian<f64>(11, 2, {.normalize = true});
        auto profile_smooth = like(profile);
        ns::convolve(profile, profile_smooth, kernel, {.border = noa::Border::REFLECT});
        noa::normalize(profile_smooth, profile_smooth, {.mode = noa::Norm::MIN_MAX});

        save_plot_xy({}, profile_smooth, output_directory / "thickness_profile.txt", {
            .title = "Median of Deviations (MADs) per z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "MAD",
            .label = "MAD",
        });

        return profile_smooth;
    }

    auto subtract_background(
        const Array<f64>& profile,
        const Path& output_directory
    ) {
        // Compute the baseline.
        constexpr auto SMOOTHING = GaussianSlider{
            .peak_coordinate = 0.5,
            .peak_value = 70'000,
            .base_width = 0.25,
            .base_value = 20'000,
        };
        constexpr auto ASYMMETRY = GaussianSlider{
            .peak_coordinate = 0.,
            .peak_value = 0.0001,
            .base_width = 1.,
            .base_value = 0.0001,
        };
        auto profile_bs = like(profile);
        // asymmetric_least_squares_smoothing(profile.span_1d().as_const(), profile_bs.span_1d(), {
        //     .smoothing = SMOOTHING, .asymmetry = ASYMMETRY, .relaxation = 0.8
        // });
        // save_plot_xy({}, profile_bs, output_directory / "thickness_profile.txt", {.label = "baseline"});

        // Subtract the baseline.
        for (auto&& [in, out]: noa::zip(profile.span_1d(), profile_bs.span_1d()))
            out = in - out;

        noa::normalize(profile_bs, profile_bs, {.mode = noa::Norm::MIN_MAX});
        save_plot_xy({}, profile_bs, output_directory / "thickness_profile_bs.txt", {
            .title = "Baseline-subtracted Median of Deviations (MADs) per z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "MAD - baseline",
        });

        return profile_bs;
    }

    auto analyse_profile(const View<const f64>& profile, f64 spacing_nm) {
        // Find the threshold between background noise and signal.
        const auto threshold = [&] {
            const f64 median = noa::median(profile);
            f64 sum{};
            f64 sum_squares{};
            i64 count{};
            for (const auto& e: profile.span_1d()) {
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
            for (i64 i{}, start{}; const auto& e: profile.span_1d()) {
                if (not is_within_window and e >= threshold) {
                    is_within_window = true;
                    start = i;
                } else if (is_within_window and (e < threshold or i == profile.ssize() - 1)) {
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
            const i64 center = profile.ssize() / 2;
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
        const i64 specimen_offset_from_center = profile.ssize() / 2 - specimen_window_center; // FIXME
        const f64 specimen_offset_from_center_nm = static_cast<f64>(specimen_offset_from_center) * spacing_nm;
        Logger::info(
            "specimen_window_size={} ({:.2f}nm)\n"
            "specimen_offset_from_center={} ({:.2f}nm)",
            specimen_window_size, specimen_window_size_nm,
            specimen_offset_from_center, specimen_offset_from_center_nm
        );

        return Pair{specimen_window_size_nm, specimen_offset_from_center_nm};
    }

    auto estimate(
        const View<f32>& images,
        const Metadata::Stack& metadata,
        f64 spacing_nm,
        const Path& output_directory
    ) {
        const auto n_images = images.shape()[0];
        const auto image_shape = images.shape().filter(2, 3);

        auto profile = compute_mad_profile(images, metadata, spacing_nm, output_directory);
        auto profile_bs = subtract_background(profile, output_directory);
        return analyse_profile(profile_bs.view(), spacing_nm);
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

        // Adjust the shifts to move the specimen to the new tomogram center and set sample thickness.
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

        // save_stack(input_images.view(), metadata.spacing, metadata.stack, options.output_directory / "input_stack.mrc");

        // auto input_images = noa::read_image<f32>(options.output_directory / "input_stack.mrc").data;
        // auto stack_spacing_nm = 1.2;
        // metadata.set_spacing(12.);
        // const auto original_spacing = metadata.spacing;
        const auto [specimen_window_size_nm, specimen_offset_from_center_nm] = estimate(
            input_images.view(), metadata.stack, stack_spacing_nm, options.output_directory
        );

        // Adjust the shifts to move the specimen to the new tomogram center and set sample thickness.
        metadata.stack.add_volume_shift({-specimen_offset_from_center_nm * 1e1, 0., 0.});
        metadata.sample.thickness = specimen_window_size_nm;

        return specimen_window_size_nm;
    }
}
