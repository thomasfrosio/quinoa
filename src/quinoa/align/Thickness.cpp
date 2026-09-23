#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>

#include "quinoa/Plot.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/align/Thickness.hpp"
#include "quinoa/postprocessing/FilterStack.hpp"

namespace {
    using namespace qn;

    template<typename T>
    struct BackwardProjection {
    public:
        using input_span_t = SpanContiguous<const f32, 3, i32>;
        using interpolator_t = nx::Interpolator<2, nx::Interp::NEAREST, noa::Border::ZERO, input_span_t>;

    public:
        interpolator_t images{};
        SpanContiguous<const Mat<f32, 2, 4>, 1, i32> projection_matrices{};
        SpanContiguous<T, 3, i32> tomogram{};

    public:
        NOA_HD void operator()(const Vec<i32, 3>& indices) const {
            const auto volume_coordinates = indices.as<f32>().push_back(1);
            f32 value{};
            for (i64 i{}; i < projection_matrices.ssize(); ++i) {
                const auto image_coordinates = projection_matrices[i] * volume_coordinates;
                value += images.interpolate_at(image_coordinates, i);
            }
            tomogram(indices) = static_cast<T>(value);
        }
    };

    template<typename T>
    struct Histogram {
        SpanContiguous<const T, 2, i32> inputs; // (n,w)
        SpanContiguous<i32, 2, i32> histograms; // (n,b)

        static constexpr void init(nt::compute_handle auto& handle) {
            // Zero-initialize the per-block histogram if it exists.
            const auto& block = handle.block();
            block.template zeroed_scratch<i32>();
            block.synchronize();
        }

        constexpr void operator()(nt::compute_handle auto& handle, i32 b, i32 i) const {
            // Compute the bin of the current value.
            const auto n_bins = histograms.shape()[1]; // TODO constexpr
            const auto value_scaled = static_cast<f32>(inputs(b, i)) * static_cast<f32>(n_bins - 1);
            const auto bin = static_cast<i32>(noa::round(value_scaled));

            // Increment the bin count.
            // If the block has its own histogram, increment it
            // instead of incrementing the global histogram.
            const auto& grid = handle.grid();
            const auto& block = handle.block();
            if (block.has_scratch()) {
                auto scratch = block.template scratch<i32>();
                grid.atomic_add(1, scratch, bin);
            } else {
                grid.atomic_add(1, histograms[b], bin);
            }
        }

        constexpr void deinit(nt::compute_handle auto& handle, i32 b) const {
            const auto& block = handle.block();
            const auto& thread = handle.thread();
            if (not block.has_scratch())
                return;

            // If the block has its own histogram, add it to the global histogram.
            block.synchronize();
            const auto& grid = handle.grid();
            auto scratch = block.template scratch<i32>();
            for (i32 i = thread.lid(); i < scratch.n_elements(); i += block.size())
                grid.atomic_add(scratch[i], histograms, b, i);
        }
    };

    template<typename T>
    auto compute_mad_profile(
        const View<const f32>& images,
        const Metadata::Stack& metadata,
        const View<T>& tomogram,
        const Path& output_directory
    ) -> Array<f64> {
        auto _ = Logger::trace_scope_time<false>("Tomogram thickness profile");

        const auto n_images = images.shape()[0];
        const auto image_shape = images.shape().filter(2, 3);
        const isize volume_depth = tomogram.shape()[1];
        const auto device = tomogram.device();

        const auto volume_shape = Shape3{volume_depth, image_shape[0], image_shape[1]};
        const auto image_center = (image_shape.vec / 2).as<f64>();
        const auto volume_center = (volume_shape.vec / 2).as<f64>();

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
        if (device.is_gpu())
            matrices = std::move(matrices).to({.device = device, .allocator = Allocator::ASYNC});

        // Compute the entire tomogram.
        // Despite being low-resolution, the tomogram can be multiple GB. Note that if we were to compute the
        // per-slice variance, we wouldn't need to store the tomogram at all and could compute the variance
        // on-the-fly. For the MAD, it's not possible, but it seems worth it.
        using interp_t = BackwardProjection<T>::interpolator_t;
        noa::iwise(volume_shape.as<i32>(), device, BackwardProjection{
            .images = interp_t(images.span_contiguous<const f32, 3, i32>(), image_shape.as<i32>()),
            .projection_matrices = matrices.span_contiguous<const Mat<f32, 2, 4>, 1, i32>(),
            .tomogram = tomogram.template span_contiguous<T, 3, i32>(),
        });

        // Compute the MAD.
        constexpr isize N_BINS = 1024;
        const auto tomogram_1d = tomogram.reshape({volume_depth, 1, 1, -1});
        const auto shape_dw = Shape2{volume_depth, tomogram_1d.shape()[3]}.as<i32>();
        const auto options_unified = ArrayOption{.device = device, .allocator = Allocator::UNIFIED};
        const auto histograms = Array<i32>({volume_depth, 1, 1, N_BINS}, options_unified);
        auto medians = Array<f32>(volume_depth, options_unified);
        auto get_medians_from_histograms = [&] {
            auto histogram_2d = histograms.reinterpret_as_cpu().eval().span_contiguous<i32, 2>();
            auto medians_1d = medians.span_1d();
            const auto half = shape_dw[1] / 2;
            const auto bin_step = 1.f / static_cast<f32>(N_BINS - 1); // linspace(0, 1, N_BINS, endpoint=true)
            for (isize i{}; i < shape_dw[0]; ++i) {
                i32 count{};
                i32 previous_count{};
                for (isize j{}; j < N_BINS; ++j) {
                    if (count >= half) {
                        const auto distance = count - previous_count;
                        const auto offset = half - previous_count;
                        const auto ratio = static_cast<f32>(offset) / static_cast<f32>(distance);
                        const auto previous_bin = bin_step * static_cast<f32>(std::max(j - 1, isize{}));
                        const auto current_bin = bin_step * static_cast<f32>(j);
                        const auto median = std::lerp(previous_bin, current_bin, ratio);
                        medians_1d[i] = median;
                        break;
                    }
                    previous_count = count;
                    count += histogram_2d(i, j);
                }
            }
        };
        constexpr auto OPTIONS = noa::ReduceIwiseOptions{
            .generate_cpu = false,
            .gpu_block_shape = {1, 512}, // 1d block
            .gpu_optimize_block_shape = false, // enforce the block shape
            .gpu_number_of_indices_per_threads = {1, 16}, // increase the value of the per-block histogram by working on it more
            .gpu_scratch_size = N_BINS * sizeof(i32), // per block histogram
        };

        // Compute the median for each Z slice.
        noa::normalize(tomogram, tomogram, {.mode = noa::Norm::MIN_MAX});
        noa::fill(histograms, 0);
        noa::reduce_axes_iwise<OPTIONS>(shape_dw, device, {}, ReduceAxes{.width = true}, Histogram{
            .inputs = tomogram_1d.template span_contiguous<const T, 2, i32>(),
            .histograms = histograms.span_contiguous<i32, 2, i32>(),
        });
        get_medians_from_histograms();

        // Compute the absolute deviations from the medians.
        noa::ewise(medians.flat(0), tomogram_1d, []NOA_HD(f32 median, T& value) {
            value = static_cast<T>(noa::abs(static_cast<f32>(value) - median));
        });

        // Compute the MAD for each Z slice.
        noa::normalize(tomogram, tomogram, {.mode = noa::Norm::MIN_MAX});
        noa::fill(histograms, 0);
        noa::reduce_axes_iwise<OPTIONS>(shape_dw, device, {}, ReduceAxes{.width = true}, Histogram{
            .inputs = tomogram_1d.template span_contiguous<const T, 2, i32>(),
            .histograms = histograms.span_contiguous<i32, 2, i32>(),
        });
        get_medians_from_histograms();
        medians = medians.reinterpret_as_cpu();
        for (f32& median: medians.span_1d())
            median *= median;
        noa::normalize(medians, medians, {.mode = noa::Norm::MIN_MAX});
        save_plot_xy({}, medians.span_1d(), output_directory / "thickness_profile_rough.txt", {
            .title = "Median of Deviations (MADs) per z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "MAD",
            .label = "MAD",
        });

        // Small Gaussian blur and normalize between [0, 1].
        auto kernel = ns::window_gaussian<f64>(11, 2, {.normalize = true});
        auto medians_smooth = noa::Array<f64>(medians.shape());
        ns::convolve(medians, medians_smooth, kernel, {.border = noa::Border::REFLECT});
        noa::normalize(medians_smooth, medians_smooth, {.mode = noa::Norm::MIN_MAX});

        save_plot_xy({}, medians_smooth, output_directory / "thickness_profile_smooth.txt", {
            .title = "Median of Deviations (MADs) per z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "MAD",
            .label = "MAD",
        });

        return medians_smooth;
    }

    auto subtract_background(
        const View<const f64>& profile,
        const Path& output_directory
    ) {
        // Compute the baseline.
        auto x = noa::linspace<f64>(profile.n_elements(), noa::Linspace{0., 1.});
        auto profile_bs = noa::like(profile);
        asymmetric_least_squares_smoothing(x.span_1d(), profile.span_1d(), profile_bs.span_1d(), {
            .smoothing = {
                // TODO Use the current thickness estimate to increase the smoothness within the specimen window,
                // and decrease it significantly outside of it to follow the background closely.
                .peak_coordinate = 0.5, // specimen should be roughly at the center
                .peak_value = 1e-6,
                .base_width = 0.15,
                .base_value = 1e-7,
            },
            .asymmetry = GaussianSlider::from_constant(0.05),
            .max_iter = 50,
            .relaxation = 0.9,
        });

        save_plot_xy({}, profile_bs, output_directory / "thickness_profile_smooth.txt", {.label = "baseline"});

        // Subtract the baseline.
        for (auto&& [in, out]: noa::zip(profile.span_1d(), profile_bs.span_1d()))
            out = in - out;

        noa::normalize(profile_bs, profile_bs, {.mode = noa::Norm::MIN_MAX});
        save_plot_xy({}, profile_bs, output_directory / "thickness_profile.txt", {
            .title = "Baseline-subtracted Median of Deviations (MADs) per z-slice of the tomogram",
            .x_name = "depth (in pixels)",
            .y_name = "MAD",
        });

        return profile_bs;
    }

    auto analyse_profile(const View<const f64>& profile, f64 spacing_nm) {
        const auto profile_1d = profile.span_1d();
        const isize center = profile_1d.ssize() / 2;

        // Find the threshold between background noise and signal.
        const auto threshold = [&] {
            const f64 median = noa::median(profile);
            f64 sum{};
            f64 sum_squares{};
            i64 count{};
            for (const auto& e: profile_1d) {
                if (e < median) {
                    sum += e;
                    sum_squares += e * e;
                    ++count;
                }
            }
            const f64 background_mean = sum / static_cast<f64>(count);
            const f64 background_variance = sum_squares / static_cast<f64>(count) - (background_mean * background_mean);
            const f64 background_stddev = std::sqrt(background_variance);

            f64 signal_threshold = std::min(0.5, background_mean + 5 * background_stddev);
            Logger::trace("signal_threshold={:.4f} (bg_mean={:.4f}, bg_stddev={:.4f}, signal_scale=5.)",
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
            const i64 smallest_window_size = static_cast<i64>(30 / spacing_nm);
            const i64 maximum_distance_between_windows = static_cast<i64>(50 / spacing_nm);
            const i64 biggest_window_size = static_cast<i64>(500 / spacing_nm);
            const i64 maximum_distance_from_center = static_cast<i64>(100 / spacing_nm);

            // First, collect the regions above the threshold.
            bool is_within_window{};
            auto possible_windows = std::vector<Vec<i64, 2>>{};
            for (i64 i{}, start{}; const auto& e: profile_1d) {
                if (not is_within_window and e >= threshold) {
                    is_within_window = true;
                    start = i;
                } else if (is_within_window and (e < threshold or i == profile_1d.ssize() - 1)) {
                    is_within_window = false;
                    const auto window_size = i - start;
                    if (window_size >= smallest_window_size)
                        possible_windows.push_back({start, i});
                }
                ++i;
            }
            Logger::trace(
                "possible_windows={} (range={}, center={})",
                possible_windows, Vec<isize, 2>{0, profile_1d.ssize()}, center
            );
            check(not possible_windows.empty(), "No possible windows found. Please report this issue");

            // Then, fuse windows that are close to each other.
            auto window_peak = [&](const Vec<i64, 2>& window) {
                f64 v{-1.};
                for (i64 i{window[0]}; i <= window[1]; ++i)
                    v = std::max(v, profile_1d[i]);
                return v;
            };
            for (size_t i{}; i < possible_windows.size() - 1; ++i) {
                const i64 distance = possible_windows[i + 1][0] - possible_windows[i][1];
                if (distance <= maximum_distance_between_windows) {
                    // The two windows are close enough to be fused,
                    // but only fuse if the peaks are within a thrid of each other.
                    const auto peak_0 = window_peak(possible_windows[i]);
                    const auto peak_1 = window_peak(possible_windows[i + 1]);
                    if (peak_0 >= peak_1 / 3. and peak_1 >= peak_0 / 3.) {
                        possible_windows[i + 1][0] = possible_windows[i][0];
                        possible_windows[i][0] = -1; // mark to be removed
                    }
                }
            }
            std::erase_if(possible_windows, [](const auto& window) { return window[0] == -1; });
            Logger::trace("possible_windows={} (after fuse)", possible_windows);

            // Remove windows that are too small or too far away rom the center.
            auto window_distance = [&center](const Vec<i64, 2>& window) {
                const i64 window_edge =
                    window[1] < center ? window[1] :
                    window[0] > center ? window[0] :
                    center;
                return std::abs(window_edge - center);
            };
            for (auto& window: possible_windows) {
                const i64 window_size = window[1] - window[0];
                const i64 distance_from_center = window_distance(window);
                if (window_size > biggest_window_size or distance_from_center > maximum_distance_from_center)
                    window[0] = -1; // mark to be removed
            }
            std::erase_if(possible_windows, [](const auto& window) { return window[0] == -1; });
            Logger::trace("possible_windows={} (after sanitize)", possible_windows);
            check(
                not possible_windows.empty(),
                "No possible windows found. Please report this issue. All windows are either too small or too far away from the center."
            );

            // Get the most centered window.
            auto best_window = Vec<i64, 2>{};
            for (const auto& window: possible_windows)
                if (window_distance(window) < window_distance(best_window))
                    best_window = window;
            Logger::trace("best_window={}", best_window);

            // The window is overestimated at this point because of the backprojection artefacts that don't produce
            // a sharp specimen edge. Since this thickness estimate is intended for projection matching, prefer to keep
            // it tight to the specimen and do so by cutting the edges of the window at 10% of the window peak.
            const auto edge_threshold = (window_peak(best_window) - threshold) * 0.10;
            for (i64 i{best_window[0]}; i <= best_window[1]; ++i) {
                if (profile_1d[i] - threshold > edge_threshold) {
                    best_window[0] = i;
                    break;
                }
            }
            for (i64 i{best_window[1]}; i >= best_window[0]; --i) {
                if (profile_1d[i] - threshold > edge_threshold) {
                    best_window[1] = i;
                    break;
                }
            }
            Logger::trace("best_window={} (after cutting edges)", best_window);
            return best_window;
        }();

        // Center on the specimen window.
        // TODO For the CTF correction, it may be better to center on the COM to better match the defocus estimates.
        const i64 specimen_window_size = specimen_window[1] - specimen_window[0];
        const f64 specimen_window_size_nm = static_cast<f64>(specimen_window_size) * spacing_nm;
        const i64 specimen_window_center = specimen_window[0] + specimen_window_size / 2;
        const i64 specimen_offset_from_center = center - specimen_window_center;
        const f64 specimen_offset_from_center_nm = static_cast<f64>(specimen_offset_from_center) * spacing_nm;
        Logger::info(
            "specimen_window_size={}pix ({:.2f}nm)\n"
            "specimen_offset_from_center={}pix ({:.2f}nm)\n",
            specimen_window_size, specimen_window_size_nm,
            specimen_offset_from_center, specimen_offset_from_center_nm
        );

        return Pair{specimen_window_size_nm, specimen_offset_from_center_nm};
    }
}

namespace qn {
    SpecimenThickness::SpecimenThickness(
        const Path& stack_filename,
        Metadata& metadata, // updated: .shifts
        Device device
    ) {
        const auto b0 = noa::Allocator::bytes_currently_allocated(device);

        // Load the stack as sorted in the metadata.
        metadata.stack.sort("index");
        auto result = load_stack(stack_filename, metadata.stack, {
            .compute_device = device,
            .allocator = Allocator::ASYNC,

            // Fourier cropping:
            .precise_cutoff = true, // ensure isotropic spacing
            .rescale_target_resolution = 30,
            .rescale_min_size = 500,
            .rescale_max_size = 1000,

            // Signal processing after cropping:
            .bandpass{
                .highpass_cutoff = 0.02,
                .highpass_width = 0.02,
                .lowpass_cutoff = 1,
                .lowpass_width = 0,
            },
            .bandpass_mirror_padding_factor = 0.,
            .fake_sirt_iterations = 20,
            .exposure_filter_voltage = 0,

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        m_tilt_series = std::move(result).stack;
        m_spacing_nm = mean(result.stack_spacing) * 0.1;

        // Compute the volume depth.
        // 1. The backward projection can only reconstruct within a sphere of image_min_size diameter.
        //    While the specimen is likely much thinner than this, this is our theoretical thickness limit.
        // 2. The actual limit is 500 nm (technically the algorithm can go above this), but we reconstruct
        //    at least twice as much to include the background from the backward-projection so that it can
        //    be detected more easily (see baseline fitting below). This is also necessary in case the
        //    specimen is offset in Z.
        const auto image_shape = m_tilt_series.shape().filter(2, 3);
        const auto image_min_size = static_cast<f64>(noa::min(image_shape));
        const auto maximum_specimen_thickness = std::min(500. / m_spacing_nm, image_min_size);
        m_volume_depth = static_cast<isize>(std::round(maximum_specimen_thickness * 3));

        const auto b1 = noa::Allocator::bytes_currently_allocated(device);
        Logger::trace(
            "Specimen thickness:\n"
            "   tomogram_shape={}\n"
            "   allocated={:.3f}GB (device={}, {})\n",
            image_shape.push_front(m_volume_depth),
            static_cast<f64>(b1 - b0) / 1e9, device, m_tilt_series.allocator()
        );
    }

    auto SpecimenThickness::shared_buffer_bytes() const -> isize {
        return m_volume_depth * m_tilt_series.shape()[2] * m_tilt_series.shape()[3] * static_cast<isize>(sizeof(value_type));
    }

    void SpecimenThickness::set_shared_buffer(const View<std::byte>& shared_buffer) {
        // Don't use managed memory, allocate and keep the tomogram close to the device.
        // We use f16 to reduce the memory requirement and since this is low-res it shouldn't be an issue.
        const auto shape = Shape4{1, m_volume_depth, m_tilt_series.shape()[2], m_tilt_series.shape()[3]};
        m_tomogram = shared_buffer
            .reinterpret_as<value_type>()
            .subregion(Ellipsis{}, Slice{0, shape.n_elements()})
            .reshape(shape);
    }

    auto SpecimenThickness::estimate(
        Metadata& metadata, // updated: stack.shifts, sample.thickness
        const Path& output_directory
    ) const -> f64 {
        // Rescale to our tilt-series and sort the images
        const auto original_spacing = metadata.spacing;
        metadata.set_spacing(m_spacing_nm * 10);
        metadata.stack.sort("index");

        auto profile = compute_mad_profile(m_tilt_series.view(), metadata.stack, m_tomogram.view(), output_directory);
        auto profile_bs = subtract_background(profile.view(), output_directory);
        auto [thickness_nm, offset_from_center_nm] = analyse_profile(profile_bs.view(), m_spacing_nm);

        // Adjust the shifts to move the tomogram center onto the specimen.
        metadata.stack.add_volume_shift({-offset_from_center_nm * 10, 0., 0.});
        metadata.sample.thickness = thickness_nm;

        metadata.set_spacing(original_spacing);
        return thickness_nm;
    }
}
