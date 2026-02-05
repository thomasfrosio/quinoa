#include "quinoa/align/Align.hpp"

#include "quinoa/align/Coarse.hpp"
#include "quinoa/align/Projection.hpp"
#include "quinoa/align/Rotation.hpp"

#include "quinoa/Logger.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Thickness.hpp"
#include "quinoa/Reconstruct.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/SplineGrid.hpp"

namespace qn {
    void coarse_alignment(
        const Path& stack_path,
        Metadata& metadata,
        const CoarseAlignmentSettings& settings
    ) {
        auto t0 = Logger::status_scope_time("Coarse alignment");

        // The stage leveling needs the stack sorted with the tilt in ascending order and will throw otherwise.
        metadata.stack.sort("tilt").reset_indices();

        const auto tilt_series = load_stack(stack_path, metadata, {
            .compute_device = settings.device,
            .allocator = Allocator::DEFAULT_ASYNC,

            // Fourier cropping:
            // Keep everything at low resolution, high frequencies are useless here.
            .precise_cutoff = false,
            .rescale_target_resolution = 12,
            .rescale_min_size = 1000,
            .rescale_max_size = 1280,

            // Signal processing after cropping:
            .bandpass{
                .highpass_cutoff = 0.03, // FIXME use resolution2fftfreq with a min, check with rescale_min_size
                .highpass_width = 0.03,
                .lowpass_cutoff = 0.35,
                .lowpass_width = 0.15,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .exposure_filter_voltage = metadata.sample.voltage,

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.05,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        auto aligner = AlignmentCoarse(tilt_series.shape(), tilt_series.device());

        // TODO Detect for view with huge shifts and remove them?
        //      Maybe only for higher tilts, e.g. >20deg, I don't want to remove valuable low tilts...

        // We require the rotation angle from the mdoc, but we can check that this rotation matches the images.
        // In this case, do a quick shift alignment and search for the rotation offset across the full angle range.
        // The resulting angle isn't the most accurate but should be close enough (+-5deg) to the provided rotation.
        if (settings.check_rotation) {
            auto t1 = Logger::info_scope_time("Rotation check");
            auto metadata_check = metadata.stack;

            for (auto i: noa::irange(2)) {
                aligner.align_shifts(tilt_series.view(), metadata_check, {
                    .cosine_stretch = i > 0,
                    .update_count = 1,
                    .fov_mask = false,
                    .smooth_edge_percent = 0.08,
                    .max_shift_percent = 0.5,
                    .output_directory = &settings.output_directory,
                });

                find_rotation_offset(tilt_series.view(), metadata_check, {
                    .angle_range = -1, // full range
                    .output_directory = &settings.output_directory,
                });
            }

            const auto expected_rotation = Metadata::Image::to_angle_range(metadata.stack[0].angles[0]);
            const auto measured_rotation_1 = Metadata::Image::to_angle_range(metadata_check[0].angles[0]);
            const auto measured_rotation_2 = Metadata::Image::to_angle_range(metadata_check[0].angles[0] + 180);
            if (std::abs(expected_rotation - measured_rotation_1) > 5 and
                std::abs(expected_rotation - measured_rotation_2) > 5) {
                panic(
                    "The tilt-axis from the mdoc file or the experiment.tilt_axis setting (rotation={:.2f}) does not "
                    "seem to match the tilt images (rotation_estimate={:.2f}deg, or equivalently {:.2f}deg). Since "
                    "this check is fairly reliable, the program will stop now. If you are certain that the provided "
                    "tilt-axis is correct, this check can be turned off using alignment.coarse.check_rotation=false",
                    expected_rotation, measured_rotation_1, measured_rotation_2
                );
            }
        }

        auto angle_offsets = Vec{0., 0., 0.};
        for (auto i: noa::irange(4)) {
            aligner.align_shifts(tilt_series.view(), metadata.stack, {
                .cosine_stretch = i != 0,
                .update_count = i > 2 ? 5 : 10, // FIXME i >= 2?
                .fov_mask = i > 2,
                .smooth_edge_percent = i == 0 ? 0.08 : 0.3,
                .max_shift_percent = i == 0 ? 0.5 : 0.1,
                .output_directory = &settings.output_directory,
            });

            if (settings.fit_rotation_offset) {
                find_rotation_offset(tilt_series.view(), metadata.stack, angle_offsets, {
                    .angle_range = i > 2 ? 10. : 5.,
                    .output_directory = &settings.output_directory,
                });
            }

            if (settings.fit_tilt_offset or settings.fit_pitch_offset) {
                aligner.level_stage(tilt_series.view(), metadata.stack, angle_offsets, {
                    .tilt_search_range = not settings.fit_tilt_offset ? 0. : i == 0 ? 20. : 5.,
                    .pitch_search_range = not settings.fit_pitch_offset ? 0. : i == 0 ? 10. : 5.,
                    .fov_mask = i > 2,
                    .smooth_edge_percent = i == 0 ? 0.08 : 0.3,
                    .max_shift_percent = i == 0 ? 0.5 : 0.1,
                    .output_directory = &settings.output_directory,
                });
            }
        }

        aligner.align_shifts(tilt_series.view(), metadata.stack, {
            .cosine_stretch = true,
            .update_count = 15,
            .fov_mask = true,
            .smooth_edge_percent = 0.1,
            .max_shift_percent = 0.1,
            .output_directory = &settings.output_directory,
        });
    }

    void refine_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        const RefineAlignmentSettings& settings
    ) {
        auto timer = Logger::status_scope_time("Refine alignment");

        auto loader = StackLoader(stack_filename, {
            .compute_device = settings.compute_device,
            .allocator = Allocator::MANAGED,

            // Fourier cropping:
            .precise_cutoff = true, // ensure isotropic spacing
            .rescale_target_resolution = 10,
            .rescale_min_size = 1000,
            .rescale_max_size = 2000,

            // Signal processing after cropping:
            .bandpass{
                .highpass_cutoff = 0.03,
                .highpass_width = 0.01,
                .lowpass_cutoff = 0.45,
                .lowpass_width = 0.05,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .exposure_filter_voltage = metadata.sample.voltage,

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        // Load and filter the tilt-series.
        // This corrects for the CTF at the "center of the sample", where most of the signal comes from.
        // It is up to the thickness estimation and tomogram centering to place that "center of the sample"
        // at the center of the tomogram. On the other hand, this is quite low resolution and the CTF correction
        // has little to no effect; it's mostly about the B-factor filtering.
        const auto stack_spacing = mean(loader.stack_spacing());
        metadata.set_spacing(stack_spacing);
        metadata.stack.sort("tilt").reset_indices();
        const auto tilt_series = filter_stack(std::move(loader), metadata, {
            .ramp_filter = false,
            .correct_ctf = settings.correct_ctf,
            .phase_flip_strength = settings.phase_flip_strength,
            .defocus_step_nm = 15, // TODO probably not worth it, decrease it
            .bfactor = -50,
        });

        // Prepare for the projection matching.
        const auto n_images = metadata.stack.ssize();
        const auto image_shape = tilt_series.shape().filter(2, 3);
        auto projection_matcher = ProjectionMatcher(n_images, image_shape, tilt_series.device());

        // FIXME PLOT sincs

        // Set up the central-slice insertion.
        constexpr auto INSERT_SINC_OSCILLATIONS = 8;
        const f64 virtual_volume_size = static_cast<f64>(projection_matcher.spectrum_size()); // TODO -10%?
        const f64 fftfreq_sinc = 1 / virtual_volume_size;
        const f64 fftfreq_blackman = INSERT_SINC_OSCILLATIONS * fftfreq_sinc;
        Logger::trace(
            "Central-slice insertion bounds:\n"
            "  fftfreq_sinc={:.4f}cpp|{:.1f}pix (virtual_volume_size={})\n"
            "  fftfreq_blackman={:.4f}cpp",
            fftfreq_sinc, fftfreq_sinc * virtual_volume_size,
            virtual_volume_size, fftfreq_blackman
        );

        auto extraction_sinc = [&] {
            // Set up the central-slice extraction.
            constexpr auto EXTRACT_SINC_OSCILLATIONS = 4;
            const f64 thickness_estimate_pixels = metadata.sample.thickness / (stack_spacing * 1e-1);
            const f64 fftfreq_z_sinc = 1 / thickness_estimate_pixels;
            const f64 fftfreq_z_blackman = EXTRACT_SINC_OSCILLATIONS * fftfreq_z_sinc;
            Logger::trace(
                "Central-slice extraction bounds:\n"
                "  fftfreq_sinc={:.4f}cpp (sample_thickness={}pix|{:.2f}nm)\n"
                "  fftfreq_blackman={:.4f}cpp (w_window_size=~{}pix)",
                fftfreq_z_sinc, std::round(thickness_estimate_pixels), metadata.sample.thickness,
                fftfreq_z_blackman, std::round(fftfreq_z_blackman * virtual_volume_size * 2 + 1)
            );
            return nx::WindowedSinc{fftfreq_z_sinc, fftfreq_z_blackman};
        };

        constexpr f64 MAX_TILT_DIFFERENCE = 20;
        constexpr f64 SMOOTH_EDGE_PERCENT = 0.1;

        auto angle_offsets = Vec{0., 0., 0.};

        const i32 N_ITERATIONS = settings.fit_rotation_offset ? 2 : 1;
        for (auto _: noa::irange(N_ITERATIONS)) {
            // TODO Fourier crop stack to lower resolution and estimate thickness
            // metadata.sample.thickness = estimate_sample_thickness(stack_filename, metadata, {
            //     .device = settings.compute_device,
            //     .allocator = Allocator::ASYNC,
            //     .resolution = 24.,
            //     .output_directory = settings.output_directory / "thickness",
            // });

            projection_matcher.update_shifts(tilt_series.view(), metadata.stack, {
                .max_tilt_difference = MAX_TILT_DIFFERENCE,
                .smooth_edge_percent = SMOOTH_EDGE_PERCENT,
                .insertion_sinc = {fftfreq_sinc, fftfreq_blackman},
                .extraction_sinc = extraction_sinc(),
                // .debug_directory = settings.output_directory / "projection_matching",
            });

            if (settings.fit_rotation_offset) {
                find_rotation_offset(tilt_series.view(), metadata.stack, angle_offsets, {
                    .angle_range = 4,
                    .output_directory = &settings.output_directory,
                });
            }

            // Note that we don't recompute the CTF correction despite the possible change of tilt-axis.
            // Usually the axis barely changes, and while we could check and recompute it if the change is significant,
            // the CTF correction has little effect and is already an approximation anyway.
        }

        if (settings.fit_rotation_offset) {
            const auto esinc = extraction_sinc();
            const auto projection_matching_options = ProjectionMatchingParameters{
                .max_tilt_difference = MAX_TILT_DIFFERENCE,
                .smooth_edge_percent = SMOOTH_EDGE_PERCENT,
                .insertion_sinc = {fftfreq_sinc, fftfreq_blackman},
                .extraction_sinc = esinc,
                .debug_directory = settings.output_directory / "projection_matching",
            };

            // TODO store shifts for each
            // Optimizer optimizer(NLOPT_GN_DIRECT, std::ssize(buffer));
            // optimizer.set_max_number_of_evaluations(75);

            auto grid = GridSearch(Vec{-1., 1., 0.05}); // 0.1 then local opt?
            std::vector<f64> rotations;
            std::vector<f64> ccs;
            grid.for_each([&](const f64& offset) {
                auto tmp = metadata;
                tmp.stack.add_image_angles({offset, 0., 0.});
                auto score = projection_matcher.update_shifts(tilt_series.view(), tmp.stack, projection_matching_options);

                rotations.emplace_back(tmp.stack[0].angles[0]);
                ccs.emplace_back(score);
                fmt::println("rot={:.2f}, cc={}", tmp.stack[0].angles[0], score);

                // TODO try common-line with this alignment?
            });
            save_plot_xy(rotations, ccs, settings.output_directory / "projection_matching" / "pm_scores.txt");

            // TODO set shifts from best
        }

        projection_matcher = ProjectionMatcher{};

        // TODO stage level

        // TODO thickness center the sample ?
        // metadata.sample.thickness = estimate_sample_thickness(stack_filename, metadata, {
        //     .device = settings.compute_device,
        //     .allocator = Allocator::ASYNC,
        //     .resolution = 24.,
        //     .output_directory = settings.output_directory / "thickness",
        // });
    }
}


// auto rotations = std::array{
        //     174.9, 175.0, 175.0, 175.1, 175.1, 175.1, 175.1, 175.2, 175.2, 175.2, 175.2, 175.2, 175.2, 175.2, 175.2,
        //     175.2, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3, 175.3,
        //     175.3, 175.3, 175.3, 175.4, 175.4, 175.4, 175.4, 175.4, 175.4, 175.4
        // };
        //
        // auto rotations2 = std::array{
        //     175.012, 175.023, 175.032, 175.042, 175.052, 175.061, 175.070, 175.079, 175.089, 175.098, 175.108, 175.118,
        //     175.129, 175.140, 175.152, 175.165, 175.178, 175.192, 175.207, 175.223, 175.241, 175.259, 175.278, 175.298,
        //     175.319, 175.340, 175.362, 175.385, 175.408, 175.431, 175.455, 175.479, 175.503, 175.527, 175.551, 175.575,
        //     175.598, 175.621, 175.644, 175.667
        // };
        // for (usize i{}; auto& image: metadata.stack) {
        //     image.angles[0] = 175;//rotations2[i++];
        // }
        //
        // std::array buffer{0., 0., 0.};
        // const auto max_tilt = max(abs(metadata.stack.tilt_range()));
        // const auto min_tilt = -max_tilt;
        // //
        // using spline_t = SplineGrid<const f64, 1, nx::Interp::CUBIC>;
        // const auto spline = spline_t(SpanContiguous(buffer.data(), std::ssize(buffer)));
        //
        // Optimizer optimizer(NLOPT_GN_DIRECT, std::ssize(buffer));
        // optimizer.set_max_number_of_evaluations(75);
        // optimizer.set_x_tolerance_abs(0.05);
        // optimizer.set_bounds(-1., 1.);
        // optimizer.set_max_objective([&](u32 n, const f64* p, f64* g) -> f64 {
        //     check(g == nullptr);
        //     const auto spline = spline_t(SpanContiguous(p, std::ssize(buffer)));
        //     auto tmp = metadata.stack;
        //     for (auto& image: tmp) {
        //         const auto coordinate = (image.angles[1] - min_tilt) / (max_tilt - min_tilt);
        //         const auto rotation_offset = spline.interpolate_at(coordinate);
        //         image.angles[0] += rotation_offset;
        //     }
        //     // Logger::trace("rot={::.1f}", tmp | stdv::transform([](auto&i) { return i.angles[0]; }));
        //
        //     auto score = projection_matcher.update_shifts(tilt_series.view(), tmp, {
        //         .max_tilt_difference = MAX_TILT_DIFFERENCE,
        //         .smooth_edge_percent = SMOOTH_EDGE_PERCENT,
        //         .insertion_sinc = {fftfreq_sinc, fftfreq_blackman},
        //         .extraction_sinc = e_sinc,
        //         // .debug_directory = settings.output_directory / "projection_matching",
        //     });
        //     Logger::trace("rot={::.4f}, cc={:.6f}", spline.span, score);
        //     return score;
        // });
        // auto s = optimizer.optimize(buffer.data());
        // Logger::trace("n={}, s={}", optimizer.n_evaluations(), s);
        //
        // for (auto& image: metadata.stack) {
        //     const auto coordinate = (image.angles[1] - min_tilt) / (max_tilt - min_tilt);
        //     const auto rotation_offset = spline.interpolate_at(coordinate);
        //     image.angles[0] += rotation_offset;
        // }
        // Logger::trace("rot={::.3f}", metadata.stack | stdv::transform([](auto&i) { return i.angles[0]; }));
