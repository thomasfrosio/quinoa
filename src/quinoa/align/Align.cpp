#include "quinoa/align/Align.hpp"
#include "quinoa/align/Coarse.hpp"
#include "quinoa/align/Projection.hpp"
#include "quinoa/align/Rotation.hpp"

#include "quinoa/Logger.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Thickness.hpp"

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
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.03, // FIXME use resolution2fftfreq with a min, check with rescale_min_size
                .highpass_width = 0.03,
                .lowpass_cutoff = 0.35,
                .lowpass_width = 0.15,
            },
            .bandpass_mirror_padding_factor = 0.5,

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
        for (i32 i: noa::irange(4)) {
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

        metadata.stack.sort("tilt").reset_indices();
        const auto tilt_series = load_stack(stack_filename, metadata, {
            .compute_device = settings.compute_device,
            .allocator = Allocator::MANAGED,

            // Fourier cropping:
            .precise_cutoff = true, // ensure isotropic spacing
            .rescale_target_resolution = settings.maximum_resolution,
            .rescale_min_size = 512,

            // Signal processing after cropping:
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.02,
                .highpass_width = 0.02,
                .lowpass_cutoff = 0.5,
                .lowpass_width = 0.05,
            },
            .bandpass_mirror_padding_factor = 0.5,

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        const auto stack_spacing = mean(metadata.spacing);
        const auto thickness_settings = EstimateSampleThicknessFromFileOptions{
            .device = settings.compute_device,
            .allocator = Allocator::ASYNC,
            .resolution = 24.,
            .output_directory = settings.output_directory / "thickness",
        };
        estimate_sample_thickness(stack_filename, metadata, thickness_settings);
        panic();

        auto rotation_settings = RotationOffsetParameters{
            .angle_range = 4,
            .output_directory = &settings.output_directory,
        };

        const auto image_shape = tilt_series.shape().filter(2, 3);
        const auto projection_matcher = ProjectionMatcher(metadata.stack.ssize(), image_shape, tilt_series.device());

        // Set up the Fourier insertion.
        const f64 virtual_volume_size = static_cast<f64>(projection_matcher.spectrum_size()); // FIXME
        const f64 fftfreq_sinc = 1 / virtual_volume_size;
        const f64 fftfreq_blackman = 8 * fftfreq_sinc;
        Logger::trace(
            "Fourier insertion bounds:\n"
            "  fftfreq_sinc={:.4f}\n"
            "  fftfreq_blackman={:.4f}",
            fftfreq_sinc, fftfreq_blackman
        );

        // save_stack(tilt_series.view(), stack_spacing, metadata, settings.output_directory / "projection_matching" / "stack_coarse.mrc");

        // for (auto &s: metadata)
            // s.shifts += {noa::random_value(noa::Uniform{-10., 10.}), noa::random_value(noa::Uniform{-10., 10.})};

        // auto shift_fitter = PairwiseShift(tilt_series.shape(), tilt_series.device());
        // auto shift_settings = PairwiseShiftParameters{.output_directory = &settings.output_directory};

        for (auto i: noa::irange(1)) {
            // shift_settings.cosine_stretch = true;
            // shift_settings.smooth_edge_percent = 0.08;
            // shift_settings.max_shift_percent = 0.15;
            // shift_settings.area_match = true;
            // shift_settings.update_count = 15;
            // shift_fitter.update(tilt_series.view(), metadata.stack, shift_settings);

            f64 thickness_nm = 220; // estimate_sample_thickness(stack_filename, metadata.stack, thickness_settings) - 40;
            Logger::trace("thickness={:.1f}nm", thickness_nm);
            metadata.sample.thickness = thickness_nm;

            const f64 thickness_estimate_pixels = thickness_nm / (stack_spacing * 1e-1);
            const f64 fftfreq_z_sinc = 1 / thickness_estimate_pixels;
            const f64 fftfreq_z_blackman = 8 * fftfreq_z_sinc;
            Logger::trace(
                "Fourier extraction bounds:\n"
                "  fftfreq_sinc={:.4f} (sample_thickness={}pixels)\n"
                "  fftfreq_blackman={:.4f} (window_size=~{}pixels)",
                fftfreq_z_sinc, std::round(thickness_estimate_pixels), fftfreq_z_blackman,
                std::round(fftfreq_z_blackman * virtual_volume_size * 2 + 1)
            );

            projection_matcher.update_shifts(tilt_series.view(), metadata.stack, {
                .shift_tolerance = 0.005,
                .max_tilt_difference = 120.,
                .smooth_edge_percent = 0.1,

                .insertion_sinc = {fftfreq_sinc, fftfreq_blackman},
                .extraction_sinc = {fftfreq_z_sinc, fftfreq_z_blackman},
                .bandpass = {0., 0., 0.5, 0.},
                .debug_directory = settings.output_directory / "projection_matching",
            });

            find_rotation_offset(tilt_series.view(), metadata.stack, rotation_settings);
        }
        // save_stack(tilt_series.view(), metadata.spacing, metadata.stack, settings.output_directory / "projection_matching" / "stack_refine.mrc");
        // panic();
    }
}
