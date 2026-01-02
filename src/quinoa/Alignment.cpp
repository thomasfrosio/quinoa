#include <noa/FFT.hpp>

#include "quinoa/Alignment.hpp"
#include "quinoa/AlignmentCoarse.hpp"
#include "quinoa/CTF.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/PairwiseShift.hpp"
#include "quinoa/ProjectionMatching.hpp"
#include "quinoa/RotationOffset.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Thickness.hpp"

namespace qn {
    void coarse_alignment(
        const Path& stack_path,
        Metadata& metadata,
        const CoarseAlignmentParameters& parameters
    ) {
        auto t0 = Logger::status_scope_time("Coarse alignment");

        // To keep it simple, work with the stack sorted with its tilt in ascending order.
        // The stage leveling relies on this and will throw an error if the stack isn't ordered.
        metadata.stack.sort("tilt").reset_indices();

        // Keep everything at low resolution, high frequencies are useless here.
        const auto tilt_series = load_stack(stack_path, metadata, {
            .compute_device = parameters.compute_device,
            .allocator = Allocator::DEFAULT_ASYNC,

            // Fourier cropping:
            .precise_cutoff = false,
            .rescale_target_resolution = parameters.maximum_resolution,
            .rescale_min_size = 1000,
            .rescale_max_size = 1280,

            // Signal processing after cropping:
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.03, // FIXME use resolution2fftfreq with a min, check with size 670
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
        // The resulting angle isn't the most accurate but should be close enough (+-10deg) to the provided rotation.
        if (parameters.check_rotation) {
            auto t1 = Logger::info_scope_time("Rotation check");
            t1.set_newline(false);
            auto metadata_check = metadata.stack;

            for (auto i: noa::irange(2)) {
                aligner.align_shifts(tilt_series.view(), metadata.stack, {
                    .cosine_stretch = i > 0,
                    .update_count = 1,
                    .fov_mask = false,
                    .smooth_edge_percent = 0.08,
                    .max_shift_percent = 0.5,
                    .output_directory = &parameters.output_directory,
                });

                find_rotation_offset(tilt_series.view(), metadata_check, {
                    .check_rotation = true,
                    .output_directory = &parameters.output_directory,
                });
            }

            const auto expected_rotation = Metadata::Image::to_angle_range(metadata.stack[0].angles[0]);
            const auto measured_rotation_1 = Metadata::Image::to_angle_range(metadata_check[0].angles[0]);
            const auto measured_rotation_2 = Metadata::Image::to_angle_range(metadata_check[0].angles[0] + 180);
            if (std::abs(expected_rotation - measured_rotation_1) > 10 and
                std::abs(expected_rotation - measured_rotation_2) > 10) {
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
                .update_count = i > 2 ? 5 : 10,
                .fov_mask = i > 2,
                .smooth_edge_percent = i == 0 ? 0.08 : 0.3,
                .max_shift_percent = i == 0 ? 0.5 : 0.1,
                .output_directory = &parameters.output_directory,
            });

            if (parameters.fit_rotation_offset) {
                find_rotation_offset(tilt_series.view(), metadata.stack, angle_offsets, {
                    .check_rotation = false,
                    .angle_range = i > 2 ? 10. : 5.,
                    .output_directory = &parameters.output_directory,
                });
            }

            if (parameters.fit_tilt_offset or parameters.fit_pitch_offset) {
                aligner.level_stage(tilt_series.view(), metadata.stack, angle_offsets, {
                    .tilt_search_range = not parameters.fit_tilt_offset ? 0. : i == 0 ? 20. : 5.,
                    .pitch_search_range = not parameters.fit_pitch_offset ? 0. : i == 0 ? 10. : 5.,
                    .fov_mask = i > 2,
                    .smooth_edge_percent = i == 0 ? 0.08 : 0.3,
                    .max_shift_percent = i == 0 ? 0.5 : 0.1,
                    .output_directory = &parameters.output_directory,
                });
            }
        }

        aligner.align_shifts(tilt_series.view(), metadata.stack, {
            .cosine_stretch = true,
            .update_count = 15,
            .fov_mask = true,
            .smooth_edge_percent = 0.1,
            .max_shift_percent = 0.1,
            .output_directory = &parameters.output_directory,
        });
    }

    void ctf_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        const CTFAlignmentParameters& parameters
    ) {
        auto timer = Logger::status_scope_time("CTF alignment");

        auto stack_loader = StackLoader(stack_filename, {
            .compute_device = parameters.compute_device,
            .allocator = Allocator::MANAGED,

            // Fourier cropping:
            .precise_cutoff = true, // ensure isotropic spacing
            .rescale_target_resolution = 0, // load at original spacing

            // Signal processing after cropping:
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.02,
                .highpass_width = 0.02,
                .lowpass_cutoff = 0.5,
                .lowpass_width = 0.05,
            },

            // Image processing after cropping:
            .normalize_and_standardize = true, // TODO do we need any kind of preprocessing here?
            .smooth_edge_percent = 0.03, // TODO is this necessary?
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });

        metadata.set_spacing(stack_loader.stack_spacing());
        const auto spacing = mean(metadata.spacing);

        // Patch size.
        // It should be big enough to have sufficient signal and the Thon rings are somewhat visible in a single
        // patch, but it shouldn't be too big otherwise the defocus range within one patch at high tilt becomes too
        // big resulting in less Thon rings.
        i64 patch_size = static_cast<i64>(std::round(parameters.patch_size_ang / spacing));
        patch_size = nf::next_fast_size(noa::clamp(patch_size, 512, 1024)); // TODO Document 300 is unnecessary small?

        // The patches are Fourier cropped to fftfreq_range[1] and zero-padded to this size to increase the sampling.
        // At this point, we don't know what defocus to expect, but this should be enough to get us started and
        // remove aliasing in most cases.
        i64 patch_size_padded = noa::clamp(patch_size, 512, 1024);

        const auto grid = ctf::Grid(stack_loader.slice_shape(), patch_size, patch_size / 2);

        // Load and process images in the same order they were collected.
        // TODO This may cause issues for cases where highest tilts are collected first.
        metadata.stack.sort("time").reset_indices();

        // If the exposure of the first image is significantly higher than the second and third, it may also
        // be collected at a much lower defocus (see TYGRESS-like schemes), so keep track of this.
        const bool first_image_has_higher_exposure = [&] {
            // The first image that was collected should be the lowest tilt.
            auto metadata_time_sorted = metadata.stack;
            metadata_time_sorted.sort("time");
            if (metadata_time_sorted.find_lowest_tilt_index() != 0)
                return false;

            f64 exposure_first = metadata_time_sorted[0].exposure[1] - metadata_time_sorted[0].exposure[0];
            f64 exposure_second = metadata_time_sorted[1].exposure[1] - metadata_time_sorted[1].exposure[0];
            f64 exposure_third = metadata_time_sorted[2].exposure[1] - metadata_time_sorted[2].exposure[0];
            if (exposure_first > exposure_second * 2 and exposure_first > exposure_third * 2) {
                Logger::info(
                    "Hybrid mode detected (exposure of images 1:{:.1f}, 2:{:.1f}, 3:{:.1f})",
                    exposure_first, exposure_second, exposure_third
                );
                return true;
            }
            return false;
        }();

        // Get an initial defocus, phase-shift and fitting-range based on the first few images.
        auto metadata_initial = metadata;
        metadata_initial.stack.exclude_if([&](auto& s) {
            return (first_image_has_higher_exposure and s.index == 0) or
                   s.index >= parameters.n_images_in_initial_average;
        });
        auto patches = ctf::Patches::from_stack(
            stack_loader, metadata_initial.stack, grid, parameters.resolution_range,
            patch_size, patch_size_padded
        );
        const auto initial_fit = ctf::initial_fit(
            metadata_initial, grid, patches, {
                .n_slices_to_average = parameters.n_images_in_initial_average,
                .fit_phase_shift = parameters.fit_phase_shift,
                .output_directory = parameters.output_directory,
            });

        for (auto& image: metadata.stack) {
            image.defocus.value = initial_fit.defocus;
            image.phase_shift = initial_fit.phase_shift;
        }

        {
            // Using the initial defocus estimate, we can compute an estimate of the aliasing-free size.
            const auto estimated_max_defocus = initial_fit.defocus + 0.5;
            auto target_ctf = metadata.empty_ctf();
            target_ctf.set_defocus(estimated_max_defocus);
            const i64 aliasing_free_size = ctf::aliasing_free_size(target_ctf, patches.rho_vec());
            constexpr i64 MAX_PADDED_SIZE = 2048;
            patch_size_padded = noa::clamp(aliasing_free_size, patch_size, MAX_PADDED_SIZE);
            patch_size_padded = nf::next_fast_size(patch_size_padded);

            Logger::trace(
                "Aliasing-free size:\n"
                "  estimated_max_defocus={:.2f}\n"
                "  aliasing_free_size={}\n"
                "  padded_size={} (clamped between [{}, {}]\n",
                estimated_max_defocus, aliasing_free_size, patch_size_padded,
                patch_size, MAX_PADDED_SIZE
            );
        }

        // Extract the entire stack and sample the patches using the aliasing-free size.
        metadata.stack.sort("tilt").reset_indices();
        const auto bin_angle = parameters.fit_astigmatism ? 3 : -1;
        patches = ctf::Patches{}; // erase initial patches
        patches = ctf::Patches::from_stack(
            stack_loader, metadata.stack, grid, parameters.resolution_range,
            patch_size, patch_size_padded, bin_angle
        );
        stack_loader = StackLoader{}; // erase buffers

        // Run the coarse CTF alignment.
        // This is a simple alignment of the patches near the tilt-axis to get initial per-image
        // estimates of the defocus and to check that the per-image defocus gradient matches the tilt geometry.
        ctf::coarse_fit(
            metadata, grid, patches, { // ctf.defocus|phase_shift and metadata.defocus|phase_shift are updated
                .initial_fitting_range = initial_fit.fitting_range,
                .first_image_has_higher_exposure = first_image_has_higher_exposure,
                .fit_phase_shift = parameters.fit_phase_shift,
                .check_rotation = parameters.check_rotation,
                .output_directory = parameters.output_directory,
            });

        // Find the specimen thickness by fitting the variance withing the tomogram.
        // While we technically could fit the thickness from the spectrum like in CTFFIND5, using the tomogram
        // seems more reliable. The thickness value we get here can then be plugged into the CTF model for the
        // final refine fit.
        if (parameters.fit_thickness) {
            // To fit the thickness from the tomogram, we first need to find the stage angles.
            // Since we don't need to be very accurate here, turning off the astigmatism for significantly
            // faster compute time should be fine.
            ctf::refine_fit(
                metadata, grid, patches, {
                    .fit_rotation = parameters.fit_rotation,
                    .fit_tilt = parameters.fit_tilt,
                    .fit_pitch = parameters.fit_pitch,
                    .fit_phase_shift = parameters.fit_phase_shift,
                    .fit_astigmatism = false,
                    .output_directory = parameters.output_directory,
                });
            metadata.sample.thickness = estimate_sample_thickness(
                stack_filename, metadata, {
                    .device = parameters.compute_device,
                    .allocator = Allocator::DEFAULT,
                    .resolution = 24,
                    .output_directory = parameters.output_directory
                });
        }

        // Final CTF alignment where the tilt-resolved astigmatism can be fitted.
        // Fitting the astigmatism is the slowest step of the CTF alignment, by far.
        ctf::refine_fit(
            metadata, grid, patches, {
                .fit_rotation = parameters.fit_rotation,
                .fit_tilt = parameters.fit_tilt,
                .fit_pitch = parameters.fit_pitch,
                .fit_phase_shift = parameters.fit_phase_shift,
                .fit_astigmatism = parameters.fit_astigmatism,
                .output_directory = parameters.output_directory,
            });
    }

    void refine_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        const RefineAlignmentParameters& parameters
    ) {
        auto timer = Logger::status_scope_time("Refine alignment");

        metadata.stack.sort("tilt").reset_indices();
        const auto tilt_series = load_stack(stack_filename, metadata, {
            .compute_device = parameters.compute_device,
            .allocator = Allocator::MANAGED,

            // Fourier cropping:
            .precise_cutoff = true, // ensure isotropic spacing
            .rescale_target_resolution = parameters.maximum_resolution,
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
        const auto thickness_parameters = EstimateSampleThicknessFromFileOptions{
            .device = parameters.compute_device,
            .allocator = Allocator::ASYNC,
            .resolution = 24.,
            .output_directory = parameters.output_directory / "thickness",
        };

        auto rotation_parameters = RotationOffsetParameters{
            .bandpass = {0., 0., 0.5, 0.}, // off
            .angle_range = 4,
            .output_directory = &parameters.output_directory,
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

        // save_stack(tilt_series.view(), stack_spacing, metadata, parameters.output_directory / "projection_matching" / "stack_coarse.mrc");

        // for (auto &s: metadata)
            // s.shifts += {noa::random_value(noa::Uniform{-10., 10.}), noa::random_value(noa::Uniform{-10., 10.})};

        // auto shift_fitter = PairwiseShift(tilt_series.shape(), tilt_series.device());
        // auto shift_parameters = PairwiseShiftParameters{.output_directory = &parameters.output_directory};

        for (auto i: noa::irange(1)) {
            // shift_parameters.cosine_stretch = true;
            // shift_parameters.smooth_edge_percent = 0.08;
            // shift_parameters.max_shift_percent = 0.15;
            // shift_parameters.area_match = true;
            // shift_parameters.update_count = 15;
            // shift_fitter.update(tilt_series.view(), metadata.stack, shift_parameters);

            f64 thickness_nm = 220; // estimate_sample_thickness(stack_filename, metadata.stack, thickness_parameters) - 40;
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
                .debug_directory = parameters.output_directory / "projection_matching",
            });

            find_rotation_offset(tilt_series.view(), metadata.stack, rotation_parameters);
        }
        // save_stack(tilt_series.view(), metadata.spacing, metadata.stack, parameters.output_directory / "projection_matching" / "stack_refine.mrc");
        // panic();
    }
}
