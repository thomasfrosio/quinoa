#include "quinoa/Alignment.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/PairwiseShift.hpp"
#include "quinoa/PairwiseTilt.hpp"
#include "quinoa/RotationOffset.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Thickness.hpp"
#include "quinoa/CTF.hpp"

namespace qn {
    auto coarse_alignment(
        const Path& stack_path,
        MetadataStack& metadata,
        const CoarseAlignmentParameters& parameters
    ) -> f64 {
        auto timer = Logger::status_scope_time("Coarse alignment");

        // To keep it simple, work with the stack sorted with its tilt in ascending order.
        // PairwiseTilt relies on this and will throw an error if the stack isn't ordered.
        metadata.sort("tilt").reset_indices();

        // There alignments are quite robust at low tilts.
        // Keep everything at low resolution, high frequencies are useless here.
        const auto [tilt_series, stack_spacing, file_spacing, file_slice_shape] = load_stack(stack_path, metadata, {
            .compute_device = parameters.compute_device,
            .allocator = Allocator::DEFAULT_ASYNC,

            // Fourier cropping:
            .precise_cutoff = false,
            .rescale_target_resolution = std::max(parameters.maximum_resolution, 24.),
            .rescale_min_size = 512,
            .rescale_max_size = 1280,

            // Signal processing after cropping:
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.05,
                .highpass_width = 0.05,
                .lowpass_cutoff = 0.25,
                .lowpass_width = 0.25,
            },

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        const auto debug = not parameters.debug_directory.empty();
        const auto basename = stack_path.stem().string();
        auto extension = stack_path.extension().string();
        if (not noa::io::ImageFile::is_supported_extension(extension))
            extension = ".mrc";

        if (not parameters.debug_directory.empty()) {
            const auto filename = parameters.debug_directory / fmt::format("{}_preprocessed{}", basename, extension);
            noa::write(tilt_series, stack_spacing, filename);
            Logger::debug("{} saved", filename);
        }

        // Scale the metadata shifts to the current sampling rate.
        metadata.rescale_shifts(file_spacing, stack_spacing);

        auto shift_fitter = PairwiseShift(tilt_series.shape(), tilt_series.device());
        auto shift_parameters = PairwiseShiftParameters{
            .interp = noa::Interp::LINEAR_FAST,
            .output_directory = parameters.output_directory,
            .debug_directory = debug ? parameters.debug_directory / "debug_pairwise_shift" : "",
        };

        auto rotation_fitter = RotationOffset{};
        auto rotation_parameters = RotationOffsetParameters{
            .interp{noa::Interp::LANCZOS8},
            .output_directory = parameters.output_directory,
        };

        // TODO Detect for view with huge shifts and remove them?
        //      Maybe only for higher tilts, e.g. >20deg, I don't want to remove valuable low tilts...

        bool has_rotation = parameters.has_user_rotation;
        bool has_tilt{};
        f64 tilt_offset{};
        for (auto smooth_edge_percent: std::array{0.08, 0.3, 0.3}) {
            // First, get the large shifts out of the way. Once these are removed, focus on the center.
            // If we have an estimate of the rotation from the user, use cosine-stretching but don't use
            // area-matching yet in case of large shifts.
            shift_parameters.cosine_stretch = has_rotation;
            shift_parameters.area_match = false;
            shift_parameters.smooth_edge_percent = smooth_edge_percent;
            shift_parameters.max_shift_percent = 1;
            shift_parameters.update_count = 5;
            shift_fitter.update(tilt_series.view(), metadata, shift_parameters);

            // Once we have estimates for the shifts, do the rotation search.
            // If we don't have an initial rotation from the user, search the entire range.
            // Otherwise, refine whatever rotation the user gave us.
            if (parameters.fit_rotation_offset) {
                rotation_parameters.reset_rotation = not has_rotation;
                rotation_parameters.angle_range = not has_rotation ? 90. : 10.;
                rotation_parameters.angle_step = not has_rotation ? 1. : 0.1;
                // rotation_parameters.line_range = not has_rotation ? 0. : 0.2;
                // rotation_parameters.line_step = not has_rotation ? 1. : 0.1;
                rotation_fitter.search(tilt_series.view(), metadata, rotation_parameters);
                has_rotation = true;
            }

            // Once we have an estimate for the rotation, do the tilt search.
            if (parameters.fit_tilt_offset) {
                coarse_fit_tilt(tilt_series.view(), metadata, tilt_offset, {
                    .grid_search_range = not has_tilt ? 25. : 15.,
                    .grid_search_step = not has_tilt ? 5. : 0.5,
                    .output_directory = parameters.output_directory,
                });
                has_tilt = true;
            }
        }

        // Once we have a first good estimate of the rotation and shifts, start again using the common area masks.
        // At each iteration, the rotation should be better, improving the cosine stretching for the shifts.
        // Similarly, the shifts should get better, allowing a better estimate of the common area and the rotation.
        for (auto [angle_range, angle_step]: noa::zip(std::array{5., 2., 1.}, std::array{0.1, 0.02, 0.01})) {
            shift_parameters.cosine_stretch = true;
            shift_parameters.area_match = true;
            shift_parameters.smooth_edge_percent = 0.3;
            shift_parameters.max_shift_percent = 0.05;
            shift_parameters.update_count = 10;
            shift_fitter.update(tilt_series.view(), metadata, shift_parameters);

            if (parameters.fit_rotation_offset) {
                rotation_parameters.angle_range = angle_range;
                rotation_parameters.angle_step = angle_step;
                // rotation_parameters.line_range = 0.2;
                // rotation_parameters.line_step = 0.01;
                rotation_fitter.search(tilt_series.view(), metadata, rotation_parameters);
            }

            if (parameters.fit_tilt_offset) {
                coarse_fit_tilt(tilt_series.view(), metadata, tilt_offset, {
                    .grid_search_range = 10.,
                    .grid_search_step = 0.1,
                    .output_directory = parameters.output_directory,
                });
            }
        }

        // Final shift alignment. We are done after that.
        shift_parameters.update_count = 15;
        shift_fitter.update(tilt_series.view(), metadata, shift_parameters);

        save_stack(
            tilt_series.view(), stack_spacing, metadata,
            parameters.output_directory / fmt::format("{}_coarse_aligned{}", basename, extension), {
                .correct_rotation = true,
                .dtype = noa::io::Encoding::F16,
            });

        // Scale the metadata back to the original resolution.
        metadata.rescale_shifts(stack_spacing, file_spacing);

        const Path csv_filename = parameters.output_directory / fmt::format("{}_coarse_aligned.csv", basename);
        metadata.save_csv(csv_filename, file_slice_shape, file_spacing);
        Logger::info("{} saved", csv_filename);

        return noa::mean(stack_spacing);
    }

    auto ctf_alignment(
        const Path& stack_filename,
        MetadataStack& metadata,
        const CTFAlignmentParameters& parameters
    ) -> f64 {
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
                .lowpass_cutoff = 0.45,
                .lowpass_width = 0.05,
            },

            // Image processing after cropping:
            .normalize_and_standardize = true, // TODO do we need any kind of preprocessing here?
            .smooth_edge_percent = 0.03, // TODO is this necessary?
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });

        auto input_ctf = ns::CTFAnisotropic<f64>({
            .pixel_size = stack_loader.stack_spacing(),
            .defocus = {0., 0., 0.},
            .voltage = parameters.voltage,
            .amplitude = parameters.amplitude,
            .cs = parameters.cs,
            .phase_shift = parameters.phase_shift,
            .bfactor = 0,
            .scale = 1.,
        });
        auto ctf = ns::CTFIsotropic(input_ctf);

        // Patch size.
        // It should be big enough so there's sufficient signal and the Thon rings are somewhat visible in a single
        // patch, but it shouldn't be too big otherwise the defocus range within one patch at high tilt becomes too
        // big resulting in less Thon rings.
        const auto spacing = mean(stack_loader.stack_spacing()); // assume isotropic spacing by this point
        i64 patch_size = static_cast<i64>(std::round(parameters.patch_size_ang / spacing));
        patch_size = noa::fft::next_fast_size(noa::clamp(patch_size, 512, 1024)); // TODO Document 300 is unnecessary small?

        // The patches are Fourier cropped to fftfreq_range[1] and zero-padded to this size to increase the sampling.
        // At this point, we don't know what defocus to expect, but this should be enough to get us started and
        // remove aliasing in most cases.
        i64 patch_size_padded = noa::clamp(patch_size, 512, 1024);

        // TODO allow maximum number of patch to limit memory usage, maybe by controlling the overlap?
        auto grid = ctf::Grid(stack_loader.slice_shape(), patch_size, patch_size / 2);

        // Load and process images in the same order they were collected.
        metadata.sort("time").reset_indices();

        // If the exposure of the first image is significantly higher than the second and third, it may also
        // be collected at a much lower defocus (see TYGRESS-like schemes), so keep track of this.
        const bool first_image_has_higher_exposure = [&] {
            // The first image that was collected should be the lowest tilt.
            auto metadata_time_sorted = metadata;
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

        // Get an initial CTF based on the first few images.
        // This will be used to compute the aliasing-free size of the patches.
        auto metadata_initial = metadata;
        metadata_initial.exclude_if([&](auto& s) {
            return (first_image_has_higher_exposure and s.index == 0) or
                   s.index >= parameters.n_images_in_initial_average;
        });
        auto patches = ctf::Patches::from_stack(
            stack_loader, metadata_initial, grid, parameters.resolution_range,
            patch_size, patch_size_padded, patch_size
        );
        auto fitting_range = ctf::initial_fit(
            grid, patches, metadata_initial, ctf, {
                .n_slices_to_average = parameters.n_images_in_initial_average,
                .fit_phase_shift = parameters.fit_phase_shift,
                .output_directory = parameters.output_directory,
            });

        // Now that we have an idea of the defocus range within the stack, we can compute a size that should be
        // big enough to prevent any aliasing of the CTF. Extract the entire stack and sample the patches at that size.
        i64 aliasing_free_size = ctf::aliasing_free_size(ctf, patches.rho_vec()); // FIXME average vs max defocus, add +0.5
        patch_size_padded = noa::clamp(aliasing_free_size, patch_size, 1280);
        patch_size_padded = noa::fft::next_fast_size(patch_size_padded);
        Logger::trace("aliasing_free_size={} -> patch_size_padded={}", aliasing_free_size, patch_size_padded);

        patches = ctf::Patches{}; // erase initial patches
        patches = ctf::Patches::from_stack(
            stack_loader, metadata, grid, parameters.resolution_range,
            patch_size, patch_size_padded, patch_size
        );

        // Run the coarse CTF alignment.
        // This is a simple alignment of the patches near the tilt-axis to get initial per-image estimates of the
        // defocus, and check that the defocus ramp matches the tilt geometry.
        ctf::coarse_fit(
            grid, patches, ctf, metadata, { // ctf.defocus|phase_shift and metadata.defocus|phase_shift are updated
                .initial_fitting_range = fitting_range,
                .first_image_has_higher_exposure = first_image_has_higher_exposure,
                .fit_phase_shift = parameters.fit_phase_shift,
                .has_user_rotation = parameters.has_user_rotation,
                .output_directory = parameters.output_directory,
            });

        // Final CTF fitting.
        // Get the stage angles (mostly tilt and pitch), accurate per-image astigmatic defocus
        // and time-resolve phase-shift.
        ctf::refine_fit(
            metadata, grid, patches, ctf, {
                .fit_rotation = parameters.fit_rotation,
                .fit_tilt = parameters.fit_tilt,
                .fit_pitch = parameters.fit_pitch,
                .fit_phase_shift = parameters.fit_phase_shift,
                .fit_astigmatism = parameters.fit_astigmatism,
                .output_directory = parameters.output_directory,
            });

        return 0;
    }
}
