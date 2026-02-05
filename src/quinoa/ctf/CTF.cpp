#include "quinoa/ctf/CTF.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Thickness.hpp"

namespace qn::ctf {
    void fit(
        const Path& stack_filename,
        Metadata& metadata,
        const FitSettings& settings
    ) {
        auto timer = Logger::status_scope_time("CTF alignment");

        auto stack_loader = StackLoader(stack_filename, {
            .compute_device = settings.compute_device,
            .allocator = Allocator::MANAGED,

            // Fourier cropping:
            .precise_cutoff = true, // ensure isotropic spacing
            .rescale_target_resolution = 0, // load at original spacing

            // Signal processing after cropping:
            .bandpass{
                .highpass_cutoff = 0.02, // FIXME is this necessary?
                .highpass_width = 0.02,
                .lowpass_cutoff = 0.5,
                .lowpass_width = 0.05,
            },
            .exposure_filter_voltage = 0, // turn off

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
        i64 patch_size = static_cast<i64>(std::round(settings.patch_size_ang / spacing));
        patch_size = nf::next_fast_size(noa::clamp(patch_size, 512, 1024)); // TODO Document 300 is unnecessary small?

        // The patches are Fourier cropped to fftfreq_range[1] and zero-padded to this size to increase the sampling.
        // At this point, we don't know what defocus to expect, but this should be enough to get us started and
        // remove aliasing in most cases.
        i64 patch_size_padded = noa::clamp(patch_size, 512, 1024);

        const auto grid = Grid(stack_loader.slice_shape(), patch_size, patch_size / 2);

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
                   s.index >= settings.n_images_in_initial_average;
        });
        auto patches = Patches::from_stack(
            stack_loader, metadata_initial.stack, grid, settings.resolution_range,
            patch_size, patch_size_padded
        );
        const auto initial_fit = ctf::initial_fit(
            metadata_initial, grid, patches, {
                .n_slices_to_average = settings.n_images_in_initial_average,
                .fit_phase_shift = settings.fit_phase_shift,
                .output_directory = settings.output_directory,
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
        const auto bin_angle = settings.fit_astigmatism ? 3 : 180;
        patches = Patches{}; // erase initial patches
        patches = Patches::from_stack(
            stack_loader, metadata.stack, grid, settings.resolution_range,
            patch_size, patch_size_padded, bin_angle
        );
        stack_loader = StackLoader{}; // erase buffers

        // Run the coarse CTF alignment.
        // This is a simple alignment of the patches near the tilt-axis to get initial per-image
        // estimates of the defocus and to check that the per-image defocus gradient matches the tilt geometry.
        coarse_fit(
            metadata, grid, patches, { // ctf.defocus|phase_shift and metadata.defocus|phase_shift are updated
                .initial_fitting_range = initial_fit.fitting_range,
                .first_image_has_higher_exposure = first_image_has_higher_exposure,
                .fit_phase_shift = settings.fit_phase_shift,
                .check_defocus_gradient = settings.check_defocus_gradient,
                .output_directory = settings.output_directory,
            });

        // Find the specimen thickness by fitting the variance withing the tomogram.
        // While we technically could fit the thickness from the spectrum like in CTFFIND5, using the tomogram
        // seems more reliable. The thickness value we get here can then be plugged into the CTF model for the
        // final refine fit.
        if (settings.fit_thickness) {
            // To fit the thickness from the tomogram, we first need to find the stage angles.
            // Since we don't need to be very accurate here, turning off the astigmatism for significantly
            // faster compute time should be fine.
            refine_fit(
                metadata, grid, patches, {
                    .fit_rotation = settings.fit_rotation,
                    .fit_tilt = settings.fit_tilt,
                    .fit_pitch = settings.fit_pitch,
                    .fit_phase_shift = settings.fit_phase_shift,
                    .fit_astigmatism = false,
                    .output_directory = settings.output_directory,
                });
            metadata.sample.thickness = estimate_sample_thickness(
                stack_filename, metadata, {
                    .device = settings.compute_device,
                    .allocator = Allocator::DEFAULT,
                    .resolution = 24,
                    .output_directory = settings.output_directory
                });
        }

        // Final CTF alignment where the tilt-resolved astigmatism can be fitted.
        // Fitting the astigmatism is the slowest step of the CTF alignment, by far.
        refine_fit(
            metadata, grid, patches, {
                .fit_rotation = settings.fit_rotation,
                .fit_tilt = settings.fit_tilt,
                .fit_pitch = settings.fit_pitch,
                .fit_phase_shift = settings.fit_phase_shift,
                .fit_astigmatism = settings.fit_astigmatism,
                .output_directory = settings.output_directory,
            });
    }
}
