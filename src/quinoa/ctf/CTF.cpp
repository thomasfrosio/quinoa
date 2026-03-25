#include "quinoa/ctf/CTF.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Thickness.hpp"

namespace {
    auto is_hybrid_scheme_(const Metadata::Stack& metadata) -> bool {
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
    }

    auto patch_aliasing_free_size_(
        const Metadata& metadata,
        const Vec<f64, 2>& fftfreq_range,
        isize patch_min_size,
        isize patch_max_size,
        f64 defocus_offset
    ) -> isize {
        const auto estimated_max_defocus = metadata.stack.defocus_range(true)[1] + defocus_offset;
        auto target_ctf = metadata.empty_ctf();
        target_ctf.set_defocus(estimated_max_defocus);
        const i64 aliasing_free_size = ctf::aliasing_free_size(target_ctf, fftfreq_range);
        auto patch_padded_size = noa::clamp(aliasing_free_size, patch_min_size, patch_max_size);
        patch_padded_size = nf::next_fast_size(patch_padded_size);

        Logger::trace(
            "Aliasing-free size:\n"
            "  estimated_max_defocus={:.2f}\n"
            "  aliasing_free_size={}\n"
            "  padded_size={} (clamped between [{}, {}]\n",
            estimated_max_defocus, aliasing_free_size, patch_padded_size,
            patch_min_size, patch_max_size
        );
        return patch_padded_size;
    }

    auto is_high_resolution_recovery_needed_(
        const Metadata& metadata,
        const Metadata::Stack& previous_metadata,
        const Vec<f64, 2>& fftfreq_range,
        const SpanContiguous<Vec<f64, 2>>& fitting_ranges,
        isize patch_padded_size
    ) {
        // For severe cases of astigmatism, the equiphase-binning will increase the signal for the bins
        // between the two defoci where the defocus varies quickly between lines. However, the optimizer
        // is unlikely to need this because signal from the bins near the two defoci is fully preserved
        // during the initial binning. This is just to do things right, use all the available signal and
        // improve the reconstructed spectra.
        const bool need_new_equiphase_binning =
            metadata.stack.has_astigmatism(0.2) and
            metadata.stack.has_astigmatism_changed(previous_metadata);

        // The resolution limit may be cutting valuable signal.
        // Check if we are passed 90% of the spectrum current range.
        f64 highest_fitted_fftfreq{};
        for (const auto& fitting_range: fitting_ranges)
            highest_fitted_fftfreq = std::max(highest_fitted_fftfreq, fitting_range[1]);
        const auto fftfreq_threshold = fftfreq_range[0] + (fftfreq_range[1] - fftfreq_range[0]) * 0.9;
        const bool fitting_has_reached_the_end = highest_fitted_fftfreq > fftfreq_threshold;

        // If so, increase the resolution limit by 25%. This should be more than enough, and given that increasing
        // this will also significantly increase the aliasing-free size, be conservative.
        const auto spacing = mean(metadata.spacing);
        const auto current_resolution_limit = fftfreq_to_resolution(spacing, fftfreq_range[1]);
        auto target_resolution_limit = fitting_has_reached_the_end ?
            std::max(std::max(current_resolution_limit * 0.75, spacing * 2), 1.) :
            current_resolution_limit;

        // Only register >10% increases; otherwise it doesn't seem worth it. Of course, if we need to
        // recompute the patches because of the astigmatism anyway, update it regardless.
        const bool need_new_resolution_limit = target_resolution_limit < (current_resolution_limit * 0.9);
        if (not need_new_equiphase_binning or not need_new_resolution_limit)
            target_resolution_limit = current_resolution_limit;

        // The max (astigmatic) defocus, or the resolution limit, may have increased
        // to the point that the original aliasing-free size is too small.
        const auto new_fftfreq_limit = resolution_to_fftfreq(spacing, target_resolution_limit);
        const auto new_fftfreq_range = Vec{fftfreq_range[0], new_fftfreq_limit};
        const auto new_patch_padded_size = patch_aliasing_free_size_(
            metadata, new_fftfreq_range, patch_padded_size, 2048, 0.);
        const bool need_new_aliasing_free_size = (new_patch_padded_size - patch_padded_size) > 50;

        Logger::trace(
            "High-resolution recovery:\n"
            "  need_new_equiphase_binning={}\n"
            "  need_new_resolution_limit={} ({:.2f}A->{:.2f}A)\n"
            "  need_new_aliasing_free_size={} ({}pix->{}pix)",
            need_new_equiphase_binning,
            need_new_resolution_limit, current_resolution_limit, target_resolution_limit,
            need_new_aliasing_free_size, patch_padded_size, new_patch_padded_size
        );

        const bool need_recovery = need_new_aliasing_free_size or need_new_resolution_limit or need_new_equiphase_binning;
        return noa::make_tuple(need_recovery, new_patch_padded_size, target_resolution_limit);
    }
}

namespace qn::ctf {
    void fit(
        const Path& stack_filename,
        Metadata& metadata,
        const FitSettings& settings
    ) {
        auto timer = Logger::status_scope_time("CTF alignment");

        // Load and process images in the same order they were collected.
        // TODO This may cause issues for cases where highest tilts are collected first.
        metadata.stack.sort("time").reset_indices();

        // If the exposure of the first image is significantly higher than the second and third, it may also
        // be collected at a much lower defocus (see TYGRESS-like schemes), so keep track of this.
        const auto first_image_has_higher_exposure = is_hybrid_scheme_(metadata.stack);

        // Open the stack file and prepare for loading.
        auto stack_loading_parameters = Patches::LOADING_STACK_PARAMETERS;
        stack_loading_parameters.compute_device = settings.compute_device;
        stack_loading_parameters.allocator = Allocator::MANAGED;
        auto stack_loader = StackLoader(stack_filename, stack_loading_parameters);

        metadata.set_spacing(stack_loader.stack_spacing());
        const auto spacing = mean(metadata.spacing);

        // Patch size.
        // This is the size of the patches used to compute the grid; decreasing patch_size_ang increases the
        // number of patches, making the alignment more costly, but can increase the quality of the
        // oscillations at higher frequencies.
        const f64 patch_size_ang = noa::clamp(settings.patch_size_ang, 500., 1000.);
        i64 patch_size = static_cast<i64>(std::round(patch_size_ang / spacing));
        patch_size = nf::next_fast_size(noa::clamp(patch_size, 250, 2000));

        // The patches are Fourier cropped to the resolution limit and zero-padded to this size to increase the sampling.
        // At this point, we don't know what defocus to expect, but this should be enough to get us started and
        // remove aliasing in most cases.
        i64 patch_padded_size = 512;

        // Compute the centers of the 50% overlapped patches.
        const auto grid = Grid(stack_loader.slice_shape(), patch_size, patch_size / 2);

        // Get an initial defocus, phase-shift and fitting-range based on the first few images.
        auto metadata_initial = metadata;
        metadata_initial.stack.exclude_if([&](auto& s) {
            return (first_image_has_higher_exposure and s.index == 0) or
                   s.index >= settings.n_images_in_initial_average;
        });
        auto patches = Patches::from_stack(
            stack_loader, metadata_initial, grid, settings.resolution_range,
            patch_size, patch_padded_size
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

        // Using the initial defocus estimate, we can compute an estimate of the aliasing-free size.
        patch_padded_size = patch_aliasing_free_size_(metadata, patches.rho_vec(), patch_padded_size, 2048, 0.5);

        // Extract the entire stack and sample the patches using the aliasing-free size.
        metadata.stack.sort("tilt").reset_indices();
        auto bin_angle = settings.fit_astigmatism ? 3.75 : 180;
        patches = Patches{}; // erase initial patches
        patches = Patches::from_stack(
            stack_loader, metadata, grid, settings.resolution_range,
            patch_size, patch_padded_size, bin_angle
        );
        if (not settings.fit_astigmatism)
            stack_loader = StackLoader{}; // erase buffers

        // Coarse CTF alignment.
        coarse_fit(
            metadata, grid, patches, { // ctf.defocus|phase_shift and metadata.defocus|phase_shift are updated
                .initial_fitting_range = initial_fit.fitting_range,
                .first_image_has_higher_exposure = first_image_has_higher_exposure,
                .fit_phase_shift = settings.fit_phase_shift,
                .check_defocus_gradient = settings.check_defocus_gradient,
                .output_directory = settings.output_directory,
            });

        // Prepare for the full tilt-series alignment.
        auto previous_metadata = metadata.stack;
        auto refine_fit_state = FitRefineState{
            .phase_shift = Array<f64>(2),
            .astigmatism_value = Array<f64>(5),
            .astigmatism_angle = Array<f64>(5),
            .fitting_ranges = Array<Vec<f64, 2>>(patches.n_images()),
            .angle_offsets = Vec<f64, 3>{},
        };
        auto refine_fit_settings = FitRefineOptions{
            .full_fit = true,
            .fit_rotation = settings.fit_rotation,
            .fit_tilt = settings.fit_tilt,
            .fit_pitch = settings.fit_pitch,
            .fit_phase_shift = settings.fit_phase_shift,
            .fit_astigmatism = settings.fit_astigmatism,
            .output_directory = settings.output_directory,
        };

        // Full tilt-series alignment.
        refine_fit(metadata, refine_fit_state, grid, patches, refine_fit_settings);

        // High-resolution recovery.
        auto [recompute_patches, new_patch_padded_size, new_resolution_limit] = is_high_resolution_recovery_needed_(
            metadata, previous_metadata, patches.rho_vec(),
            refine_fit_state.fitting_ranges.span_1d(), patch_padded_size
        );
        if (not recompute_patches)
            return;

        // Recompute patches.
        bin_angle = 2.8125;
        patches = Patches::from_stack(
            stack_loader, metadata, grid, Vec{settings.resolution_range[0], new_resolution_limit},
            patch_size, new_patch_padded_size, bin_angle
        );

        // Full tilt-series refinement.
        refine_fit_settings.full_fit = false;
        refine_fit(metadata, refine_fit_state, grid, patches, refine_fit_settings);

        //
        bin_angle = 2.8125;
        patches = Patches::from_stack(
            stack_loader, metadata, grid, Vec{settings.resolution_range[0], new_resolution_limit},
            patch_size, new_patch_padded_size, bin_angle
        );

        // Full tilt-series refinement.
        refine_fit_settings.full_fit = false;
        refine_fit(metadata, refine_fit_state, grid, patches, refine_fit_settings);
    }
}
