#include "quinoa/core/Alignment.h"
#include "Utilities.h"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/Stack.hpp"

namespace qn {
    void pairwise_alignment(
            const Path& stack_filename,
            MetadataStack& metadata,
            const PairwiseAlignmentParameters& parameters
    ) {
        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*median_filter_window=*/ int32_t{1},
                /*precise_cutoff=*/ false,
                /*rescale_target_resolution=*/ std::max(parameters.maximum_resolution, 16.),
                /*rescale_min_size=*/ 1024,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.03, 0.03},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.03f,
                /*zero_pad_to_fast_fft_shape=*/ true,
        };

        const auto rotation_parameters = GlobalRotationParameters{
                /*highpass_filter=*/ {0.05, 0.05},
                /*lowpass_filter=*/ {0.25, 0.15},
                /*absolute_max_tilt_difference=*/ 70,
                /*solve_using_estimated_gradient=*/ false,
                /*interpolation_mode=*/ noa::InterpMode::LINEAR_FAST
        };

        const auto pairwise_shift_parameters = PairwiseShiftParameters{
                /*highpass_filter=*/ {0.03, 0.03},
                /*lowpass_filter=*/ {0.25, 0.1},
                /*interpolation_mode=*/ noa::InterpMode::LINEAR_FAST,
                /*debug_directory=*/ parameters.debug_directory
        };

        // These alignments are quite robust, and we shouldn't need to
        // run multiple resolution cycles. Just load the stack once and align it.
        const auto [tilt_series, stack_scaling, file_scaling] =
                load_stack(stack_filename, metadata, loading_parameters);

        if (!parameters.debug_directory.empty()) {
            const auto cropped_stack_filename =
                    parameters.debug_directory /
                    noa::string::format("{}_preprocessed_0{}",
                                        stack_filename.stem().string(),
                                        stack_filename.extension().string());
            noa::io::save(tilt_series, stack_scaling.as<f32>(), cropped_stack_filename);
        }

        // Scale the metadata shifts to the current sampling rate.
        const auto pre_scale = file_scaling / stack_scaling;
        for (auto& slice: metadata.slices())
            slice.shifts *= pre_scale;

        // TODO tilt offset can be estimated using CC first. Also try the quick profile projection and COM?
        auto pairwise_shift = PairwiseShift(tilt_series.shape(), tilt_series.device());
        auto global_rotation =
                !parameters.search_rotation_offset ?
                GlobalRotation() :
                GlobalRotation(
                        tilt_series, metadata,
                        tilt_series.device(), rotation_parameters);

        const bool has_initial_rotation = !noa::math::are_almost_equal(
                MetadataSlice::UNSET_ROTATION_VALUE, metadata[0].angles[0]);
        QN_CHECK(parameters.search_rotation_offset || has_initial_rotation,
                 "No rotation offset was provided and rotation offset alignment is turned off");

        if (!has_initial_rotation) {
            // Find a good estimate of the shifts, without cosine-stretching.
            // Similarly, we cannot use area match because we don't have the rotation.
            std::array smooth_edge_percents{0.08, 0.3}; // once large shifts removed, focus on center.
            for (auto smooth_edge_percent: smooth_edge_percents) {
                pairwise_shift.update(
                        tilt_series, metadata,
                        pairwise_shift_parameters,
                        /*cosine_stretch=*/ false,
                        /*area_match=*/ false,
                        /*smooth_edge_percent=*/ smooth_edge_percent);
            }

            // Once we have estimates for the shifts, do the global rotation search.
            global_rotation.initialize(metadata, rotation_parameters);

        } else {
            // If we have an estimate of the rotation from the user, use cosine-stretching
            // but don't use area-matching yet in order to allow large shifts.
            std::array smooth_edge_percents{0.08, 0.3}; // once large shifts removed, focus on center.
            for (auto smooth_edge_percent: smooth_edge_percents) {
                pairwise_shift.update(
                        tilt_series, metadata,
                        pairwise_shift_parameters,
                        /*cosine_stretch=*/ true,
                        /*area_match=*/ false,
                        /*smooth_edge_percent=*/ smooth_edge_percent);
            }
        }

        // Once we have a first good estimate of the rotation and shifts, start again using the common area masks.
        // At each iteration the rotation should be better, improving the cosine stretching for the shifts.
        // Similarly, the shifts should get better, allowing a better estimate of the common area and the rotation.
        const std::array rotation_bounds{5., 2.5, 1.25};
        for (auto rotation_bound: rotation_bounds) {
            pairwise_shift.update(
                    tilt_series, metadata,
                    pairwise_shift_parameters,
                    /*cosine_stretch=*/ true,
                    /*area_match=*/ true,
                    /*smooth_edge_percent=*/ 0.08,
                    /*max_size_loss_percent=*/ 0.1);
            global_rotation.update(
                    metadata, rotation_parameters, rotation_bound);
        }
        pairwise_shift.update(
                tilt_series, metadata,
                pairwise_shift_parameters,
                /*cosine_stretch=*/ true,
                /*area_match=*/ true,
                /*smooth_edge_percent=*/ 0.08,
                /*max_size_loss_percent=*/ 0.1); // TODO Double phase?

        // Scale the metadata back to the original resolution.
        const auto post_scale = 1 / pre_scale;
        for (auto& slice: metadata.slices())
            slice.shifts *= post_scale;

        const auto saving_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*median_filter_window=*/ int32_t{1},
                /*precise_cutoff=*/ true,
                /*rescale_target_resolution=*/ std::max(parameters.maximum_resolution, 16.),
                /*rescale_min_size=*/ 1024,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.03, 0.03},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.02f,
                /*zero_pad_to_fast_fft_shape=*/ false,
        };

        const auto aligned_stack_filename =
                parameters.output_directory /
                noa::string::format("{}_coarse_aligned{}",
                                    stack_filename.stem().string(),
                                    stack_filename.extension().string());
        qn::save_stack(stack_filename, aligned_stack_filename,
                       metadata, saving_parameters,
                       noa::InterpMode::LINEAR_FAST);
    }

    void ctf_alignment(
            const Path& tilt_series_filename,
            MetadataStack& tilt_series_metadata
    ) {
        // Load stack
        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ Device("gpu"),
                /*median_filter_window=*/ int32_t{1},
                /*precise_cutoff=*/ true,
                /*rescale_target_resolution=*/ -1,
                /*rescale_min_size=*/ 1024,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.02, 0.02},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.02f,
                /*zero_pad_to_fast_fft_shape=*/ false,
        };
        auto stack_loader = StackLoader(tilt_series_filename, loading_parameters);

        rescale_shifts(tilt_series_metadata, stack_loader.file_spacing(), stack_loader.stack_spacing());

        const f64 spacing = noa::math::sum(stack_loader.stack_spacing()) / 2 ;
        const i64 patch_size = noa::fft::next_fast_size(512);
        const i64 patch_step = noa::fft::next_fast_size(256);
        const auto delta_z_range_nanometers = Vec2<f64>{-50, 50}; // nm
        const auto delta_z_shift_nanometers = 150.;
        const auto max_tilt_for_average = 90.;
        const bool fit_phase_shift = false;
        const bool fit_astigmatism = true;
        const bool flip_rotation_to_match_defocus_ramp = false;

        CTF::FittingRange fitting_range({40, 8}, spacing, patch_size);

        auto ctf_isotropic = CTFAnisotropic64(
                {spacing, spacing},
                /*defocus=*/ {2.5, 0, 0},
                /*voltage=*/ 300,
                /*amplitude=*/ 0.07,
                /*cs=*/ 2.7,
                /*phase_shift=*/ 0,
                /*bfactor=*/ 0
        );

        auto ctf_fitter = CTF(stack_loader.slice_shape(), patch_size, patch_step, Device("gpu"));

        ctf_fitter.fit_average(
                stack_loader,
                tilt_series_metadata,
                fitting_range,
                ctf_isotropic,
                delta_z_range_nanometers,
                delta_z_shift_nanometers,
                max_tilt_for_average,
                fit_phase_shift,
                fit_astigmatism,
                flip_rotation_to_match_defocus_ramp,
                ""
                );

        ctf_fitter.fit_global(

                );
    }

    MetadataStack projection_matching_alignment(
            const Path& tilt_series_filename,
            const MetadataStack& tilt_series_metadata,
            const ProjectionMatchingAlignmentParameters& parameters
    ) {

        auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*median_filter_window=*/ 1,
                /*precise_cutoff=*/ false,
                /*rescale_target_resolution=*/ -1.,
                /*rescale_min_size=*/ 1024,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.03, 0.03},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.03f,
                /*zero_pad_to_fast_fft_shape=*/ true,
        };

        auto projection_matching_parameters = ProjectionMatchingParameters{
                /*zero_pad_to_fast_fft_size=*/ true,

                /*rotation_tolerance_abs=*/ 0.005,
                /*rotation_range=*/ 0,
                /*rotation_initial_step=*/ 0.15,

                /*smooth_edge_percent=*/ 0.1,

                /*projection_slice_z_radius=*/ 0.0001f,
                /*projection_cutoff=*/ 0.5f,
                /*projection_max_tilt_angle_difference=*/ 120,

                /*highpass_filter=*/ {0.08, 0.05},
                /*lowpass_filter=*/ {0.4, 0.1}, // FIXME

                /*allowed_shift_percent=*/ 0.005,

                /*debug_directory=*/ parameters.debug_directory
        };

        //
        MetadataStack metadata = tilt_series_metadata;

        // To find the global rotation, start at low resolution and progressively decrease the resolution
        // to focus and increase the accuracy of the grid search. The final iteration is there to find
        // the best shifts, without any changes on the rotation.
        std::vector target_size{512, 1024, 2048};
        std::vector target_resolutions{20., 16., 10.};
        std::vector cubic_spline_resolution{1, 1, 1};
        std::vector global_rotation_bound{5., 2., 2.};
        std::vector global_rotation_tolerance{0.05, 0.05, 0.01};

        for (size_t i : noa::irange(3)) {

            loading_parameters.rescale_target_resolution = 20.;
            loading_parameters.rescale_min_size = target_size[i];

            //
            projection_matching_parameters.rotation_tolerance_abs = global_rotation_tolerance[i];
            projection_matching_parameters.rotation_range = global_rotation_bound[i];
            projection_matching_parameters.rotation_initial_step = global_rotation_bound[i] / 3;

            // Load stack at the current resolution.

            const auto [tilt_series, stack_spacing, file_spacing] =
                    load_stack(tilt_series_filename, metadata, loading_parameters);
            const auto stack = tilt_series.view();

            // Rescale metadata to this resolution.
            center_shifts(metadata);
            rescale_shifts(metadata, file_spacing / stack_spacing);

            // Set the common area. For projection matching it should be more important to make sure
            // the views that do not fully contain the common area are excluded. So 1) allow smaller
            // common areas to exclude fewer views, and 2) exclude the views that are still too far off.
            constexpr f64 maximum_size_loss = 0.2; // FIXME Make this a user parameter
            auto common_area = CommonArea(metadata.size(), stack.device());
            const std::vector<i64> excluded_indexes = common_area.set_geometry(
                    stack.shape().filter(2, 3), metadata, maximum_size_loss);
            if (!excluded_indexes.empty())
                metadata.exclude(excluded_indexes);

            // TODO Clean fft cache
            // TODO Move stack back to CPU to save GPU memory

            auto projection_matching = ProjectionMatching(
                    stack.shape().filter(2, 3),
                    parameters.compute_device,
                    metadata, projection_matching_parameters
                    );

            // Initialize the rotation model.
            auto cubic_spline_grid = CubicSplineGrid<f64, 1>(/*resolution=*/ cubic_spline_resolution[i]);
            cubic_spline_grid.data()[0] = metadata[0].angles[0];

            projection_matching.update(stack, metadata, common_area,
                                       cubic_spline_grid,
                                       projection_matching_parameters);

            // Rescale metadata back to original resolution.
            rescale_shifts(metadata, stack_spacing / file_spacing);
        }

        // Final centering of the alignment's field-of-view.
        center_shifts(metadata);

        const auto saving_parameters = LoadStackParameters{
                parameters.compute_device, // Device compute_device
                int32_t{0}, // int32_t median_filter_window
                /*precise_cutoff=*/ true,
                parameters.maximum_resolution,
                1024,
                false, // bool exposure_filter
                {0.01, 0.01}, // float2_t highpass_parameters
                {0.5, 0.05}, // float2_t lowpass_parameters
                true, // bool normalize_and_standardize
                0.02f, // float smooth_edge_percent
        };

        const auto aligned_stack_filename =
                parameters.output_directory /
                noa::string::format("{}_aligned{}",
                                    tilt_series_filename.stem().string(),
                                    tilt_series_filename.extension().string());
        qn::save_stack(tilt_series_filename, aligned_stack_filename, metadata, saving_parameters);

        return metadata;
    }
}
