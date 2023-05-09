#pragma once

#include "quinoa/core/Metadata.h"
#include "quinoa/core/PairwiseShift.hpp"
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Stack.hpp"
#include "quinoa/core/GlobalRotation.hpp"

namespace qn {
    struct PairwiseAlignmentParameters {
        Device compute_device;
        f64 maximum_resolution;

        bool search_rotation_offset{true};
        bool search_tilt_offset{true};

        Path output_directory;
        Path debug_directory;
    };

    // Initial global alignment.
    // Updates the tilt-series geometry:
    //  - The rotation offset can be either measured or refined if a value is already set.
    //  - The shifts are measured using pairwise comparisons.
    void pairwise_alignment(
            const Path& tilt_series_filename,
            MetadataStack& tilt_series_metadata,
            const PairwiseAlignmentParameters& parameters) {

        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*median_filter_window=*/ int32_t{1},
                /*rescale_target_resolution=*/ std::max(parameters.maximum_resolution, 16.),
                /*rescale_min_shape=*/ {1024, 1024},
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.03, 0.03},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.03f,
                /*zero_pad_to_fast_fft_shape=*/ true,
        };

        const auto saving_parameters = SaveStackParameters{
                parameters.compute_device, // Device compute_device
                int32_t{0}, // int32_t median_filter_window
                std::max(parameters.maximum_resolution, 16.),
                {1024, 1024},
                false, // bool exposure_filter
                {0.03, 0.03}, // float2_t highpass_parameters
                {0.5, 0.05}, // float2_t lowpass_parameters
                true, // bool normalize_and_standardize
                0.02f, // float smooth_edge_percent
                noa::InterpMode::LINEAR,
                noa::BorderMode::ZERO,
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
        const auto [tilt_series, current_scaling, original_scaling] =
                load_stack(tilt_series_filename, tilt_series_metadata, loading_parameters);

        if (!parameters.debug_directory.empty()) {
            const auto cropped_stack_filename =
                    parameters.debug_directory /
                    noa::string::format("{}_preprocessed_0{}",
                                        tilt_series_filename.stem().string(),
                                        tilt_series_filename.extension().string());
            noa::io::save(tilt_series, current_scaling.as<f32>(), cropped_stack_filename);
        }

        // Scale the metadata shifts to the current sampling rate.
        const auto pre_scale = original_scaling / current_scaling;
        for (auto& slice: tilt_series_metadata.slices())
            slice.shifts *= pre_scale;

        // TODO tilt offset can be estimated using CC first. Also try the quick profile projection and COM?
        auto pairwise_shift = PairwiseShift(tilt_series.shape(), tilt_series.device());
        auto global_rotation =
                !parameters.search_rotation_offset ?
                GlobalRotation() :
                GlobalRotation(
                        tilt_series, tilt_series_metadata,
                        tilt_series.device(), rotation_parameters);

        const bool has_initial_rotation = !noa::math::are_almost_equal(
                MetadataSlice::UNSET_ROTATION_VALUE, tilt_series_metadata[0].angles[0]);
        QN_CHECK(parameters.search_rotation_offset || has_initial_rotation,
                 "No rotation offset was provided and rotation offset alignment is turned off");

        if (!has_initial_rotation) {
            // Find a good estimate of the shifts, without cosine-stretching.
            // Similarly, we cannot use area match because we don't have the rotation.
            std::array smooth_edge_percents{0.08, 0.3}; // once large shifts removed, focus on center.
            for (auto smooth_edge_percent: smooth_edge_percents) {
                pairwise_shift.update(
                        tilt_series, tilt_series_metadata,
                        pairwise_shift_parameters,
                        /*cosine_stretch=*/ false,
                        /*area_match=*/ false,
                        /*smooth_edge_percent=*/ smooth_edge_percent);
            }

            // Once we have estimates for the shifts, do the global rotation search.
            global_rotation.initialize(tilt_series_metadata, rotation_parameters);

        } else {
            // If we have an estimate of the rotation from the user, use cosine-stretching
            // but don't use area-matching yet in order to allow large shifts.
            std::array smooth_edge_percents{0.08, 0.3}; // once large shifts removed, focus on center.
            for (auto smooth_edge_percent: smooth_edge_percents) {
                pairwise_shift.update(
                        tilt_series, tilt_series_metadata,
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
                    tilt_series, tilt_series_metadata,
                    pairwise_shift_parameters,
                    /*cosine_stretch=*/ true,
                    /*area_match=*/ true,
                    /*smooth_edge_percent=*/ 0.08,
                    /*max_size_loss_percent=*/ 0.1);
            global_rotation.update(
                    tilt_series_metadata, rotation_parameters, rotation_bound);
        }
        pairwise_shift.update(
                tilt_series, tilt_series_metadata,
                pairwise_shift_parameters,
                /*cosine_stretch=*/ true,
                /*area_match=*/ true,
                /*smooth_edge_percent=*/ 0.08,
                /*max_size_loss_percent=*/ 0.1); // TODO Double phase?

        // Scale the metadata back to the original resolution.
        const auto post_scale = 1 / pre_scale;
        for (auto& slice: tilt_series_metadata.slices())
            slice.shifts *= post_scale;

        const auto aligned_stack_filename =
                parameters.output_directory /
                noa::string::format("{}_aligned{}",
                                    tilt_series_filename.stem().string(),
                                    tilt_series_filename.extension().string());
        qn::save_stack(tilt_series_filename, tilt_series_metadata, aligned_stack_filename, saving_parameters);
    }

    struct ProjectionMatchingAlignmentParameters {
        bool shift_only{false};
        bool save_input_stack{false};
        bool save_aligned_stack{false};
        Path output_directory;
    };

    void projection_matching_alignment(
            const Path& tilt_series_filename,
            MetadataStack& tilt_series_metadata,
            const LoadStackParameters& loading_parameters,
            const ProjectionMatchingParameters& projection_matching_parameters,
            const SaveStackParameters& saving_parameters) {

        // TODO Clean fft cache
        // TODO Move stack back to CPU to save GPU memory

        // TODO Loop in resolution. As resolution increases, decrease global reference rotation range.
        //      20A: +-2, step 0.5,
        //      15A: +-0.75, step 0.25
        //      10A: +- 0.25, step 0.1
        // TODO Maybe don't update the shifts at low res?

        // center_shifts(optimizer_data.metadata);

        // Projection matching alignment.
        // At this point, the global geometry should be pretty much on point.
//        if (alignment_parameters.projection_matching) {
//            auto projection_matching = qn::ProjectionMatching(
//                    tilt_series.shape(), tilt_series.device(),
//                    tilt_series_metadata, projection_matching_parameters);
//
//            projection_matching.update(
//                    tilt_series, tilt_series_metadata,
//                    projection_matching_parameters,
//                    alignment_parameters.projection_matching_shift_only);
//        }

//            // Enforce 3rd degree polynomial on the rotation angles.
//            // For the first iteration, this should just fit a line because the rotation is likely
//            // to be constant. In this case, it will simply update the shift, which is what we want anyway.
//            const ThirdDegreePolynomial polynomial = poly_fit_rotation(optimizer_data.metadata);
//
//            // Compute the rotation of the global reference using the polynomial. So whilst we cannot align
//            // the global reference, we can still move the average rotation (including the global reference's)
//            // progressively using projection matching.
//            optimizer_data.metadata[0].angles[0] = polynomial(optimizer_data.metadata[0].angles[1]);
    }
}
