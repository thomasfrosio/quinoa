#pragma once

#include "quinoa/core/Metadata.h"
#include "quinoa/core/PairwiseShift.hpp"
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Stack.hpp"
#include "quinoa/core/GlobalRotation.hpp"

namespace qn {
    struct InitialGlobalAlignmentParameters {
        bool rotation_offset{true};
        bool tilt_offset{true};
        bool shifts{true};

        bool save_input_stack{false};
        bool save_aligned_stack{false};
        Path output_directory;
    };

    // Initial global alignment.
    // Updates the tilt-series geometry:
    //  - The rotation offset can be either measured or refined if a value is already set.
    //  - The shifts are measured using pairwise comparisons.
    void initial_global_alignment(
            const Path& tilt_series_filename,
            MetadataStack& tilt_series_metadata,
            const LoadStackParameters& loading_parameters,
            const InitialGlobalAlignmentParameters& alignment_parameters,
            const GlobalRotationParameters& rotation_offset_parameters,
            const PairwiseShiftParameters& pairwise_shift_parameters,
            const SaveStackParameters& saving_parameters) {

        // These alignments are quite robust, and we shouldn't need to
        // run multiple resolution cycles. Just load the stack once and align it.
        const auto [tilt_series, current_pixel_size, original_pixel_size] =
                load_stack(tilt_series_filename, tilt_series_metadata, loading_parameters);

        if (alignment_parameters.save_input_stack) {
            const auto cropped_stack_filename =
                    alignment_parameters.output_directory /
                    noa::string::format("{}_preprocessed_0{}",
                                        tilt_series_filename.stem().string(),
                                        tilt_series_filename.extension().string());
            noa::io::save(tilt_series, current_pixel_size.as<f32>(), cropped_stack_filename);
        }

        // Scale the metadata shifts to the current sampling rate.
        const auto pre_scale = original_pixel_size / current_pixel_size;
        for (auto& slice: tilt_series_metadata.slices())
            slice.shifts *= pre_scale;

        // TODO tilt offset can be estimated using CC first. Also try the quick profile projection and COM?

        {
            const bool has_initial_rotation = noa::math::are_almost_equal(
                    MetadataSlice::UNSET_ROTATION_VALUE, tilt_series_metadata[0].angles[0]);

            auto pairwise_shift =
                    !alignment_parameters.pairwise_shift ?
                    PairwiseShift() :
                    PairwiseShift(tilt_series.shape(), tilt_series.device());

            auto global_rotation =
                    !alignment_parameters.rotation_offset ?
                    GlobalRotation() :
                    GlobalRotation(
                            tilt_series, tilt_series_metadata,
                            tilt_series.device(), rotation_offset_parameters);

            if (!has_initial_rotation) {
                // Find a good estimate of the shifts, without cosine-stretching.
                // Similarly, we cannot use area match because we don't have the rotation.
                std::array smooth_edge_percents{0.08, 0.3};
                for (auto smooth_edge_percent: smooth_edge_percents) {
                    pairwise_shift.update(
                            tilt_series, tilt_series_metadata,
                            pairwise_shift_parameters,
                            /*cosine_stretch=*/ false,
                            /*area_match=*/ false,
                            /*smooth_edge_percent=*/ smooth_edge_percent);
                }

                // Once we have estimates for the shifts, do the global rotation search.
                global_rotation.initialize(tilt_series_metadata, rotation_offset_parameters);
            } else {
                // If we have an estimate of the rotation from the user, use cosine-stretching
                // without area-matching to find estimates of the shifts.
                std::array smooth_edge_percents{0.08, 0.3};
                for (auto smooth_edge_percent: smooth_edge_percents) {
                    pairwise_shift.update(
                            tilt_series, tilt_series_metadata,
                            pairwise_shift_parameters,
                            /*cosine_stretch=*/ true,
                            /*area_match=*/ false,
                            /*smooth_edge_percent=*/ smooth_edge_percent);
                }
            }

            // Once we have a first good estimate of the rotation and shifts, start again.
            // At each iteration the rotation should be better, improving the cosine stretching for the shifts.
            // Similarly, the shifts should improve, allowing a better estimate of the field-of-view and the rotation.
            const std::array rotation_bounds{5., 2., 0.5};
            for (auto rotation_bound : rotation_bounds) {
                pairwise_shift.update(
                        tilt_series, tilt_series_metadata,
                        pairwise_shift_parameters,
                        /*cosine_stretch=*/ true,
                        /*area_match=*/ true,
                        /*smooth_edge_percent=*/ 0.1,
                        /*max_size_loss_percent=*/ 0.1);
                global_rotation.update(
                        tilt_series_metadata, rotation_offset_parameters, rotation_bound);
            }
            pairwise_shift.update(
                    tilt_series, tilt_series_metadata,
                    pairwise_shift_parameters,
                    /*cosine_stretch=*/ true,
                    /*area_match=*/ true,
                    /*smooth_edge_percent=*/ 0.1,
                    /*max_size_loss_percent=*/ 0.1); // TODO Double phase?
        }

        // TODO Estimate tilt and elevation offset using CTF.
        // TODO Rerun pairwise shift and rotation offset.

        // Scale the metadata back to the original resolution.
        const auto post_scale = 1 / pre_scale;
        for (auto& slice: tilt_series_metadata.slices())
            slice.shifts *= post_scale;

        if (alignment_parameters.save_aligned_stack) {
            const auto aligned_stack_filename =
                    alignment_parameters.output_directory /
                    noa::string::format("{}_aligned{}",
                                        tilt_series_filename.stem().string(),
                                        tilt_series_filename.extension().string());
            qn::save_stack(tilt_series_filename, tilt_series_metadata, aligned_stack_filename, saving_parameters);
        }
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
    }
}
