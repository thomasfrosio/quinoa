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
        bool elevation_offset{true};

        bool pairwise_shift{true};
        bool projection_matching_shift{true};
        bool projection_matching_rotation{true};

        bool save_input_stack{false};
        bool save_aligned_stack{false};
        Path output_directory;
    };

    /// Initial global alignment.
    /// \details Updates the tilt-series geometry. The tilt-series should come with a good first
    ///          estimate of the angles, since this function can only slightly refine around
    ///          these first estimates.
    void initial_global_alignment(
            const Path& tilt_series_filename,
            MetadataStack& tilt_series_metadata,
            const LoadStackParameters& loading_parameters,
            const InitialGlobalAlignmentParameters& alignment_parameters,
            const GlobalRotationParameters& rotation_offset_parameters,
            const PairwiseShiftParameters& pairwise_shift_parameters,
            const ProjectionMatchingParameters& projection_matching_parameters,
            const SaveStackParameters& saving_parameters) {

        // TODO load stack in the "exposure" order.
        const auto [tilt_series, preprocessed_pixel_size, original_pixel_size] =
                load_stack(tilt_series_filename, tilt_series_metadata, loading_parameters);

        if (alignment_parameters.save_input_stack) {
            const auto cropped_stack_filename =
                    alignment_parameters.output_directory /
                    noa::string::format("{}_preprocessed{}",
                                        tilt_series_filename.stem().string(),
                                        tilt_series_filename.extension().string());
            noa::io::save(tilt_series, preprocessed_pixel_size.as<f32>(), cropped_stack_filename);
        }

        // Scale the metadata shifts to the alignment resolution.
        const auto pre_scale = (original_pixel_size / preprocessed_pixel_size).as<f32>();
        for (auto& slice: tilt_series_metadata.slices())
            slice.shifts *= pre_scale;

        // TODO tilt offset can be estimated using CC first. Also try the quick profile projection and COM?

        // Initial alignment using neighbouring views as reference.
        {
            const bool has_initial_rotation = MetadataSlice::UNSET_ROTATION_VALUE != tilt_series_metadata[0].angles[0];

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
                // Use area-matching only once we've got big shifts out of the picture.
                std::array shift_area_match{false, false, true, true};
                for (auto area_match: shift_area_match) {
                    pairwise_shift.update(
                            tilt_series, tilt_series_metadata,
                            pairwise_shift_parameters,
                            /*cosine_stretch=*/ false,
                            /*area_match=*/ area_match);
                }

                // Once we have estimates for the shifts, do the global rotation search.
                global_rotation.initialize(tilt_series_metadata, rotation_offset_parameters);
            } else {
                // If we have an estimate of the rotation from the user, use cosine-stretching
                // without area-matching to find estimates of the shifts.
                for (i64 i = 0; i < 2; ++i) {
                    pairwise_shift.update(
                            tilt_series, tilt_series_metadata,
                            pairwise_shift_parameters,
                            /*cosine_stretch=*/ true,
                            /*area_match=*/ false);
                }
            }

            // Trust the user and only allow a +-2deg offset on the user-provided rotation angle.
            const std::array<f32, 3> user_rotation_bounds{2, 1, 0.5};
            const std::array<f32, 3> estimate_rotation_bounds{10, 4, 0.5};

            // Once we have a first good estimate of the rotation and shifts, start again.
            // At each iteration the rotation should be better, improving the cosine stretching for the shifts.
            // Similarly, the shifts should improve, allowing a better estimate of the field-of-view and the rotation.
            for (auto i : noa::irange<size_t>(3)) {
                pairwise_shift.update(
                        tilt_series, tilt_series_metadata,
                        pairwise_shift_parameters,
                        /*cosine_stretch=*/ true,
                        /*area_match=*/ true);
                global_rotation.update(
                        tilt_series_metadata, rotation_offset_parameters,
                        has_initial_rotation ? user_rotation_bounds[i] : estimate_rotation_bounds[i]);
            }
            pairwise_shift.update(
                    tilt_series, tilt_series_metadata,
                    pairwise_shift_parameters,
                    /*cosine_stretch=*/ true,
                    /*area_match=*/ true); // TODO Double phase?
        }

        // TODO Estimate tilt and elevation offset using CTF.
        // TODO Rerun pairwise shift and rotation offset.

        // Projection matching alignment.
        // At this point, the global geometry should be pretty much on point.
        if (alignment_parameters.projection_matching_shift ||
            alignment_parameters.projection_matching_rotation) {
            auto projection_matching = qn::ProjectionMatching(
                    tilt_series.shape(), tilt_series.device(),
                    tilt_series_metadata, projection_matching_parameters);

            for (auto i : noa::irange<size_t>(1)) {
                projection_matching.update(
                        tilt_series, tilt_series_metadata,
                        projection_matching_parameters,
                        alignment_parameters.projection_matching_shift,
                        alignment_parameters.projection_matching_rotation);
            }
        }

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
}
