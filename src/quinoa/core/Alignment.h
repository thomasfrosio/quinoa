#pragma once

#include "quinoa/core/Metadata.h"
#include "quinoa/core/PairwiseCosine.h"
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Preprocessing.h"

namespace qn {
    struct InitialGlobalAlignmentParameters {
        bool save_preprocessed_stack{false};
        bool do_pairwise_cosine_alignment{true};
        bool do_projection_matching_alignment{true};
        bool save_aligned_stack{false};
        path_t output_directory;
    };

    /// Initial global alignment.
    /// \details Updates the tilt-series geometry. The tilt-series should come with a good first
    ///          estimate of the angles, since this function can only slightly refine around
    ///          these first estimates.
    void initialGlobalAlignment(
            const path_t& tilt_series_filename,
            MetadataStack& tilt_series_metadata,
            const LoadStackParameters& loading_parameters,
            const InitialGlobalAlignmentParameters& alignment_parameters,
            const PairwiseCosineParameters& pairwise_cosine_parameters,
            const ProjectionMatchingParameters& projection_matching_parameters,
            const SaveStackParameters& saving_parameters) {

        const auto [tilt_series, preprocessed_pixel_size, original_pixel_size] =
                loadStack(tilt_series_filename, tilt_series_metadata, loading_parameters);

        if (alignment_parameters.save_preprocessed_stack) {
            const auto cropped_stack_filename =
                    alignment_parameters.output_directory /
                    noa::string::format("{}_preprocessed{}",
                                        tilt_series_filename.stem().string(),
                                        tilt_series_filename.extension().string());
            noa::io::save(tilt_series, float2_t(preprocessed_pixel_size), cropped_stack_filename);
        }

        // Scale the metadata shifts to the alignment resolution.
        const auto pre_scale = float2_t(original_pixel_size / preprocessed_pixel_size);
        for (auto& slice: tilt_series_metadata.slices())
            slice.shifts *= pre_scale;

        // Stretching alignment:
        if (alignment_parameters.do_pairwise_cosine_alignment) {
            auto pairwise_cosine = qn::PairwiseCosine(
                    tilt_series.shape(), tilt_series.device(),
                    pairwise_cosine_parameters.interpolation_mode);
            pairwise_cosine.updateShifts(tilt_series, tilt_series_metadata, pairwise_cosine_parameters);

            // Once we have a first estimate, start again. At this point the average shift should
            // have been centered to 0, and we have a much better estimate of the common field-of-view.
            pairwise_cosine.updateShifts(tilt_series, tilt_series_metadata, pairwise_cosine_parameters);
            pairwise_cosine.updateShifts(tilt_series, tilt_series_metadata, pairwise_cosine_parameters);
        }

        // Projection matching alignment:
        if (alignment_parameters.do_projection_matching_alignment) {
            auto projection_matching = qn::ProjectionMatching(
                    tilt_series.shape(), tilt_series.device());
            projection_matching.align(tilt_series, tilt_series_metadata, projection_matching_parameters);
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
            qn::saveStack(tilt_series_filename, tilt_series_metadata, aligned_stack_filename, saving_parameters);
        }
    }
}
