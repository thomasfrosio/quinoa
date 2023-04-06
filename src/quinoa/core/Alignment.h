#pragma once

#include "quinoa/core/Metadata.h"
#include "quinoa/core/PairwiseCosine.h"
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Stack.hpp"
#include "quinoa/core/YawFinder.h"

namespace qn {
    struct InitialGlobalAlignmentParameters {
        bool global_tilt_axis_angle{true};
        bool pairwise_cosine{true};
        bool projection_matching{true};

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
            const GlobalYawOffsetParameters& yaw_offset_parameters,
            const PairwiseCosineParameters& pairwise_cosine_parameters,
            const ProjectionMatchingParameters& projection_matching_parameters,
            const SaveStackParameters& saving_parameters) {

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

        // Initial alignment using neighbouring views as reference.
        {
            const bool has_initial_yaw = MetadataSlice::UNSET_YAW_VALUE != tilt_series_metadata[0].angles[0];

            auto pairwise_cosine = PairwiseCosine(
                    tilt_series.shape(), tilt_series.device(),
                    pairwise_cosine_parameters.interpolation_mode);

            auto global_yaw = GlobalYawSolver(
                    tilt_series, tilt_series_metadata,
                    tilt_series.device(), yaw_offset_parameters);

            if (!has_initial_yaw) {
                pairwise_cosine.update_shifts(tilt_series, tilt_series_metadata,
                                              pairwise_cosine_parameters,
                                              /*cosine_stretch=*/ false);
                global_yaw.initialize_yaw(tilt_series_metadata, yaw_offset_parameters);
                pairwise_cosine.update_shifts(tilt_series, tilt_series_metadata,
                                              pairwise_cosine_parameters,
                        /*cosine_stretch=*/ false);
                global_yaw.initialize_yaw(tilt_series_metadata, yaw_offset_parameters);
            }

            // Once we have a first estimate, start again. At each iteration the yaw should be better, improving
            // the cosine stretching for the shifts. Similarly, the shifts should improve, allowing a better estimate
            // of the common field-of-view.
            const std::array<f32, 3> yaw_bounds{5, 2, 1};
            for (auto yaw_bound: yaw_bounds) {
                if (alignment_parameters.pairwise_cosine) {
                    pairwise_cosine.update_shifts(
                            tilt_series, tilt_series_metadata,
                            pairwise_cosine_parameters);
                }
                if (alignment_parameters.global_tilt_axis_angle) {
                    global_yaw.update_yaw(
                            tilt_series_metadata, yaw_offset_parameters,
                            -yaw_bound, yaw_bound);
                }
            }
            if (alignment_parameters.pairwise_cosine) {
                pairwise_cosine.update_shifts(
                        tilt_series, tilt_series_metadata,
                        pairwise_cosine_parameters);
            }
        }

        // Projection matching alignment.
        // At this point, the global geometry should be pretty much on point.
        if (alignment_parameters.projection_matching) {
            auto projection_matching = qn::ProjectionMatching(
                    tilt_series.shape(), tilt_series.device());
            projection_matching.update_geometry(
                    tilt_series, tilt_series_metadata,
                    projection_matching_parameters);
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
