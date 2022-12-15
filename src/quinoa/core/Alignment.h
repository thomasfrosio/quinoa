#pragma once

#include "quinoa/core/Metadata.h"
#include "quinoa/core/PairwiseCosine.h"
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Preprocessing.h"


namespace qn {
    struct InitialGlobalAlignmentParameters {
        bool do_pairwise_cosine_alignment{true};
        bool do_projection_matching_alignment{true};
        path_t debug_directory{};
    };

    /// Initial global alignment.
    /// \details Updates the tilt-series geometry. The tilt-series should come with a good estimate
    ///          of the view angles, since this function can only slightly refine around these first
    ///          estimates.
    void initialGlobalAlignment(
            const Array<float>& tilt_series,
            MetadataStack& tilt_series_metadata,
            const InitialGlobalAlignmentParameters& global_parameters,
            const PairwiseCosineParameters& pairwise_cosine_parameters,
            const ProjectionParameters& projection_matching_parameters) {

        // Cosine stretching alignment:
        if (global_parameters.do_pairwise_cosine_alignment) {
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
        if (global_parameters.do_projection_matching_alignment) {
//            auto projection_matching = qn::ProjectionMatching(tilt_series.shape(), global_parameters.compute_device);
//            projection_matching.align(tilt_series, tilt_series_metadata, projection_matching_parameters);
        }
    }
}
