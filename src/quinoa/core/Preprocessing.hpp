//#include "quinoa/core/Metadata.h"
//#include "quinoa/core/Stack.hpp"
//#include "quinoa/core/YawFinder.h"
//
//namespace qn {
//    struct PreProcessingParameters {
//        bool find_tilt_axis_angle{true};
//        bool find_tilt_angle_offset{true};
//        bool exclude_blank_views{true};
//        std::vector<i32> exclude_view_indexes{true};
//        f64 absolute_max_tilt_difference{16};
//        Path output_directory;
//    };
//
//    void preprocess(
//            const Path& tilt_series_filename,
//            MetadataStack& tilt_series_metadata,
//            const LoadStackParameters& loading_parameters,
//            const PreProcessingParameters& pre_processing_parameters) {
//
//        const auto [tilt_series, preprocessed_pixel_size, original_pixel_size] =
//                load_stack(tilt_series_filename, tilt_series_metadata, loading_parameters);
//
//        // Scale the metadata shifts to the alignment resolution.
//        const auto pre_scale = (original_pixel_size / preprocessed_pixel_size).as<f32>();
//        for (auto& slice: tilt_series_metadata.slices())
//            slice.shifts *= pre_scale;
//
//        // Rotation angle:
//        if (pre_processing_parameters.find_tilt_axis_angle) {
//
//        }
//
//        // Scale the metadata back to the original resolution.
//        const auto post_scale = 1 / pre_scale;
//        for (auto& slice: tilt_series_metadata.slices())
//            slice.shifts *= post_scale;
//
//
//    }
//}
