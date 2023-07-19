#include "quinoa/core/CTF.hpp"
#include "quinoa/core/GlobalRotation.hpp"
#include "quinoa/core/GridSearch1D.hpp"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/PairwiseShift.hpp"
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Stack.hpp"

namespace {
    using namespace ::qn;

    auto parse_angle_offset(const Options& options) {
        constexpr f64 MAX = std::numeric_limits<f64>::max();

        auto has_user_angle_offset = Vec3<bool>{
                !noa::math::are_almost_equal(MAX, options.tilt_scheme.rotation_offset),
                !noa::math::are_almost_equal(MAX, options.tilt_scheme.tilt_offset),
                !noa::math::are_almost_equal(MAX, options.tilt_scheme.elevation_offset),
        };
        auto angle_offset = Vec3<f64>{
                has_user_angle_offset[0] ? options.tilt_scheme.rotation_offset : 0,
                has_user_angle_offset[1] ? options.tilt_scheme.tilt_offset : 0,
                has_user_angle_offset[2] ? options.tilt_scheme.elevation_offset : 0,
        };
        const auto fit_angle_offset = Vec3<bool>{
                options.alignment.fit_rotation_offset,
                options.alignment.fit_tilt_offset,
                options.alignment.fit_elevation_offset,
        };

        return std::tuple{has_user_angle_offset, angle_offset, fit_angle_offset};
    }

    void save_average_ctf_(
            const CTFAnisotropic64& average_ctf,
            const std::array<f64, 3>& defocus_ramp,
            const Path& output_filename
    ) {
        YAML::Emitter out;
        out << YAML::BeginMap;

        out << YAML::Key << "defocus";
        out << YAML::Value << average_ctf.defocus().value;
        out << YAML::Comment("micrometers");

        out << YAML::Key << "defocus_astigmatism";
        out << YAML::Value << average_ctf.defocus().astigmatism;
        out << YAML::Comment("micrometers");

        out << YAML::Key << "defocus_angle";
        out << YAML::Value << noa::math::rad2deg(average_ctf.defocus().angle);
        out << YAML::Comment("degrees");

        out << YAML::Key << "phase_shift";
        out << YAML::Value << noa::math::rad2deg(average_ctf.phase_shift());
        out << YAML::Comment("degrees");

        out << YAML::Key << "defocus_ramp";
        out << YAML::Comment("micrometers");
        out << YAML::BeginSeq;
        out << YAML::Value << defocus_ramp[0] << YAML::Comment("above tilt-axis");
        out << YAML::Value << defocus_ramp[1] << YAML::Comment("at tilt-axis");
        out << YAML::Value << defocus_ramp[2] << YAML::Comment("below tilt-axis");
        out << YAML::EndSeq;

        out << YAML::EndMap;
        out << YAML::Newline;

        noa::io::save_text(out.c_str(), output_filename);
    }

    struct InitialPairwiseAlignmentParameters {
        Device compute_device;
        Path debug_directory;

        f64 maximum_resolution;
        Vec3<bool> fit_angle_offset;
        Vec3<bool> has_user_angle_offset;
    };

    void initial_pairwise_alignment(
            const Path& stack_filename,
            MetadataStack& metadata,
            const InitialPairwiseAlignmentParameters& parameters
    ) {
        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ Allocator::DEFAULT_ASYNC,
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

        if (Logger::is_debug()) {
            const auto cropped_stack_filename =
                    parameters.debug_directory /
                    noa::string::format("{}_preprocessed{}",
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
                !parameters.fit_angle_offset[0] ?
                GlobalRotation() :
                GlobalRotation(
                        tilt_series, metadata,
                        tilt_series.device(), rotation_parameters);

        if (!parameters.has_user_angle_offset[0]) {
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
        // At each iteration, the rotation should be better, improving the cosine stretching for the shifts.
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
                /*allocator=*/ Allocator::DEFAULT_ASYNC,
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

        if (Logger::is_debug()) {
            const auto aligned_stack_filename =
                    parameters.debug_directory /
                    noa::string::format("{}_coarse_aligned{}",
                                        stack_filename.stem().string(),
                                        stack_filename.extension().string());
            qn::save_stack(stack_filename, aligned_stack_filename,
                           metadata, saving_parameters,
                           noa::InterpMode::LINEAR_FAST);
        }
    }

    struct CTFAlignmentParameters {
        Device compute_device;
        Path output_directory;
        Path debug_directory;

        i64 patch_size;
        i64 patch_step;
        bool fit_phase_shift;
        bool fit_astigmatism;
        Vec2<f64> resolution_range;
        bool refine_pairwise_shift;

        // CTF.
        f64 voltage;
        f64 amplitude;
        f64 cs;
        f64 phase_shift;

        // Average fitting.
        Vec2<f64> delta_z_range_nanometers;
        f64 delta_z_shift_nanometers;
        f64 max_tilt_for_average;
        bool flip_rotation_to_match_defocus_ramp;

        // Global fitting.
        Vec3<bool> fit_angle_offset;
        f64 max_tilt_for_global;
    };

    auto ctf_alignment(
            const Path& stack_filename,
            const CTFAlignmentParameters& parameters,
            MetadataStack& metadata
    ) -> std::pair<CTFAnisotropic64, std::vector<f64>> {
        // Load stack
        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ Allocator::DEFAULT_ASYNC,
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
        auto stack_loader = StackLoader(stack_filename, loading_parameters);
        Vec2<f64> spacing_file = stack_loader.file_spacing();
        Vec2<f64> spacing_fitting = stack_loader.stack_spacing(); // isotropic

        auto grid = CTFFitter::Grid(stack_loader.slice_shape(), parameters.patch_size, parameters.patch_step);
        auto fitting_range = CTFFitter::FittingRange(
                parameters.resolution_range, noa::math::sum(spacing_fitting) / 2, parameters.patch_size);

        auto average_ctf = CTFAnisotropic64(
                spacing_fitting,
                /*defocus=*/ {},
                parameters.voltage,
                parameters.amplitude,
                parameters.cs,
                noa::math::deg2rad(parameters.phase_shift),
                /*bfactor=*/ 0
        );

        {
            // Fit on the average power-spectrum.
            auto metadata_for_average_fitting = metadata;
            rescale_shifts(metadata_for_average_fitting, spacing_file, spacing_fitting);
            std::array<f64, 3> defocus_ramp = CTFFitter::fit_average_ps(
                    stack_loader, grid, metadata_for_average_fitting, parameters.debug_directory / "ctf_average",
                    parameters.delta_z_range_nanometers, parameters.delta_z_shift_nanometers,
                    parameters.max_tilt_for_average, parameters.fit_phase_shift, parameters.fit_astigmatism,
                    parameters.compute_device,
                    fitting_range, // .background is updated
                    average_ctf // (astigmatic) .defocus and .phase_shift are updated.
            );

            // Check that the defocus ramp matches what we would expect from the rotation and tilt angles.
            const auto region_below_eucentric_has_higher_defocus = defocus_ramp[0] > defocus_ramp[1];
            const auto region_above_eucentric_has_lower_defocus = defocus_ramp[0] > defocus_ramp[1];

            if (region_below_eucentric_has_higher_defocus & region_above_eucentric_has_lower_defocus) {
                qn::Logger::info("Defocus ramp matches the angles. All good! "
                                 "defocus={::.3f} (below, at, and above eucentric height)",
                                 defocus_ramp);

            } else if (!region_below_eucentric_has_higher_defocus && !region_above_eucentric_has_lower_defocus) {
                if (parameters.flip_rotation_to_match_defocus_ramp) {
                    add_global_angles(metadata, {180, 0, 0});
                    std::swap(defocus_ramp[0], defocus_ramp[2]);
                    qn::Logger::info("Defocus ramp was reversed, so flipping rotation by 180 degrees. "
                                     "defocus={::.3f} (below, at, and above eucentric height)",
                                     defocus_ramp);
                } else {
                    qn::Logger::warn("Defocus ramp is reversed. defocus={::.3f} (below, at, and above eucentric height). "
                                     "This could be a really bad sign. Check that the rotation offset and tilt angles "
                                     "are correct, and make sure the images were not flipped",
                                     defocus_ramp);
                }
            } else {
                qn::Logger::warn("Defocus ramp isn't conclusive. defocus={::.3f} (below, at, and above eucentric height). "
                                 "This could be due to a lack of signal, but note that this isn't really expected, "
                                 "so please check your data and results carefully before proceeding");
            }

            // Save average fit to an output yaml file.
            save_average_ctf_(
                    average_ctf, defocus_ramp,
                    parameters.output_directory /
                    noa::string::format("{}_average_ctf.yaml", stack_filename.stem().string()));
        }

//        // For the global fitting, allow to remove high tilts to save memory.
//        auto metadata_for_global_fitting = metadata;
//        metadata_for_global_fitting.exclude(
//                [&](const MetadataSlice& slice) {
//                    return std::abs(slice.angles[1]) > parameters.max_tilt_for_global;
//                });
//        rescale_shifts(metadata_for_global_fitting, spacing_file, spacing_fitting);
//
//        // Compute the patches.
//        const CTFFitter::Patches patches_rfft_ps = CTFFitter::compute_patches_rfft_ps(
//                parameters.compute_device, stack_loader, metadata_for_global_fitting,
//                fitting_range, grid, debug_directory);
//
//        // From that point, everything is loaded, release the loader.
//        stack_loader = StackLoader();
//
//        // Initialize pairwise shift alignment, to refine the shifts after finding the stage angles.
//        struct PairwiseShiftAlignment {
//            MetadataStack metadata;
//            Array<f32> stack;
//            Vec2<f64> spacing;
//            PairwiseShift runner;
//            PairwiseShiftParameters parameters;
//        } pairwise_shift;
//
//        if (parameters.refine_pairwise_shift) {
//            const auto pairwise_loading_parameters = LoadStackParameters{
//                    /*compute_device=*/ parameters.compute_device,
//                    /*allocator=*/ Allocator::MANAGED,
//                    /*median_filter_window=*/ int32_t{1},
//                    /*precise_cutoff=*/ false,
//                    /*rescale_target_resolution=*/ 10.,
//                    /*rescale_min_size=*/ 1024,
//                    /*exposure_filter=*/ false,
//                    /*highpass_parameters=*/ {0.03, 0.03},
//                    /*lowpass_parameters=*/ {0.5, 0.05},
//                    /*normalize_and_standardize=*/ true,
//                    /*smooth_edge_percent=*/ 0.03f,
//                    /*zero_pad_to_fast_fft_shape=*/ true,
//            };
//            auto [stack, stack_spacing, _] = load_stack(stack_filename, metadata, loading_parameters);
//            pairwise_shift.metadata = metadata;
//            pairwise_shift.stack = stack;
//            pairwise_shift.spacing = stack_spacing;
//            pairwise_shift.runner = PairwiseShift(
//                    pairwise_shift.stack.shape(), parameters.compute_device, Allocator::MANAGED);
//            pairwise_shift.parameters = {
//                    /*highpass_filter=*/ {0.03, 0.03},
//                    /*lowpass_filter=*/ {0.3, 0.1},
//                    /*interpolation_mode=*/ noa::InterpMode::LINEAR_FAST,
//                    /*debug_directory=*/ parameters.debug_directory / "debug_pairwise_shifts",
//            };
//            rescale_shifts(pairwise_shift.metadata, spacing_file, pairwise_shift.spacing);
//        }
//
//        CTFFitter::GlobalFitOutput output;
//        for (i64 i = 0; i < 1 + parameters.refine_pairwise_shift; ++i) {
//            // Global fit.
//             output = CTFFitter::fit_ctf_to_patches(
//                    metadata_for_global_fitting, patches_rfft_ps,
//                    fitting_range, grid, average_ctf, global_fit,
//                    debug_directory
//            );
//
//            // Update and output.
//            add_global_angles(metadata_for_global_fitting, output.angle_offsets);
//            average_ctf.set_defocus({output.average_defocus, output.astigmatism_value, output.astigmatism_angle});
//            average_ctf.set_phase_shift(output.phase_shift);
//
//            // Scaling shift: pairwise->fitting
//            if (parameters.refine_pairwise_shift) {
//                // Update angles.
//                add_global_angles(pairwise_shift.metadata, output.angle_offsets);
//
//                pairwise_shift.runner.update(
//                        pairwise_shift.stack, pairwise_shift.metadata,
//                        pairwise_shift_parameters,
//                        /*cosine_stretch=*/ true,
//                        /*area_match=*/ true,
//                        /*smooth_edge_percent=*/ 0.08,
//                        /*max_size_loss_percent=*/ 0.1);
//
//                // Update shifts for the CTF fitting.
//                const auto scaling_factor = pairwise_shift.spacing / spacing_fitting;
//                for (const MetadataSlice& shifted_slice: pairwise_shift.metadata.slices()) {
//                    for (MetadataSlice& fitting_slice: metadata_for_global_fitting.slices()) {
//                        if (shifted_slice.index == fitting_slice.index) {
//                            fitting_slice.shifts = shifted_slice.shifts * scaling_factor;
//                        }
//                    }
//                }
//            }
//        }
//
//        // Update metadata.
//        if (parameters.refine_pairwise_shift) {
//            // Here the angle offsets are already added, so we just need to add the shifts.
//            rescale_shifts(pairwise_shift.metadata, pairwise_shift.spacing, spacing_file);
//            metadata = pairwise_shift.metadata;
//        } else {
//            add_global_angles(metadata, output.angle_offsets);
//        }

        // TODO print per-view defocus

        return {average_ctf, std::vector<f64>{}};
    }

//    struct ProjectionMatchingAlignmentParameters {
//        Device compute_device;
//        f64 maximum_resolution;
//
//        bool search_rotation{true};
//
//        Path output_directory;
//        Path debug_directory;
//    };
//
//    MetadataStack projection_matching_alignment(
//            const Path& tilt_series_filename,
//            const MetadataStack& tilt_series_metadata,
//            const ProjectionMatchingAlignmentParameters& parameters
//    ) {
//
//        auto loading_parameters = LoadStackParameters{
//                /*compute_device=*/ parameters.compute_device,
//                /*median_filter_window=*/ 1,
//                /*precise_cutoff=*/ false,
//                /*rescale_target_resolution=*/ -1.,
//                /*rescale_min_size=*/ 1024,
//                /*exposure_filter=*/ false,
//                /*highpass_parameters=*/ {0.03, 0.03},
//                /*lowpass_parameters=*/ {0.5, 0.05},
//                /*normalize_and_standardize=*/ true,
//                /*smooth_edge_percent=*/ 0.03f,
//                /*zero_pad_to_fast_fft_shape=*/ true,
//        };
//
//        auto projection_matching_parameters = ProjectionMatchingParameters{
//                /*zero_pad_to_fast_fft_size=*/ true,
//
//                /*rotation_tolerance_abs=*/ 0.005,
//                /*rotation_range=*/ 0,
//                /*rotation_initial_step=*/ 0.15,
//
//                /*smooth_edge_percent=*/ 0.1,
//
//                /*projection_slice_z_radius=*/ 0.0001f,
//                /*projection_cutoff=*/ 0.5f,
//                /*projection_max_tilt_angle_difference=*/ 120,
//
//                /*highpass_filter=*/ {0.08, 0.05},
//                /*lowpass_filter=*/ {0.4, 0.1}, // FIXME
//
//                /*allowed_shift_percent=*/ 0.005,
//
//                /*debug_directory=*/ parameters.debug_directory
//        };
//
//        //
//        MetadataStack metadata = tilt_series_metadata;
//
//        // To find the global rotation, start at low resolution and progressively decrease the resolution
//        // to focus and increase the accuracy of the grid search. The final iteration is there to find
//        // the best shifts, without any changes on the rotation.
//        std::vector target_size{512, 1024, 2048};
//        std::vector target_resolutions{20., 16., 10.};
//        std::vector cubic_spline_resolution{1, 1, 1};
//        std::vector global_rotation_bound{5., 2., 2.};
//        std::vector global_rotation_tolerance{0.05, 0.05, 0.01};
//
//        for (size_t i : noa::irange(3)) {
//
//            loading_parameters.rescale_target_resolution = 20.;
//            loading_parameters.rescale_min_size = target_size[i];
//
//            //
//            projection_matching_parameters.rotation_tolerance_abs = global_rotation_tolerance[i];
//            projection_matching_parameters.rotation_range = global_rotation_bound[i];
//            projection_matching_parameters.rotation_initial_step = global_rotation_bound[i] / 3;
//
//            // Load stack at the current resolution.
//
//            const auto [tilt_series, stack_spacing, file_spacing] =
//                    load_stack(tilt_series_filename, metadata, loading_parameters);
//            const auto stack = tilt_series.view();
//
//            // Rescale metadata to this resolution.
//            center_shifts(metadata);
//            rescale_shifts(metadata, file_spacing / stack_spacing);
//
//            // Set the common area. For projection matching it should be more important to make sure
//            // the views that do not fully contain the common area are excluded. So 1) allow smaller
//            // common areas to exclude fewer views, and 2) exclude the views that are still too far off.
//            constexpr f64 maximum_size_loss = 0.2; // FIXME Make this a user parameter
//            auto common_area = CommonArea(metadata.size(), stack.device());
//            const std::vector<i64> excluded_indexes = common_area.set_geometry(
//                    stack.shape().filter(2, 3), metadata, maximum_size_loss);
//            if (!excluded_indexes.empty())
//                metadata.exclude(excluded_indexes);
//
//            // TODO Clean fft cache
//            // TODO Move stack back to CPU to save GPU memory
//
//            auto projection_matching = ProjectionMatching(
//                    stack.shape().filter(2, 3),
//                    parameters.compute_device,
//                    metadata, projection_matching_parameters
//            );
//
//            // Initialize the rotation model.
//            auto cubic_spline_grid = CubicSplineGrid<f64, 1>(/*resolution=*/ cubic_spline_resolution[i]);
//            cubic_spline_grid.data()[0] = metadata[0].angles[0];
//
//            projection_matching.update(stack, metadata, common_area,
//                                       cubic_spline_grid,
//                                       projection_matching_parameters);
//
//            // Rescale metadata back to original resolution.
//            rescale_shifts(metadata, stack_spacing / file_spacing);
//        }
//
//        // Final centering of the alignment's field-of-view.
//        center_shifts(metadata);
//
//        const auto saving_parameters = LoadStackParameters{
//                parameters.compute_device, // Device compute_device
//                int32_t{0}, // int32_t median_filter_window
//                /*precise_cutoff=*/ true,
//                parameters.maximum_resolution,
//                1024,
//                false, // bool exposure_filter
//                {0.01, 0.01}, // float2_t highpass_parameters
//                {0.5, 0.05}, // float2_t lowpass_parameters
//                true, // bool normalize_and_standardize
//                0.02f, // float smooth_edge_percent
//        };
//
//        const auto aligned_stack_filename =
//                parameters.output_directory /
//                noa::string::format("{}_aligned{}",
//                                    tilt_series_filename.stem().string(),
//                                    tilt_series_filename.extension().string());
//        qn::save_stack(tilt_series_filename, aligned_stack_filename, metadata, saving_parameters);
//
//        return metadata;
//    }
}

namespace qn {
    auto align(
            const Options& options,
            const MetadataStack& metadata
    ) -> std::tuple<MetadataStack, CTFAnisotropic64, std::vector<f64>> {
        MetadataStack output_metadata = metadata;

        // First, extract the angle offsets.
        const auto [has_user_angle_offset, angle_offset, fit_angle_offset] = parse_angle_offset(options);
        QN_CHECK(has_user_angle_offset[0] || fit_angle_offset[0],
                 "An initial estimate of rotation-angle offset was not provided and "
                 "the rotation-angle offset alignment was turned off");
        add_global_angles(output_metadata, angle_offset);

        // 1. Initial pairwise alignment:
        //  -
        //  -
        qn::Logger::set_level("trace");
        const auto pairwise_alignment_parameters = InitialPairwiseAlignmentParameters{
                /*compute_device=*/ options.compute.device,
                /*debug_directory=*/ options.files.output_directory / "debug_pairwise_matching",
                /*maximum_resolution=*/ 10.,
                /*fit_angle_offset=*/ fit_angle_offset,
                /*has_user_angle_offset=*/ has_user_angle_offset,
        };
        initial_pairwise_alignment(options.files.input_stack, output_metadata, pairwise_alignment_parameters);

        // 2. CTF alignment:
        //  -
        //  -
        qn::Logger::set_level("debug");
        const auto ctf_alignment_parameters = CTFAlignmentParameters{
                /*compute_device=*/ options.compute.device,
                /*output_directory=*/ options.files.output_directory,
                /*debug_directory=*/ options.files.output_directory / "debug_ctf",

                /*patch_size=*/ 512,
                /*patch_step=*/ 256,
                /*fit_phase_shift=*/ options.alignment.fit_phase_shift,
                /*fit_astigmatism=*/ options.alignment.fit_astigmatism,
                /*resolution_range=*/ Vec2<f64>{ 40, 8 },
                /*refine_pairwise_shift=*/ options.alignment.use_pairwise_matching,

                /*voltage=*/ options.tilt_scheme.voltage,
                /*amplitude=*/ options.tilt_scheme.amplitude,
                /*cs=*/ options.tilt_scheme.cs,
                /*phase_shift=*/ options.tilt_scheme.phase_shift,

                /*delta_z_range_nanometers=*/ Vec2<f64>{-50, 50},
                /*delta_z_shift_nanometers=*/ 150,
                /*max_tilt_for_average=*/ 90,
                /*flip_rotation_to_match_defocus_ramp=*/ !has_user_angle_offset[0],

                /*fit_angle_offset=*/ fit_angle_offset,
                /*max_tilt_for_global=*/ 90,
        };
        auto [average_ctf, per_view_defocus] = ctf_alignment(
                options.files.input_stack,
                ctf_alignment_parameters,
                output_metadata
        );

//        const auto projection_matching_alignment_parameters = ProjectionMatchingAlignmentParameters{
//                /*compute_device=*/ options.compute.device,
//                /*maximum_resolution=*/ 10.,
//                /*search_rotation_offset=*/ options.alignment.fit_rotation_offset,
//                /*output_directory=*/ options.files.output_directory,
//                /*debug_directory=*/ options.files.output_directory / "debug_projection_matching"
//        };




//        const auto projection_matching_alignment_parameters = ProjectionMatchingAlignmentParameters{
//                /*compute_device=*/ compute_device,
//                /*maximum_resolution=*/ alignment_max_resolution,
//                //alignment_rotation_offset
//                /*search_rotation_offset=*/ options["alignment_rotation_offset"].as<bool>(true),
//                /*output_directory=*/ output_directory,
//                /*debug_directory=*/ ""//output_directory / "debug_projection_matching"
//
//
//                metadata = projection_matching_alignment(
//                original_stack_filename, metadata,
//                projection_matching_alignment_parameters);

        return {output_metadata, average_ctf, per_view_defocus};
    }
}
