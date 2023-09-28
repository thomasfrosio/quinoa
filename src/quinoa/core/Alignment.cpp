#include "quinoa/core/Alignment.h"
#include "quinoa/core/CTF.hpp"
#include "quinoa/core/GlobalRotation.hpp"
#include "quinoa/core/GridSearch1D.hpp"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/PairwiseShift.hpp"
#include "quinoa/core/ProjectionMatching.hpp"
#include "quinoa/core/Thickness.hpp"
#include "quinoa/core/Stack.hpp"

#include "quinoa/core/Ewise.hpp"

namespace {
    using namespace ::qn;

    auto parse_angle_offset(const Options& options) {
        constexpr f64 MAX = std::numeric_limits<f64>::max();

        auto has_user_angle_offset = Vec3<bool>{
                !noa::math::are_almost_equal(MAX, options.experiment.rotation_offset),
                !noa::math::are_almost_equal(MAX, options.experiment.tilt_offset),
                !noa::math::are_almost_equal(MAX, options.experiment.elevation_offset),
        };
        auto angle_offset = Vec3<f64>{
                has_user_angle_offset[0] ? options.experiment.rotation_offset : 0,
                has_user_angle_offset[1] ? options.experiment.tilt_offset : 0,
                has_user_angle_offset[2] ? options.experiment.elevation_offset : 0,
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
            const std::array<f64, 3>& ncc_ramp,
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
        out << YAML::BeginMap;

        std::array keys{"above", "at", "below"};
        for (size_t i = 0; i < 3; ++i) {
            out << YAML::Key << keys[i];
            out << YAML::BeginMap;
            out << YAML::Key << "defocus" << YAML::Value << defocus_ramp[i] << YAML::Comment("micrometers");
            out << YAML::Key << "ncc" << YAML::Value << ncc_ramp[i];
            out << YAML::EndMap;
        }
        out << YAML::EndMap;

        out << YAML::EndMap;
        out << YAML::Newline;

        noa::io::save_text(out.c_str(), output_filename);
    }

    void save_global_ctfs_(
            const MetadataStack& metadata,
            const CTFAnisotropic64& average_ctf,
            const Vec3<f64>& total_angles_offset,
            const Path& output_filename
    ) {
        YAML::Emitter out;
        out << YAML::BeginMap;

        out << YAML::Key << "total_angle_offsets";
        out << YAML::Value << total_angles_offset;
        out << YAML::Comment("in degrees");

        out << YAML::Key << "defocus";
        out << YAML::Value << average_ctf.defocus().value;
        out << YAML::Comment("in micrometers");

        out << YAML::Key << "defocus_astigmatism";
        out << YAML::Value << average_ctf.defocus().astigmatism;
        out << YAML::Comment("in micrometers");

        out << YAML::Key << "defocus_angle";
        out << YAML::Value << noa::math::rad2deg(average_ctf.defocus().angle);
        out << YAML::Comment("in degrees");

        out << YAML::Key << "phase_shift";
        out << YAML::Value << noa::math::rad2deg(average_ctf.phase_shift());
        out << YAML::Comment("in degrees");

        out << YAML::Key << "defoci" << YAML::Comment("(slice file-index, defocus in micrometers)");
        out << YAML::BeginMap;
        for (const auto& slice: metadata.slices())
            out << YAML::Key << slice.index_file << YAML::Value << slice.defocus;
        out << YAML::EndMap;

        out << YAML::EndMap;
        out << YAML::Newline;

        noa::io::save_text(out.c_str(), output_filename);
    }

    // Loads the input stack and wraps PairwiseShift.
    struct PairwiseShiftWrapper {
    private:
        MetadataStack m_metadata;
        Array<f32> m_stack;
        Vec2<f64> m_stack_spacing;
        Vec2<f64> m_file_spacing;
        PairwiseShift m_runner;
        PairwiseShiftParameters m_runner_parameters;

    public:
        PairwiseShiftWrapper() = default;

        PairwiseShiftWrapper(
                const LoadStackParameters& loading_parameters,
                const Path& stack_filename,
                const MetadataStack& metadata,
                const Path& debug_directory
        ) : m_metadata(metadata) {

            // Load the stack eagerly.
            auto [stack, stack_spacing, file_spacing] = qn::load_stack(stack_filename, m_metadata, loading_parameters);
            m_stack = stack;
            m_stack_spacing = stack_spacing;
            m_file_spacing = file_spacing;

            // Prepare for the pairwise alignment.
            m_metadata.rescale_shifts(m_file_spacing, m_stack_spacing);
            m_runner = PairwiseShift(stack.shape(), loading_parameters.compute_device, loading_parameters.allocator);
            m_runner_parameters = {
                    /*highpass_filter=*/ {0.03, 0.03},
                    /*lowpass_filter=*/ {0.3, 0.1},
                    /*interpolation_mode=*/ noa::InterpMode::LINEAR_FAST,
                    /*debug_directory=*/ debug_directory,
            };
        }

        void align(
                bool cosine_stretch,
                bool area_match,
                f64 smooth_edge_percent,
                f64 max_area_loss_percent,
                f64 max_shift_percent
        ) {
            m_runner.update(
                    stack(), metadata(), m_runner_parameters,
                    cosine_stretch, area_match, smooth_edge_percent,
                    max_area_loss_percent, max_shift_percent);
        }

        [[nodiscard]] auto metadata() -> MetadataStack& { return m_metadata; }
        [[nodiscard]] auto metadata() const -> const MetadataStack& { return m_metadata; }
        [[nodiscard]] auto stack() const -> const Array<f32>& { return m_stack; }
        [[nodiscard]] auto stack_spacing() const -> const Vec2<f64>& { return m_stack_spacing; }
    };
}

namespace {
    struct InitialPairwiseAlignmentParameters {
        Device compute_device;
        Path debug_directory;

        f64 maximum_resolution;
        Vec3<bool> fit_angle_offset;
        Vec3<bool> has_user_angle_offset;
    };

    auto initial_pairwise_alignment(
            const Path& stack_filename,
            MetadataStack& metadata,
            const InitialPairwiseAlignmentParameters& parameters
    ) -> f64 {
        noa::Timer timer;
        timer.start();
        qn::Logger::status("Initial alignment...");

        const auto debug = !parameters.debug_directory.empty();
        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ Allocator::DEFAULT_ASYNC,
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
                /*interpolation_mode=*/ noa::InterpMode::LINEAR_FAST,
                /*debug_directory=*/ debug ? parameters.debug_directory / "rotation_offset" : "",
        };

        const auto pairwise_shift_parameters = PairwiseShiftParameters{
                /*highpass_filter=*/ {0.03, 0.03},
                /*lowpass_filter=*/ {0.25, 0.1},
                /*interpolation_mode=*/ noa::InterpMode::LINEAR_FAST,
                /*debug_directory=*/  debug ? parameters.debug_directory / "pairwise_shift" : "",
        };

        // These alignments are quite robust, and we shouldn't need to
        // run multiple resolution cycles. Just load the stack once and align it.
        const auto [tilt_series, stack_spacing, file_spacing] =
                load_stack(stack_filename, metadata, loading_parameters);

        if (debug) {
            const auto cropped_stack_filename =
                    parameters.debug_directory /
                    noa::string::format("{}_preprocessed{}",
                                        stack_filename.stem().string(),
                                        stack_filename.extension().string());
            noa::io::save(tilt_series, stack_spacing.as<f32>(), cropped_stack_filename);
        }

        // Scale the metadata shifts to the current sampling rate.
        metadata.rescale_shifts(file_spacing, stack_spacing);

        // TODO Aretomo: tilt offset can be apparently estimated using a simple CC method?

        auto pairwise_shift = PairwiseShift(tilt_series.shape(), tilt_series.device());
        auto global_rotation =
                !parameters.fit_angle_offset[0] ?
                GlobalRotation() :
                GlobalRotation(
                        tilt_series, metadata, rotation_parameters,
                        tilt_series.device(), Allocator::DEFAULT_ASYNC);

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
                /*max_size_loss_percent=*/ 0.1);

        // Scale the metadata back to the original resolution.
        metadata.rescale_shifts(stack_spacing, file_spacing);

        const auto saving_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ Allocator::DEFAULT_ASYNC,
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

        if (debug) {
            const auto aligned_stack_filename =
                    parameters.debug_directory /
                    noa::string::format("{}_coarse_aligned{}",
                                        stack_filename.stem().string(),
                                        stack_filename.extension().string());
            qn::save_stack(stack_filename, aligned_stack_filename,
                           metadata, saving_parameters,
                           noa::InterpMode::LINEAR_FAST);
            qn::Logger::debug("{} saved", aligned_stack_filename);

            const Path csv_filename =
                    parameters.debug_directory /
                    noa::string::format("{}_coarse_aligned.csv", stack_filename.stem().string());
            metadata.save(csv_filename, tilt_series.shape().pop_front<2>(), file_spacing);
            qn::Logger::debug("{} saved", csv_filename);
        }

        qn::Logger::status("Initial alignment... done. Took {:.3f}s", timer.elapsed() * 1e-3);
        return noa::math::sum(stack_spacing) / 2;
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

        // Average fitting.
        Vec2<f64> delta_z_range_nanometers;
        f64 delta_z_shift_nanometers;
        f64 max_tilt_for_average;
        bool flip_rotation_to_match_defocus_ramp;

        // Global fitting.
        Vec3<bool> fit_angle_offset;
    };

    void ctf_alignment(
            const Path& stack_filename,
            const CTFAlignmentParameters& parameters,
            MetadataStack& metadata,
            CTFAnisotropic64& average_ctf
    ) {
        noa::Timer timer;
        timer.start();
        qn::Logger::status("CTF alignment...");

        const bool debug = !parameters.debug_directory.empty();
        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ Allocator::MANAGED,
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
        const Vec2<f64> spacing_file = stack_loader.file_spacing();
        const Vec2<f64> spacing_fitting = stack_loader.stack_spacing(); // precise_cutoff=true ensures it is isotropic
        const f64 spacing_fitting_iso = noa::math::sum(spacing_fitting) / 2;

        auto grid = CTFFitter::Grid(stack_loader.slice_shape(), parameters.patch_size, parameters.patch_step);
        auto fitting_range = CTFFitter::FittingRange(
                parameters.resolution_range, spacing_fitting_iso, parameters.patch_size);
        average_ctf.set_pixel_size(spacing_fitting);

        {
            // Fit on the average power-spectrum.
            auto metadata_for_average_fitting = metadata;
            metadata_for_average_fitting.rescale_shifts(spacing_file, spacing_fitting);
            auto [defocus_ramp, ncc_ramp] = CTFFitter::fit_average_ps(
                    stack_loader, grid, metadata_for_average_fitting,
                    debug ? parameters.debug_directory / "ctf_average" : "",
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
                    metadata.add_global_angles({180, 0, 0});
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
            if (debug) {
                const auto filename =
                        parameters.debug_directory /
                        noa::string::format("{}_average_ctf.yaml", stack_filename.stem().string());
                save_average_ctf_(average_ctf, defocus_ramp, ncc_ramp, filename);
                qn::Logger::debug("{} saved", filename);
            }
        }

        // At this point, we have a first estimate of the (astigmatic) defocus and background.
        // So prepare for the global fitting.
        auto metadata_for_global_fitting = metadata;
        metadata_for_global_fitting.rescale_shifts(spacing_file, spacing_fitting);

        // Initialize the vector of CTFs (one per slice, same order as in metadata) with the average CTF.
        for (auto& slice: metadata_for_global_fitting.slices())
            slice.defocus = average_ctf.defocus().value;

        const CTFFitter::GlobalFit global_fit{
                /*rotation=*/ parameters.fit_angle_offset[0],
                /*tilt=*/ parameters.fit_angle_offset[1],
                /*elevation=*/ parameters.fit_angle_offset[2],
                /*phase_shift=*/ parameters.fit_phase_shift,
                /*astigmatism=*/ parameters.fit_astigmatism,
        };

        // Compute the patches. This is where most of the memory is allocated.
        const CTFFitter::Patches patches_rfft_ps = CTFFitter::compute_patches_rfft_ps(
                parameters.compute_device, stack_loader, metadata_for_global_fitting,
                fitting_range, grid, parameters.debug_directory);

        // From that point, everything is loaded, so release the loader.
        stack_loader = StackLoader();

        // Cycle through stages of angle and shift refinements. The CTF fitting refines the stage angles,
        // and the pairwise alignment refines the shifts. Iterate since both steps depend on the global geometry.
        const auto pairwise_loading_parameters = LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ Allocator::MANAGED,
                /*precise_cutoff=*/ false,
                /*rescale_target_resolution=*/ 8.,
                /*rescale_min_size=*/ 1024,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.03, 0.03},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.03f,
                /*zero_pad_to_fast_fft_shape=*/ true,
        };
        auto pairwise_shift =
                !parameters.refine_pairwise_shift ?
                PairwiseShiftWrapper() :
                PairwiseShiftWrapper(pairwise_loading_parameters,stack_filename, metadata,
                                     debug ? parameters.debug_directory / "debug_pairwise_shifts" : "");

        Vec3<f64> total_angles_offset;
        for (i64 i = 0; i < 2 + parameters.refine_pairwise_shift * 2; ++i) {
            // Global fit. Metadata and ctfs are updated.
            const Vec3<f64> angles_offset = CTFFitter::fit_ctf_to_patches(
                    metadata_for_global_fitting, average_ctf,
                    patches_rfft_ps, fitting_range, grid, global_fit,
                    parameters.debug_directory);
            total_angles_offset += angles_offset;

            // Pairwise shift alignment.
            pairwise_shift.metadata().add_global_angles(angles_offset);
            pairwise_shift.align(
                    /*cosine_stretch=*/ true,
                    /*area_match=*/ true,
                    /*smooth_edge_percent=*/ 0.08,
                    /*max_area_loss_percent=*/ 0.1,
                    /*max_shift_percent=*/ 1);

            // Update metadata used for CTF fitting.
            metadata_for_global_fitting.update_shift_from(
                    pairwise_shift.metadata(),
                    pairwise_shift.stack_spacing(),
                    spacing_fitting);
        }

        // Update original metadata.
        metadata.add_global_angles(total_angles_offset);
        metadata.update_shift_from(pairwise_shift.metadata(), pairwise_shift.stack_spacing(), spacing_file);
        metadata.update_defocus_from(metadata_for_global_fitting);

        if (debug) {
            const Path csv_filename =
                    parameters.debug_directory /
                    noa::string::format("{}_ctf_aligned.csv", stack_filename.stem().string());
            metadata.save(csv_filename, grid.slice_shape(), spacing_file,
                          average_ctf.defocus().astigmatism,
                          noa::math::rad2deg(average_ctf.defocus().angle),
                          noa::math::rad2deg(average_ctf.phase_shift()));
            qn::Logger::debug("{} saved", csv_filename);
        }

        qn::Logger::status("Sample angles (in degrees): "
                           "rotation={:.3f} ({:+.3f}), tilt={:.3f} ({:+.3f}), elevation={:.3f} ({:+.3f})",
                           metadata[0].angles[0], total_angles_offset[0],
                           metadata[0].angles[1], total_angles_offset[1],
                           metadata[0].angles[2], total_angles_offset[2]);
        qn::Logger::status("CTF alignment... done. Took {:.3f}s", timer.elapsed() * 1e-3);
    }

    struct ProjectionMatchingAlignmentParameters {
        Device compute_device;
        Path debug_directory;
        f64 thickness_estimate_nm;
        i64 rotation_spline_resolution{0};
        f64 maximum_resolution{10};
        bool use_ctf{false};
    };

    auto projection_matching_alignment(
            const Path& stack_filename,
            const ProjectionMatchingAlignmentParameters& parameters,
            const CTFAnisotropic64& average_ctf,
            MetadataStack& metadata
    ) -> f64 {
        noa::Timer timer;
        timer.start();
        qn::Logger::status("Projection-matching alignment...");

        // 1. Initial run:
        //    Run projection-matching at a lower resolution and search for the best rotation.
        //    This should also pretty much solve for the remaining shifts.
        // 2. Final run:
        //    Run projection-matching at the maximum resolution, without searching for the rotation.
        std::array resolution{28., parameters.maximum_resolution};
        std::array min_size{512, 1024};
        std::array rotation_range{parameters.rotation_spline_resolution ? 5. : 0., 0.};

        f64 final_resolution{};
        for (auto i: noa::irange(resolution.size())) {
            // Load the stack. Make sure spacing is isotropic.
            const auto loading_parameters = LoadStackParameters{
                    /*compute_device=*/ parameters.compute_device,
                    /*allocator=*/ Allocator::MANAGED,
                    /*precise_cutoff=*/ true,
                    /*rescale_target_resolution=*/ resolution[i],
                    /*rescale_min_size=*/ min_size[i],
                    /*exposure_filter=*/ false,
                    /*highpass_parameters=*/ {0.01, 0.01},
                    /*lowpass_parameters=*/ {0.5, 0.01},
                    /*normalize_and_standardize=*/ true,
                    /*smooth_edge_percent=*/ 0.1f,
                    /*zero_pad_to_fast_fft_shape=*/ false,
            };
            const auto [stack, stack_spacing, file_spacing] =
                    load_stack(stack_filename, metadata, loading_parameters);
            metadata.center_shifts();
            metadata.rescale_shifts(file_spacing, stack_spacing);

            // Just in case memory is tight, make sure to free everything that is not needed,
            // even if it means recomputing it later.
            noa::Session::clear_fft_cache(parameters.compute_device);

            // Set the spacing and CTF.
            f64 spacing = noa::math::sum(stack_spacing) / 2;
            CTFAnisotropic64 ctf = average_ctf;
            ctf.set_pixel_size(stack_spacing);

            // Set the common area. For projection-matching, it should be more important to make sure
            // the views that do not fully contain the common area are excluded. So 1) allow smaller
            // common areas to exclude fewer views, and 2) exclude the views that are still too far off.
            auto common_area = CommonArea(stack.shape().filter(2, 3), metadata, 0.05); // FIXME

            final_resolution = spacing * 2;
            qn::Logger::info("Running projection-matching at resolution={:.2f}A, with rotation_range={:.2f}deg",
                             final_resolution, rotation_range[i]);

            auto projection_matching = ProjectionMatching(
                    metadata.ssize(),
                    stack.shape().filter(2, 3),
                    parameters.compute_device);

            // Set up and run the projection-matching.
            const f64 virtual_volume_size = static_cast<f64>(projection_matching.projection_size());
            const f64 fftfreq_sinc = 1 / virtual_volume_size;
            const f64 fftfreq_blackman = 4 * fftfreq_sinc;
            const f64 thickness_estimate_pixels = parameters.thickness_estimate_nm / (spacing * 1e-1);
            const f64 fftfreq_z_sinc = 1 / thickness_estimate_pixels;
            const f64 fftfreq_z_blackman = 4 * fftfreq_z_sinc;
            qn::Logger::trace("Fourier insertion and extraction bounds:\n"
                              "fftfreq_sinc={:.4f}\n"
                              "fftfreq_blackman={:.4f}\n"
                              "fftfreq_z_sinc={:.4f} (sample_thickness={}pixels)\n"
                              "fftfreq_z_blackman={:.4f} (window_size=~{}pixels)\n",
                              fftfreq_sinc, fftfreq_blackman,
                              fftfreq_z_sinc, std::round(thickness_estimate_pixels), fftfreq_z_blackman,
                              std::round(fftfreq_z_blackman * virtual_volume_size * 2 + 1));

            // FIXME Multiplying by the CTF is removing a lot of low frequencies (everything before the first peak).
            //       Since these frequencies seem to be _really_ useful for the alignment (they "anchor" the images
            //       and make sure images aren't aligned to high frequency noise, also orthogonal to the tilt we
            //       mostly only have low frequencies...), so turn off the CTF weighting for now.
            auto projection_matching_parameters = ProjectionMatchingParameters{
                    /*use_ctfs=*/ false, // FIXME parameters.use_ctf
                    /*rotation_range=*/ rotation_range[i],
                    /*rotation_spline_resolution=*/ parameters.rotation_spline_resolution,
                    /*shift_tolerance=*/ 0.001,
                    /*smooth_edge_percent=*/ 0.10,
                    /*fftfreq_sinc=*/ fftfreq_sinc,
                    /*fftfreq_blackman=*/ fftfreq_blackman,
                    /*fftfreq_z_sinc=*/ fftfreq_z_sinc,
                    /*fftfreq_z_blackman=*/ fftfreq_z_blackman,
                    /*highpass_filter=*/ {0.1, 0.03},
                    /*lowpass_filter=*/ {0.35, 0.15},
                    /*debug_directory=*/ parameters.debug_directory,
            };
            projection_matching.update(stack.view(), common_area, projection_matching_parameters, ctf, metadata);

            // Rescale back to original spacing and center.
            metadata.rescale_shifts(stack_spacing, file_spacing);
            metadata.center_shifts();
        }

        qn::Logger::status("Projection-matching alignment... done. Took {:.3f}s", timer.elapsed() * 1e-3);
        return final_resolution;
    }
}

namespace qn {
    auto tilt_series_alignment(
            const Options& options,
            const MetadataStack& metadata
    ) -> AlignmentOutputs {
        // First, extract the angle offsets.
        const auto [has_user_angle_offset, angle_offset, fit_angle_offset] = parse_angle_offset(options);
//        QN_CHECK(has_user_angle_offset[0] or fit_angle_offset[0],
//                 "An initial estimate of the rotation-angle offset was not provided and "
//                 "the rotation-angle offset alignment was turned off"); // FIXME

        AlignmentOutputs outputs;
        outputs.aligned_metadata = metadata;
        outputs.aligned_metadata.add_global_angles(angle_offset);

        // Then check the sample thickness can be computed.
        QN_CHECK(options.alignment.use_thickness_estimate or
                 options.experiment.thickness != std::numeric_limits<f64>::max(),
                 "An initial estimate of the sample thickness was not provided and "
                 "the sampled-thickness estimation was turned off"); // FIXME

        // 1. Initial pairwise alignment:
        //  - Find the shifts, using the pairwise cosine-stretching alignment.
        //  - Find/refine the rotation offset, using the global cosine-stretching method.
        if (options.alignment.use_initial_pairwise_alignment) {
            const auto pairwise_alignment_parameters = InitialPairwiseAlignmentParameters{
                    /*compute_device=*/ options.compute.device,
                    /*debug_directory=*/ qn::Logger::is_debug() ? options.files.output_directory / "debug_initial_alignment" : "",
                    /*maximum_resolution=*/ 12.,
                    /*fit_angle_offset=*/ fit_angle_offset, //{false, false, false}, // FIXME fit_angle_offset,
                    /*has_user_angle_offset=*/ has_user_angle_offset,
            };
            outputs.alignment_resolution = initial_pairwise_alignment(
                    options.files.input_stack,
                    outputs.aligned_metadata, // updated: .angles[0], .shifts
                    pairwise_alignment_parameters);
        }

        // 2. CTF alignment:
        //  - Fit the CTF to the average power spectrum. This outputs the (astigmatic) defocus and the phase shift.
        //  - Fit the CTF globally. This outputs a per-slice defocus, can refine the astigmatism and phase shift,
        //    but most importantly, returns the very accurate stage angle offsets (rotation, tilt, elevation).
        //  - The shifts are refined using the new angle offsets, using the pairwise cosine-stretching alignment.
        outputs.average_ctf = CTFAnisotropic64(
                /*pixel_size=*/ {0, 0},
                {0, options.experiment.astigmatism_value, noa::math::deg2rad(options.experiment.astigmatism_angle)},
                options.experiment.voltage,
                options.experiment.amplitude,
                options.experiment.cs,
                noa::math::deg2rad(options.experiment.phase_shift),
                /*bfactor=*/ 0,
                /*scale=*/ 1
        );
        if (options.alignment.use_ctf_estimate) {
            const auto ctf_alignment_parameters = CTFAlignmentParameters{
                    /*compute_device=*/ options.compute.device,
                    /*output_directory=*/ options.files.output_directory,
                    /*debug_directory=*/ qn::Logger::is_debug() ? options.files.output_directory / "debug_ctf" : "",

                    /*patch_size=*/ 1024,
                    /*patch_step=*/ 512, // FIXME
                    /*fit_phase_shift=*/ options.alignment.fit_phase_shift,
                    /*fit_astigmatism=*/ options.alignment.fit_astigmatism,
                    /*resolution_range=*/ Vec2<f64>{40, 8},
                    /*refine_pairwise_shift=*/ false, // FIXME

                    /*delta_z_range_nanometers=*/ Vec2<f64>{-50, 50},
                    /*delta_z_shift_nanometers=*/ 150,
                    /*max_tilt_for_average=*/ 90,
                    /*flip_rotation_to_match_defocus_ramp=*/ !has_user_angle_offset[0],

                    /*fit_angle_offset=*/ fit_angle_offset,
                    };
            ctf_alignment(
                    options.files.input_stack,
                    ctf_alignment_parameters,
                    outputs.aligned_metadata, // updated: .angles, .shifts, .defocus
                    outputs.average_ctf // updated: .pixel_size, .defocus, .phase_shift
            );
        }

        // 3. Sample thickness.
        outputs.sample_thickness_nm = options.experiment.thickness;
        if (options.alignment.use_thickness_estimate) {
            outputs.sample_thickness_nm = estimate_sample_thickness(options.files.input_stack, outputs.aligned_metadata, {
                    /*resolution=*/ 16.,
                    /*initial_thickness_nm=*/ outputs.sample_thickness_nm,
                    /*maximum_thickness_nm=*/ 400.,
                    /*adjust_com=*/ true,
                    /*compute_device=*/ options.compute.device,
                    /*allocator=*/ Allocator::MANAGED,
                    /*debug_directory=*/ qn::Logger::is_debug() ? options.files.output_directory / "debug_thickness" : "",
            });
        }

        // 4. Projection matching.
        if (options.alignment.use_projection_matching) {
            const auto projection_matching_alignment_parameters = ProjectionMatchingAlignmentParameters{
                    /*compute_device=*/ options.compute.device,
                    /*debug_directory=*/ "", // qn::Logger::is_debug() ? options.files.output_directory / "debug_projection_matching" : "",
                    /*thickness_estimate_nm=*/ outputs.sample_thickness_nm,
                    /*rotation_spline_resolution=*/ fit_angle_offset[0] ? options.alignment.rotation_spline_resolution : 0,
                    /*maximum_resolution=*/ 10.,
                    /*use_ctf=*/ options.alignment.use_ctf_estimate,
            };

            outputs.alignment_resolution = projection_matching_alignment(
                    options.files.input_stack,
                    projection_matching_alignment_parameters,
                    outputs.average_ctf,
                    outputs.aligned_metadata // updated: .angles[1], .shifts
            );
        }

        const Path csv_filename =
                options.files.output_directory /
                noa::string::format("{}.csv", options.files.input_stack.stem().string());
        const auto input_file = noa::io::ImageFile(options.files.input_stack, noa::io::READ);
        outputs.aligned_metadata.save(
                csv_filename,
                input_file.shape().pop_front<2>(),
                input_file.pixel_size().pop_front().as<f64>(),
                outputs.average_ctf.defocus().astigmatism,
                noa::math::rad2deg(outputs.average_ctf.defocus().angle),
                noa::math::rad2deg(outputs.average_ctf.phase_shift()));
        qn::Logger::info("{} saved", csv_filename);

        return outputs;
    }
}
