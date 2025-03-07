#include "quinoa/Alignment.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/PairwiseShift.hpp"
#include "quinoa/PairwiseTilt.hpp"
#include "quinoa/RotationOffset.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Thickness.hpp"
#include "quinoa/CTF.hpp"

namespace {
    using namespace ::qn;

    // // Loads the input stack and wraps PairwiseShift.
    // struct PairwiseShiftWrapper {
    // private:
    //     MetadataStack m_metadata;
    //     Array<f32> m_stack;
    //     Vec2<f64> m_stack_spacing;
    //     Vec2<f64> m_file_spacing;
    //     PairwiseShift m_runner;
    //     PairwiseShiftParameters m_runner_parameters;
    //
    // public:
    //     PairwiseShiftWrapper() = default;
    //
    //     PairwiseShiftWrapper(
    //             const LoadStackParameters& loading_parameters,
    //             const Path& stack_filename,
    //             const MetadataStack& metadata,
    //             const Path& debug_directory
    //     ) : m_metadata(metadata) {
    //
    //         // Load the stack eagerly.
    //         auto [stack, stack_spacing, file_spacing] = qn::load_stack(stack_filename, m_metadata, loading_parameters);
    //         m_stack = stack;
    //         m_stack_spacing = stack_spacing;
    //         m_file_spacing = file_spacing;
    //
    //         // Prepare for the pairwise alignment.
    //         m_metadata.rescale_shifts(m_file_spacing, m_stack_spacing);
    //         m_runner = PairwiseShift(stack.shape(), loading_parameters.compute_device, loading_parameters.allocator);
    //         m_runner_parameters = {
    //                 /*highpass_filter=*/ {0.03, 0.03},
    //                 /*lowpass_filter=*/ {0.3, 0.1},
    //                 /*interpolation_mode=*/ noa::InterpMode::LINEAR_FAST,
    //                 /*debug_directory=*/ debug_directory,
    //         };
    //     }
    //
    //     void align(
    //             bool cosine_stretch,
    //             bool area_match,
    //             f64 smooth_edge_percent,
    //             f64 max_area_loss_percent,
    //             f64 max_shift_percent
    //     ) {
    //         m_runner.update(
    //                 stack(), metadata(), m_runner_parameters,
    //                 cosine_stretch, area_match, smooth_edge_percent,
    //                 max_area_loss_percent, max_shift_percent);
    //     }
    //
    //     [[nodiscard]] auto metadata() -> MetadataStack& { return m_metadata; }
    //     [[nodiscard]] auto metadata() const -> const MetadataStack& { return m_metadata; }
    //     [[nodiscard]] auto stack() const -> const Array<f32>& { return m_stack; }
    //     [[nodiscard]] auto stack_spacing() const -> const Vec2<f64>& { return m_stack_spacing; }
    // };

    // inline void save_average_ctf(
    //     const ns::CTFAnisotropic<f64>& average_ctf,
    //     const std::array<f64, 3>& defocus_ramp,
    //     const std::array<f64, 3>& ncc_ramp,
    //     const Path& output_filename
    // ) {
    //     YAML::Emitter out;
    //     out << YAML::BeginMap;
    //
    //     out << YAML::Key << "defocus";
    //     out << YAML::Value << average_ctf.defocus().value;
    //     out << YAML::Comment("micrometers");
    //
    //     out << YAML::Key << "defocus_astigmatism";
    //     out << YAML::Value << average_ctf.defocus().astigmatism;
    //     out << YAML::Comment("micrometers");
    //
    //     out << YAML::Key << "defocus_angle";
    //     out << YAML::Value << noa::rad2deg(average_ctf.defocus().angle);
    //     out << YAML::Comment("degrees");
    //
    //     out << YAML::Key << "phase_shift";
    //     out << YAML::Value << noa::rad2deg(average_ctf.phase_shift());
    //     out << YAML::Comment("degrees");
    //
    //     out << YAML::Key << "defocus_ramp";
    //     out << YAML::BeginMap;
    //
    //     std::array keys{"above", "at", "below"};
    //     for (size_t i = 0; i < 3; ++i) {
    //         out << YAML::Key << keys[i];
    //         out << YAML::BeginMap;
    //         out << YAML::Key << "defocus" << YAML::Value << defocus_ramp[i] << YAML::Comment("micrometers");
    //         out << YAML::Key << "ncc" << YAML::Value << ncc_ramp[i];
    //         out << YAML::EndMap;
    //     }
    //     out << YAML::EndMap;
    //
    //     out << YAML::EndMap;
    //     out << YAML::Newline;
    //
    //     noa::write_text(out.c_str(), output_filename);
    // }
    //
    // inline void save_global_ctfs_(
    //     const MetadataStack& metadata,
    //     const ns::CTFAnisotropic<f64>& average_ctf,
    //     const Vec3<f64>& total_angles_offset,
    //     const Path& output_filename
    // ) {
    //     YAML::Emitter out;
    //     out << YAML::BeginMap;
    //
    //     out << YAML::Key << "total_angle_offsets";
    //     out << YAML::Value << total_angles_offset;
    //     out << YAML::Comment("in degrees");
    //
    //     out << YAML::Key << "defocus";
    //     out << YAML::Value << average_ctf.defocus().value;
    //     out << YAML::Comment("in micrometers");
    //
    //     out << YAML::Key << "defocus_astigmatism";
    //     out << YAML::Value << average_ctf.defocus().astigmatism;
    //     out << YAML::Comment("in micrometers");
    //
    //     out << YAML::Key << "defocus_angle";
    //     out << YAML::Value << noa::rad2deg(average_ctf.defocus().angle);
    //     out << YAML::Comment("in degrees");
    //
    //     out << YAML::Key << "phase_shift";
    //     out << YAML::Value << noa::rad2deg(average_ctf.phase_shift());
    //     out << YAML::Comment("in degrees");
    //
    //     out << YAML::Key << "defoci" << YAML::Comment("(slice file-index, defocus in micrometers)");
    //     out << YAML::BeginMap;
    //     for (const auto& slice: metadata)
    //         out << YAML::Key << slice.index_file << YAML::Value << slice.defocus;
    //     out << YAML::EndMap;
    //
    //     out << YAML::EndMap;
    //     out << YAML::Newline;
    //
    //     noa::write_text(out.c_str(), output_filename);
    // }
}

namespace qn {
    auto coarse_alignment(
        const Path& stack_filename,
        MetadataStack& metadata,
        const CoarseAlignmentParameters& parameters
    ) -> f64 {
        auto timer = Logger::status_scope_time("Coarse alignment");

        // These alignments are quite robust, and we shouldn't need to
        // run multiple resolution cycles. Instead, load the stack once and align it.
        // TODO Run very low res and higher res? some tilt-series are very noisy.
        //      Use lowpass filter in the pairwise search to progressively allow higher res?
        const auto [tilt_series, stack_spacing, file_spacing] = load_stack(stack_filename, metadata, {
            .compute_device = parameters.compute_device,
            .allocator = Allocator::DEFAULT_ASYNC,

            // Fourier cropping:
            .precise_cutoff = false,
            .rescale_target_resolution = std::max(parameters.maximum_resolution, 28.), // FIXME
            .rescale_min_size = 512,

            // Signal processing after cropping:
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.03,
                .highpass_width = 0.03,
                .lowpass_cutoff = 0.50,
                .lowpass_width = 0.05,
            },

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        const auto basename = stack_filename.stem().string();
        const auto extension = stack_filename.extension().string();
        const auto debug = not parameters.debug_directory.empty();

        if (debug) {
            const auto filename = parameters.debug_directory / fmt::format("{}_preprocessed{}", basename, extension);
            noa::write(tilt_series, stack_spacing, filename);
            Logger::debug("{} saved", filename);
        }

        // Scale the metadata shifts to the current sampling rate.
        metadata.rescale_shifts(file_spacing, stack_spacing);

        auto rotation_parameters = RotationOffsetParameters{
            .interp = noa::Interp::LINEAR,
            .bandpass{
                .highpass_cutoff = 0.05,
                .highpass_width = 0.05,
                .lowpass_cutoff = 0.25, // FIXME
                .lowpass_width = 0.15,
            },
            .grid_search_line_range = 1.2,
            .grid_search_line_delta = 0.1,
            .local_search_line_range = 1.2,
            .local_search_line_delta = 0.1,
            .local_search_using_estimated_gradient = false,
            .output_directory = parameters.output_directory,
            .debug_directory = debug ? parameters.debug_directory / "rotation_offset" : "",
        };

        const auto shift_parameters = PairwiseShiftParameters{
            .bandpass{
                .highpass_cutoff = 0.03,
                .highpass_width = 0.03,
                .lowpass_cutoff = 0.25, // FIXME
                .lowpass_width = 0.10,
            },
            .interp = noa::Interp::LINEAR_FAST,
            .debug_directory = "", //debug ? parameters.debug_directory / "pairwise_shift" : "", // FIXME
        };

        auto shift_fitter = PairwiseShift(tilt_series.shape(), tilt_series.device());
        auto rotation_fitter = parameters.fit_rotation_offset ? RotationOffset(tilt_series.view(), metadata) : RotationOffset();

        // TODO Detect for view with huge shifts and remove them?

        bool has_rotation = parameters.has_user_rotation;
        bool has_tilt{};
        for (auto smooth_edge_percent: std::array{0.08, 0.3}) {
            // First, get the large shifts out of the way. Once these are removed, focus on the center.
            // If we have an estimate of the rotation from the user, use cosine-stretching but don't use
            // area-matching yet in case of large shifts.
            shift_fitter.update(tilt_series.view(), metadata, shift_parameters, {
                .cosine_stretch = has_rotation,
                .area_match = false,
                .smooth_edge_percent = smooth_edge_percent,
            });

            // Once we have estimates for the shifts, do the rotation search.
            // If we don't have an initial rotation from the user, do a full search.
            // Otherwise, refine whatever rotation the user gave us.
            rotation_parameters.reset_rotation = not has_rotation;
            rotation_parameters.grid_search_range = not has_rotation ? 90. : 10.;
            rotation_parameters.grid_search_step = not has_rotation ? 1. : 0.5;
            rotation_parameters.local_search_range = 5.;
            rotation_fitter.search(tilt_series.view(), metadata, rotation_parameters);
            has_rotation = true;

            // Once we have an estimate for the rotation, do the tilt search.
            if (parameters.fit_tilt_offset) {
                coarse_fit_tilt(tilt_series.view(), metadata, {
                    .grid_search_range = not has_tilt ? 25. : 5.,
                    .grid_search_step = not has_tilt ? 1. : 0.2,
                });
                has_tilt = true;
            }
        }

        // Once we have a first good estimate of the rotation and shifts, start again using the common area masks.
        // At each iteration, the rotation should be better, improving the cosine stretching for the shifts.
        // Similarly, the shifts should get better, allowing a better estimate of the common area and the rotation.
        for (auto rotation_bound: std::array{2.5, 1.25}) {
            shift_fitter.update(tilt_series.view(), metadata, shift_parameters, {
                .cosine_stretch = true,
                .area_match = true,
                .smooth_edge_percent = 0.08,
                .max_shift_percent = 0.1,
            });

            rotation_parameters.grid_search_range = rotation_bound;
            rotation_parameters.grid_search_step = 0.25;
            rotation_parameters.local_search_range = 1.;
            rotation_parameters.local_search_line_delta = 0.05;
            rotation_fitter.search(tilt_series.view(), metadata, rotation_parameters);

            if (parameters.fit_tilt_offset) {
                coarse_fit_tilt(tilt_series.view(), metadata, {
                    .grid_search_range = 4.,
                    .grid_search_step = 0.1,
                });
            }
        }

        shift_fitter.update(tilt_series.view(), metadata, shift_parameters, {
            .cosine_stretch = true,
            .area_match = true,
            .smooth_edge_percent = 0.08,
            .max_shift_percent = 0.1,
        });

        if (debug) {
            const auto filename = parameters.output_directory / fmt::format("{}_coarse_aligned{}", basename, extension);
            save_stack(tilt_series.view(), stack_spacing, metadata, filename);
            Logger::debug("{} saved", filename);
        }

        // Scale the metadata back to the original resolution.
        metadata.rescale_shifts(stack_spacing, file_spacing);

        const Path csv_filename = parameters.output_directory / fmt::format("{}_coarse_aligned.csv", basename);
        metadata.save_csv(csv_filename, tilt_series.shape().pop_front<2>(), file_spacing);
        Logger::info("{} saved", csv_filename);

        return noa::mean(stack_spacing);
    }

    auto ctf_alignment(
        const Path& stack_filename,
        MetadataStack& metadata,
        const CTFAlignmentParameters& parameters
    ) -> f64 {
        auto timer = Logger::status_scope_time("CTF alignment");

        auto stack_loader = StackLoader(stack_filename, {
            .compute_device = parameters.compute_device,
            .allocator = Allocator::MANAGED,

            // Fourier cropping:
            .precise_cutoff = true, // ensure isotropic spacing
            .rescale_target_resolution = 0, // load at original spacing
            .rescale_min_size = 512,

            // Signal processing after cropping:
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.02,
                .highpass_width = 0.02,
                .lowpass_cutoff = 0.50,
                .lowpass_width = 0.05,
            },

            // Image processing after cropping:
            .normalize_and_standardize = true, // TODO do we need any kind of preprocessing here?
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });

        // Resolution to fftfreq range.
        // The patch is Fourier cropped to the integer frequency closest to the resolution.
        const auto spacing = mean(stack_loader.stack_spacing()); // assume isotropic spacing by this point
        const auto [fourier_cropped_size, fourier_cropped_fftfreq] = fourier_crop_to_resolution(
            parameters.patch_size, spacing, parameters.resolution_range[1]
        );
        const auto fftfreq_range = Vec{
            resolution_to_fftfreq(spacing, parameters.resolution_range[0]),
            fourier_cropped_fftfreq,
        };
        Logger::info(
            "CTF fitting frequency range:\n"
            "  resolution_range={::.3f}A\n"
            "  fftfreq_range={::.5f}",
            fftfreq_to_resolution(spacing, fftfreq_range),
            fftfreq_range
        );

        const auto grid = ctf::Grid(stack_loader.slice_shape(), parameters.patch_size, parameters.patch_size / 2);
        const auto patches = ctf::Patches::from_stack(stack_loader, metadata, grid, fourier_cropped_size);

        auto input_ctf = ns::CTFAnisotropic<f64>({
            .pixel_size = stack_loader.stack_spacing(),
            .defocus = {0, parameters.astigmatism_value, parameters.astigmatism_angle},
            .voltage = parameters.voltage,
            .amplitude = parameters.amplitude,
            .cs = parameters.cs,
            .phase_shift = parameters.phase_shift,
            .bfactor = -200,
            .scale = 1.,
        });
        auto ctf = ns::CTFIsotropic(input_ctf);

        ctf::coarse_fit(
            grid, patches, ctf, metadata, {
                .fftfreq_range = fftfreq_range,
                .fit_phase_shift = false,
                .has_user_rotation = parameters.has_user_rotation,
            });


        // auto [defocus_ramp, ncc, fourier_crop_size, fftfreq_range, average_ctf, background] =
        //     ctf::coarse_fit_average_ps(stack_loader, grid, metadata, input_ctf, {
        //         .delta_z_range_nanometers = parameters.delta_z_range_nanometers,
        //         .delta_z_shift_nanometers = parameters.delta_z_shift_nanometers,
        //         .max_tilt_for_average = parameters.max_tilt_for_average,
        //         .resolution_range = parameters.resolution_range,
        //         .fit_envelope = parameters.fit_envelope,
        //         .fit_phase_shift = parameters.fit_phase_shift,
        //         .fit_astigmatism = parameters.fit_astigmatism,
        //         .debug_directory = parameters.debug_directory / "ctf_coarse",
        //     });
        //
        // const auto [defocus_at, defocus_below, defocus_above] = defocus_ramp;
        // Logger::info("defocus={::.3f}, ncc={::.3f} (at, below, and above)", defocus_ramp, ncc);
        //
        // // Check that the defocus ramp matches what we would expect from the rotation and tilt angles.
        // const auto region_below_eucentric_has_higher_defocus = defocus_below > defocus_at;
        // const auto region_above_eucentric_has_lower_defocus = defocus_above < defocus_at;
        // if (region_below_eucentric_has_higher_defocus and region_above_eucentric_has_lower_defocus) {
        //     Logger::info("Defocus ramp matches the stage angles");
        // } else if (not region_below_eucentric_has_higher_defocus and not region_above_eucentric_has_lower_defocus) {
        //     if (not parameters.has_user_rotation) { // the rotation is from common-lines
        //         metadata.add_global_angles({180, 0, 0});
        //         std::swap(defocus_ramp[1], defocus_ramp[2]);
        //         Logger::info("Defocus ramp is reversed, so flipping the computed rotation by 180 degrees");
        //     } else {
        //         Logger::warn(
        //             "Defocus ramp is reversed. This is a bad sign!\n"
        //             "Check that the rotation offset and tilt angles are correct, "
        //             "and make sure the images were not flipped along one axis"
        //         );
        //     }
        // } else {
        //     Logger::warn(
        //         "Defocus ramp isn't conclusive. This could be due to a lack of signal, "
        //         "but is likely a bad sign, so please check your data and results carefully before proceeding"
        //     );
        // }

        // // Save average fit to an output yaml file.
        // const bool debug = not parameters.debug_directory.empty();
        // if (debug) {
        //     const auto filename =
        //             parameters.debug_directory /
        //             fmt::format("{}_average_ctf.yaml", stack_filename.stem().string());
        //     save_average_ctf_(average_ctf, defocus_ramp, ncc_ramp, filename);
        //     qn::Logger::debug("{} saved", filename);
        // }

        // // Collect the patches.
        // auto patches = ctf::compute_patches_ps(stack_loader, grid, metadata, fourier_crop_size);
        //
        //
        // // Set the phase-shift and astigmatism as splines.
        // // These will be updated (and their resolution will be increased) by the CTF refinement.
        // auto phase_shift = CubicSplineGrid<f64, 1>(1);
        // phase_shift.span()[0][0] = average_ctf.phase_shift();
        // auto astigmatism = CubicSplineGrid<f64, 1>(1, 2);
        // phase_shift.span()[0][0] = average_ctf.defocus().astigmatism;
        // phase_shift.span()[1][0] = average_ctf.defocus().angle;
        //
        // auto isotropic_ctf = ns::CTFIsotropic(average_ctf);
        // auto rotational_average = Array<f32>(patches.shape().width(), {
        //     .device = parameters.compute_device,
        //     .allocator = Allocator::ASYNC,
        // });
        //
        // const auto fitting = ctf::FitRefineOptions{
        //     .fit_rotation = parameters.fit_rotation,
        //     .fit_tilt = parameters.fit_tilt,
        //     .fit_pitch = parameters.fit_pitch,
        //     .fit_phase_shift = parameters.fit_phase_shift,
        //     .fit_astigmatism = parameters.fit_astigmatism,
        // };
        //
        // ctf::refine_fit_patches_ps(
        //     metadata, grid, patches, fftfreq_range, isotropic_ctf,
        //     background, phase_shift, astigmatism, fitting,
        //     rotational_average.view()
        // );

        // TODO save rotational_average (background subtracted) with simulated CTF.
        // TODO save defocus and splines.

        return 0;
    }
}
