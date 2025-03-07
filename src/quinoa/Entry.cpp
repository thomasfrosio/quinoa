#include <noa/Session.hpp>
#include <noa/Utils.hpp>

// #include "quinoa/core/CommonArea.hpp"
// #include "quinoa/core/Reconstruction.hpp"

#include "Reconstruction.hpp"
#include "quinoa/Alignment.hpp"
#include "quinoa/ExcludeViews.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Options.hpp"
#include "quinoa/Stack.hpp"

// #include "quinoa/Tests.hpp"

auto main(int argc, char* argv[]) -> int {
    using namespace qn;

    // test_pairwise_shift_data();
    // test_reconstruct();
    // return 0;

    try {
        // Initialize logger before doing anything else.
        Logger::initialize();
        auto timer = Logger::status_scope_time("Main");

        // Parse the options.
        auto options = Options(argc, argv);

        // Adjust global settings.
        Logger::add_logfile(options.files.output_directory / "quinoa.log");
        Logger::set_level(options.compute.log_level);
        noa::Session::set_gpu_lazy_loading();
        noa::Session::set_thread_limit(options.compute.n_cpu_threads);

        // Create a user-async stream for the GPU, but ensure that the CPU stream is synchronous.
        if (options.compute.device.is_gpu())
            noa::Stream::set_current(noa::Stream(options.compute.device, noa::Stream::ASYNC));
        noa::Stream::set_current(noa::Stream({}, noa::Stream::DEFAULT));

        // Initialize the metadata early in case the parsing fails.
        auto metadata = MetadataStack(options);
        const auto basename = options.files.input_stack.stem().string();

        // Register the input stack. The application loads the input stack many times. To save computation,
        // load the input (and unspoiled) stack once and save it inside a static array. The StackLoader will
        // check for it when needed. This is optional as the user may want to save memory...
        if (options.compute.register_input_stack)
            StackLoader::register_input_stack(options.files.input_stack);

        // Preprocessing.
        if (options.preprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Preprocessing");

            if (not options.preprocessing.exclude_view_indexes.empty()) {
                Logger::info("Excluding views: {}", options.preprocessing.exclude_view_indexes);
                metadata.exclude([&](const MetadataSlice& slice) {
                    for (i64 e: options.preprocessing.exclude_view_indexes)
                        if (e == slice.index)
                            return true;
                    return false;
                });
            }

            // TODO Remove hot pixels?

            if (options.preprocessing.exclude_blank_views) {
                detect_and_exclude_blank_views(
                    options.files.input_stack, metadata, {
                        .compute_device = options.compute.device,
                        .resolution = 20.,
                    });
            }
        }

        // Alignment.
        const bool has_user_rotation = not noa::allclose(
            options.experiment.rotation_offset,
            std::numeric_limits<f64>::max()
        );
        f64 sample_thickness_nm = options.experiment.thickness;
        if (options.alignment.run) {
            auto scope_timer = Logger::status_scope_time("Alignment");

            // 1. Coarse alignment:
            //  - Find the shifts, using the pairwise cosine-stretching alignment.
            //  - Find/refine the rotation offset, using the common-lines method.
            //  - Find the tilt offset, using a pairwise cosine-stretching alignment.
            if (options.alignment.do_coarse_alignment) {
                const auto pairwise_alignment_parameters = CoarseAlignmentParameters{
                    .compute_device = options.compute.device,
                    .maximum_resolution = 12.,
                    .fit_rotation_offset = options.alignment.fit_rotation_offset,
                    .fit_tilt_offset = options.alignment.fit_tilt_offset,
                    .has_user_rotation = has_user_rotation,
                    .output_directory = options.files.output_directory,
                    .debug_directory = Logger::is_debug() ? options.files.output_directory / "debug_coarse_alignment" : "",
                };
                coarse_alignment(
                    options.files.input_stack,
                    metadata, // updated: .angles[0], .shifts
                    pairwise_alignment_parameters
                );
            }

            // 2. CTF alignment:
            //  - Fit the CTF to the average power spectrum. This outputs the (astigmatic) defocus and the phase shift.
            //  - Fit the CTF globally. Outputs a per-slice defocus, can refine the astigmatism and phase shift,
            //    but most importantly, returns stage angle offsets (tilt, pitch).
            //  - The sample thickness is estimated (which requires a horizontal specimen).
            //  - The shifts are refined using low-res projection matching (which requires the thickness estimate).
            if (options.alignment.do_ctf_alignment) {
                ctf_alignment(
                    options.files.input_stack,
                    metadata, // updated: .angles[1, 2], .shifts
                    {
                        .compute_device = options.compute.device,
                        .output_directory = options.files.output_directory,
                        .debug_directory = options.files.output_directory / "debug_ctf_alignment",

                        .voltage = options.experiment.voltage,
                        .cs = options.experiment.cs,
                        .amplitude = options.experiment.amplitude,
                        .astigmatism_value = options.experiment.astigmatism_value,
                        .astigmatism_angle = options.experiment.astigmatism_angle,
                        .phase_shift = options.experiment.phase_shift,

                        .patch_size = 512,
                        .resolution_range = {30, 8},
                        .fit_envelope = false,
                        .fit_phase_shift = options.alignment.fit_phase_shift,
                        .fit_astigmatism = options.alignment.fit_astigmatism,

                        // Coarse:
                        .delta_z_range_nanometers = {-50., 50.},
                        .delta_z_shift_nanometers = 150.,
                        .max_tilt_for_average = 90.,
                        .has_user_rotation = has_user_rotation,

                        // Refine:
                        .fit_rotation = options.alignment.fit_rotation_offset,
                        .fit_tilt = options.alignment.fit_tilt_offset,
                        .fit_pitch = options.alignment.fit_pitch_offset,
                    }
                );
            }

            // 3. Refine alignment.
            if (options.alignment.do_refine_alignment) {
            }

            // Save the metadata.
            const auto csv_filename = options.files.output_directory / fmt::format("{}_aligned.csv", basename);
            const auto input_file = noa::io::ImageFile(options.files.input_stack, {.read = true});
            metadata.save_csv(csv_filename, input_file.shape().pop_front<2>(), input_file.spacing().pop_front()); // TODO Add ctf
            Logger::info("{} saved", csv_filename);
        }

        // Postprocessing.
        if (options.postprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Postprocessing");

            const auto [stack, stack_spacing, file_spacing] = load_stack(options.files.input_stack, metadata, {
                .compute_device = options.compute.device,
                .allocator = Allocator::DEFAULT_ASYNC,
                .precise_cutoff = true,
                .rescale_target_resolution = options.postprocessing.resolution,
                .rescale_min_size = 1024,
                .exposure_filter = false,
                .bandpass{
                    .highpass_cutoff = 0.01,
                    .highpass_width = 0.01,
                    .lowpass_cutoff = 0.49,
                    .lowpass_width = 0.01,
                },
                .normalize_and_standardize = true,
                .smooth_edge_percent = 0.02,
                .zero_pad_to_fast_fft_shape = false,
                .zero_pad_to_square_shape = false,
            });
            auto postprocessing_metadata = metadata;
            postprocessing_metadata.rescale_shifts(file_spacing, stack_spacing);

            if (options.postprocessing.save_aligned_stack) {
                const auto filename = options.files.output_directory / fmt::format("{}_aligned.mrc", basename);
                save_stack(stack.view(), stack_spacing, postprocessing_metadata, filename);
            }

            if (options.postprocessing.reconstruct_tomogram) {
                auto tomogram = tomogram_reconstruction(stack.view(), stack_spacing, postprocessing_metadata, {
                    .sample_thickness_nm = sample_thickness_nm,
                    .mode = options.postprocessing.reconstruct_mode,
                    .weighting = options.postprocessing.reconstruct_weighting,
                    .z_padding_percent = options.postprocessing.reconstruct_z_padding,
                    .cube_size = 128,
                    .debug_directory = Logger::is_debug() ? options.files.output_directory / "debug_reconstruction" : "",
                });
                const auto filename = options.files.output_directory / fmt::format("{}_tomogram.mrc", basename);
                noa::write(tomogram, filename);
                Logger::info("{} saved", filename);
            }
        }
    } catch (...) {
        for (int i{}; auto& message : noa::Exception::backtrace())
            fmt::print("[{}]: {}\n", i++, message);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
