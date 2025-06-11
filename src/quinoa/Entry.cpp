#include <noa/Array.hpp>
#include <noa/Session.hpp>
#include <noa/Signal.hpp>
#include <noa/Utils.hpp>

// #include "Reconstruction.hpp"
#include "quinoa/Alignment.hpp"
#include "quinoa/ExcludeViews.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Stack.hpp"

// #include "quinoa/FiguresBCI.hpp"
// #include "quinoa/Tests.hpp"
#include "Thickness.hpp"
#include "quinoa/Test2.hpp"

auto main(int argc, char* argv[]) -> int {
    using namespace qn;

    // test_00();
    // test_01();
    // {
    //     auto slice_angles = noa::deg2rad(Vec{90., 40., 0.});
    //     auto slice_shape = Shape2<i64>{512, 512};
    //     auto slice_spacing = 1.;
    //     auto patch_center = Vec{512., 0.};
    //     const Mat<f64, 1, 3> patch2z = (
    //             ng::rotate_x(slice_angles[2]) *
    //             ng::rotate_y(slice_angles[1]) *
    //             ng::rotate_z(-slice_angles[0])
    //         ).filter_rows(0); // z only
    //
    //     const auto slice_center = (slice_shape / 2).vec.as<f64>();
    //     const auto scale = slice_spacing * 1e-4; // pixels->micrometers
    //     const auto patch_center_um = (patch_center - slice_center) * scale;
    //     const f64 patch_center_z_um = (patch2z * patch_center_um.push_front(0))[0];
    //     fmt::println("{}", patch_center_z_um);
    //     return EXIT_SUCCESS;
    // }
    // test03();
    // return EXIT_SUCCESS;

    try {
        // Initialize logger before doing anything else.
        Logger::initialize();
        auto timer = Logger::status_scope_time("Main");

        // Parse the settings.
        auto settings = Settings{};
        if (not settings.parse(argc, argv))
            return EXIT_SUCCESS;

        // Adjust global settings.
        Logger::add_logfile(settings.files.output_directory / "quinoa.log");
        Logger::set_level(settings.compute.log_level);
        noa::Session::set_gpu_lazy_loading();
        noa::Session::set_thread_limit(settings.compute.n_threads);

        // Create a user-async stream for the GPU, but ensure that the CPU stream is synchronous.
        if (settings.compute.device.is_gpu())
            noa::Stream::set_current(noa::Stream(settings.compute.device, noa::Stream::ASYNC));
        noa::Stream::set_current(noa::Stream({}, noa::Stream::DEFAULT));

        // Initialize the metadata early in case the parsing fails.
        auto metadata = MetadataStack::load_from_settings(settings);
        const auto basename = settings.files.stack_file.stem().string();

        // Register the input stack. The application loads the input stack many times. To save computation,
        // load the input (and unspoiled) stack once and save it inside a static array. The StackLoader will
        // check for it when needed. This is optional as the user may want to save memory...
        if (settings.compute.register_stack)
            StackLoader::register_input_stack(settings.files.stack_file);

        // Preprocessing.
        if (settings.preprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Preprocessing");

            if (not settings.preprocessing.exclude_stack_indices.empty()) {
                Logger::info("Excluding views: {}", settings.preprocessing.exclude_stack_indices);
                metadata.exclude_if([&](const MetadataSlice& slice) {
                    for (i64 e: settings.preprocessing.exclude_stack_indices)
                        if (e == slice.index)
                            return true;
                    return false;
                });
            }

            // TODO Remove hot pixels?

            if (settings.preprocessing.exclude_blank_views) {
                detect_and_exclude_blank_views(
                    settings.files.stack_file, metadata, {
                        .compute_device = settings.compute.device,
                        .output_directory = settings.files.output_directory,
                    });
            }
        }

        // Alignment.
        const bool has_user_rotation = not noa::allclose(
            settings.experiment.tilt_axis,
            std::numeric_limits<f64>::max()
        );
        if (settings.alignment.coarse_run or settings.alignment.ctf_run or settings.alignment.refine_run) {
            auto scope_timer = Logger::status_scope_time("Alignment");

            // 1. Coarse alignment:
            //  - Find the shifts, using the pairwise cosine-stretching alignment.
            //  - Find/refine the rotation offset, using the common-lines method.
            //  - Find the tilt offset, using a pairwise cosine-stretching alignment.
            if (settings.alignment.coarse_run) {
                const auto pairwise_alignment_parameters = CoarseAlignmentParameters{
                    .compute_device = settings.compute.device,
                    .maximum_resolution = 12.,
                    .fit_rotation_offset = settings.alignment.coarse_fit_rotation,
                    .fit_tilt_offset = settings.alignment.coarse_fit_tilt,
                    .has_user_rotation = has_user_rotation,
                    .output_directory = settings.files.output_directory,
                    .debug_directory = Logger::is_debug() ? settings.files.output_directory / "debug_coarse_alignment" : "",
                };
                coarse_alignment(
                    settings.files.stack_file,
                    metadata, // updated: .angles[0], .shifts
                    pairwise_alignment_parameters
                );
            }

            // 2. CTF alignment:
            //  - Coarse: per-slice defocus, one average phase-shift.
            //  - Refine: stage angles, per-slice defocus, time resolved astigmatism and phase shift.
            if (settings.alignment.ctf_run) {
                ctf_alignment(
                    settings.files.stack_file,
                    metadata, // updated: .angles[1, 2], .shifts
                    {
                        .compute_device = settings.compute.device,
                        .output_directory = settings.files.output_directory,

                        .voltage = settings.experiment.voltage,
                        .cs = settings.experiment.cs,
                        .amplitude = settings.experiment.amplitude,
                        .phase_shift = settings.experiment.phase_shift,

                        .patch_size_ang = 680,
                        .n_images_in_initial_average = 3,
                        .resolution_range = {40, 4.5},
                        .fit_phase_shift = settings.alignment.ctf_fit_phase_shift,
                        .fit_astigmatism = settings.alignment.ctf_fit_astigmatism,

                        // Coarse:
                        .has_user_rotation = has_user_rotation,

                        // Refine:
                        .fit_rotation = settings.alignment.ctf_fit_rotation, // false, // common-lines are more accurate
                        .fit_tilt = settings.alignment.ctf_fit_tilt,
                        .fit_pitch = settings.alignment.ctf_fit_pitch,
                    }
                );
            }

            // 3. Refine alignment.
            //  - The sample thickness is estimated (which requires a horizontal specimen).
            //  - TODO
            if (settings.alignment.refine_run) {
                // metadata = MetadataStack::load_from_csv(
                //     "/Users/cix56657/Tmp/quinoa/10304/001/tilt1_aligned.csv");
                estimate_sample_thickness(
                    settings.files.stack_file, metadata, {
                        .resolution = 30,
                        .initial_thickness_nm = 100,
                        .maximum_thickness_nm = 300,
                        .adjust_com = true,
                        .compute_device = Device{},
                        .allocator = Allocator::DEFAULT,
                        .debug_directory = settings.files.output_directory / "thickness"
                    });
            }

            // Save the metadata.
            const auto csv_filename = settings.files.output_directory / fmt::format("{}_aligned.csv", basename);
            const auto input_file = noa::io::ImageFile(settings.files.stack_file, {.read = true});
            metadata.save_csv(csv_filename, input_file.shape().pop_front<2>(), input_file.spacing().pop_front()); // TODO Add ctf
            Logger::info("{} saved", csv_filename);
        }

        // Postprocessing.
        if (settings.postprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Postprocessing");

            const auto [stack, stack_spacing, file_spacing, _] = load_stack(settings.files.stack_file, metadata, {
                .compute_device = settings.compute.device,
                .allocator = Allocator::DEFAULT_ASYNC,
                .precise_cutoff = true,
                .rescale_target_resolution = settings.postprocessing.resolution,
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

            if (settings.postprocessing.save_aligned_stack) {
                const auto filename = settings.files.output_directory / fmt::format("{}_aligned.mrc", basename);
                save_stack(stack.view(), stack_spacing, postprocessing_metadata, filename);
            }

            if (settings.postprocessing.reconstruct_tomogram) {
                // auto tomogram = tomogram_reconstruction(stack.view(), stack_spacing, postprocessing_metadata, {
                //     .sample_thickness_nm = sample_thickness_nm,
                //     .mode = settings.postprocessing.reconstruct_mode,
                //     .weighting = settings.postprocessing.reconstruct_weighting,
                //     .z_padding_percent = settings.postprocessing.reconstruct_z_padding,
                //     .cube_size = 128,
                //     .debug_directory = Logger::is_debug() ? settings.files.output_directory / "debug_reconstruction" : "",
                // });
                // const auto filename = settings.files.output_directory / fmt::format("{}_tomogram.mrc", basename);
                // noa::write(tomogram, filename);
                // Logger::info("{} saved", filename);
            }
        }
    } catch (...) {
        for (int i{}; auto& message : noa::Exception::backtrace())
            Logger::error("[{}]: {}", i++, message);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
