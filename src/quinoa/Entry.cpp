#include <noa/Runtime.hpp>
#include <noa/Session.hpp>
#include <noa/Signal.hpp>

#include "quinoa/align/Align.hpp"
#include "quinoa/ctf/CTF.hpp"

#include "quinoa/ExcludeViews.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Thickness.hpp"
#include "quinoa/PostProcessing.hpp"

// #include "quinoa/Tests.hpp"

auto process_tilt_series() {

}

auto main(int argc, char* argv[]) -> int {
    using namespace qn;

    try {
        // Initialize the logger before doing anything else.
        Logger::initialize();
        auto timer = Logger::status_scope_time<false>("Main");

        // Parse the settings.
        auto settings = Settings{};
        if (not settings.parse(argc, argv))
            return EXIT_SUCCESS;

        // Adjust global settings.
        Logger::add_logfile(settings.files.output_directory / "quinoa.log");
        Logger::set_level(settings.compute.log_level);
        Session::set_gpu_lazy_loading();
        Session::set_thread_limit(settings.compute.n_threads);

        // Create a user-async stream for the GPU and ensure that the CPU stream is synchronous.
        if (settings.compute.device.is_gpu())
            Stream::set_current(Stream(settings.compute.device, Stream::DEFAULT)); // FIXME ASYNC
        Stream::set_current(Stream({}, Stream::SYNC));

        // Initialize the metadata early in case the parsing fails.
        auto metadata = Metadata::load_from_settings(settings);
        const auto basename = settings.files.stack_file.stem().string();

        // tests::test_ctf_grid();
        // tests::simulate_tilt_series();
        // tests::test_stage_leveling();
        // tests::test_find_shifts();
        // tests::test_star_file();
        // tests::test_common_fov2();
        // tests::test_image_cross_correlation();
        // tests::test_frc();
        // return 0;

        // Register the input stack. The application loads the input stack many times. To save computation,
        // load the stack to memory once and save it inside a static array. The StackLoader will
        // check for it next time it needs it.
        // TODO By default, register only if file.is_compressed?
        if (settings.compute.register_stack)
            StackLoader::register_input_stack(settings.files.stack_file);

        if (settings.preprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Preprocessing");

            if (not settings.preprocessing.exclude_stack_indices.empty()) {
                Logger::info("Excluding views: {}", settings.preprocessing.exclude_stack_indices);
                metadata.stack.exclude_if([&](const auto& image) {
                    for (isize e: settings.preprocessing.exclude_stack_indices)
                        if (e == image.index)
                            return true;
                    return false;
                });
            }

            // TODO Frame alignment

            if (settings.preprocessing.exclude_blank_views) {
                detect_and_exclude_blank_views(
                    settings.files.stack_file, metadata.stack, {
                        .compute_device = settings.compute.device,
                        .output_directory = settings.files.output_directory,
                    });
            }
        }

        // Alignment.
        if (settings.alignment.coarse_run or settings.alignment.ctf_run or settings.alignment.refine_run) {
            auto scope_timer = Logger::status_scope_time("Alignment");

            if (settings.alignment.coarse_run) {
                coarse_alignment(
                    settings.files.stack_file, metadata, {
                        .device = settings.compute.device,
                        .check_rotation = settings.alignment.coarse_check_rotation,
                        .fit_rotation_offset = settings.alignment.coarse_fit_rotation,
                        .fit_tilt_offset = settings.alignment.coarse_fit_tilt,
                        .fit_pitch_offset = settings.alignment.coarse_fit_pitch,
                        .output_directory = settings.files.output_directory / "diagnostics" / "coarse",
                    }
                );
            }

            if (settings.alignment.ctf_run) {
                ctf::fit(
                    settings.files.stack_file, metadata, {
                        .compute_device = settings.compute.device,
                        .output_directory = settings.files.output_directory / "diagnostics" / "ctf",

                        .patch_size_ang = 680,
                        .n_images_in_initial_average = 3,
                        .resolution_range = {30, 4.}, // FIXME 4.5
                        .fit_phase_shift = settings.alignment.ctf_fit_phase_shift,
                        .fit_astigmatism = settings.alignment.ctf_fit_astigmatism,
                        .fit_thickness = settings.alignment.ctf_fit_thickness,
                        .check_defocus_gradient = settings.alignment.ctf_check_defocus_gradient,

                        .fit_rotation = settings.alignment.ctf_fit_rotation,
                        .fit_tilt = settings.alignment.ctf_fit_tilt,
                        .fit_pitch = settings.alignment.ctf_fit_pitch,
                    }
                );
            }

            if (settings.alignment.refine_run) {
                refine_alignment(
                    settings.files.stack_file, metadata, {
                        .compute_device = settings.compute.device,
                        .maximum_resolution = 12.,
                        .fit_rotation_offset = settings.alignment.coarse_fit_rotation,
                        .fit_tilt_offset = settings.alignment.coarse_fit_tilt,
                        .output_directory = settings.files.output_directory,
                    }
                );
            }

            // Save the metadata.
            const auto star_filename = settings.files.output_directory / fmt::format("{}.star", basename);
            metadata.save_star(star_filename);
            Logger::info("{} saved", star_filename);
        }

        // Postprocessing.
        if (settings.postprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Postprocessing");
            post_processing(settings.files.stack_file, metadata,
                {
                    .compute_device = settings.compute.device,
                    .target_resolution = settings.postprocessing.resolution,
                    .min_size = 512,
                    .output_directory = settings.files.output_directory,
                }, {
                    .save_aligned_stack = settings.postprocessing.stack_run,
                    .correct_rotation = settings.postprocessing.stack_correct_rotation,
                    .interp = settings.postprocessing.stack_interpolation,
                    .dtype = settings.postprocessing.stack_dtype,
                }, {
                    .save_tomogram = settings.postprocessing.tomogram_run,
                    .correct_ctf = settings.postprocessing.tomogram_correct_ctf,
                    .phase_flip_strength = settings.postprocessing.tomogram_phase_flip_strength,
                    .defocus_step_nm = 15,
                    .z_padding_percent = settings.postprocessing.tomogram_z_padding_percent / 100,
                    .correct_rotation = settings.postprocessing.tomogram_correct_rotation,
                    .oversample = settings.postprocessing.tomogram_oversample,
                    .interp = settings.postprocessing.tomogram_interpolation,
                    .dtype = settings.postprocessing.tomogram_dtype,
            });
        }
    } catch (...) {
        for (i32 i{}; auto& message : noa::Exception::backtrace())
            Logger::error("[{}]: {}", i++, message);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
