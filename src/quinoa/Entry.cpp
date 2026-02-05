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
#include "quinoa/Reconstruct.hpp"

namespace {
    using namespace qn;

    // auto test01() {
    //     const auto spacing = 0.2;
    //     const auto angles = noa::deg2rad(Vec{0., 60., 60.});
    //     const auto plane_rotation = ( // TODO check
    //         nx::rotate_z(angles[0]) *
    //         nx::rotate_y(angles[1]) *
    //         nx::rotate_x(angles[2])
    //     );
    //     const auto plane_normal = (plane_rotation * Vec{1., 0., 0.}).as<f32>();
    //
    //     auto image = Array<f32>({1, 1, 100, 100});
    //     auto span = image.span_contiguous<f32, 2>();
    //     noa::iwise(span.shape(), image.device(), [&](Vec<isize, 2> indices) {
    //         auto coordinates = indices.as<f32>() - (image.shape().filter(2, 3).vec / 2).as<f32>();
    //         const auto& [c, b, a] = plane_normal;
    //         const auto volume_z_coordinate = -(a * coordinates[1] + b * coordinates[0]) / c;
    //         const auto volume_z_coordinate_nm = volume_z_coordinate * 1;
    //         span(indices) = volume_z_coordinate_nm;
    //     });
    //
    //     noa::write_image(image, "~/Tmp/image_z.mrc");
    // }
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

        // test01();
        // return EXIT_SUCCESS;

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
                        .output_directory = settings.files.output_directory / "diagnostics" / "preprocessing",
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
                        .resolution_range = {30, 4.}, // TODO 4.5?
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
                        .correct_ctf = settings.alignment.refine_correct_ctf,
                        .phase_flip_strength = settings.alignment.refine_phase_flip_strength,
                        .fit_thickness = settings.alignment.refine_fit_thickness,
                        .fit_rotation_offset = settings.alignment.refine_fit_rotation,
                        .fit_tilt_offset = settings.alignment.refine_fit_tilt,
                        .fit_pitch_offset = settings.alignment.refine_fit_pitch,
                        .output_directory = settings.files.output_directory / "diagnostics" / "refine",
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
            post_processing(settings.files.stack_file, metadata, {
                .compute_device = settings.compute.device,
                .target_resolution = settings.postprocessing.resolution,
                .min_size = 512,
                .output_directory = settings.files.output_directory,
                .save_aligned_stack = settings.postprocessing.stack_run,
                .stack_dtype = settings.postprocessing.stack_dtype,
                .stack_correct_rotation = settings.postprocessing.stack_correct_rotation,
                .stack_interp = settings.postprocessing.stack_interpolation,

                .save_tomogram = settings.postprocessing.tomogram_run,
                .tomogram_dtype = settings.postprocessing.tomogram_dtype,
            }, {
                .ramp_filter = settings.postprocessing.tomogram_ramp_filter,
                .correct_ctf = settings.postprocessing.tomogram_correct_ctf,
                .phase_flip_strength = settings.postprocessing.tomogram_phase_flip_strength,
                .defocus_step_nm = 15,
            }, {
                .algorithm = settings.postprocessing.tomogram_algorithm,
                .z_padding_percent = settings.postprocessing.tomogram_z_padding_percent / 100,
                .correct_rotation = settings.postprocessing.tomogram_correct_rotation,
                .oversampling_factor = settings.postprocessing.tomogram_oversampling_factor,
                .interp = settings.postprocessing.tomogram_interpolation,
            });
        }
    } catch (...) {
        for (i32 i{}; auto& message : noa::Exception::backtrace())
            Logger::error("[{}]: {}", i++, message);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
