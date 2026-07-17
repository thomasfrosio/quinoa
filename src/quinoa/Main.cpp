#include <noa/Runtime.hpp>
#include <noa/Session.hpp>

#include "quinoa/ExcludeViews.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/ctf/CTF.hpp"
#ifndef QN_CTF_ONLY
#   include "quinoa/align/Align.hpp"
#   include "quinoa/align/Thickness.hpp"
#   include "quinoa/align/Reconstruct.hpp"
#endif

namespace {
    using namespace qn;

    void process_data(const Settings& settings, const Series& series, const Device& device) {
        const auto basename = series.stem();
        const auto diagnostics_directory = series.output_directory / "quinoa-diagnostics" / basename;

        auto t = Logger::ScopeTimer{};
        if (not settings.compute.dry) {
            Logger::set_logfile(series.output_directory / fmt::format("{}.log", basename));
            t = Logger::status_scope_time("{}", basename);
        } else {
            Logger::info("{}:", basename);
        }
        Logger::trace(
            "  mdoc={}\n"
            "  stack={}\n"
            "  rawtlt={}\n"
            "  output={}{}",
            series.mdoc_file, series.stack_file,
            series.rawtlt_file.empty() ? "<none>" : series.rawtlt_file.native(),
            series.output_directory,
            settings.compute.dry ? "" : fmt::format("\n  device={}", device)
        );

        // Initialize the metadata early in case the parsing fails.
        // In dry mode, turn off console logging, except for errors and warnings, and return.
        auto metadata = Metadata::load_from_settings(settings, series);
        if (settings.compute.dry)
            return;

        // Create a user-async stream for the GPU and ensure that the CPU stream is synchronous.
        if (device.is_gpu()) {
            Session::set_gpu_lazy_loading();
            Device::set_current(device);
            Stream::set_current(Stream(device, Stream::ASYNC));
        }
        Session::set_thread_limit(settings.compute.n_threads);
        Stream::set_current(Stream({}, Stream::SYNC));

        // Register the input stack. The application loads the input stack many times. To save computation,
        // load the stack to memory once and save it inside a static array. The StackLoader will
        // check for it the next time it needs it.
        if (settings.compute.register_stack)
            StackLoader::register_input_stack(series.stack_file);

        if (settings.preprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Preprocessing");

            if (not settings.preprocessing.exclude_stack_indices.empty()) {
                metadata.stack.exclude_if([&](const auto& image) {
                    for (isize e: settings.preprocessing.exclude_stack_indices)
                        if (e == image.index) {
                            Logger::info("Excluding view: index={} (tilt={:+.2f})", image.index, image.angles[1]);
                            return true;
                        }
                    return false;
                });
            }

            // TODO Hot pixels correction
            // TODO Frame alignment

            if (settings.preprocessing.exclude_blank_views) {
                detect_and_exclude_blank_views(
                    series.stack_file, metadata.stack, {
                        .compute_device = device,
                        .output_directory = diagnostics_directory / "preprocessing",
                    });
            }
        }

        // Alignment.
        if (settings.alignment.coarse_run or settings.alignment.ctf_run or settings.alignment.refine_run) {
            auto scope_timer = Logger::status_scope_time("Alignment");

            if (settings.alignment.coarse_run) {
                #ifndef QN_CTF_ONLY
                coarse_alignment(
                    series.stack_file, metadata, {
                        .device = device,
                        .check_rotation = settings.alignment.coarse_check_rotation,
                        .fit_rotation_offset = settings.alignment.coarse_fit_rotation,
                        .fit_tilt_offset = settings.alignment.coarse_fit_tilt,
                        .fit_pitch_offset = settings.alignment.coarse_fit_pitch,
                        .output_directory = diagnostics_directory / "coarse",
                    }
                );
                #else
                Logger::warn("Build does not include tilt-series alignment");
                #endif
            }

            if (settings.alignment.ctf_run) {
                ctf::fit(
                    series.stack_file, metadata, {
                        .compute_device = device,
                        .output_directory = diagnostics_directory / "ctf",

                        .patch_size_ang = settings.alignment.ctf_patch_size_ang,
                        .patch_size_min_pix = settings.alignment.ctf_patch_size_min_pix,
                        .nb_images_in_initial_average = settings.alignment.ctf_nb_images_in_initial_average,
                        .max_nb_high_resolution_recovery = settings.alignment.ctf_max_nb_high_resolution_recovery,
                        .astigmatism_tilt_resolution =  settings.alignment.ctf_astigmatism_tilt_resolution.as<isize>(),
                        .phase_shift_time_resolution = settings.alignment.ctf_phase_shift_time_resolution.as<isize>(),
                        .resolution_range = settings.alignment.ctf_resolution_range,
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
                #ifndef QN_CTF_ONLY
                refine_alignment(
                    series.stack_file, metadata, {
                        .compute_device = device,
                        .correct_ctf = settings.alignment.refine_correct_ctf,
                        .phase_flip_strength = settings.alignment.refine_phase_flip_strength,
                        .fit_thickness = settings.alignment.refine_fit_thickness,
                        .fit_rotation_offset = settings.alignment.refine_fit_rotation,
                        .fit_tilt_offset = settings.alignment.refine_fit_tilt,
                        .fit_pitch_offset = settings.alignment.refine_fit_pitch,
                        .output_directory = diagnostics_directory / "refine",
                    }
                );
                #else
                Logger::warn("Build does not include tilt-series alignment");
                #endif
            }

            // Save the metadata.
            const auto star_filename = series.output_directory / fmt::format("{}.star", basename);
            metadata.save_star(star_filename);
            Logger::info("{} saved", star_filename);
        }

        // Postprocessing.
        if (settings.postprocessing.run) {
            #ifndef QN_CTF_ONLY
            auto scope_timer = Logger::status_scope_time("Postprocessing");
            post_processing(series.stack_file, metadata, {
                .compute_device = device,
                .target_resolution = settings.postprocessing.resolution,
                .min_size = 512,
                .output_directory = series.output_directory,
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
            #else
            Logger::warn("Build does not include reconstruction");
            #endif
        }
    }

    void distribute_work(const Settings& settings, std::vector<Series>& series) {
        const auto batch_processing = series.size() > 1;
        auto mutex = std::mutex{};
        auto remaining = std::ssize(series);
        auto work = [&] (Device device) {
            Logger::initialize();
            Logger::activate_console();
            auto ts = Series{};
            while (true) {
                {
                    const auto lock = std::scoped_lock(mutex);
                    if (series.empty())
                        return;
                    ts = std::move(series.back());
                    series.pop_back();
                }
                try {
                    if (batch_processing)
                        Logger::deactivate_console();

                    process_data(settings, ts, device);

                    if (batch_processing) {
                        Logger::activate_console();
                        const auto lock = std::scoped_lock(mutex);
                        Logger::info("Processing {} done. Remaining stacks: {}", ts.stem(), --remaining);
                    }
                } catch (...) {
                    Logger::activate_console();
                    Logger::error("Error occurred while processing {}:", ts.stem());
                    for (i32 i{}; auto& message : noa::Exception::backtrace())
                        Logger::error("[{}]: {}", i++, message);

                    if (settings.compute.stop_at_first_error) {
                        const auto lock = std::scoped_lock(mutex);
                        series.clear();
                    } else {
                        // To not lose this thread and continue the processing,
                        // reset the device, and get to the next tilt-series.
                        device.reset();
                    }
                }
            }
        };

        Logger::set_console_level(settings.compute.log_level);
        if (settings.compute.dry) {
            for (const auto& ts: series)
                process_data(settings, ts, settings.compute.devices[0]);
            return;
        }

        if (batch_processing) {
            Logger::info("Batch processing:");
            Logger::trace("  n_stacks={}\n  devices={}\n  output={}\n",
                series.size(), settings.compute.devices, series[0].output_directory);
            Logger::info("Running...");
        }

        // Create one worker per device.
        auto workers = noa::ThreadPool(settings.compute.devices.size());
        auto results = std::vector<std::future<void>>{};
        for (auto& device: settings.compute.devices)
            results.emplace_back(workers.enqueue(work, device));
        for (auto& result: results)
            result.get();
    }
}

auto main(int argc, char* argv[]) -> int {
    using namespace qn;

    try {
        Logger::initialize();
        Logger::activate_console();
        auto timer = Logger::status_scope_time<false>("Main");

        // Parse the settings and do the work.
        auto settings = Settings{};
        auto series = settings.parse(argc, argv);
        if (not series.empty())
            distribute_work(settings, series);

    } catch (...) {
        for (i32 i{}; auto& message : noa::Exception::backtrace())
            Logger::error("[{}]: {}", i++, message);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
