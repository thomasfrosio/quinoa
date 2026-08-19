#include <noa/Runtime.hpp>
#include <noa/Session.hpp>

#ifndef QN_CTF_ONLY
#   include "quinoa/align/Align.hpp"
#   include "quinoa/Thickness.hpp"
#   include "quinoa/Reconstruct.hpp"
#endif

#include "quinoa/ExcludeViews.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/ctf/CTF.hpp"

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

        // Capture the stack shape and spacing for when exporting to IMOD.
        auto exported_image_shape = Shape2{};
        auto exported_spacing = Vec<f64, 2>{};
        if (settings.compute.register_stack) {
            // Register the input stack. The application loads the input stack many times. To save computation,
            // load the stack to memory once and save it inside a static array. The StackLoader will
            // check for it the next time it needs it.
            noa::tie(exported_image_shape, exported_spacing) = StackLoader::register_input_stack(series.stack_file);
        } else {
            auto file = ni::ImageFile(series.stack_file, {.read = true});
            exported_image_shape = file.shape().filter(2, 3);
            exported_spacing = file.spacing().pop_front(); // remove z
        }

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
                        .allow_90_and_flip_rotation_from_mdoc = settings.alignment.coarse_allow_90_and_flip_rotation_from_mdoc,
                        .is_tilt_axis_from_mdoc = settings.alignment.coarse_is_tilt_axis_from_mdoc,
                        .fit_rotation_offset = settings.alignment.coarse_fit_rotation,
                        .fit_tilt_offset = settings.alignment.coarse_fit_tilt,
                        .fit_pitch_offset = settings.alignment.coarse_fit_pitch,
                        .output_directory = diagnostics_directory / "coarse",
                    }
                );
                #else
                Logger::warn("Build does not include refine tilt-series alignment");
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
                Logger::warn("Build does not include refine tilt-series alignment");
                #endif
            }

            // Save the metadata.
            const auto star_filename = series.output_directory / fmt::format("{}.star", basename);
            metadata.save_star(star_filename);
            Logger::info("{} saved", star_filename);
        }

        // Save IMOD files.
        const auto imod_directory = series.output_directory / "quinoa-exports" / "imod";
        metadata.save_imod(series.stack_file, imod_directory, basename, exported_image_shape, exported_spacing);
        Logger::info("{} files saved", imod_directory);

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
            Logger::warn("Build does not include refine tilt-series alignment");
            #endif
        }
    }

    void distribute_work(const Settings& settings, std::vector<Series>& series) {
        const auto single_ts = series.size() == 1;
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
                    if (not single_ts)
                        Logger::deactivate_console();

                    process_data(settings, ts, device);

                    if (not single_ts) {
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

        if (not single_ts) {
            Logger::info("Batch processing:");
            Logger::trace("  n_stacks={}\n  devices={}\n  output={}\n",
                series.size(), settings.compute.devices, series[0].output_directory);
        }

        // Distribute one per device.
        auto workers = noa::ThreadPool(settings.compute.devices.size());
        auto results = std::vector<std::future<void>>{};
        for (auto& device: settings.compute.devices)
            results.emplace_back(workers.enqueue(work, device));
        for (auto& result: results)
            result.get();
    }

    // void fix_baseline() {
    //     auto spectra = noa::read_image<f32>("~/Tmp/quinoa/test_baseline/test_image_spectra_3A.mrc").data;
    //     auto spectrum = spectra.subregion(19).span_1d();
    //     auto background = noa::like(spectra.subregion(19));
    //     auto fftfreq = Vec{0.02269, 0.23333};
    //     auto linspace = noa::Linspace<f64>::from_vec(fftfreq);
    //
    //     save_plot_xy(linspace, spectrum, "~/Tmp/quinoa/test_baseline/fit.txt", {.label = "spectrum"});
    //
    //     ctf::Baseline baseline{};
    //     baseline.fit(spectrum, fftfreq, fftfreq);
    //     baseline.sample(background.span_1d(), fftfreq);
    //     save_plot_xy(linspace, background, "~/Tmp/quinoa/test_baseline/fit.txt", {.label = "baseline1"});
    //
    //     auto ctf = ns::CTFIsotropic<f64>::Parameters{
    //         .pixel_size = 0.675,
    //         .defocus = 1.4477,
    //         .voltage = 300,
    //         .amplitude = 0.07,
    //         .cs = 2.7,
    //         .phase_shift = 0,
    //         .bfactor = 0,
    //         .scale = 1,
    //     }.to_ctf();
    //     baseline.fit(spectrum, fftfreq, ctf);
    //     baseline.sample(background.span_1d(), fftfreq);
    //     save_plot_xy(linspace, background, "~/Tmp/quinoa/test_baseline/fit.txt", {.label = "baseline2"});
    // }
}

auto main(int argc, char* argv[]) -> int {
    using namespace qn;

    try {
        Logger::initialize();
        Logger::activate_console();
        auto timer = Logger::status_scope_time<false>("Main");

        // fix_baseline();
        // return EXIT_FAILURE;

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
