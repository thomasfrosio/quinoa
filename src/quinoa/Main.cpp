#include <noa/Runtime.hpp>
#include <noa/Session.hpp>

#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Stack.hpp"

#include "quinoa/preprocessing/Run.hpp"
#include "quinoa/ctf/Run.hpp"
#ifndef QN_CTF_ONLY
#   include "quinoa/align/Run.hpp"
#   include "quinoa/postprocessing/Run.hpp"
#endif

namespace {
    using namespace qn;

    void process_data(const Settings& settings, const Series& series, const Device& device) {
        const auto basename = series.stem();
        const auto diagnostics_directory = series.output_directory / "quinoa_diagnostics" / basename;

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
            preprocess(series.stack_file, metadata, device, settings.preprocessing, diagnostics_directory / "preprocessing");
        }

        // Alignment.
        if (settings.alignment.coarse.run or settings.alignment.refine.run or settings.ctf.run) {
            auto scope_timer = Logger::status_scope_time("Alignment");

            if (settings.alignment.coarse.run) {
                #ifndef QN_CTF_ONLY
                coarse_alignment(series.stack_file, metadata, device, settings.alignment.coarse, diagnostics_directory / "alignment_coarse");
                #else
                Logger::warn("Build does not include tilt-series alignment");
                #endif
            }

            if (settings.ctf.run) {
                ctf::full_fit(series.stack_file, metadata, device, settings.ctf, diagnostics_directory / "ctf");
            }

            if (settings.alignment.refine.run) {
                #ifndef QN_CTF_ONLY
                refine_alignment(series.stack_file, metadata, device, settings.alignment.refine, diagnostics_directory / "alignment_refine");
                #else
                Logger::warn("Build does not include tilt-series alignment");
                #endif
            }

            // Save the metadata.
            const auto star_filename = series.output_directory / fmt::format("{}.star", basename);
            metadata.save_star(star_filename);
            Logger::info("{} saved", star_filename);
        }

        // Save IMOD files.
        const auto imod_directory = series.output_directory / "quinoa_exports" / "imod";
        metadata.save_imod(series.stack_file, imod_directory, basename, exported_image_shape, exported_spacing);
        Logger::info("{} files saved", imod_directory);

        // Postprocessing.
        if (settings.postprocessing.run) {
            #ifndef QN_CTF_ONLY
            auto scope_timer = Logger::status_scope_time("Postprocessing");
            postprocess(series.stack_file, metadata, device, settings.postprocessing, series.output_directory);
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
