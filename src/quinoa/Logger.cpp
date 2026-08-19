#include "quinoa/Logger.hpp"

#include <noa/runtime/core/Random.hpp>

// Include this after our Logger.hpp to properly set the spdlog levels.
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace {
    using console_sink_t = std::shared_ptr<spdlog::sinks::stdout_color_sink_mt>;
    console_sink_t s_console_sink;
}

namespace qn {
    thread_local spdlog::logger Logger::s_logger("quinoa");
    thread_local usize Logger::s_uuid = noa::random_value(noa::Uniform<usize>{0, std::numeric_limits<usize>::max()});
    Path Logger::s_debug_path{};

    void Logger::initialize() {
        // Configure the console sink. All processing threads share the same console.
        if (not s_console_sink) {
            s_console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            s_console_sink->set_color(spdlog::level::critical, s_console_sink->red_bold); // our error
            s_console_sink->set_color(spdlog::level::err, s_console_sink->yellow_bold); // our warn
            s_console_sink->set_color(spdlog::level::warn, s_console_sink->blue); // our status
            s_console_sink->set_color(spdlog::level::info, s_console_sink->green); // our info
            s_console_sink->set_color(spdlog::level::debug, s_console_sink->reset); // our trace
            s_console_sink->set_color(spdlog::level::trace, s_console_sink->cyan); // our debug
            s_console_sink->set_pattern("%^%v%$"); // colored log
            s_console_sink->set_level(spdlog::level::trace); // default to our into
        }

        // Configure the logger if not configured yet.
        s_logger.set_level(spdlog::level::trace); // no limits for the logger; the sinks set the levels.
        s_logger.flush_on(spdlog::level::err); // our warn
    }

    void Logger::activate_console() {
        for (const auto& sink : s_logger.sinks())
            if (sink == s_console_sink)
                return;
        s_logger.sinks().push_back(s_console_sink);
    }

    void Logger::deactivate_console() {
        std::erase_if(s_logger.sinks(), [] (const auto& sink) { return sink == s_console_sink; });
    }

    void Logger::set_console_level(const std::string& level_name) {
        // Level should be ["off", "error", "warn", "status", "info", "trace", "debug"].
        const spdlog::level::level_enum level = spdlog::level::from_str(level_name);
        s_console_sink->set_level(level);
    }

    void Logger::set_logfile(const std::filesystem::path& logfile) {
        // Configure the single-threaded file sink.
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_st>(logfile.string());
        file_sink->set_pattern("[%T][%l]: %v"); // [time][level]: log
        file_sink->set_level(spdlog::level::debug); // our trace (debug would be console only)

        // Erase any previous file sinks and add the new one.
        std::erase_if(s_logger.sinks(), [](const auto& sink) { return sink != s_console_sink; });
        s_logger.sinks().push_back(std::move(file_sink));
    }

    Logger::ScopeTimer::~ScopeTimer() {
        if (not timer.is_running())
            return;

        std::chrono::duration elapsed = timer.elapsed();
        const char* end = newline ? "\n" : "";
        if (elapsed > std::chrono::minutes(1)) {
            auto minutes = stdc::floor<stdc::minutes>(elapsed);
            auto seconds = stdc::duration_cast<stdc::seconds>(elapsed - minutes);
            s_logger.log(level, "{}... done. Took {}{}.{}", name, minutes, seconds, end);
        } else if (elapsed > std::chrono::seconds(1)) {
            auto seconds = stdc::floor<stdc::seconds>(elapsed);
            auto milliseconds = stdc::duration_cast<stdc::milliseconds>(elapsed - seconds);
            s_logger.log(level, "{}... done. Took {}{}.{}", name, seconds, milliseconds, end);
        } else {
            auto milliseconds = stdc::round<stdc::milliseconds>(elapsed);
            s_logger.log(level, "{}... done. Took {}.{}", name, milliseconds, end);
        }
    }
}
