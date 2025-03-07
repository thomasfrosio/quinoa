#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "quinoa/Logger.hpp"

namespace qn {
    spdlog::logger Logger::s_logger(std::string{}); // empty logger
    bool Logger::s_is_debug = false;

    void Logger::initialize() {
        // Configure the console sink.
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_color(spdlog::level::critical, console_sink->red_bold); // our error
        console_sink->set_color(spdlog::level::err, console_sink->yellow_bold); // our warn
        console_sink->set_color(spdlog::level::warn, console_sink->green); // our status
        console_sink->set_color(spdlog::level::info, console_sink->reset); // our info
        console_sink->set_color(spdlog::level::debug, console_sink->white); // our trace
        console_sink->set_color(spdlog::level::trace, console_sink->cyan); // our debug
        console_sink->set_pattern("%^%v%$"); // colored log
        console_sink->set_level(spdlog::level::info); // default to our into

        // Configure the logger.
        s_logger = spdlog::logger("quinoa", std::move(console_sink));
        s_logger.set_level(spdlog::level::trace); // no limits for the logger; the sinks set the levels.
        s_logger.flush_on(spdlog::level::err);
    }

    void Logger::add_logfile(const std::filesystem::path& logfile) {
        // Configure the file sink.
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logfile.string());
        file_sink->set_pattern("[%T][%l]: %v"); // [time][level]: log
        file_sink->set_level(spdlog::level::debug); // default to our trace
        s_logger.sinks().push_back(std::move(file_sink));
    }

    void Logger::set_level(const std::string& level_name) {
        if (s_logger.sinks().empty()) // if not initialized
            return;

        // Level should be ["off", "error", "warn", "status", "info", "trace", "debug"].
        const spdlog::level::level_enum level = spdlog::level::from_str(level_name);

        // The log level from the user does not affect the logfile.
        // The logfile is always at the trace level (which is effectively our maximum verbosity in production).
        const auto console_sink = dynamic_cast<spdlog::sinks::stdout_color_sink_mt*>(s_logger.sinks()[0].get());
        if (console_sink)
            console_sink->set_level(level);

        // Save for easy access whether we are in debug mode.
        s_is_debug = level == spdlog::level::trace; // our debug is spdlog's trace
    }
}
