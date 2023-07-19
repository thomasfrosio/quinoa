#include "quinoa/io/Logging.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace qn {
    spdlog::logger Logger::s_logger(std::string{}); // empty logger
    bool Logger::s_is_debug = false;

    void Logger::initialize(std::string_view logfile) {
        // Configure the console sink.
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_color(spdlog::level::debug, console_sink->reset); // our trace
        console_sink->set_color(spdlog::level::trace, console_sink->white); // our debug
        console_sink->set_pattern("%^%v%$");
        console_sink->set_level(spdlog::level::debug); // our trace

        std::array<spdlog::sink_ptr, 2> sinks;
        sinks[0] = console_sink;

        const bool has_filename = !logfile.empty();
        if (has_filename) {
            sinks[1] = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logfile.data());
            sinks[1]->set_pattern("[%T][%l]: %v");
            sinks[1]->set_level(spdlog::level::debug); // our trace
        }

        s_logger = spdlog::logger("", sinks.data(), sinks.data() + 1 + has_filename);
        s_logger.set_level(spdlog::level::trace); // be limited by the sink levels.
        s_logger.flush_on(spdlog::level::err);
    }

    void Logger::set_level(const std::string& level_name) {
        if (s_logger.sinks().empty()) // if not initialized
            return;

        // Level should be ["off", "error", "warn", "info", "trace", "debug"].
        const spdlog::level::level_enum level = spdlog::level::from_str(level_name);

        const auto console_sink = dynamic_cast<spdlog::sinks::stdout_color_sink_mt*>(s_logger.sinks()[0].get());
        if (console_sink)
            console_sink->set_level(level);

        // Save for easy access whether we are in debug mode.
        s_is_debug = level == spdlog::level::trace; // our debug is spdlog's trace
    }
}
