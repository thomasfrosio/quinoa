#include "quinoa/io/Logging.h"

#include <spdlog/common.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace qn {
    spdlog::logger Logger::s_logger(std::string{}); // empty logger

    void Logger::initialize(
            std::string_view name,
            std::string_view filename,
            const std::string& level
    ) {
        const spdlog::level::level_enum level_ = spdlog::level::from_str(level);

        std::array<spdlog::sink_ptr, 2> sinks;
        sinks[0] = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sinks[0]->set_pattern("%^%v%$");
        sinks[0]->set_level(level_);

        const bool has_filename = !filename.empty();
        if (has_filename) {
            sinks[1] = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename.data());
            sinks[1]->set_pattern("[%T] [%n::%l]: %v");
            sinks[1]->set_level(level_);
        }

        s_logger = spdlog::logger(name.data(), sinks.data(), sinks.data() + 1 + has_filename);
        s_logger.set_level(spdlog::level::trace); // be limited by the sink levels.
        s_logger.flush_on(spdlog::level::err);
    }
}
