#pragma once

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

namespace qn {
    // Static logger, with priority error > warn > info > trace > debug.
    class Logger {
    public:
        // Initializes the logger.
        // - If "logfile" can be empty, the logs are not saved to a file.
        // - The log level is set to debug (our lowest level). Use set_level() to change it.
        static void initialize(std::string_view logfile);

        // Set the level of the console sink.
        static void set_level(const std::string& level_name);

        template<typename... Args>
        static void error(Args&& ... args) { s_logger.error(std::forward<Args>(args)...); }

        template<typename... Args>
        static void warn(Args&& ... args) { s_logger.warn(std::forward<Args>(args)...); }

        template<typename... Args>
        static void info(Args&& ... args) { s_logger.info(std::forward<Args>(args)...); }

        template<typename... Args>
        static void trace(Args&& ... args) { s_logger.debug(std::forward<Args>(args)...); }

        template<typename... Args>
        static void debug(Args&& ... args) { s_logger.trace(std::forward<Args>(args)...); }

        static bool is_debug() { return s_is_debug; };

    private:
        static spdlog::logger s_logger;
        static bool s_is_debug;
    };
}
