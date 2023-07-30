#pragma once

#include <filesystem>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

namespace qn {
    // Static logger.
    class Logger {
    public:
        static void initialize();
        static void add_logfile(const std::filesystem::path& logfile);
        static void set_level(const std::string& level_name);

        template<typename... Args>
        static void error(Args&& ... args) { s_logger.critical(std::forward<Args>(args)...); }

        template<typename... Args>
        static void warn(Args&& ... args) { s_logger.error(std::forward<Args>(args)...); }

        template<typename... Args>
        static void status(Args&& ... args) { s_logger.warn(std::forward<Args>(args)...); }

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
