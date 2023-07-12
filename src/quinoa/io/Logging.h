#pragma once

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

namespace qn {
    class Logger {
    public:
        static void initialize(std::string_view name, std::string_view filename, const std::string& level);

        template<typename... Args>
        static void error(Args&& ... args) { s_logger.error(std::forward<Args>(args)...); }

        template<typename... Args>
        static void warn(Args&& ... args) { s_logger.warn(std::forward<Args>(args)...); }

        template<typename... Args>
        static void info(Args&& ... args) { s_logger.info(std::forward<Args>(args)...); }

        template<typename... Args>
        static void trace(Args&& ... args) { s_logger.trace(std::forward<Args>(args)...); }

        template<typename... Args>
        static void debug([[maybe_unused]] Args&& ... args) {
            #ifdef QN_DEBUG
            s_logger.debug(std::forward<Args>(args)...);
            #endif
        }

    private:
        static spdlog::logger s_logger;
    };
}
