#pragma once

// For now, let noa handle logging.
#include <noa/Session.h>

namespace qn {
    // Log messages.
    struct Logger {
        template<typename... Args>
        static void trace(Args&& ... args) { ::noa::Session::logger.trace(std::forward<Args>(args)...); }

        template<typename... Args>
        static void info(Args&& ... args) { ::noa::Session::logger.info(std::forward<Args>(args)...); }

        template<typename... Args>
        static void warn(Args&& ... args) { ::noa::Session::logger.warn(std::forward<Args>(args)...); }

        template<typename... Args>
        static void error(Args&& ... args) { ::noa::Session::logger.error(std::forward<Args>(args)...); }

        template<typename... Args>
        static void debug([[maybe_unused]] Args&& ... args) {
            #ifdef QN_DEBUG
            ::noa::Session::logger.debug(std::forward<Args>(args)...);
            #endif
        }
    };
}
