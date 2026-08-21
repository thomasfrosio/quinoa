#pragma once

#include <filesystem>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <quinoa/Types.hpp>

namespace qn {
    class Logger {
    public:
        static void initialize();
        static void activate_console();
        static void deactivate_console();
        static void set_console_level(const std::string& level_name);
        static void set_logfile(const std::filesystem::path& logfile);

        template<typename... Args>
        static void error(fmt::format_string<Args...>&& fmt, Args&&... args) {
            s_logger.critical(fmt::runtime(fmt), std::forward<Args>(args)...);
        }

        template<typename... Args>
        static void warn(fmt::format_string<Args...>&& fmt, Args&&... args) {
            s_logger.error(fmt::runtime(fmt), std::forward<Args>(args)...);
        }

        template<typename... Args>
        static void status(fmt::format_string<Args...>&& fmt, Args&&... args) {
            s_logger.warn(fmt::runtime(fmt), std::forward<Args>(args)...);
        }

        template<typename... Args>
        static void info(fmt::format_string<Args...>&& fmt, Args&&... args) {
            s_logger.info(fmt::runtime(fmt), std::forward<Args>(args)...);
        }

        template<typename... Args>
        static void trace(fmt::format_string<Args...>&& fmt, Args&&... args) {
            s_logger.debug(fmt::runtime(fmt), std::forward<Args>(args)...);
        }

        template<typename... Args>
        static void debug(fmt::format_string<Args...>&& fmt, Args&&... args) {
            s_logger.trace(fmt::runtime(fmt), std::forward<Args>(args)...);
        }

    public:
        struct ScopeTimer {
            noa::Timer timer{};
            std::string name{};
            spdlog::level::level_enum level{};
            bool newline{true};

            explicit ScopeTimer() = default;
            explicit ScopeTimer(
                std::string_view name_,
                spdlog::level::level_enum level_,
                bool newline_ = true
            ) : name(name_), level(level_), newline(newline_)
            {
                s_logger.log(level, "{}...", name);
                timer.start();
            }

            auto set_newline(bool add_newline) -> ScopeTimer& {
                newline = add_newline;
                return *this;
            }

            // Define move-semantics to explicitly "destruct" the object
            // by resetting the timer, which turns off the logging.
            ScopeTimer(ScopeTimer&& t) noexcept {
                timer = std::exchange(t.timer, noa::Timer{});
                name = std::move(t.name);
                level = t.level;
                newline = t.newline;
            }
            ScopeTimer& operator=(ScopeTimer&& t) noexcept {
                if (this != &t) {
                    timer = std::exchange(t.timer, noa::Timer{});
                    name = std::move(t.name);
                    level = t.level;
                    newline = t.newline;
                }
                return *this;
            }

            ~ScopeTimer();
        };

        template<bool NEW_LINE = true, typename... Args>
        [[nodiscard]] static auto status_scope_time(fmt::format_string<Args...>&& fmt, Args&&... args) -> ScopeTimer {
            return ScopeTimer(fmt::format(fmt::runtime(fmt), std::forward<Args>(args)...), spdlog::level::warn, NEW_LINE);
        }
        template<bool NEW_LINE = true, typename... Args>
        [[nodiscard]] static auto info_scope_time(fmt::format_string<Args...>&& fmt, Args&&... args) -> ScopeTimer {
            return ScopeTimer(fmt::format(fmt::runtime(fmt), std::forward<Args>(args)...), spdlog::level::info, NEW_LINE);
        }
        template<bool NEW_LINE = false, typename... Args>
        [[nodiscard]] static auto trace_scope_time(fmt::format_string<Args...>&& fmt, Args&&... args) -> ScopeTimer {
            return ScopeTimer(fmt::format(fmt::runtime(fmt), std::forward<Args>(args)...), spdlog::level::debug, NEW_LINE);
        }

    public:
        static thread_local spdlog::logger s_logger;
        static thread_local usize s_uuid;
        static Path s_debug_path;
    };
}
