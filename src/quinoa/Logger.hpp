#pragma once

#include <filesystem>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <noa/Utils.hpp>

namespace qn {
    // Static logger.
    class Logger {
    public:
        static void initialize();
        static void add_logfile(const std::filesystem::path& logfile);
        static void set_level(const std::string& level_name);

        template<typename... Args>
        static void error(fmt::format_string<Args...>&& fmt, Args&&... args) {
            s_logger.critical(fmt::runtime(fmt), std::forward<Args>(args)...);
        }

        template<typename... Args>
        static void warn(fmt::format_string<Args...>&& fmt, Args&&... args) {
            s_logger.warn(fmt::runtime(fmt), std::forward<Args>(args)...);
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

        static bool is_debug() { return s_is_debug; };


    public:
        struct ScopeTimer {
            noa::Timer timer{};
            std::string name{};
            spdlog::level::level_enum level{};

            explicit ScopeTimer(
                std::string_view name_,
                spdlog::level::level_enum level_
            ) : name(name_), level(level_)
            {
                s_logger.log(level, "{}...", name);
                timer.start();
            }

            ~ScopeTimer() {
                std::chrono::duration<double, std::milli> elapsed = timer.elapsed();

                // TODO There's probably a better way to do this.
                if (elapsed > std::chrono::minutes(1)) {
                    s_logger.log(level, "{}... done. Took {:.3f} mins.\n", name, elapsed.count() * 60e-3);
                } else if (elapsed > std::chrono::seconds(1)) {
                    s_logger.log(level, "{}... done. Took {:.3f} secs.\n", name, elapsed.count() * 1e-3);
                } else {
                    s_logger.log(level, "{}... done. Took {:.3f} ms.\n", name, elapsed.count());
                }
            }
        };

        template<typename... Args>
        [[nodiscard]] static auto status_scope_time(fmt::format_string<Args...>&& fmt, Args&&... args) -> ScopeTimer {
            return ScopeTimer(fmt::format(fmt::runtime(fmt), std::forward<Args>(args)...), spdlog::level::warn);
        }
        template<typename... Args>
        [[nodiscard]] static auto info_scope_time(fmt::format_string<Args...>&& fmt, Args&&... args) -> ScopeTimer {
            return ScopeTimer(fmt::format(fmt::runtime(fmt), std::forward<Args>(args)...), spdlog::level::info);
        }
        template<typename... Args>
        [[nodiscard]] static auto trace_scope_time(fmt::format_string<Args...>&& fmt, Args&&... args) -> ScopeTimer {
            return ScopeTimer(fmt::format(fmt::runtime(fmt), std::forward<Args>(args)...), spdlog::level::debug);
        }

    public:
        static spdlog::logger s_logger;
        static bool s_is_debug;
    };
}
