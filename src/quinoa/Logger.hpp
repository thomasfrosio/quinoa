#pragma once

#include <filesystem>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <noa/Utils.hpp>
#include <quinoa/Types.hpp>

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

        static bool is_debug() { return s_is_debug; };


        template<typename... Args>
        static auto warn_once(noa::guts::FormatWithLocation<std::type_identity_t<Args>...> fmt, Args&&... args) -> bool {
            static std::vector<std::string> hashes;
            std::string hash = fmt::format("{}:{}", fmt.location.file_name(), fmt.location.line());
            if (std::ranges::find(hashes, hash) == hashes.end()) {
                hashes.push_back(std::move(hash));
                return false;
            }
            s_logger.warn(fmt::runtime(fmt.fmt), std::forward<Args>(args)...);
            return true;
        }

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

            ~ScopeTimer();
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
        static uint64_t s_uuid;
        static bool s_is_debug;
    };
}
