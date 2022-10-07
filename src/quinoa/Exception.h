#pragma once

#include <string>
#include <exception>
#include <filesystem>

#include <noa/String.h>

namespace qn {
    // Global (within ::qn) exception. Usually caught in main().
    class Exception : public std::exception {
    public:
        template<typename... Args>
        Exception(const char* file, const char* function, int line, Args&& ... args) {
            m_buffer = format_(file, function, line, noa::string::format(args...));
        }

        [[nodiscard]] const char* what() const noexcept override {
            return m_buffer.data();
        }

        [[nodiscard]] static std::string backtrace();

    protected:
        static std::string format_(const char* file, const char* function, int line,
                                   const std::string& message) {
            namespace fs = std::filesystem;
            const size_t idx = std::string(file).rfind(std::string("quinoa") + fs::path::preferred_separator);
            return noa::string::format("ERROR:{}:{}:{}: {}",
                                       idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                                       function, line, message);
        }

    protected:
        std::string m_buffer{};
    };
}

#define QN_THROW(...) std::throw_with_nested(::qn::Exception(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__))
#define QN_THROW_FUNC(func, ...) std::throw_with_nested(::qn::Exception(__FILE__, func, __LINE__, __VA_ARGS__))

#if defined(QN_DEBUG) || defined(QN_ENABLE_CHECKS_RELEASE)
#define QN_CHECK(cond, ...) if (!(cond)) QN_THROW(__VA_ARGS__)
#else
#define QN_CHECK(cond, ...)
#endif
