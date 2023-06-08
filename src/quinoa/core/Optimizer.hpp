#pragma once

#include <noa/core/Definitions.hpp>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wconversion"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

#include <nlopt.h>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif

#include <optional>
#include "quinoa/Types.h"
#include "quinoa/Exception.h"
#include "quinoa/io/Logging.h"

namespace qn {
    /// Thin wrapper of the NLopt library.
    /// \note I prefer to use the C API over the C++ API because:
    ///       1) The C++ API throws exceptions if the optimization failed.
    ///       2) It is incomplete (e.g. result_to_string doesn't exist).
    ///       3) It uses std::vector but in our case a std::array would be more appropriate.
    struct Optimizer {
        Optimizer(nlopt_algorithm algorithm, u32 variables) {
            pointer = nlopt_create(algorithm, variables);
            QN_CHECK(pointer != nullptr, "Failed to create the optimizer");
            qn::Logger::trace("Optimizer created, using: {}", nlopt_algorithm_name(algorithm));
        }

        ~Optimizer() {
            nlopt_destroy(pointer);
        }

        void set_max_objective(nlopt_func function, void* data = nullptr) const {
            const nlopt_result result = nlopt_set_max_objective(pointer, function, data);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void set_min_objective(nlopt_func function, void* data = nullptr) const {
            const nlopt_result result = nlopt_set_min_objective(pointer, function, data);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void set_bounds(f64 lower_bounds, f64 upper_bounds) const {
            nlopt_result result = nlopt_set_lower_bounds1(pointer, lower_bounds);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
            result = nlopt_set_upper_bounds1(pointer, upper_bounds);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void set_bounds(const f64* lower_bounds, const f64* upper_bounds) const {
            nlopt_result result = nlopt_set_lower_bounds(pointer, lower_bounds);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
            result = nlopt_set_upper_bounds(pointer, upper_bounds);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void set_initial_step(f64 dx) const {
            const nlopt_result result = nlopt_set_initial_step1(pointer, dx);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void set_x_tolerance_abs(f64 tolerance) const {
            const nlopt_result result = nlopt_set_xtol_abs1(pointer, tolerance);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void set_fx_tolerance_abs(f64 tolerance) const {
            const nlopt_result result = nlopt_set_ftol_abs(pointer, tolerance);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void optimize(f64* x, f64* fx) const {
            const nlopt_result result = nlopt_optimize(pointer, x, fx);
            if (result < 0)
                qn::Logger::trace("Optimizer terminated with status code = {}", nlopt_result_to_string(result));
        }

        f64 optimize(f64* x) const {
            f64 fx{};
            optimize(x, &fx);
            return fx;
        }

        [[nodiscard]] i32 number_of_evaluations() const noexcept {
            return nlopt_get_numevals(pointer);
        }

    private:
        nlopt_opt pointer;
    };

    template<typename Input, typename Output>
    class Memoizer {
    public:
        std::optional<Output> find(const Input& input) {
            const auto iter = std::find_if(
                    m_cache.begin(), m_cache.end(),
                    [=](const auto& value) {
                        return noa::math::are_almost_equal(value.first, input);
                    });
            return iter == m_cache.end() ? std::optional<Output>{} : iter->second;
        }

        template<typename Predicate>
        std::optional<Output> find(Predicate&& predicate) {
            const auto iter = std::find_if(
                    m_cache.begin(), m_cache.end(),
                    [&](const auto& value) {
                        return predicate(value.first);
                    });
            return iter == m_cache.end() ? std::optional<Output>{} : iter->second;
        }

        void record(const Input& input, const Output& output) {
            m_cache.emplace_back(input, output);
        }

    private:
        std::vector<std::pair<Input, Output>> m_cache;
    };

    struct CentralFiniteDifference {
        template<typename Real>
        [[nodiscard]] static constexpr Real delta(Real x) noexcept {
            Real eps = std::numeric_limits<Real>::epsilon();
            Real h = std::pow(3 * eps, Real{1} / Real{3});
            return make_delta_representable(x, h);
        }

        template<typename Real>
        [[nodiscard]] static constexpr Real make_delta_representable(Real x, Real h) noexcept {
            // From https://github.com/boostorg/math/blob/develop/include/boost/math/differentiation/finite_difference.hpp
            // Redefine h so that x + h is representable. Not using this trick leads to
            // large error. The compiler flag -ffast-math evaporates these operations...
            Real temp = x + h;
            h = temp - x;
            // Handle the case x + h == x:
            if (h == 0)
                h = std::nextafter(x, (std::numeric_limits<Real>::max)()) - x;
            return h;
        }
    };
}
