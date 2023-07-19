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
        Optimizer(nlopt_algorithm algorithm, i64 variables) {
            pointer = nlopt_create(algorithm, safe_cast<u32>(variables));
            QN_CHECK(pointer != nullptr, "Failed to create the optimizer");
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

        void set_x_tolerance_abs(const f64* tolerance) const {
            const nlopt_result result = nlopt_set_xtol_abs(pointer, tolerance);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void set_fx_tolerance_abs(f64 tolerance) const {
            const nlopt_result result = nlopt_set_ftol_abs(pointer, tolerance);
            QN_CHECK(result >= 0, "Failed with status: {}", nlopt_result_to_string(result));
        }

        void optimize(f64* x, f64* fx) const {
            const nlopt_result result = nlopt_optimize(pointer, x, fx);
            qn::Logger::debug("Optimizer terminated with status code = {}", nlopt_result_to_string(result));
        }

        f64 optimize(f64* x) const {
            f64 fx{};
            optimize(x, &fx);
            return fx;
        }

        void force_stop() const {
            const nlopt_result result = nlopt_force_stop(pointer);
            if (result < 0)
                qn::Logger::debug("Optimizer failed to stop, status={}", nlopt_result_to_string(result));
        }

        [[nodiscard]] i32 number_of_evaluations() const noexcept {
            return nlopt_get_numevals(pointer);
        }

    private:
        nlopt_opt pointer;
    };

    // Memoize optimization steps, to save computation.
    class Memoizer {
    private:
        struct Key {
            f64 value;
            bool has_value;
            bool has_gradient;
        };

    public:
        Memoizer() = default;

        Memoizer(i64 n_parameters, i64 resolution) {
            if (resolution > 0) {
                // Allocate everything upfront.
                m_cache_input = noa::memory::zeros<f64>({resolution, 1, 2, n_parameters});
                m_cache_lines = std::vector<Key>(static_cast<size_t>(resolution), {0, false, false});
            }
        }

        // If the input is found:
        //  - Returns the corresponding score.
        //  - If "gradients" is valid, set the corresponding gradients, if any.
        std::optional<f64> find(const f64* input, f64* gradients = nullptr, f64 epsilon = 1e-7) {
            if (m_cache_lines.empty())
                return std::nullopt;

            const auto cache = m_cache_input.view().subregion(
                    noa::indexing::FullExtent{}, 0, 0, noa::indexing::FullExtent{});

            // If the caller passes the gradients, we need a record with the gradients too.
            const bool requires_gradients = gradients != nullptr;

            // Check the cache lines for a perfect match.
            i64 line = m_circular_index;
            i64 successful_line{-1};
            for (i64 i = 0; i < resolution(); ++i) {
                const Key& cache_line = m_cache_lines[static_cast<size_t>(line)];
                if (!cache_line.has_value)
                    continue;

                // Check this cache line.
                const Span cache_line_input = cache.subregion(line).span();
                bool success{true};
                for (i64 parameter = 0; parameter < n_parameters(); ++parameter) {
                    if (!noa::math::are_almost_equal<4>(cache_line_input[parameter], input[parameter], epsilon)) {
                        success = false;
                        break;
                    }
                }

                // We have a match, but make sure this match has the gradients too.
                if (success && cache_line.has_gradient == requires_gradients) {
                    successful_line = line;
                    break;
                }

                // This line isn't a good match. Step to the previous one.
                line -= 1;
                if (line < 0)
                    line += resolution();
            }

            // We don't have a match.
            if (successful_line == -1)
                return std::nullopt;

            // We have a match. Set the recorded gradients and return recorded value.
            if (gradients) {
                const auto cache_line_gradient = m_cache_input.view().subregion(
                        successful_line, 0, 1, noa::indexing::FullExtent{}).span();
                for (i64 i = 0; i < cache_line_gradient.ssize(); ++i)
                    gradients[i] = cache_line_gradient[i];
            }
            return m_cache_lines[static_cast<size_t>(successful_line)].value;
        }

        void record(const f64* inputs, f64 score, const f64* gradients = nullptr) {
            if (m_cache_lines.empty())
                return;

            // Circular buffer. Increment at every call.
            // We start at +1 to make the find() increment simpler.
            m_circular_index = (m_circular_index + 1) % resolution();

            const Span cache_line_input = m_cache_input.view().subregion(
                    m_circular_index, 0, 0, noa::indexing::FullExtent{}).span();
            for (i64 i = 0; i < cache_line_input.ssize(); ++i) {
                cache_line_input[i] = inputs[i];
            }

            if (gradients) {
                const Span cache_line_gradient = m_cache_input.view().subregion(
                        m_circular_index, 0, 1, noa::indexing::FullExtent{}).span();
                for (i64 i = 0; i < cache_line_gradient.ssize(); ++i)
                    cache_line_gradient[i] = gradients[i];
            }

            m_cache_lines[static_cast<size_t>(m_circular_index)] = {score, true, gradients != nullptr};
        }

        [[nodiscard]] i64 resolution() const noexcept { return m_cache_input.shape()[0]; }
        [[nodiscard]] i64 n_parameters() const noexcept { return m_cache_input.shape()[3]; }

        void reset_cache() {
            for (auto& line: m_cache_lines)
                line.has_value = false;
        }

    private:
        Array<f64> m_cache_input;
        std::vector<Key> m_cache_lines;
        i64 m_circular_index{0};
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

        template<typename Real>
        [[nodiscard]] static constexpr Real get(
                Real fx_minus_delta,
                Real fx_plus_delta,
                Real delta
        ) noexcept {
            const auto diff = fx_plus_delta - fx_minus_delta;
            return diff / (2 * delta);
        }
    };
}
