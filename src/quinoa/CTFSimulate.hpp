#pragma once

#include "quinoa/Types.hpp"

namespace qn::ctf {
    /// Range-like type designed to iterate through a |CTF| curve and efficiently analyze its gradient.
    /// This is used when we need to know where are the |CTF| vertexes (peaks and zeros) and sample a spectrum
    /// at these locations.
    class Simulate {
    public:
        constexpr static size_t SIMULATED_LOGICAL_SIZE = 8192;
        constexpr static f64 SIMULATED_FREQ_STEP = 1 / static_cast<f64>(SIMULATED_LOGICAL_SIZE);
    public:
        /// Sample the spectrum at the given fftfreq.
        [[nodiscard]] static auto sample_at(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            f64 fftfreq
        ) -> f64 {
            const auto spectrum_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
            const auto spectrum_frequency = (fftfreq - fftfreq_range[0]) / spectrum_step;
            const auto floored_f64 = std::floor(spectrum_frequency);
            const auto floored_i64 = static_cast<i64>(floored_f64);
            const auto fraction = spectrum_frequency - floored_f64;

            // Lerp.
            const auto index_0 = ni::index_at<noa::Border::REFLECT>(floored_i64 + 0, spectrum.ssize());
            const auto index_1 = ni::index_at<noa::Border::REFLECT>(floored_i64 + 1, spectrum.ssize());
            const auto interpolated =
                static_cast<f64>(spectrum[index_0]) * (1 - fraction) +
                static_cast<f64>(spectrum[index_1]) * fraction;
            return interpolated;
        }

    public:
        constexpr explicit Simulate(
            const ns::CTFIsotropic<f64>& ctf,
            const Vec<f64, 2>& fftfreq_range
        ) :
            m_ctf{&ctf},
            m_simulated_range_index{noa::round(fftfreq_range * SIMULATED_LOGICAL_SIZE).as<i64>()}
        {}

    public: // range-for loop support
        struct Iterator {
        public:
            [[nodiscard]] constexpr auto fftfreq() const -> f64 {
                return static_cast<f64>(m_index) * SIMULATED_FREQ_STEP;
            }

            /// Given the current position i, retrieve the slope [i-1,i] and [i,i+1].
            [[nodiscard]] constexpr auto slopes() const -> Vec<f64, 2> {
                const f64 ctf_value_0 = circular_buffer_get_(-2); // i - 1
                const f64 ctf_value_1 = circular_buffer_get_(-1); // i
                const f64 ctf_value_2 = circular_buffer_get_( 0); // i + 1
                return {
                    ctf_value_1 - ctf_value_0,
                    ctf_value_2 - ctf_value_1
                };
            }

            [[nodiscard]] constexpr auto is_ctf_zero() const -> bool {
                auto [slope_0, slope_1] = slopes();
                return slope_0 < 0 and slope_1 >= 0;
            }
            [[nodiscard]] constexpr auto is_ctf_peak() const -> bool {
                auto [slope_0, slope_1] = slopes();
                return slope_0 > 0 and slope_1 <= 0;
            }
            [[nodiscard]] constexpr auto is_ctf_vertex() const -> bool {
                auto [slope_0, slope_1] = slopes();
                return slope_0 * slope_1 < 0.; // if different sign
            }

        public: // minimal range-for support
            constexpr explicit Iterator(const Simulate* parent, i64 index) noexcept: m_parent{parent}, m_index{index} {
                circular_buffer_next_(std::max(i64{0}, m_index - 1)); // prevent negative frequencies
                circular_buffer_next_(m_index);
                circular_buffer_next_(m_index + 1);
            }
            constexpr bool operator!=(const i64& end) const noexcept { return m_index != end; }
            constexpr auto operator*() const noexcept -> const Iterator& { return *this; }
            constexpr Iterator& operator++() noexcept {
                ++m_index;
                // Sample one ahead, so that when this returns, we have:
                // m_index-1 at circular_buffer_get_(offset: -2)
                // m_index   at circular_buffer_get_(offset: -1)
                // m_index+1 at circular_buffer_get_(offset:  0)
                circular_buffer_next_(m_index + 1);
                return *this;
            }

        private:
            const Simulate* m_parent;
            i64 m_index;

            // Circular buffer.
            constexpr static i64 CIRCULAR_BUFFER_SIZE = 3;
            Vec<f64, CIRCULAR_BUFFER_SIZE> m_circular_buffer{};
            i64 m_circular_index{};

            constexpr void circular_buffer_next_(i64 simulated_index) {
                const auto fftfreq = static_cast<f64>(simulated_index) * SIMULATED_FREQ_STEP;
                m_circular_index = (m_circular_index + 1) % CIRCULAR_BUFFER_SIZE;
                m_circular_buffer[static_cast<size_t>(m_circular_index)] = std::abs(m_parent->m_ctf->value_at(fftfreq));
            }
            [[nodiscard]] constexpr auto circular_buffer_get_(i64 offset) const -> f64 {
                auto current = (m_circular_index + CIRCULAR_BUFFER_SIZE + offset) % CIRCULAR_BUFFER_SIZE;
                return m_circular_buffer[static_cast<size_t>(current)];
            }
        };

        [[nodiscard]] constexpr auto begin() const -> Iterator { return Iterator(this, m_simulated_range_index[0]); }
        [[nodiscard]] constexpr auto end() const -> i64 { return m_simulated_range_index[1]; }

    private:
        const ns::CTFIsotropic<f64>* m_ctf{};
        Vec<i64, 2> m_simulated_range_index{};
    };
}
