#include <noa/Signal.hpp>
#include <noa/Geometry.hpp>

#include "quinoa/CTF.hpp"

namespace {
    using namespace qn;
    using namespace qn::ctf;

    ///
    class Simulator {
    public:
        constexpr static size_t SIMULATED_LOGICAL_SIZE = 8192;
        constexpr static f64 SIMULATED_FREQ_STEP = 1 / static_cast<f64>(SIMULATED_LOGICAL_SIZE);

    public:
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
        constexpr explicit Simulator(
            const ns::CTFIsotropic<f64>& ctf,
            const Vec<f64, 2>& fftfreq_range
        ) :
            m_ctf{&ctf},
            m_simulated_range_index{noa::round(fftfreq_range * SIMULATED_LOGICAL_SIZE).as<i64>()}
        {}

    public: // range-for loop support
        struct Iterator {
        public:
            [[nodiscard]] auto fftfreq() const -> f64 {
                return static_cast<f64>(m_index) * SIMULATED_FREQ_STEP;
            }

            /// Given the current position i, retrieve the slope [i-1,i] and [i,i+1].
            [[nodiscard]] auto slopes() const -> Vec<f64, 2> {
                const f64 ctf_value_0 = circular_buffer_get_(-2); // i - 1
                const f64 ctf_value_1 = circular_buffer_get_(-1); // i
                const f64 ctf_value_2 = circular_buffer_get_( 0); // i + 1
                return {
                    ctf_value_1 - ctf_value_0,
                    ctf_value_2 - ctf_value_1
                };
            }

            [[nodiscard]] auto is_ctf_zero() const -> bool {
                auto [slope_0, slope_1] = slopes();
                return slope_0 < 0 and slope_1 >= 0;
            }
            [[nodiscard]] auto is_ctf_peak() const -> bool {
                auto [slope_0, slope_1] = slopes();
                return slope_0 > 0 and slope_1 <= 0;
            }

        public: // minimal range-for support
            constexpr explicit Iterator(const Simulator* parent, i64 index) noexcept: m_parent{parent}, m_index{index} {
                circular_buffer_next_(m_index - 1);
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
            const Simulator* m_parent;
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

        [[nodiscard]] auto begin() const -> Iterator { return Iterator(this, m_simulated_range_index[0]); }
        [[nodiscard]] auto end() const -> i64 { return m_simulated_range_index[1]; }

    private:
        // Simulate CTF.
        const ns::CTFIsotropic<f64>* m_ctf{};
        Vec<i64, 2> m_simulated_range_index{};
    };
}

namespace qn::ctf {
    auto Background::smallest_defocus_for_fitting(
        const Vec<f64, 2>& fftfreq_range,
        ns::CTFIsotropic<f64> ctf,
        i64 n_zeroes
    ) -> f64 {
        // There's probably a better way to do this, but since this only used once per image, brute force.
        auto get_n_zeroes = [&] {
            i32 count{};
            for (auto e: Simulator(ctf, fftfreq_range))
                if (e.is_ctf_zero() and ++count >= n_zeroes)
                    break;
            return count;
        };

        f64 defocus = 1.;
        for (f64 step: {0.1, 0.01}) {
            if (get_n_zeroes() >= n_zeroes) {
                // Decrease the defocus as much as possible.
                while (true) {
                    defocus -= step;
                    ctf.set_defocus(defocus);
                    if (get_n_zeroes() < n_zeroes)
                        break;
                }
            } else {
                // Increase the defocus until we get enough zeros.
                while (true) {
                    defocus += step;
                    ctf.set_defocus(defocus);
                    if (get_n_zeroes() >= n_zeroes)
                        break;
                }
            }
        }
        return defocus;
    }

    void Background::fit(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const ns::CTFIsotropic<f64>& new_ctf
    ) {
        std::vector<f64> zeros_x;
        std::vector<f64> zeros_y;
        zeros_x.reserve(16);
        zeros_y.reserve(16);
        for (auto e: Simulator(new_ctf, fftfreq_range)) {
            if (e.is_ctf_zero()) {
                f64 fftfreq = e.fftfreq();
                zeros_x.push_back(fftfreq);
                zeros_y.push_back(Simulator::sample_at(spectrum, fftfreq_range, fftfreq));
            }
        }

        if (zeros_x.size() < 2) {
            Logger::error(
                "CTF background fitting failed: less than two CTF zeros are within the frequency range. "
                "This is usually due to a very low defocus estimate, combined with a large pixel size "
                "and/or low resolution cutoff. Try to increase the resolution cutoff, and make sure the input data "
                "is not Fourier cropped to a low-resolution (the smaller the pixel size the better).\n"
                "defocus={:.2f}, phase_shift={:.2f}, spacing={:.2f}, fftfreq_cutoff={:.2f})",
                new_ctf.defocus(), new_ctf.phase_shift(), new_ctf.pixel_size(), fftfreq_range[1]
            );
            panic();
        }
        if (zeros_x.size() == 2) {
            Logger::warn(
               "Only two CTF zeros are located within the current frequency range. This is not great... "
               "This is usually due to a very low defocus estimate, combined with a large pixel size "
               "and/or low resolution cutoff. Instead of fitting a cubic spline, a line is fitted through "
               "these two points and it will be used as a background. However, this is unlikely to give good results."
               "Instead, you may want to increase the resolution cutoff, and make sure the input data is not Fourier "
               "cropped to a low-resolution (the smaller the pixel size the better).\n"
               "defocus={:.2f}, phase_shift={:.2f}, spacing={:.2f}, fftfreq_cutoff={:.2f})",
               new_ctf.defocus(), new_ctf.phase_shift(), new_ctf.pixel_size(), fftfreq_range[1]
           );
            // Add an extra point between the two and make it monotonic.
            zeros_x.push_back(zeros_x[0] + (zeros_x[1] - zeros_x[0]) / 2);
            zeros_y.push_back(zeros_y[0] + (zeros_y[1] - zeros_y[0]) / 2);
            std::swap(zeros_x[1], zeros_x[2]);
            std::swap(zeros_y[1], zeros_y[2]);
        }

        m_ctf = new_ctf;
        m_spline.fit(
            SpanContiguous(zeros_x.data(), static_cast<i64>(zeros_x.size())),
            SpanContiguous(zeros_y.data(), static_cast<i64>(zeros_y.size())), {
                .type = zeros_x.size() == 3 ? Spline::LINEAR : Spline::CSPLINE,
                .monotonic = true,
                .left = Spline::SECOND_DERIVATIVE,
                .right = Spline::SECOND_DERIVATIVE,
                .left_value = 0.,
                .right_value = 0.,
            });
    }

    auto Background::tune_fitting_range(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        f64 threshold,
        i32 keep_minimum
    ) const -> Vec<f64, 2> {
        // Collect fftfreq of zeros and peaks.
        std::vector<f64> zeros{};
        std::vector<f64> peaks{};
        zeros.reserve(10);
        peaks.reserve(10);
        for (auto& e: Simulator(m_ctf, fftfreq_range)) {
            if (e.is_ctf_zero())
                zeros.push_back(e.fftfreq());
            else if (e.is_ctf_peak())
                peaks.push_back(e.fftfreq());
        }

        // Tune low frequency base on the height of the first (or second) peak.
        auto fitting_range = fftfreq_range;
        const f64 fftfreq_peak = peaks[zeros[0] < peaks[0] ? 0 : 1];
        threshold *= (Simulator::sample_at(spectrum, fftfreq_range, fftfreq_peak) - sample_at(fftfreq_peak));

        const f64 fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
        for (i64 i{}; i < spectrum.ssize(); ++i) {
            const f64 fftfreq = fftfreq_range[0] + static_cast<f64>(i) * fftfreq_step;
            const f64 bs_spectrum = static_cast<f64>(spectrum[i]) - sample_at(fftfreq);
            if (bs_spectrum <= threshold) {
                fitting_range[0] = fftfreq;
                break;
            }
        }

        // Tune high frequency based on the quality of the peaks.
        // Keep the first 3 peaks and start tuning.
        for (auto i = static_cast<size_t>(keep_minimum); i < zeros.size() - 1; ++i) {
            auto peak_range = Vec{zeros[i], zeros[i + 1]};
            const f64 ncc = normalized_cross_correlation(spectrum, m_ctf, fftfreq_range, peak_range, *this);
            if (ncc < 0.45) {
                fitting_range[1] = zeros[i];
                break;
            }
        }
        return fitting_range;
    }

    void Background::sample(
        SpanContiguous<f32> spectrum,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
        for (i64 i{}; i < spectrum.ssize(); ++i) {
            const auto fftfreq = static_cast<f64>(i) * fftfreq_step + fftfreq_range[0];
            spectrum[i] = static_cast<f32>(m_spline.interpolate_at(fftfreq));
        }
    }

    void Background::sample(
        const View<f32>& spectrum,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        auto [b, d, h, w] = spectrum.shape();
        check(b == 1 and d == 1 and h == 1);
        sample(spectrum.reinterpret_as_cpu().span_1d_contiguous(), fftfreq_range);
    }

    void Background::subtract(
        SpanContiguous<const f32> input,
        SpanContiguous<f32> output,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        check(input.ssize() == output.ssize());
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(output.ssize() - 1);
        for (i64 i{}; i < output.ssize(); ++i) {
            const auto fftfreq = static_cast<f64>(i) * fftfreq_step + fftfreq_range[0];
            output[i] = input[i] - static_cast<f32>(m_spline.interpolate_at(fftfreq));
        }
    }

    void Background::subtract(
        const View<const f32>& input,
        const View<f32>& output,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        auto [b, d, h, w] = output.shape();
        check(noa::all(input.shape() == output.shape()));
        check(d == 1 and h == 1);

        const auto input_2d = input.reinterpret_as_cpu().span().filter(0, 3).as_contiguous();
        const auto output_2d = output.reinterpret_as_cpu().span().filter(0, 3).as_contiguous();
        for (i64 i{}; i < b; ++i)
            subtract(input_2d[i], output_2d[i], fftfreq_range);
    }
}
