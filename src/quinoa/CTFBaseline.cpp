#include "quinoa/CTF.hpp"
#include "quinoa/CTFBaseline.hpp"
#include "quinoa/Utilities.hpp"

namespace qn::ctf {
    auto Baseline::best_fitting_range(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const ns::CTFIsotropic<f64>& ctf
    ) -> Vec<f64, 2> {
        // The signal at very low frequencies can be quite off the rest of the spectrum to the point that
        // despite the strong smoothing of the baseline, the overall fit at low-mid frequency might be degraded.
        // As such, remove this region from the fit.

        // Get the midpoint value between the first zero and the following peak.
        f64 target_value{};
        bool got_zero{};
        for (auto e: Simulate(ctf, fftfreq_range)) {
            if (not got_zero and e.is_ctf_zero()) {
                target_value += Simulate::sample_at(spectrum, fftfreq_range, e.fftfreq());
                got_zero = true;
            }
            if (got_zero and e.is_ctf_peak()) {
                target_value += Simulate::sample_at(spectrum, fftfreq_range, e.fftfreq());
                break;
            }
        }
        target_value /= 2;
        target_value *= 1.2;

        // Stop at the frequency that crosses target_value.
        auto fitting_range = fftfreq_range;
        for (i64 i{}; i < spectrum.ssize(); ++i) {
            if (spectrum[i] <= static_cast<f32>(target_value)) {
                auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
                fitting_range[0] = fftfreq_range[0] + static_cast<f64>(i) * fftfreq_step;
                break;
            }
        }

        return fitting_range;
    }

    void Baseline::fit(SpanContiguous<const f32> spectrum, const Vec<f64, 2>& fftfreq_range, const Vec<f64, 2>& fitting_range) {
        // Adjust the spectrum window to only include the fitting range.
        auto [start, fftfreq_start] = nearest_integer_fftfreq(spectrum.ssize(), fftfreq_range, fitting_range[0], true);
        auto [end, fftfreq_end] = nearest_integer_fftfreq(spectrum.ssize(), fftfreq_range, fitting_range[1], true);
        auto actual_fitting_range = Vec{fftfreq_start, fftfreq_end};
        spectrum = spectrum.subregion(ni::Slice{start, end + 1});

        allocate_(spectrum.ssize());

        // Save fftfreq range for sample_at().
        m_fftfreq_start = actual_fitting_range[0];
        m_fftfreq_step = noa::Linspace<f64>::from_vec(actual_fitting_range).for_size(spectrum.ssize()).step;

        // Baseline fit using an Asymmetric Least Squares Smoothing (ALS) algorithm.
        // - The asymmetric part isn't used (p=0.5) since we want to go through the Thon rings.
        // - We use varying penalization (smoothing), since Thon rings decay with the frequency.
        //   This helps to guarantee that the baseline follows the spectrum at high frequency and
        //   doesn't drift off to accommodate the fitting at low frequencies.
        constexpr auto SMOOTHING = GaussianSlider{
            .peak_coordinate = 0.,
            .peak_value = 70'000,
            .base_width = 0.75,
            .base_value = 10'000,
        };
        asymmetric_least_squares_smoothing(spectrum, m_a, {.smoothing = SMOOTHING, .asymmetric_penalty = 0.5});

        // While we could have used a smoothing spline, the ALS seems to be giving better results at the edges.
        // Plus, for fast evaluation we need to express the spline as a piecewise polynomial, which isn't practical
        // when fitting a penalized spline, so we would have needed to do a conversion anyway.

        // Instead, fit an interpolating spline directly onto the baseline.
        // This spline is stored and will be queried many times for evaluation (interpolation and extrapolation).
        interpolating_uniform_cubic_spline(m_a, m_b, m_c, m_d);
    }

    auto Baseline::tune_fitting_range(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const ns::CTFIsotropic<f64>& ctf,
        const BaselineTuningOptions& options
    ) const -> Vec<f64, 2> {
        const auto thickness_modulation = ThicknessModulation{
            .wavelength = ctf.wavelength(),
            .spacing = ctf.pixel_size(),
            .thickness = options.thickness_um * 1e4
        };

        // Collect fftfreq of zeros and peaks.
        std::vector<f64> zeros{};
        std::vector<f64> peaks{};
        zeros.reserve(10);
        peaks.reserve(10);
        for (auto& e: Simulate(ctf, fftfreq_range)) {
            if (not e.is_ctf_vertex())
                continue;

            const auto fftfreq = e.fftfreq();
            const auto modulation = thickness_modulation.sample_at(fftfreq);
            if (std::abs(modulation) >= 0.9) { // if too close to a node, skip it
                bool is_zero = e.is_ctf_zero();
                if (modulation < 0)
                    is_zero = !is_zero; // flipped, zero<->peak

                if (is_zero)
                    zeros.push_back(fftfreq);
                else
                    peaks.push_back(fftfreq);
            }
        }
        check(zeros.size() >= 2 and peaks.size() >= 2,
              "Something is wrong... Too few CTF zeros and peaks detected. "
              "n_zeros={}, n_peaks={}, fftfreq_range={::.3f}, defocus={:.3f}",
              zeros.size(), peaks.size(), fftfreq_range, ctf.defocus());

        // Tune low frequency based on the height of the first (or second) peak within fftfreq_range.
        auto fitting_range = fftfreq_range;
        const f64 fftfreq_peak = peaks[zeros[0] < peaks[0] ? 0 : 1];
        const f64 bs_peak = Simulate::sample_at(spectrum, fftfreq_range, fftfreq_peak) - sample_at(fftfreq_peak);
        const f64 threshold = options.threshold * bs_peak;

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
        const f64 minimum_ncc_for_recovery =
            options.minimum_ncc_for_recovery == 0 ?
            options.minimum_ncc : options.minimum_ncc_for_recovery;

        i32 n_recoveries{};
        i32 n_consecutive_bad_peaks{};
        size_t last_zero{};
        for (auto i = static_cast<size_t>(options.keep_first_nth_peaks); i < zeros.size() - 1; ++i) {
            auto peak_range = Vec{zeros[i], zeros[i + 1]};
            if (thickness_modulation.is_frequency_range_within_node_transition(peak_range, 0.9))
                continue;

            const f64 ncc = zero_normalized_cross_correlation(spectrum, ctf, fftfreq_range, peak_range, *this);

            f64 minimum_ncc = n_consecutive_bad_peaks > 0 ? minimum_ncc_for_recovery : options.minimum_ncc;
            if (ncc < minimum_ncc) {
                // If recovery isn't allowed, or if we passed the number of recoveries allowed,
                // or if the maximum number of consecutive bad peaks has been reached, save the
                // end of the last good peak and break.
                if (n_recoveries >= options.n_recoveries_allowed or
                    n_consecutive_bad_peaks >= options.maximum_n_consecutive_bad_peaks) {
                    last_zero = i - static_cast<size_t>(n_consecutive_bad_peaks);
                    break;
                }

                // A bad peak was detected, but we may still recover from it.
                n_consecutive_bad_peaks++;
            } else if (n_consecutive_bad_peaks > 0) {
                // A good peak was found, we managed to recover.
                n_consecutive_bad_peaks = 0;
                n_recoveries++;
                last_zero = i + 1;
            } else {
                last_zero = i + 1;
            }
        }

        last_zero += static_cast<size_t>(options.n_extra_peaks_to_append);
        fitting_range[1] = zeros[std::min(last_zero, zeros.size() - 1)];
        return fitting_range;
    }

    void Baseline::sample(
        SpanContiguous<f32> spectrum,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
        for (i64 i{}; i < spectrum.ssize(); ++i) {
            const auto fftfreq = static_cast<f64>(i) * fftfreq_step + fftfreq_range[0];
            spectrum[i] = static_cast<f32>(sample_at(fftfreq));
        }
    }

    void Baseline::sample(
        const View<f32>& spectrum,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        auto [b, d, h, w] = spectrum.shape();
        check(b == 1 and d == 1 and h == 1);
        sample(spectrum.reinterpret_as_cpu().span_1d_contiguous(), fftfreq_range);
    }

    void Baseline::subtract(
        SpanContiguous<const f32> input,
        SpanContiguous<f32> output,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        check(input.ssize() == output.ssize());
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(output.ssize() - 1);
        for (i64 i{}; i < output.ssize(); ++i) {
            const auto fftfreq = static_cast<f64>(i) * fftfreq_step + fftfreq_range[0];
            output[i] = input[i] - static_cast<f32>(sample_at(fftfreq));
        }
    }

    void Baseline::subtract(
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
