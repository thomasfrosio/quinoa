#pragma once

#include <noa/Array.hpp>
#include "quinoa/Types.hpp"

namespace qn::ctf {
    struct BaselineTuningOptions {
        /// Low-frequency exclusion threshold.
        /// The fitting range will start at the first point in the spectrum that goes below the height of
        /// first peak times this threshold.
        f64 threshold = 1.5;

        /// NCC between the baseline-subtracted peak and simulated CTF at which the peak is considered bad.
        f64 minimum_ncc = 0.45;

        /// Keep at least that number of peaks.
        i32 keep_first_nth_peaks = 3;

        /// Number of peaks to add after the last good peak.
        i32 n_extra_peaks_to_append = 0;

        /// Number of recoveries allowed. If we detect a bad peak, we look ahead to the next peak(s) until the
        /// maximum_n_consecutive_bad_peaks is reached. If a good peak is found, the tuning will continue extending
        /// the fitting range.
        i32 n_recoveries_allowed = 1;

        /// The number of consecutive bad peaks allowed before the next good peak.
        i32 maximum_n_consecutive_bad_peaks = 1;

        /// Same as minimum_ncc, but for what is considered a "good peak" for a recovery.
        /// Zero means minimum_ncc. This can be used to recover only very good peaks.
        f64 minimum_ncc_for_recovery = 0;

        /// Thickness of the sample. This depends on the stage angles (see effective_thickness).
        /// Zero means that the classic CTF model should be used. Otherwise, the thickness modulation applied.
        /// Note that regions near the nodes are skipped from the tuning and automatically included.
        f64 thickness_um = 0;
    };

    /// Smooth baseline of a 1d power-spectrum.
    class Baseline {
    public:
        /// Get the best fitting-range for the baseline fit.
        static auto best_baseline_fitting_range(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFIsotropic<f64>& ctf
        ) -> Vec<f64, 2>;

    public:
        /// Creates an empty baseline. Use fit().
        Baseline() = default;

        /// Fits a smooth baseline through the spectrum.
        /// Only the region within the given fitting range is used for the fitting.
        void fit(SpanContiguous<const f32> spectrum, const Vec<f64, 2>& fftfreq_range, const Vec<f64, 2>& fitting_range);

        /// Fits a smooth baseline through the spectrum.
        /// Uses the CTF to find the best fitting range.
        void fit(SpanContiguous<const f32> spectrum, const Vec<f64, 2>& fftfreq_range, const ns::CTFIsotropic<f64>& ctf) {
            auto fitting_range = best_baseline_fitting_range(spectrum, fftfreq_range, ctf);
            fit(spectrum, fftfreq_range, fitting_range);
        }

        /// Tunes the fitting range for subsequent cross-correlation between a CTF and the baseline-subtracted spectrum.
        /// \details Finds the first frequency where the baseline-subtracted spectrum is getting below the height
        ///          of the first CTF peak (*threshold). Then, sets the last fftfreq based on the quality of the
        ///          per-peak NCC.
        /// \param spectrum         Raw power spectrum (not background subtracted).
        /// \param fftfreq_range    Frequency range of the spectrum. The returned frequency is within this range.
        /// \param options          Tuning options.
        [[nodiscard]] auto tune_fitting_range(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFIsotropic<f64>& ctf,
            const BaselineTuningOptions& options = {}
        ) const -> Vec<f64, 2>;

        /// Fits the baseline and tune the fitting range, iteratively.
        auto fit_and_tune_fitting_range(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFIsotropic<f64>& ctf,
            const BaselineTuningOptions& options = {}
        ) -> Vec<f64, 2> {
            fit(spectrum, fftfreq_range, ctf);
            return tune_fitting_range(spectrum, fftfreq_range, ctf, options);
        }

        /// Samples the background on 1d spectrum.
        void sample(
            SpanContiguous<f32> spectrum,
            const Vec<f64, 2>& fftfreq_range
        ) const;
        void sample(
            const View<f32>& spectrum,
            const Vec<f64, 2>& fftfreq_range
        ) const;

        /// Subtract the background from 1d spectrum(s).
        void subtract(
            SpanContiguous<const f32> input,
            SpanContiguous<f32> output,
            const Vec<f64, 2>& fftfreq_range
        ) const;
        void subtract(
            const View<const f32>& input,
            const View<f32>& output,
            const Vec<f64, 2>& fftfreq_range
        ) const;

        /// Evaluate the baseline at a given fftfreq.
        /// Values outside the spectrum frequency-range are extrapolated.
        /// The extrapolation is set to preserve the spline slope at the edges.
        [[nodiscard]] auto sample_at(f64 fftfreq) const -> f64 {
            const i64 n = m_a.ssize();
            const i64 last_index = n - 1;
            if (n == 0)
                return 0;

            // Denormalize fftfreq back to frequencies.
            const f64 frequency = (fftfreq - m_fftfreq_start) / m_fftfreq_step;
            const i64 index = static_cast<i64>(std::floor(frequency));
            const f64 fraction = frequency - noa::clamp(std::floor(frequency), 0, static_cast<f64>(last_index));

            // Polynomial evaluation (Horner's scheme) - interpolation.
            // This is assuming x is uniform [0,n).
            if (index < 0)
                return m_a[0] + fraction * m_b[0]; // extrapolation to the left
            if (index >= last_index)
                return  m_a[last_index] + fraction * m_b[last_index]; // extrapolation to the right
            return ((m_d[index] * fraction + m_c[index]) * fraction + m_b[index]) * fraction + m_a[index];
        }

    private:
        void allocate_(i64 n) {
            const i64 n_to_allocate = n * 4;
            if (m_buffer == nullptr or n_allocated < n_to_allocate) {
                // Allocate a bit more to increase the chance of reusing the buffer
                // when the fitting is done with slightly different fitting ranges.
                n_allocated = n_to_allocate + 20 * 4;
                m_buffer = std::make_unique<f64[]>(static_cast<size_t>(n_allocated));
            }
            m_a = SpanContiguous(m_buffer.get() + n * 0, n);
            m_b = SpanContiguous(m_buffer.get() + n * 1, n);
            m_c = SpanContiguous(m_buffer.get() + n * 2, n);
            m_d = SpanContiguous(m_buffer.get() + n * 3, n);
        }

    private:
        std::unique_ptr<f64[]> m_buffer; // (n*4)
        i64 n_allocated{};

        // Cubic spline expressed as a set of cubic polynomials.
        // f(x) = a_x + b_x + c_x^2 + d_x^3, where a_x = y_x, and x is uniform [0 to n-1].
        // These are all views of the main buffer.
        SpanContiguous<f64> m_a;
        SpanContiguous<f64> m_b;
        SpanContiguous<f64> m_c;
        SpanContiguous<f64> m_d;

        f64 m_fftfreq_start{};
        f64 m_fftfreq_step{};
        f64 m_fftfreq_stop{0.5};
    };
}
