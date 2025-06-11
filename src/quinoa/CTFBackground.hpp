#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/SplineCurve.hpp"
#include "quinoa/SplineGrid.hpp"

namespace qn::ctf {
    struct Background {
    public:
        /// For fitting the background (which depends on CTF zeroes),
        /// there's a limit how on how low the defocus can be.
        [[nodiscard]] static auto smallest_defocus_for_fitting(
            const Vec<f64, 2>& fftfreq_range,
            ns::CTFIsotropic<f64> ctf,
            i64 n_zeroes = 3
        ) -> f64;

    public:
        /// Creates an empty background. Use fit().
        Background() { m_spline.reserve(20); }

        /// Creates and fits the background to the spectrum. See fit().
        Background(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFIsotropic<f64>& new_ctf
        ) : Background() {
            fit(spectrum, fftfreq_range, new_ctf);
        }

        /// Fits a spline through the CTF zeros of the spectrum.
        void fit(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFIsotropic<f64>& new_ctf
        );

        /// Finds the first frequency where the spectrum is getting below the height of the first CTF peak.
        /// Then, sets the last fftfreq based on the quality of the per-peak NCC.
        /// \param spectrum         Raw power spectrum (not background subtracted).
        /// \param fftfreq_range    Frequency range of the spectrum. The returned frequency is within this range.
        /// \param threshold        Threshold relative to the first CTF peak height.
        /// \param keep_minimum     Keep at least that number of peaks.
        [[nodiscard]] auto tune_fitting_range(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            f64 threshold = 1.5,
            i32 keep_minimum = 3
        ) const -> Vec<f64, 2>;

        [[nodiscard]] auto fit_and_tune_fitting_range(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFIsotropic<f64>& new_ctf,
            f64 threshold = 1.5,
            i32 keep_minimum = 3
        ) -> Vec<f64, 2> {
            fit(spectrum, fftfreq_range, new_ctf);
            return tune_fitting_range(spectrum, fftfreq_range, threshold, keep_minimum);
        }

        [[nodiscard]] auto sample_at(f64 fftfreq) const -> f64 {
            return m_spline.interpolate_at(fftfreq);
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

    private:
        Spline m_spline{};
        ns::CTFIsotropic<f64> m_ctf{};
    };
}
