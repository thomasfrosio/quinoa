#pragma once

#include <noa/Runtime.hpp>
#include "quinoa/Types.hpp"
#include "quinoa/SplineCurve.hpp"

namespace qn::ctf {
    struct BaselineTuningOptions {
        /// Low-frequency exclusion threshold.
        /// The fitting range will start at the first point in the spectrum that goes below the height of
        /// first peak times this threshold.
        f64 threshold = 1.5;

        /// NCC between the baseline-subtracted peak and simulated CTF at which a peak is considered bad.
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

        /// Thickness of the sample used for the thickness modulation curve. Note that this depends on the stage angles
        /// (see effective_thickness). Zero is equivalent to the classic CTF model. Note that regions near the nodes
        /// are skipped from the tuning and automatically included.
        f64 thickness_um = 0;
    };

    /// Smooth baseline of a 1d power-spectrum.
    /// TODO Rename to Background
    class Baseline {
    public:
        Spline spline;

    public:
        /// Fits a smooth spline through the spectrum.
        /// Returns the adjusted fitting range used for the fitting.
        auto fit(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const Vec<f64, 2>& fitting_range
        ) -> Vec<f64, 2>;

        /// Fits a smooth spline through the spectrum.
        /// Instead of using the first half of the spectrum, where oscillations can be challenging to erase,
        /// the CTF estimate is used to get the values at the expected midpoints of the oscillations.
        /// These values, with the second half of the spectrum, are used for the fitting.
        /// Returns the adjusted fitting range used for the fitting.
        auto fit(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const CTFIsotropic64& ctf
        ) -> Vec<f64, 2>;

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
            const CTFIsotropic64& ctf,
            const BaselineTuningOptions& options = {}
        ) const -> Vec<f64, 2>;

        /// Fits the baseline and tune the fitting range, iteratively.
        auto fit_and_tune_fitting_range(
            SpanContiguous<const f32> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const CTFIsotropic64& ctf,
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
            return spline.interpolate_at(fftfreq);
        }
    };
}
