#include <algorithm>
#include <noa/Signal.hpp>

#include "quinoa/ctf/CTF.hpp"
#include "quinoa/ctf/Baseline.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/SplineCurve.hpp"

namespace {
    auto best_start_index_(SpanContiguous<const f64> spectrum, isize start, isize end) {
        auto spectrum_windowed = spectrum.subregion(Slice{start, end + 1});
        auto sn = spectrum_windowed.ssize();

        // Detect the first full oscillation (zero, then peak) and compute the midpoint.
        bool has_minima{false};
        isize maxima{};
        isize minima{};
        for (isize i = 1; i < sn - 1; ++i) {
            const auto prev = spectrum_windowed[i - 1];
            const auto curr = spectrum_windowed[i];
            const auto next = spectrum_windowed[i + 1];

            if (curr < prev and curr < next) {
                has_minima = true;
                minima = i;
            } else if (has_minima and curr > prev and curr > next) {
                maxima = i;
                break;
            }
        }
        auto midpoint = (maxima + minima) / 2;

        // Revert if midpoint is past 25% of the start.
        const auto limit = sn / 4;
        if (midpoint > limit)
            midpoint = 0;

        start += minima;
        return start;
    }

    auto best_end_index_(SpanContiguous<f64> spectrum, isize start, isize end) {
        auto spectrum_windowed = spectrum.subregion(Slice{start, end + 1});
        auto sn = spectrum_windowed.ssize();

        // Detect lowpass.
        isize c{};
        for (isize i = sn - 1; i >= 0; --i) {
            if (noa::allclose(spectrum_windowed[i], 0.)) {
                c++;
            } else {
                break;
            }
        }
        check(c < sn / 2, "Data significantly lowpass filtered (more than half of the spectrum is zero). Please use unfiltered data");

        // Update spectrum window.
        end -= c;
        spectrum_windowed = spectrum.subregion(Slice{start, end + 1});
        sn = spectrum_windowed.ssize();

        // Then try to detect if the end of the spectrum is abnormal.
        // This works for the tested datasets, but is unlikely to work for everything.

        // Compute the gradient of the smoothed spectrum.
        auto gradient = std::vector<f64>{};
        auto gradient_abs = std::vector<f64>{};
        gradient.reserve(static_cast<usize>(sn));
        gradient_abs.reserve(static_cast<usize>(sn));
        for (isize i{}; i < sn - 1; ++i) {
            auto g = spectrum_windowed[i + 1] - spectrum_windowed[i];
            gradient.push_back(g);
            gradient_abs.push_back(std::abs(g));
        }

        // Set the signal threshold.
        auto [gradient_threshold, signal_threshold] = [&gradient_abs] {
            const auto quartile_75 = (gradient_abs.size() * 3) / 4;
            auto gradient_copy = gradient_abs;
            stdr::nth_element(gradient_copy, gradient_copy.begin() + static_cast<ptrdiff_t>(quartile_75));
            const f64 gradient_threshold_ = gradient_copy.at(quartile_75);

            f64 sum{}, sum_squares{};
            i32 count{};
            for (const auto& e: gradient_abs) {
                if (e < gradient_threshold_) {
                    sum += e;
                    sum_squares += e * e;
                    ++count;
                }
            }
            // TODO MAD
            const f64 background_mean = sum / static_cast<f64>(count);
            const f64 background_variance = sum_squares / static_cast<f64>(count) - (background_mean * background_mean);
            const f64 background_stddev = std::sqrt(background_variance);
            const f64 signal_threshold_ = std::min(0.5, background_mean + 6 * background_stddev);
            return Pair{gradient_threshold_, signal_threshold_};
        }();

        // First, collect the frequency windows above the thresholds.
        // These should be the regions where the gradient varies substantially up or down.
        bool is_within_window{};
        f64 max_value_within_window{};
        auto possible_windows = std::vector<Vec<isize, 2>>{};
        const isize offset = std::ssize(gradient_abs) - sn / 4;
        for (isize i{offset}, j{}; const auto& e: gradient_abs | stdv::drop(std::max(isize{0}, offset))) {
            max_value_within_window = std::max(max_value_within_window, e);
            if (not is_within_window and e >= signal_threshold) {
                is_within_window = true;
                j = i;
                max_value_within_window = -1;
            } else if (is_within_window and (e < signal_threshold or i == std::ssize(gradient_abs) - 1)) {
                is_within_window = false;
                const auto window_size_ = i - j;
                if (window_size_ >= 3 and max_value_within_window >= gradient_threshold)
                    possible_windows.push_back({j, i});
            }
            ++i;
        }
        if (possible_windows.empty())
            return end;

        // Then, fuse windows that are close to each other.
        // To prevent fusing and removing CTF^2 oscillations, only fuse if the sign of the gradient is the same.
        auto are_gradient_signs_equal = [&](const Vec<isize, 2>& lhs, const Vec<isize, 2>& rhs) {
            const auto lhs_ = gradient[static_cast<usize>(lhs[0])];
            const auto rhs_ = gradient[static_cast<usize>(rhs[0])];
            return std::signbit(lhs_) == std::signbit(rhs_);
        };
        const auto maximum_distance_between_windows = std::max(isize{1}, sn / 20);
        for (usize i{}; i < possible_windows.size() - 1; ++i) {
            const isize distance = possible_windows[i + 1][0] - possible_windows[i][1];
            if (distance <= maximum_distance_between_windows and
                are_gradient_signs_equal(possible_windows[i], possible_windows[i + 1])) {
                possible_windows[i + 1][0] = possible_windows[i][0];
                possible_windows[i][0] = -1;
            }
        }
        std::erase_if(possible_windows, [](const auto& window) { return window[0] == -1; });

        // Finally, remove the last window if it's near the end of the spectrum.
        // Remove at most one third of the valid spectrum window.
        const auto maximum_distance_from_end = std::max(isize{1}, sn / 20);
        auto last_possible_window = possible_windows.back();
        if (last_possible_window[1] >= sn - maximum_distance_from_end and
            (last_possible_window[1] - last_possible_window[0]) < sn / 3)
            end = last_possible_window[0];

        return end;
    }

    void gaussian_smoothing_(
        SpanContiguous<const f32> spectrum,
        SpanContiguous<f64> spectrum_smooth,
        isize kernel_size,
        f64 stddev
    ) {
        const auto filter = ns::window_gaussian<f32>(kernel_size, stddev, {.normalize = true});
        ns::convolve(View(spectrum), View(spectrum_smooth), filter.view(), {.border = noa::Border::REFLECT});
    }

    void fit_spline_(
        Spline& spline,
        SpanContiguous<const f64> x,
        SpanContiguous<const f64> y,
        SpanContiguous<f64> z,
        GaussianSlider low_resolution_smoothing
    ) {
        // Least-square fitting of a cubic spline onto the midpoints.
        asymmetric_least_squares_smoothing(x, y, z, {
            .smoothing = low_resolution_smoothing,
            .asymmetry = GaussianSlider::from_constant(0.5),
            .max_iter = 50,
            .relaxation = 0.9,
        });

        // Transform the fitted spline into a piecewise polynomial for interpolation.
        spline.fit(x, z, {
            .type = Spline::CSPLINE,
            .monotonic = true,
            .left = Spline::SECOND_DERIVATIVE,
            .right = Spline::SECOND_DERIVATIVE,
            .left_value = 0,
            .right_value = 0,
        });
    }
}

namespace qn::ctf {
    auto Baseline::fit(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const Vec<f64, 2>& fitting_range
    ) -> Vec<f64, 2> {
        // Allocate temporary buffers.
        const auto buffer = Array<f64>({3, 1, 1, spectrum.ssize()});

        // Adjust the spectrum window to only include the fitting range.
        auto start = nearest_integer_fftfreq(spectrum.ssize(), fftfreq_range, fitting_range[0], true).first;
        auto end = nearest_integer_fftfreq(spectrum.ssize(), fftfreq_range, fitting_range[1], true).first;

        // Adjust the spectrum window based on the spectrum signal.
        const auto spectrum_smooth = buffer.view().subregion(0).span_1d();
        gaussian_smoothing_(spectrum, spectrum_smooth, 11, 1.5);
        start = best_start_index_(spectrum_smooth, start, end);
        end = best_end_index_(spectrum_smooth, start, end);

        // Get the final window.
        const auto original_size = spectrum.size();
        spectrum = spectrum.subregion(Slice{start, end + 1});
        const auto fftfreq_step = noa::Linspace<f64>::from_vec(fftfreq_range).for_size(original_size).step;
        const auto fftfreq_start = fftfreq_range[0] + static_cast<f64>(start) * fftfreq_step;
        const auto fftfreq_end = fftfreq_range[0] + static_cast<f64>(end) * fftfreq_step;
        const auto new_size = spectrum.ssize();

        // Compute the spline.
        const auto x = buffer.span().subregion(0).as_1d().subregion(Slice{0, new_size});
        const auto y = buffer.span().subregion(1).as_1d().subregion(Slice{0, new_size});
        const auto z = buffer.span().subregion(2).as_1d().subregion(Slice{0, new_size});
        for (isize i{}; i < new_size; ++i)
            x[i] = fftfreq_start + static_cast<f64>(i) * fftfreq_step;
        gaussian_smoothing_(spectrum, y, 21, 2);
        fit_spline_(spline, x, y, z, {
            .peak_coordinate = 0.,
            .peak_value = 1e-4,
            .base_width = 0.6,
            .base_value = 1e-5,
        });

        return {fftfreq_start, fftfreq_end};
    }

    auto Baseline::fit(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const CTFIsotropic64& ctf
    ) -> Vec<f64, 2> {
        // Normalize distance until which we use midpoints for the baseline.
        // Originally 0.5, to only use midpoints for the first half of the spectrum, but in practice in doesn't make
        // much difference. Use 1 just in case large oscillations are present even towards the end of the spectrum.
        // For very low defoci, where the CTF has only a few midpoints, the program will fall back to spline only.
        constexpr f64 PIVOT = 1.0;
        constexpr isize MINIMUM_N_MIDPOINTS = 6;

        // Allocate temporary buffers.
        const auto sn = spectrum.ssize();
        const auto buffer = Array<f64>({3, 1, 1, sn});

        // Get the fftfreq of the first zero.
        f64 fftfreq_start{};
        for (const auto& e: Simulate(ctf, fftfreq_range)) {
            if (e.is_ctf_zero()) {
                // Step back by one, just to make sure the next step doesn't miss the zero.
                fftfreq_start = e.fftfreq() - e.fftfreq_step();
                break;
            }
        }

        // Collect fftfreq at zeros and peaks of the spectrum until reaching the pivot point.
        const auto fftfreq_stop = fftfreq_start + (fftfreq_range[1] - fftfreq_start) * PIVOT;
        std::vector<f64> extrema{};
        extrema.reserve(50);
        for (const auto& e: Simulate(ctf, Vec{fftfreq_start, fftfreq_stop})) {
            if (e.is_ctf_vertex())
                extrema.push_back(e.fftfreq());
        }
        if (auto s = std::ssize(extrema); s < MINIMUM_N_MIDPOINTS or s >= sn) {
            // Too few extrema probably due to very low defocus/spacing ratio (which may be due to an incorrect
            // initial or coarse fit because of strong astigmatism). Regardless of the reason, fall back to spline only.
            return fit(spectrum, fftfreq_range, fftfreq_range);
        }

        // Substantial Gaussian-smoothing of the spectrum.
        const auto spectrum_smooth = buffer.view().subregion(0).span_1d();
        gaussian_smoothing_(spectrum, spectrum_smooth, 21, 2);

        // Collect the oscillation midpoints.
        auto midpoints_x = buffer.span().subregion(1).as_1d();
        auto midpoints_y = buffer.span().subregion(2).as_1d();
        isize n_points{};
        for (usize i{}; i < extrema.size() - 2; ++i) {
            const auto fftfreq_0 = extrema[i];
            const auto fftfreq_1 = extrema[i + 1];
            const auto fftfreq_2 = extrema[i + 2];

            const auto fftfreq_midpoint_0 = (fftfreq_1 + fftfreq_0) / 2;
            const auto fftfreq_midpoint_1 = (fftfreq_2 + fftfreq_1) / 2;
            midpoints_x[i] = (fftfreq_midpoint_1 + fftfreq_midpoint_0) / 2;

            const auto value_midpoint_0 = Simulate::sample_at(spectrum_smooth, fftfreq_range, fftfreq_midpoint_0);
            const auto value_midpoint_1 = Simulate::sample_at(spectrum_smooth, fftfreq_range, fftfreq_midpoint_1);
            midpoints_y[i] = (value_midpoint_1 + value_midpoint_0) / 2;

            ++n_points;
        }

        // Add to the midpoints the smooth spectrum from the pivot to the end.
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(sn - 1);
        const auto last_midpoint = midpoints_x[n_points - 1];
        for (isize i = 0; i < sn; ++i) {
            const auto fftfreq = fftfreq_range[0] + static_cast<f64>(i) * fftfreq_step;
            if (fftfreq > last_midpoint) {
                midpoints_x[n_points] = fftfreq;
                midpoints_y[n_points] = spectrum_smooth[i];
                ++n_points;
            }
        }

        // Adjust the spectrum end based on the spectrum signal.
        gaussian_smoothing_(spectrum, spectrum_smooth, 11, 1.5);
        const auto start = static_cast<isize>((midpoints_x[0] - fftfreq_range[0]) / fftfreq_step);
        const auto end = best_end_index_(spectrum_smooth, start, sn - 1);
        const auto end_fftfreq = fftfreq_range[0] + static_cast<f64>(end) * fftfreq_step;
        for (isize i{}; i < n_points; ++i) {
            if (midpoints_x[i] > end_fftfreq) {
                n_points = i; // stop at previous point: (i-1)+1
                break;
            }
        }

        // Compute the spline. Follow the data more closely since midpoints
        // should already be tracing a relatively smooth path through the background.
        // TODO As opposed to decrease the smoothing if certain resolution are reached (e.g. 3.7A bump of amorphous ice)
        //      or increase the smoothing if the resolution range is small, which I expect may be necessary for some
        //      data, it would probably be better to fit, subtract, and fit-again. This is already done when computing
        //      the EPA of the stack for diagnostics, but should be safe to do here. In practice, the CC is very
        //      resilient to errors in the background, so I should not worry about this now.
        const auto x = midpoints_x.subregion(Slice{0, n_points});
        const auto y = midpoints_y.subregion(Slice{0, n_points});
        const auto z = spectrum_smooth.subregion(Slice{0, n_points});
        fit_spline_(spline, x, y, z, {
            .peak_coordinate = 0.,
            .peak_value = 1e-5,
            .base_width = 0.5,
            .base_value = 1e-6,
        });

        return {midpoints_x[0], midpoints_x[n_points - 1]};
    }

    auto Baseline::tune_fitting_range(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const CTFIsotropic64& ctf,
        const BaselineTuningOptions& options
    ) const -> Vec<f64, 2> {
        const auto thickness_modulation = ThicknessModulation<true>{
            .wavelength = ctf.wavelength(),
            .spacing = ctf.pixel_size(),
            .thickness = options.thickness_um * 1e4
        };

        // Collect fftfreq of zeros and peaks.
        std::vector<f64> zeros{};
        std::vector<f64> peaks{};
        zeros.reserve(20);
        peaks.reserve(20);
        for (auto& e: Simulate(ctf, fftfreq_range)) {
            if (not e.is_ctf_vertex())
                continue;

            const auto fftfreq = e.fftfreq();
            const auto modulation = thickness_modulation.sample_at(fftfreq);
            if (std::abs(modulation) >= 0.95) { // if too close to a node, skip it
                bool is_zero = e.is_ctf_zero();
                if (modulation < 0)
                    is_zero = not is_zero; // flipped, zero<->peak

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
        const f64 high_threshold = options.threshold * bs_peak;

        const f64 fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
        for (isize i{}; i < spectrum.ssize(); ++i) {
            const f64 fftfreq = fftfreq_range[0] + static_cast<f64>(i) * fftfreq_step;
            const f64 bs_spectrum = static_cast<f64>(spectrum[i]) - sample_at(fftfreq);
            if (bs_spectrum <= high_threshold) {
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
            if (thickness_modulation.is_fftfreq_range_containing_node(peak_range))
                continue;

            const f64 ncc = zero_normalized_cross_correlation(spectrum, ctf, fftfreq_range, peak_range, *this, thickness_modulation);

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

        // Clamp the end of the fitting range with the cutoff used for the baseline.
        // This cutoff guards against big variations of the spectrum where the baseline correction
        // will be subpar at best. When adding extra peaks "blindly", this guard can be useful.
        last_zero += static_cast<size_t>(options.n_extra_peaks_to_append);
        const auto m_fftfreq_stop = spline.x()[spline.x().size() - 1];
        fitting_range[1] = std::min(m_fftfreq_stop, zeros[std::min(last_zero, zeros.size() - 1)]);
        return fitting_range;
    }

    void Baseline::sample(
        SpanContiguous<f32> spectrum,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
        for (isize i{}; i < spectrum.ssize(); ++i) {
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
        sample(spectrum.reinterpret_as_cpu().span_1d(), fftfreq_range);
    }

    void Baseline::subtract(
        SpanContiguous<const f32> input,
        SpanContiguous<f32> output,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        check(input.ssize() == output.ssize());
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(output.ssize() - 1);
        for (isize i{}; i < output.ssize(); ++i) {
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
        check(input.shape() == output.shape());
        check(d == 1 and h == 1);

        const auto input_2d = input.reinterpret_as_cpu().span().filter(0, 3).as_contiguous();
        const auto output_2d = output.reinterpret_as_cpu().span().filter(0, 3).as_contiguous();
        for (isize i{}; i < b; ++i)
            subtract(input_2d[i], output_2d[i], fftfreq_range);
    }
}
