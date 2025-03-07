#include <noa/Signal.hpp>
#include <noa/Geometry.hpp>

#include "quinoa/CTF.hpp"

namespace {
    using namespace qn;
    using namespace qn::ctf;

    template<typename T>
    struct SmoothCurveFittingData {
        SpanContiguous<const T> span{};
        CubicSplineGrid<f64, 1> spline{};
        f64 norm{};

        auto get_coordinate(i64 i) {
            if constexpr (nt::real<T>) {
                return static_cast<f64>(i) * norm; // uniform spacing
            } else {
                return span[i][0];
            }
        }

        auto get_value(i64 i) {
            if constexpr (nt::real<T>) {
                return static_cast<f64>(span[i]);
            } else {
                return static_cast<f64>(span[i][1]);
            }
        }
    };

    template<typename T>
    void smooth_curve_fitting(
        const SpanContiguous<const T>& span,
        const SpanContiguous<f64>& spline
    ) {
        // Compute the least-square score between the experimental points and the model.
        auto cost = [](u32, const f64* params, f64*, void* instance) {
            auto& opt = *static_cast<SmoothCurveFittingData<T>*>(instance);
            check(params == opt.spline.span().data());

            f64 score{};
            for (i64 i{}; i < opt.span.ssize(); ++i) {
                const f64 coordinate = opt.get_coordinate(i);
                const f64 experiment = opt.get_value(i);
                const f64 predicted = opt.spline.interpolate_at(coordinate);
                const f64 diff = experiment - predicted;
                score += diff * diff;
            }
            // fmt::println("spline={}, score={}", opt.spline.span()[0], score);
            return score;
        };

        auto optimizer_data = SmoothCurveFittingData<T>{
            .span = span,
            .spline = CubicSplineGrid<f64, 1>(
                spline.ssize(), 1,
                SpanContiguous(spline.data(), spline.shape().push_front(1))
            ),
            .norm = 1 / static_cast<f64>(span.ssize() - 1), // [0,1] inclusive
        };

        // Some stats about the data points.
        f64 mean{}, min{std::numeric_limits<f64>::max()}, max{std::numeric_limits<f64>::lowest()};
        for (i64 i{}; i < span.ssize(); ++i) {
            const auto value = optimizer_data.get_value(i);
            mean += value;
            min = std::min(min, value);
            max = std::max(max, value);
        }
        mean *= optimizer_data.norm;

        // Initialize the spline to a line at the mean.
        for (auto& v: spline)
            v = mean;

        // Here use a local derivative-less algorithm, since we assume
        // the spline resolution is <= 5, and the cost is cheap to compute.
        auto optimizer = Optimizer(NLOPT_LN_SBPLX, spline.ssize());
        optimizer.set_min_objective(cost, &optimizer_data);

        // Usually min/max are large values, but the range is small, so use that to set the bounds.
        const auto value_range = max - min;
        optimizer.set_bounds(min - value_range * 5, max + value_range * 5);
        optimizer.set_fx_tolerance_abs(1e-5);
        optimizer.optimize(spline.data());
        fmt::println("n_evaluations={}", optimizer.n_evaluations());
    }

    struct FetchedCTFValues {
        // [0]=frequency between [0,1], [1]=value
        std::vector<f64> zeros_x;
        std::vector<f64> zeros_y;
        std::vector<f64> peaks_x;
        std::vector<f64> peaks_y;
    };

    [[nodiscard]] auto fetch_ctf_values(
        const SpanContiguous<const f32>& rotational_average,
        const Vec<f64, 2>& fftfreq_range,
        const ns::CTFIsotropic<f64>& ctf,
        bool fit_envelope = false
    ) -> FetchedCTFValues {
        // Simulate ctf slope with more than enough sampling to precisely detect the CTF zeros and peaks.
        constexpr i64 SIMULATED_SIZE = 2 * 8192;
        constexpr f64 SIMULATED_FREQUENCY_STEP = 1 / static_cast<f64>(SIMULATED_SIZE);

        // Only evaluates the frequencies that are within the fftfreq range.
        const auto range_index = noa::round(fftfreq_range * SIMULATED_SIZE).as<i64>();
        i64 simulated_start = std::max(range_index[0] - 3, i64{0});
        i64 simulated_end = std::min(range_index[1] + 3, SIMULATED_SIZE - 1);

        // Go from the index in the simulated spectrum, to the index in the trimmed spectrum.
        const auto n_samples_in_range = rotational_average.ssize();
        const auto range_norm = static_cast<f64>(n_samples_in_range - 1); // -1 for inclusive
        const auto fftfreq_range_step = (fftfreq_range[1] - fftfreq_range[0]) / range_norm;

        // We use the change in the sign of the slope to detect the zeros and peaks of the ctf^2 function.
        // To compute this change at index i, we need the slope at i and i+1, which requires 3 ctf evaluations
        // (at i, i+1 and i+2). Instead, use a circular buffer to only sample the ctf once per iteration.
        // Each iteration computes the i+2 element, so we need to precompute the first two for the first iteration.
        constexpr auto index_circular_buffer = [](size_t index, i64 step) {
            if (step == 0)
                return index;
            const i64 value = static_cast<i64>(index) + step;
            return static_cast<size_t>(value < 0 ? (value + 3) : value);
        };
        size_t circular_count = 2;
        auto ctf_values = Vec<f64, 3>{
            std::abs(ctf.value_at(static_cast<f64>(simulated_start + 0) * SIMULATED_FREQUENCY_STEP)),
            std::abs(ctf.value_at(static_cast<f64>(simulated_start + 1) * SIMULATED_FREQUENCY_STEP)),
            0
        };

        // Collect zeros and peaks.
        FetchedCTFValues output;
        for (i64 i = simulated_start; i < simulated_end; ++i) {
            // Do your part, sample ith+2.
            ctf_values[circular_count] = std::abs(ctf.value_at(static_cast<f64>(i + 2) * SIMULATED_FREQUENCY_STEP));

            // Get the corresponding index in the experimental spectrum.
            // Here we want the frequency in the middle of the window, so at i + 1.
            const auto fftfreq = static_cast<f64>(i + 1) * SIMULATED_FREQUENCY_STEP;
            const auto corrected_frequency = (fftfreq - fftfreq_range[0]) / fftfreq_range_step;
            const auto index = static_cast<i64>(std::round(corrected_frequency));

            if (index >= 0 and index < rotational_average.ssize()) {
                // Compute the simulated CTF slope.
                // Based on the slope, we could lerp the index and value, but nearest is good enough here.
                const f64 ctf_value_0 = ctf_values[index_circular_buffer(circular_count, -2)];
                const f64 ctf_value_1 = ctf_values[index_circular_buffer(circular_count, -1)];
                const f64 ctf_value_2 = ctf_values[index_circular_buffer(circular_count, 0)];
                const f64 slope_0 = ctf_value_1 - ctf_value_0;
                const f64 slope_1 = ctf_value_2 - ctf_value_1;

                const auto& value = rotational_average[index];
                if (slope_0 < 0 and slope_1 >= 0) {
                    // zero: negative slope to positive slope.
                    output.zeros_x.push_back(fftfreq);
                    output.zeros_y.push_back(static_cast<f64>(value));
                }
                if (fit_envelope and slope_0 > 0 and slope_1 <= 0) {
                    // peak: positive slope to negative slope.
                    output.peaks_x.push_back(fftfreq);
                    output.peaks_y.push_back(static_cast<f64>(value));
                }
            }

            // Increment circular buffer.
            circular_count = (circular_count + 1) % 3; // 0,1,2,0,1,2,...
        }
        return output;
    }
}

namespace qn::ctf {
    auto Background::fit_coarse_background_1d(
        const View<const f32>& spectrum,
        i64 spline_resolution
    ) -> CubicSplineGrid<f64, 1> {
        auto spline = CubicSplineGrid<f64, 1>(spline_resolution);
        if (spectrum.is_dereferenceable())
            smooth_curve_fitting(spectrum.reinterpret_as_cpu().span_1d_contiguous(), spline.span()[0]);
        else
            smooth_curve_fitting(spectrum.to_cpu().span_1d_contiguous<const f32>(), spline.span()[0]);
        return spline;
    }

    auto Background::fit_coarse_background_2d(
        const View<const f32>& spectrum,
        const Vec<f64, 2>& fftfreq_range,
        i64 spline_resolution
    ) -> CubicSplineGrid<f64, 1> {
        const auto logical_size = spectrum.shape()[2];
        const auto spectrum_size = spectrum.shape()[3];

        auto rotational_averages = noa::zeros<f32>({2, 1, 1, spectrum_size}, {
            .device = spectrum.device(),
            .allocator = Allocator::ASYNC,
        });
        auto rotational_average = rotational_averages.view().subregion(0);
        auto rotational_average_weights = rotational_averages.view().subregion(1);

        ng::rotational_average<"h2h">(
            spectrum, {1, 1, logical_size, logical_size},
            rotational_average, rotational_average_weights, {
                .input_fftfreq = {0, fftfreq_range[1]},
                .output_fftfreq = {fftfreq_range[0], fftfreq_range[1]},
                .add_to_output = true
            });
        return fit_coarse_background_1d(rotational_average, spline_resolution);
    }

    void Background::fit_1d(
        const View<const f32>& spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const ns::CTFIsotropic<f64>& ctf
    ) {
        // When fetching ctf values at the ctf zeros/peaks, use a smoothed version of the spectrum.
        auto spectrum_smooth = noa::Array<f32>(spectrum.shape(), {
            .device = spectrum.device(),
            .allocator = Allocator::MANAGED
        });
        ns::convolve(
            spectrum, spectrum_smooth,
            ns::window_gaussian<f32>(7, 1.25, {.normalize = true}).to(spectrum.options()),
            {.border = noa::Border::REFLECT}
        );

        auto [zeros_x, zeros_y, peaks_x, peaks_y] = fetch_ctf_values(
            spectrum_smooth.reinterpret_as_cpu().span_1d_contiguous<const f32>(),
            fftfreq_range, ctf, false
        );

        if (zeros_x.size() < 2) {
            Logger::error(
                "CTF background fitting failed: less than two CTF zeros are within the frequency range.\n"
                "defocus={:.2f}, phase_shift={:.2f}, spacing={:.2f}, fftfreq_cutoff={:.2f})\n"
                "This is usually due to a very low defocus estimate, combined with a large pixel size "
                "and/or low resolution cutoff. Try to increase the resolution cutoff, and make sure the input data "
                "is not Fourier cropped to a low-resolution (the smaller the pixel size the better).",
                ctf.defocus(), ctf.phase_shift(), ctf.pixel_size(), fftfreq_range[1]
            );
            panic();
        }
        if (zeros_x.size() < 3) {
            Logger::warn(
               "Only two CTF zeros are located within the current frequency range. This is not great...\n"
               "defocus={:.2f}, phase_shift={:.2f}, spacing={:.2f}, fftfreq_cutoff={:.2f})\n"
               "This is usually due to a very low defocus estimate, combined with a large pixel size "
               "and/or low resolution cutoff. Instead of fitting a cubic spline, a line is fitted through "
               "these two points and it will be used as a background. However, this is unlikely to give good results."
               "Instead, you may want to increase the resolution cutoff, and make sure the input data is not Fourier "
               "cropped to a low-resolution (the smaller the pixel size the better).",
               ctf.defocus(), ctf.phase_shift(), ctf.pixel_size(), fftfreq_range[1]
           );
            // Add an extra point between the two and make it monotonic.
            zeros_x.push_back(zeros_x[0] + (zeros_x[1] - zeros_x[0]) / 2);
            zeros_y.push_back(zeros_y[0] + (zeros_y[1] - zeros_y[0]) / 2);
            std::swap(zeros_x[1], zeros_x[2]);
            std::swap(zeros_y[1], zeros_y[2]);
        }

        spline = Spline(
            SpanContiguous(zeros_x.data(), std::ssize(zeros_x)),
            SpanContiguous(zeros_y.data(), std::ssize(zeros_y)), {
                .type = zeros_x.size() == 3 ? Spline::LINEAR : Spline::CSPLINE,
                .monotonic = false,
                .left = Spline::SECOND_DERIVATIVE,
                .right = Spline::SECOND_DERIVATIVE,
                .left_value = 0.,
                .right_value = 0.,
            });
    }

    void Background::fit_2d(
        const View<const f32>& spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const ns::CTFAnisotropic<f64>& ctf
    ) {
        const auto logical_size = spectrum.shape()[2];
        const auto spectrum_size = spectrum.shape()[3];
        auto rotational_averages = noa::zeros<f32>({2, 1, 1, spectrum_size}, {
            .device = spectrum.device(),
            .allocator = Allocator::ASYNC,
        });
        auto rotational_average = rotational_averages.view().subregion(0);
        auto rotational_average_weights = rotational_averages.view().subregion(1);

        // If there's no astigmatism, this is equivalent to a normal rotational average.
        ng::rotational_average_anisotropic<"h2h">(
            spectrum, {1, 1, logical_size, logical_size}, ctf,
            rotational_average, rotational_average_weights, {
                .input_fftfreq = {0, fftfreq_range[1]},
                .output_fftfreq = {fftfreq_range[0], fftfreq_range[1]},
                .add_to_output = true,
            });

        fit_1d(rotational_average, fftfreq_range, ns::CTFIsotropic(ctf));
    }

    void Background::sample(
        const View<f32>& spectrum,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        const auto height = spectrum.shape().filter(2);
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.shape()[3] - 1);
        const auto span = spectrum.reinterpret_as_cpu().span().filter(2, 3).as_contiguous();

        for (i64 i{}; i < span.shape()[0]; ++i) {
            for (i64 j{}; j < span.shape()[1]; ++j) {
                const auto frequency = nf::index2frequency<false, true>(Vec{i, j}, height); // rfft, non-centered
                const auto fftfreq_2d = frequency.as<f64>() * fftfreq_step;
                const auto fftfreq = noa::sqrt(noa::dot(fftfreq_2d, fftfreq_2d)) + fftfreq_range[0];
                span(i, j) = static_cast<f32>(spline.interpolate_at(fftfreq));
            }
        }
    }

    void Background::subtract(
        const View<const f32>& input,
        const View<f32>& output,
        const Vec<f64, 2>& fftfreq_range
    ) const {
        check(noa::all(input.shape() == output.shape()));
        const auto height = input.shape().filter(2);
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(input.shape()[3] - 1);
        const auto input_s = input.reinterpret_as_cpu().span().filter(0, 2, 3).as_contiguous();
        const auto output_s = output.reinterpret_as_cpu().span().filter(0, 2, 3).as_contiguous();

        for (i64 b{}; b < input_s.shape()[0]; ++b) { // FIXME
            for (i64 i{}; i < input_s.shape()[1]; ++i) {
                for (i64 j{}; j < input_s.shape()[2]; ++j) {
                    const auto frequency = nf::index2frequency<false, true>(Vec{i, j}, height); // rfft, non-centered
                    const auto fftfreq_2d = frequency.as<f64>() * fftfreq_step;
                    const auto fftfreq = noa::sqrt(noa::dot(fftfreq_2d, fftfreq_2d)) + fftfreq_range[0];
                    output_s(b, i, j) = input_s(b, i, j) - static_cast<f32>(spline.interpolate_at(fftfreq));
                }
            }
        }
    }
}
