#pragma once

#include <omp.h>
#include "quinoa/Types.hpp"

// noa is quite inflexible when it comes to including its headers in C++-only files.
// At the moment, it checks and enforces that if it is built with CUDA and the header has CUDA code,
// it must be compiled as a CUDA-C++ file. This is quite sensible as it helps users to correctly compile
// their code, but in this case having an option to turn off the CUDA section of the headers would be nice.
// For now though, this works fine.

// #undef QN_INCLUDE_CPU_ONLY

#ifndef QN_INCLUDE_CPU_ONLY
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>
#include <noa/FFT.hpp>
#endif

namespace {
    using namespace qn;

    struct CenterCrossCorrelationMap {
        using span_t = SpanContiguous<const f32, 3, i32>;
        using interp_t = noa::Interpolator<2, noa::Interp::LINEAR, noa::Border::ZERO, span_t>;

        interp_t input{};
        SpanContiguous<f32, 3, i32> output{};
        SpanContiguous<const Vec<f32, 2>, 1, i32> peak_indices{};

        Mat<f32, 2, 2> rotation{};
        Vec<f32, 2> input_center{};
        Vec<f32, 2> output_center{};

        constexpr void operator()(i32 i, i32 y, i32 x) const {
            auto rotated_peak = rotation.transpose() * (peak_indices[i] - input_center) + input_center;
            auto coordinates = Vec{y, x}.as<f32>();
            coordinates += rotated_peak - output_center;

            coordinates -= input_center;
            coordinates = rotation * coordinates;
            coordinates += input_center;

            output(i, y, x) = input.interpolate_at(coordinates, i);
        }
    };
}

namespace qn {
    template<typename T>
    concept vec_or_real = nt::vec_real<T> or nt::real<T>;

    template<vec_or_real T, vec_or_real U>
    constexpr auto resolution_to_fftfreq(const T& spacing, const U& resolution) {
        return spacing / resolution;
    }

    template<vec_or_real T, vec_or_real U>
    constexpr auto fftfreq_to_resolution(const T& spacing, const U& fftfreq) {
        return spacing / fftfreq;
    }

#ifndef QN_INCLUDE_CPU_ONLY
    inline auto fourier_crop_to_resolution(i64 input_logical_size, f64 input_spacing, f64 target_resolution, bool fast_size) {
        const f64 input_size = static_cast<f64>(input_logical_size);
        const f64 target_spacing = target_resolution / 2;
        const f64 target_size = input_size * input_spacing / target_spacing;
        i64 final_size = std::max(static_cast<i64>(std::round(target_size)), i64{0});

        // Clamp within spectrum size and optionally optimize the final size for FFT.
        if (fast_size)
            final_size = nf::next_fast_size(final_size);
        final_size = std::min(input_logical_size, final_size);

        // Compute the final fftfreq. This is the fftfreq within the input reference-frame.
        const f64 spectrum_size = static_cast<f64>(input_logical_size / 2 + 1);
        const f64 fftfreq_step = nf::highest_fftfreq<f64>(input_logical_size) / (spectrum_size - 1);
        const f64 final_fftfreq = static_cast<f64>(final_size / 2) * fftfreq_step ;

        return Pair{final_size, final_fftfreq};
    }
#endif

    /// Given a spectrum size and fftfreq-range, return the index, and its corresponding fftfreq, closest to the target fftfreq.
    inline auto nearest_integer_fftfreq(
        i64 spectrum_size,
        const Vec<f64, 2>& fftfreq_range,
        f64 target_fftfreq,
        bool clamp = false
    ) {
        const auto last_index = static_cast<f64>(spectrum_size - 1);
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / last_index;
        auto frequency = std::round((target_fftfreq - fftfreq_range[0]) / fftfreq_step);
        if (clamp)
            frequency = std::clamp(frequency, 0., last_index);
        const auto actual_fftfreq = frequency * fftfreq_step + fftfreq_range[0];
        return Pair{static_cast<i64>(frequency), actual_fftfreq};
    }

    void parallel_for(i32 n_threads, i64 size, auto&& func) {
        #pragma omp parallel for num_threads(n_threads)
        for (i64 c = 0; c < size; ++c) {
            func(omp_get_thread_num(), c);
        }
    }

    template<size_t N>
    void parallel_for(i32 n_threads, const Shape<i64, N>& shape, auto&& func) {
        if constexpr (N == 1) {
            #pragma omp parallel for num_threads(n_threads)
            for (i64 i = 0; i < shape[0]; ++i) {
                func(omp_get_thread_num(), i);
            }
        } else if constexpr (N == 2) {
            #pragma omp parallel for num_threads(n_threads) collapse(2)
            for (i64 i = 0; i < shape[0]; ++i) {
                for (i64 j = 0; j < shape[1]; ++j) {
                    func(omp_get_thread_num(), i, j);
                }
            }
        } else {
            static_assert(nt::always_false<Tag<N>>);
        }
    }

    constexpr auto effective_thickness(f64 thickness, const Vec<f64, 3>& stage_angles) {
        const f64 scaling = (
            ng::rotate_y(noa::deg2rad(stage_angles[1])) *
            ng::rotate_x(noa::deg2rad(stage_angles[2]))
        )[0][0];
        return thickness / scaling;
    }

    template<typename T>
    constexpr auto simple_average(SpanContiguous<T, 1> data) -> f64 {
        f64 sum{};
        for (auto& e: data)
            sum += static_cast<f64>(e);
        return sum / static_cast<f64>(data.ssize());
    }

    struct GaussianSlider {
        /// Normalized coordinates of the Gaussian peak.
        /// [0,1] is the input range, but out-of-range coordinates are valid.
        f64 peak_coordinate{0};

        /// Value of the Gaussian peak, at peak_coordinate.
        f64 peak_value{1};

        /// Normalized width of the half-Gaussian curve.
        /// [0,1] is the input range, but out-of-range coordinates are valid.
        f64 base_width{1};

        /// Value of the Gaussian at peak_coordinate.
        f64 base_value{1e-6};
    };

    struct ALSSOptions {
        GaussianSlider smoothing{};

        /// Asymmetric penalty. 0.5 means no bias towards values higher or lower the baseline.
        f64 asymmetric_penalty = 0.5;

        /// Maximum number of maximum iterations.
        i32 max_iter = 100;

        /// Convergence criterium, ie diff tolerance between weights
        f64 tol = 1e-6;

        /// Relaxation factor. Smooths the step of the new weight.
        /// 1 means no relaxation, i.e. new weights are assigned to p or 1-p.
        f64 relaxation = 1.0;
    };

    /// Compute the baseline of the spectrum.
    template<nt::any_of<f32, f64> T>
    void asymmetric_least_squares_smoothing(
        SpanContiguous<const T> spectrum,
        SpanContiguous<f64> baseline,
        const ALSSOptions& options
    );

    void interpolating_uniform_cubic_spline(
        SpanContiguous<const f64> a,
        SpanContiguous<f64> b,
        SpanContiguous<f64> c,
        SpanContiguous<f64> d
    );

    struct FindBestPeakOptions {
        f64 distortion_angle_deg{0};
        f64 max_shift_percent{-1};
    };

    /// Finds the coordinates (relative to the input center) of the best peak.
    auto find_best_peak(const SpanContiguous<const f32, 2>& xmap_centered) -> Pair<Vec<f64, 2>, f32>;

#ifndef QN_INCLUDE_CPU_ONLY
    template<nf::Layout REMAP> requires (REMAP.is_fc2xx())
    void find_peaks(
        const View<f32>& xmap,
        const View<f32>& xmap_centered,
        const View<Vec<f32, 2>>& peak_shifts,
        const View<f32>& peak_values,
        const FindBestPeakOptions& options = {}
    ) {
        const auto xmap_shape_2d = xmap.shape().filter(2, 3);
        const auto xmap_center = (xmap_shape_2d.vec / 2).as<f64>();
        const auto xmap_centered_center = (xmap_centered.shape().filter(2, 3).vec / 2).as<f32>();
        const auto distortion_angle = noa::deg2rad(options.distortion_angle_deg);

        // Get the highest peaks within the allowed lag.
        ns::cross_correlation_peak_2d<"fc2fc">(xmap, peak_shifts, {}, {
            .registration_radius = Vec<i64, 2>{0, 0}, // turn off the registration
            .maximum_lag = Vec<f64, 2>::from_value(noa::min(xmap_center) * options.max_shift_percent),
        });

        // Due to the difference in tilt, the cross-correlation map can be distorted orthogonal to the tilt-axis.
        // To improve the accuracy of the subpixel registration, correct the tilt-axis to have the distortion along x.
        // Since the actual peak is close to argmax, focus on (and only render) a small subregion around argmax.
        using interp_t = CenterCrossCorrelationMap::interp_t;
        noa::iwise(xmap_centered.shape().filter(0, 2, 3).as<i32>(), xmap.device(), CenterCrossCorrelationMap{
            .input = interp_t(xmap.span_contiguous<const f32, 3, i32>(), xmap_shape_2d.as<i32>()),
            .output = xmap_centered.span_contiguous<f32, 3, i32>(),
            .peak_indices = peak_shifts.span_contiguous<const Vec<f32, 2>, 1, i32>(),
            .rotation = ng::rotate(distortion_angle).as<f32>(),
            .input_center = xmap_center.as<f32>(),
            .output_center = xmap_centered_center.as<f32>(),
        });

        const auto span = xmap_centered.reinterpret_as_cpu().span_contiguous<const f32, 3>();
        const auto has_values = not peak_values.is_empty();
        const auto span_peak_values = has_values ? peak_values.span_1d() : SpanContiguous<f32>{};
        for (i64 i{}; auto& peak_shift: peak_shifts.span_1d()) {
            const auto [peak_offset_rotated, peak_value] = find_best_peak(span[i]);
            const auto peak_offset = ng::rotate(distortion_angle) * peak_offset_rotated;
            const auto peak_coordinate = peak_shift.as<f64>() + peak_offset;
            peak_shift = (peak_coordinate - xmap_center).as<f32>();

            if (has_values)
                span_peak_values[i] = peak_value;

            ++i;
        }
    }

    template<nf::Layout REMAP> requires (REMAP.is_fc2xx())
    auto find_peak(
        const View<f32>& xmap,
        const View<f32>& xmap_centered,
        const FindBestPeakOptions& options = {}
    ) -> Pair<Vec<f64, 2>, f32> {
        check(xmap.shape().batch() == 1 and xmap_centered.shape().batch() == 1);
        const auto xmap_shape_2d = xmap.shape().filter(2, 3);
        const auto xmap_center = (xmap_shape_2d.vec / 2).as<f64>();
        const auto xmap_centered_center = (xmap_centered.shape().filter(2, 3).vec / 2).as<f64>();
        const auto distortion_angle = noa::deg2rad(options.distortion_angle_deg);

        // Get the highest peak within the allowed lag.
        auto [peak_indices, _] = ns::cross_correlation_peak_2d<"fc2fc">(xmap, {
            .registration_radius = Vec<i64, 2>{0, 0}, // turn off the registration
            .maximum_lag = Vec<f64, 2>::from_value(noa::min(xmap_center) * options.max_shift_percent),
        });

        // Due to the difference in tilt, the cross-correlation map can be distorted orthogonal to the tilt-axis.
        // To improve the accuracy of the subpixel registration, correct the tilt-axis to have the distortion along x.
        // Since the actual peak is close to argmax, focus on (and only render) a small subregion around argmax.
        const auto rotate_xmap =
            ng::translate(xmap_center) *
            ng::rotate<true>(-distortion_angle) *
            ng::translate(-xmap_center);
        peak_indices = (rotate_xmap * peak_indices.push_back(1)).pop_back();

        const auto rotate_and_center_peak = (
            ng::translate(xmap_center - peak_indices + xmap_centered_center) *
            ng::rotate<true>(-distortion_angle) *
            ng::translate(-xmap_center)
        ).inverse();
        ng::transform_2d(xmap, xmap_centered, rotate_and_center_peak, {.interp = noa::Interp::LINEAR});

        // Get the peak and rotate back to the original xmap reference-frame.
        const auto [peak_offset, peak_value] = find_best_peak(
            xmap_centered.reinterpret_as_cpu().span<const f32, 2>().as_contiguous());
        const auto peak_coordinate = (rotate_xmap.inverse() * (peak_indices + peak_offset).push_back(1)).pop_back();
        auto shift = peak_coordinate - xmap_center;

        // Given cross_correlation_map(lhs, rhs, xmap), this should be subtracted to the lhs to align it onto the rhs.
        return {shift, peak_value};
    }
#endif

    inline auto subdivide_axis(i64 size, i64 subsize, i64 step) -> std::vector<i64> {
        // Arange:
        const auto n_patches = noa::divide_up(size, step);
        std::vector<i64> patch_origin;
        patch_origin.reserve(static_cast<size_t>(n_patches));
        for (i64 i{}; i < n_patches; ++i)
            patch_origin.push_back(i * step);

        if (patch_origin.empty())
            return patch_origin;

        // Center:
        const i64 end = patch_origin.back() + subsize;
        const i64 offset = (size - end) / 2;
        for (auto& origin: patch_origin)
            origin += offset;

        return patch_origin;
    }

#ifndef QN_INCLUDE_CPU_ONLY
    template<size_t N>
    auto subdivide_axes(
        const Shape<i64, N>& shape,
        const Shape<i64, N>& subshape,
        const Vec<i64, N>& step
    ) {
        std::array<std::vector<i64>, N> origins;
        auto grid_shape = Shape<i64, N>::from_value(1);
        for (i64 i = 0; i < N; ++i) {
            origins[i] = subdivide_axis(shape[i], subshape[i], step[i]);
            grid_shape[4 - N] = origins[i].size();
        }

        auto output = Array<Vec<i64, N>>(grid_shape);
        if constexpr (N == 1) {
            auto output_1d = output.span_1d();
            for (size_t i{}; i < origins[0].size(); ++i)
                output_1d(i) = origins[0][i];
        } else if constexpr (N == 2) {
            auto output_2d = output.span().filter(2, 3);
            for (size_t i{}; i < origins[0].size(); ++i)
                for (size_t j{}; j < origins[1].size(); ++j)
                    output_2d(i, j) = {origins[0][i], origins[1][j]};
        } else if constexpr (N == 3) {
            auto output_3d = output.span().filter(1, 2, 3);
            for (size_t i{}; i < origins[0].size(); ++i)
                for (size_t j{}; j < origins[1].size(); ++j)
                    for (size_t k{}; k < origins[1].size(); ++k)
                        output_3d(i, j, k) = {origins[0][i], origins[1][j], origins[2][k]};
        } else {
            static_assert(nt::always_false<Vec<i64, N>>);
        }

        return output;
    }
#endif
}
