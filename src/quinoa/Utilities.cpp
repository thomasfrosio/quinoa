// Suppress Eigen warnings...
#include <noa/base/Config.hpp>
#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wsign-conversion"
#   pragma GCC diagnostic ignored "-Wnull-dereference"
#   if defined(NOA_COMPILER_GCC)
#       pragma GCC diagnostic ignored "-Wduplicated-branches"
#       pragma GCC diagnostic ignored "-Wuseless-cast"
#       pragma GCC diagnostic ignored "-Wclass-memaccess"
#   endif
#elif defined(NOA_COMPILER_MSVC)
#   pragma warning(push, 0)
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(pop)
#endif

#include <nlopt.hpp>

#include "quinoa/Types.hpp"
#define QN_INCLUDE_CPU_ONLY
#include "quinoa/Utilities.hpp"

namespace qn {
    void asymmetric_least_squares_smoothing(
        SpanContiguous<const f64> x_span,
        SpanContiguous<const f64> y_span,
        SpanContiguous<f64> baseline,
        const ALSSOptions& options
    ) {
        const auto n = y_span.ssize();
        if (n <= 3) {
            panic("Not enough samples");
        }

        // Set up internal coordinate scaling [0, 1].
        // This provides numerical stability and fixed domain length for N-invariance.
        const f64 x_start = x_span[0];
        const f64 x_end = x_span[n - 1];
        const f64 x_phys_range = x_end - x_start;
        const f64 inv_phys_range = 1.0 / x_phys_range;

        auto get_scaled_x = [&](i64 i) {
            return (static_cast<f64>(x_span[i]) - x_start) * inv_phys_range;
        };

        const auto y = Eigen::Map<const Eigen::VectorXd>(y_span.data(), n);

        // 2. Build D, the second-difference operator.
        auto D = Eigen::SparseMatrix<f64>(n - 2, n);
        {
            auto triplets = std::vector<Eigen::Triplet<f64>>{};
            triplets.reserve(static_cast<usize>(3 * (n - 2)));
            for (i64 i{}; i < n - 2; ++i) {
                const f64 h1 = get_scaled_x(i + 1) - get_scaled_x(i);
                const f64 h2 = get_scaled_x(i + 2) - get_scaled_x(i + 1);

                // 2nd derivative in [0, 1] range
                triplets.emplace_back(i, i + 0, +2.0 / (h1 * (h1 + h2)));
                triplets.emplace_back(i, i + 1, -2.0 / (h1 * h2));
                triplets.emplace_back(i, i + 2, +2.0 / (h2 * (h1 + h2)));
            }
            D.setFromTriplets(triplets.begin(), triplets.end());
        }

        // Build L, the varying smoothing (diagonal) matrix.
        auto L = Eigen::SparseMatrix<f64>(n - 2, n - 2);
        {
            auto triplets = std::vector<Eigen::Triplet<f64>>{};
            triplets.reserve(static_cast<usize>(n - 2));
            const auto& sm = options.smoothing;
            const f64 decay_cut = sm.base_width;
            const f64 gaussian_decay = std::sqrt(-std::log(sm.base_value / sm.peak_value)) / decay_cut;

            for (i64 i{}; i < n - 2; ++i) {
                const f64 x_norm = get_scaled_x(i + 1);
                const f64 x_offset = x_norm - sm.peak_coordinate;

                f64 value = sm.peak_value * std::exp(-std::pow(x_offset * gaussian_decay, 2.));
                value = std::clamp(value, sm.base_value, sm.peak_value);

                // The magic scaling for N-invariance in [0, 1] space:
                // This represents the 'dx' for the integral of curvature.
                const f64 local_dx = (get_scaled_x(i + 2) - get_scaled_x(i)) * 0.5;
                triplets.emplace_back(i, i, value * local_dx);
            }
            L.setFromTriplets(triplets.begin(), triplets.end());
        }

        // Precompute density weights and asymmetry.
        Eigen::VectorXd dx_weights(n);
        Eigen::VectorXd asymmetric_penalty(n);
        {
            const auto& as = options.asymmetry;
            const f64 decay_cut = as.base_width;
            const f64 gaussian_decay = std::sqrt(-std::log(as.base_value / as.peak_value)) / decay_cut;

            for (i64 i{}; i < n; ++i) {
                dx_weights[i] =
                    i == 0 ? get_scaled_x(1) - get_scaled_x(0) :
                    i == n - 1 ? get_scaled_x(n - 1) - get_scaled_x(n - 2) :
                    (get_scaled_x(i + 1) - get_scaled_x(i - 1)) * 0.5;

                const f64 x_norm = get_scaled_x(i);
                const f64 x_off = x_norm - as.peak_coordinate;
                asymmetric_penalty[i] =
                    std::clamp(as.peak_value * std::exp(-std::pow(x_off * gaussian_decay, 2.0)),
                               as.base_value, as.peak_value);
            }
        }

        // Solve z, as in (W+DtLD)z = (Wy), using sparse Cholesky decomposition.
        Eigen::SparseMatrix<f64> DtLD = D.transpose() * L * D;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<f64>> solver;
        Eigen::VectorXd w = asymmetric_penalty;
        Eigen::VectorXd w_new(n);
        auto z = Eigen::Map<Eigen::VectorXd>(baseline.data(), n);

        Eigen::SparseMatrix<f64> A = DtLD;
        for (i64 i{}; i < n; ++i)
            A.coeffRef(i, i) += w[i] * dx_weights[i];

        solver.analyzePattern(A);

        for (i32 iter{}; iter < options.max_iter; ++iter) {
            solver.factorize(A);
            check(solver.info() == Eigen::Success, "Decomposition failed");

            z = solver.solve((w.array() * dx_weights.array() * y.array()).matrix());
            check(solver.info() == Eigen::Success, "Solving failed");

            // Compute the new weights.
            for (i64 i{}; i < n; ++i) {
                const f64 residual = y[i] - z[i];
                const f64 target_p = residual > 0 ? asymmetric_penalty[i] : (1.0 - asymmetric_penalty[i]);
                w_new[i] = options.relaxation * target_p + (1.0 - options.relaxation) * w[i];
            }

            const f64 diff = (w - w_new).cwiseAbs().maxCoeff();
            if (diff < options.tol)
                break;

            for (i64 i{}; i < n; ++i)
                A.coeffRef(i, i) += (w_new[i] - w[i]) * dx_weights[i];
            w.swap(w_new);
        }
    }

    auto find_best_peak(const SpanContiguous<const f32, 2>& data) -> Pair<Vec<f64, 2>, f32> {
        constexpr isize BLOCK_SIZE = 5;
        constexpr isize BLOCK_RADIUS = BLOCK_SIZE / 2;
        constexpr isize N_BLOCKS_Y = 3;
        constexpr isize N_BLOCKS_X = 7;
        constexpr f32 THRESHOLD = 0.8f;

        // Get the position of the max within the block.
        auto argmax = [&](const Vec<isize, 2>& block_center) {
            auto max_value = std::numeric_limits<f32>::lowest();
            auto max_indices = Vec<isize, 2>{};
            for (isize y{-BLOCK_RADIUS}; y <= BLOCK_RADIUS; ++y) {
                for (isize x{-BLOCK_RADIUS}; x <= BLOCK_RADIUS; ++x) {
                    const auto indices = block_center + Vec{y, x};
                    if (noa::is_inbound(data.shape(), indices)) {
                        const auto& value = data(indices);
                        if (max_value < value) {
                            max_value = value;
                            max_indices = indices;
                        }
                    }
                }
            }
            return max_indices;
        };

        // Find whether the given position points to a peak
        // by checking that the 8 neighbors have lower values.
        auto is_a_peak = [&](const Vec<isize, 2>& position) {
            const auto& value = data(position);
            for (isize y{-1}; y <= 1; ++y) {
                for (isize x{-1}; x <= 1; ++x) {
                    const auto indices = position + Vec{y, x};
                    if (noa::is_inbound(data.shape(), indices)) {
                        if (value < data(indices))
                            return false;
                    }
                }
            }
            return true;
        };

        // Offset the peak height by offsetting its base to 0. To select between peaks with very similar values,
        // we adjust each peak value by looking at its base and offset the peak value to put its base at zero.
        // Indeed, it seems that the "true" peak is surrounded by low/negative CC values, as opposed to less sharp
        // peaks which are surrounded by the background noise/CC.
        // Selecting the sharpest peak doesn't seem to work well with multi-lobe peaks.
        auto peak_base_value = [&](const Vec<isize, 2>& peak_position) {
            const auto& peak_value = data(peak_position);
            auto find_base = [&](isize direction) {
                f32 previous_base = peak_value;

                // Find the base of the peak along that direction.
                // Peaks should only be a few pixels wide.
                for (isize y{1}; y < 10; ++y) {

                    // Compute the base value at this y-offset.
                    // The base value is the average of the 3 values at y-offset.
                    f32 base{};
                    for (isize x{-1}; x <= 1; ++x) {
                        auto position = peak_position + Vec{y * direction, x};
                        position = noa::index_at<noa::Border::REFLECT>(position, data.shape());
                        base += data(position);
                    }
                    base /= 3;

                    // Stop when the average CC is going back up.
                    if (base > previous_base)
                        break;
                    previous_base = base;
                }
                return previous_base;
            };
            const auto lower_base = find_base(-1);
            const auto upper_base = find_base(+1);
            return (upper_base + lower_base) / 2;
        };

        // Fit a 3-points parabola along the y and x of the peak to get
        // the peak offset and value with subpixel accuracy.
        auto subpixel_registration = [&](const Vec<isize, 2>& peak_position) {
            Vec<f64, 2> peak_offset{};
            Vec<f64, 2> peak_value{};
            for (auto i: {0, 1}) {
                Vec<f32, 3> buffer{};
                for (isize j{}; j < 3; ++j) {
                    auto indices = peak_position;
                    indices[i] = noa::index_at<noa::Border::REFLECT>(peak_position[i] + j - 1, data.shape()[i]);
                    buffer[j] = data(indices);
                }
                noa::tie(peak_offset[i], peak_value[i]) = ns::details::lstsq_fit_quadratic_vertex_3points(
                    buffer[0], buffer[1], buffer[2]
                );
            }
            return Pair{peak_offset, static_cast<f32>(noa::mean(peak_value))};
        };

        // The peak is likely where the argmax of the CCmap is, i.e. at the center of data span.
        // However, when cross-correlating tilt images, the CCmap can be distorted orthogonal to the tilt-axis
        // and peaks with multilobes can appear. In these cases, adjusting for the peak heights based on their
        // base value (where the peak starts) seems to be a good way to discern the correct peaks from the others.
        const auto center = data.shape().vec / 2;
        const auto center_peak_value = data(center);
        const auto center_peak_registration = subpixel_registration(center);
        const auto center_peak_base = peak_base_value(center);
        const auto center_peak_adjusted_value = center_peak_registration.second - center_peak_base;
        const auto center_peak_coordinates_offset = center_peak_registration.first;

        auto best_peak_value_adjusted = center_peak_adjusted_value;
        auto best_peak_coordinates_offset = center_peak_coordinates_offset;

        // for (isize y = -(N_BLOCKS_Y / 2); y <= N_BLOCKS_Y / 2; ++y) {
        //     for (isize x = -(N_BLOCKS_X / 2); x <= N_BLOCKS_X / 2; ++x) {
        //         if (y == 0 and x == 0)
        //             continue;
        //
        //         const auto block_center = center + Vec{y, x} * BLOCK_SIZE;
        //         const auto peak_position = argmax(block_center);
        //         if (not is_a_peak(peak_position))
        //             continue;
        //
        //         const auto peak_value = data(peak_position);
        //         if (peak_value >= center_peak_value * THRESHOLD) {
        //             // This peak is quite close to the central peak, so correct for its base and
        //             // do the subpixel-registration to get the "actual" peak value.
        //             const auto base = peak_base_value(peak_position);
        //             const auto registration = subpixel_registration(peak_position);
        //             const auto peak_value_adjusted = registration.second - base;
        //
        //             if (peak_value_adjusted > best_peak_value_adjusted) {
        //                 Logger::trace("found new peak: new_pos={}, peak_value_adjusted={}, orig={}", peak_position, peak_value_adjusted, best_peak_value_adjusted);
        //                 best_peak_value_adjusted = peak_value_adjusted;
        //                 best_peak_coordinates_offset = peak_position.as<f64>() - center.as<f64>() + registration.first ;
        //             }
        //         }
        //     }
        // }
        // FIXME Returning the original peak value seem to be the right choice... isn't it?
        return {best_peak_coordinates_offset, center_peak_registration.second};
    }
}
