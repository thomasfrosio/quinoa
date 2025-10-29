// Suppress Eigen warnings...
#include <noa/core/Config.hpp>
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
    template<nt::any_of<f32, f64> T>
    void asymmetric_least_squares_smoothing(
        SpanContiguous<const T> spectrum,
        SpanContiguous<f64> baseline,
        const ALSSOptions& options
    ) {
        // Convert to f64 precision fp.
        const auto n = spectrum.ssize();
        auto y = Eigen::VectorXd(n);
        for (i64 i{}; i < n; ++i)
            y[i] = static_cast<f64>(spectrum[i]);

        // Construct the second-difference operator.
        auto D = Eigen::SparseMatrix<f64>(n - 2, n);
        {
            auto triplets = std::vector<Eigen::Triplet<f64>>{};
            triplets.reserve(static_cast<size_t>(3 * (n - 2)));
            for (i64 i{}; i < n - 2; ++i) {
                triplets.emplace_back(i, i + 0, +1.0);
                triplets.emplace_back(i, i + 1, -2.0);
                triplets.emplace_back(i, i + 2, +1.0);
            }
            D.setFromTriplets(triplets.begin(), triplets.end());
        }

        // Construct the varying smoothing (diagonal matrix).
        auto L = Eigen::SparseMatrix<f64>(n - 2, n - 2);
        {
            auto triplets = std::vector<Eigen::Triplet<f64>>{};
            triplets.reserve(static_cast<size_t>(n - 2));

            // Gaussian decay.
            const auto& smoothing = options.smoothing;
            const f64 decay_cut = static_cast<f64>(n) * smoothing.base_width;
            const f64 gaussian_decay = std::sqrt(-std::log(smoothing.base_value / smoothing.peak_value)) / decay_cut;
            for (i64 i{}; i < n - 2; ++i) {
                auto x = static_cast<f64>(i + 2 * (i + 1) + (i + 2)) / 4.0;
                auto x_offset = x - smoothing.peak_coordinate * static_cast<f64>(n);
                auto lambda_i = smoothing.peak_value * std::exp(-std::pow(x_offset * gaussian_decay, 2.));
                triplets.emplace_back(i, i, lambda_i);
            }
            L.setFromTriplets(triplets.begin(), triplets.end());
        }

        // Precompute the smoothing penalty.
        Eigen::SparseMatrix<f64> DtLD = D.transpose() * L * D;

        // Diagonal weight (sparse) matrix W and related vector buffers.
        auto W = Eigen::SparseMatrix<f64>(n, n);
        auto w = Eigen::VectorXd(n);
        auto w_new = Eigen::VectorXd(n);
        W.reserve(n);
        for (i64 i{}; i < n; ++i) {
            W.insert(i, i) = 1.0;
            w[i] = 1.0;
        }
        W.makeCompressed();

        // Solve z, as in (W+DtLD)z = (Wy), using sparse Cholesky decomposition.
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<f64>> solver;
        Eigen::Map<Eigen::VectorXd> z(baseline.data(), n);
        for (i32 iter{}; iter < options.max_iter; ++iter) {
            solver.compute(W + DtLD);
            check(solver.info() == Eigen::Success, "Decomposition failed");
            z = solver.solve(w.cwiseProduct(y));
            check(solver.info() == Eigen::Success, "Solving failed");

            // Compute the new weights, with optional relaxation for faster convergence.
            for (i64 i{}; i < n; ++i) {
                const auto new_w = (y[i] > z[i]) ? options.asymmetric_penalty : (1.0 - options.asymmetric_penalty);
                w_new[i] = options.relaxation * new_w + (1.0 - options.relaxation) * w[i];
            }

            // Check convergence (and stop), otherwise, try again with updated weights.
            if ((w - w_new).cwiseAbs().maxCoeff() < options.tol)
                break;
            w.swap(w_new);
            for (i64 i{}; i < n; ++i)
                W.coeffRef(i, i) = w[i];
        }
    }
    template void asymmetric_least_squares_smoothing<f32>(SpanContiguous<const f32>, SpanContiguous<f64>, const ALSSOptions&);
    template void asymmetric_least_squares_smoothing<f64>(SpanContiguous<const f64>, SpanContiguous<f64>, const ALSSOptions&);

    void interpolating_uniform_cubic_spline(
        SpanContiguous<const f64> a,
        SpanContiguous<f64> b,
        SpanContiguous<f64> c,
        SpanContiguous<f64> d
    ) {
        const auto n = a.ssize();

        auto triplets = std::vector<Eigen::Triplet<f64>>{};
        triplets.reserve(static_cast<size_t>(n) * 3 + 4);

        // Use b as a temporary buffer for rhs.
        auto rhs = Eigen::Map<Eigen::VectorXd>(b.data(), n);

        // Left boundary conditions (clamped-spline).
        triplets.emplace_back(0, 0, 2.0);
        rhs(0) = 0; // second-derivative = 0

        for (i64 i = 1; i < n - 1; i++) {
            triplets.emplace_back(i, i - 1, 1.0);
            triplets.emplace_back(i, i,     4.0);
            triplets.emplace_back(i, i + 1, 1.0);
            rhs(i) = 3.0 * (a[i + 1] - 2.0 * a[i] + a[i - 1]);
        }

        // Right boundary conditions (clamped-spline).
        triplets.emplace_back(n - 1, n - 1, 2.0);
        rhs(n - 1) = 0.; // second-derivative = 0.

        auto A = Eigen::SparseMatrix<f64>(n, n);
        A.setFromTriplets(triplets.begin(), triplets.end());

        // Solve for Ac=w.
        auto solver = Eigen::SimplicialLLT<Eigen::SparseMatrix<f64>>{};
        solver.compute(A);
        check(solver.info() == Eigen::Success, "Decomposition failed");
        Eigen::Map<Eigen::VectorXd> cc(c.data(), n);
        cc = solver.solve(rhs); // save in-place
        check(solver.info() == Eigen::Success, "Solving failed");

        // Compute and save polynomial coefficients.
        for (i64 i = 0; i < n - 1; i++) {
            d[i] = (c[i + 1] - c[i]) / 3.0;
            b[i] = a[i + 1] - a[i] - (c[i] + d[i]);
        }
        b[n - 1] = b[n - 2] + 2.0 * c[n - 2] + 3.0 * d[n - 2];
        d[n - 1] = 0.0;
    }

    auto find_best_peak(const SpanContiguous<const f32, 2>& data) -> Vec<f64, 2> {
        constexpr i64 BLOCK_SIZE = 5;
        constexpr i64 BLOCK_RADIUS = BLOCK_SIZE / 2;
        constexpr i64 N_BLOCKS_Y = 3;
        constexpr i64 N_BLOCKS_X = 7;
        constexpr f32 THRESHOLD = 0.8f;

        // Get the position of the max within the block.
        auto argmax = [&](const Vec<i64, 2>& block_center) {
            auto max_value = std::numeric_limits<f32>::lowest();
            auto max_indices = Vec<i64, 2>{};
            for (i64 y{-BLOCK_RADIUS}; y <= BLOCK_RADIUS; ++y) {
                for (i64 x{-BLOCK_RADIUS}; x <= BLOCK_RADIUS; ++x) {
                    const auto indices = block_center + Vec{y, x};
                    if (ni::is_inbound(data.shape(), indices)) {
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
        auto is_a_peak = [&](const Vec<i64, 2>& position) {
            const auto& value = data(position);
            for (i64 y{-1}; y <= 1; ++y) {
                for (i64 x{-1}; x <= 1; ++x) {
                    const auto indices = position + Vec{y, x};
                    if (ni::is_inbound(data.shape(), indices)) {
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
        auto peak_base_value = [&](const Vec<i64, 2>& peak_position) {
            const auto& peak_value = data(peak_position);
            auto find_base = [&](i64 direction) {
                f32 previous_base = peak_value;

                // Find the base of the peak along that direction.
                // Peaks should only be a few pixels wide.
                for (i64 y{1}; y < 10; ++y) {

                    // Compute the base value at this y-offset.
                    // The base value is the average of the 3 values at y-offset.
                    f32 base{};
                    for (i64 x{-1}; x <= 1; ++x) {
                        auto position = peak_position + Vec{y * direction, x};
                        position = ni::index_at<noa::Border::REFLECT>(position, data.shape());
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
        auto subpixel_registration = [&](const Vec<i64, 2>& peak_position) {
            Vec<f64, 2> peak_offset{};
            Vec<f64, 2> peak_value{};
            for (auto i: {0, 1}) {
                Vec<f32, 3> buffer{};
                for (i64 j{}; j < 3; ++j) {
                    auto indices = peak_position;
                    indices[i] = ni::index_at<noa::Border::REFLECT>(peak_position[i] + j - 1, data.shape()[i]);
                    buffer[j] = data(indices);
                }
                noa::tie(peak_offset[i], peak_value[i]) = noa::lstsq_fit_quadratic_vertex_3points(
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

        for (i64 y = -(N_BLOCKS_Y / 2); y <= N_BLOCKS_Y / 2; ++y) {
            for (i64 x = -(N_BLOCKS_X / 2); x <= N_BLOCKS_X / 2; ++x) {
                if (y == 0 and x == 0)
                    continue;

                const auto block_center = center + Vec{y, x} * BLOCK_SIZE;
                const auto peak_position = argmax(block_center);
                if (not is_a_peak(peak_position))
                    continue;

                const auto peak_value = data(peak_position);
                if (peak_value >= center_peak_value * THRESHOLD) {
                    // This peak is quite close to the central peak, so correct for its base and
                    // do the subpixel-registration to get the "actual" peak value.
                    const auto base = peak_base_value(peak_position);
                    const auto registration = subpixel_registration(peak_position);
                    const auto peak_value_adjusted = registration.second - base;

                    if (peak_value_adjusted > best_peak_value_adjusted) {
                        best_peak_value_adjusted = peak_value_adjusted;
                        best_peak_coordinates_offset = peak_position.as<f64>() - center.as<f64>() + registration.first ;
                    }
                }
            }
        }
        return best_peak_coordinates_offset;
    }
}
