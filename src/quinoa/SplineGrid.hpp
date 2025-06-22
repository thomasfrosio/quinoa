#pragma once

#include <noa/Array.hpp>
#include <noa/core/Interpolation.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Optimizer.hpp"

namespace qn {
    /// Cubic spline interpolation on multidimensional grids (only 1d and 2d are currently supported).
    /// Grids consist of uniformly spaced points (called nodes) covering the full extent of each dimension.
    ///
    /// - We can get the value at any continuous point on the grid. Note that the grid coordinate system extends
    ///   from [0, 1] along each grid dimension, so it is independent of the number of nodes.
    /// - The grid is a simple view of the actual nodes values and doesn't own any data.
    /// - The interpolation can be CUBIC_BSPLINE or CUBIC.
    template<typename T, size_t N, noa::Interp INTERP = noa::Interp::CUBIC_BSPLINE>
    class SplineGrid {
    public:
        static_assert(N == 1 or N == 2);
        static_assert(nt::real_or_complex<T>);
        static_assert(INTERP.is_any(noa::Interp::CUBIC_BSPLINE, noa::Interp::CUBIC));

        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using const_value_type = std::add_const_t<mutable_value_type>;
        using weight_type = nt::value_type_t<mutable_value_type>;
        using span_type = SpanContiguous<value_type, N>;
        using shape_type = Shape<i64, N>;
        using vec_type = Vec<value_type, N>;

        using coord_type = f64;
        using cubic_indices_type = Vec<i64, 4>;

    public:
        span_type span;

    public:
        SplineGrid() = default;

        /// Creates an N-d grid from the provided node values.
        explicit SplineGrid(const span_type& data) : span{data} { check(not span.is_empty()); }

    public:
        [[nodiscard]] constexpr auto shape() const noexcept -> const shape_type& { return span.shape(); }
        [[nodiscard]] constexpr auto ssize() const noexcept -> i64 { return span.ssize(); }
        [[nodiscard]] constexpr auto size() const noexcept -> size_t { return span.size(); }

        /// Updates the nodes of a given channel with the provided values.
        constexpr void update_from_span(const SpanContiguous<const_value_type, N>& values) noexcept {
            check(noa::vall(noa::Equal{}, shape(), values.shape()));
            for (auto&& [i, o]: noa::zip(span.as_1d(), values.as_1d()))
                i = o;
        }

    public: // Interpolation
        /// Computes the spline at the provided coordinate.
        /// Note that the coordinate should be normalized between [0,1].
        [[nodiscard]] constexpr auto interpolate_at(
            const Vec<f64, N>& normalized_coordinate
        ) const noexcept -> mutable_value_type {
            // If there's one note in the entire grid, return its value.
            if (ssize() == 1)
                return span[0];

            // Check that it cannot be simplified to 1d case.
            if constexpr (N == 2) {
                if (noa::any(shape() == 1)) {
                    const auto dim = static_cast<i32>(shape()[0] == 1);
                    return interpolate_at_(normalized_coordinate.filter(dim), span[dim].as_const()); // 1d
                }
            }
            return interpolate_at_(normalized_coordinate, span.as_const()); // nd
        }

        [[nodiscard]] constexpr auto interpolate_at(
            f64 normalized_coordinate
        ) const noexcept -> mutable_value_type requires (N == 1) {
            return interpolate_at(Vec{normalized_coordinate});
        }

    public: // Weights
        /// Computes the node weight at the given coordinate.
        /// Note that the coordinate should be normalized between [0,1].
        [[nodiscard]] static constexpr auto weight_at(
            const Vec<f64, N>& normalized_coordinate,
            const Vec<i64, N>& node_index,
            const Shape<i64, N>& shape
        ) noexcept -> T {
            check(ni::is_inbounds(shape, node_index));

            if (shape.n_elements() == 1)
                return 1;

            // Check that it cannot be simplified to 1d case.
            if constexpr (N == 2) {
                if (noa::any(shape == 1)) {
                    const auto dim = static_cast<i32>(shape[0] == 1);
                    return weight_at_(normalized_coordinate.filter(dim), node_index.filter(dim), shape.filter(dim).vec);
                }
            }
            return weight_at_(normalized_coordinate, node_index, shape.vec);
        }

        [[nodiscard]] constexpr auto weight_at(
            const Vec<f64, N>& normalized_coordinate,
            const Vec<i64, N>& point_index
        ) const noexcept -> mutable_value_type {
            return weight_at(normalized_coordinate, point_index, shape());
        }

    private:
        // Switch coordinate from range [0, 1] to range [0, resolution-1].
        template<size_t S>
        [[nodiscard]] static constexpr auto denormalize_coordinates_(
            const Vec<f64, S>& normalized_coordinate,
            const Vec<i64, S>& resolution
        ) noexcept {
            return noa::clamp(normalized_coordinate, 0., 1.) * (resolution - 1).template as<f64>();
        }

        // Compute the interpolation indices: coordinate -> [p0, p1, p2, p3]
        // The interpolation window is positioned so that the indices are always in the range [-1, size].
        template<size_t S>
        [[nodiscard]] static constexpr auto coordinates_to_interp_indices_(
            const Vec<f64, S>& coordinate,
            const Vec<i64, S>& resolution
        ) noexcept {
            Vec<Vec<i64, 4>, S> indices;
            for (size_t i = 0; i < S; ++i) {
                indices[i][1] = static_cast<i64>(noa::floor(coordinate[i]));

                // Only allow one element in the window to be out of bound.
                // FIXME -1 shouldn't be possible here, but check anyway.
                if (indices[i][1] == -1)
                    indices[i][1] = 0; // [-2, -1, 0, 1] -> [-1, 0, 1, 2]
                if (indices[i][1] == resolution[i] - 1)
                    indices[i][1] -= 1; // [n-2, n-1, n, n+1] -> [n-3, n-2, n-1, n]

                indices[i][0] = indices[i][1] - 1;
                indices[i][2] = indices[i][1] + 1;
                indices[i][3] = indices[i][1] + 2;
            }
            return indices;
        }

        // Given the 4 values [p0, p1, p2, p3], the interpolation fraction is the value
        // in range [0, 1] covering the position interval between p1 and p2.
        template<size_t S>
        [[nodiscard]] static constexpr auto coordinates_to_interp_fraction_(
            const Vec<f64, S>& coordinate,
            const Vec<Vec<i64, 4>, S>& indices
        ) noexcept {
            Vec<f64, S> fraction;
            for (size_t i = 0; i < S; ++i)
                fraction[i] = coordinate[i] - static_cast<f64>(indices[i][1]);
            return fraction;
        }

        // Get the value at a given index. The index must be within [-1, size].
        // If the index is out of bound, i.e. equal to -1 or size, the value
        // is extrapolated using the local gradient.
        template<size_t S>
        static constexpr auto get_values_or_extrapolate_(
            SpanContiguous<const_value_type, S> data,
            Vec<Vec<i64, 4>, S> indices
        ) noexcept {
            auto get = [](SpanContiguous<const_value_type, 1> span, i64 index) {
                if (index == -1) {
                    // i(0)=5, i(1)=7 -> i(-1)=3
                    // i(0)=7, i(1)=5 -> i(-1)=9
                    const auto first = span[0];
                    const auto second = span[1];
                    return first - (second - first);
                }
                if (index == span.ssize()) {
                    // Eg. i(n-2)=5, i(n-1)=7 -> i(n)=9
                    // Eg. i(n-2)=7, i(n-1)=5 -> i(n)=3
                    const auto last = span[index - 1];
                    const auto before_last = span[index - 2];
                    return last + (last - before_last);
                }
                return span[index];
            };

            if constexpr (S == 1) {
                Vec<mutable_value_type, 4> values;
                for (i64 i = 0; i < 4; ++i)
                    values[i] = get(data, indices[0][i]);
                return values;

            } else {
                Vec<Vec<mutable_value_type, 4>, 4> values;

                // Compute every possible row.
                for (i64 r = 0; r < 4; ++r) {
                    const i64 i = indices[0][r];
                    if (i >= 0 and i < data.shape()[0]) {
                        for (i64 c = 0; c < 4; ++c)
                            values[r][c] = get(data[i], indices[1][c]);
                    }
                }
                // Now fill the gaps...
                if (indices[0][0] == -1) { // first row is out
                    for (i64 c = 0; c < 4; ++c) {
                        const auto first = values[1][c];
                        const auto second = values[2][c];
                        values[0][c] = first - (second - first);
                    }
                }
                if (indices[0][3] == data.shape()[0]) { // last row is out
                    for (i64 c = 0; c < 4; ++c) {
                        const auto last = values[2][c];
                        const auto before_last = values[1][c];
                        values[3][c] = last + (last - before_last);
                    }
                }
                return values;
            }
        }

        template<size_t S>
        static constexpr auto interpolate_at_(
            const Vec<f64, S>& normalized_coordinate,
            const SpanContiguous<const_value_type, S> data
        ) {
            const Vec<i64, S> resolution = data.shape().vec;
            const Vec<f64, S> coordinates = denormalize_coordinates_(normalized_coordinate, resolution);
            const Vec<Vec<i64, 4>, S> indices = coordinates_to_interp_indices_(coordinates, resolution);
            const Vec<f64, S> fraction = coordinates_to_interp_fraction_(coordinates, indices);
            const Vec<Vec<f64, S>, 4> weights = noa::interpolation_weights<INTERP, Vec<f64, S>>(fraction);

            const auto values = get_values_or_extrapolate_(data, indices);

            mutable_value_type interpolant{};
            if constexpr (S == 1) {
                for (i64 i = 0; i < 4; ++i)
                    interpolant += values[i] * weights[i][0];

            } else if constexpr (S == 2) {
                for (i64 y = 0; y < 4; ++y) {
                    mutable_value_type interpolant_y{};
                    for (i64 x = 0; x < 4; ++x)
                        interpolant_y += values[y][x] * weights[x][1];
                    interpolant += interpolant_y * weights[y][0];
                }
            } else {
                static_assert(nt::always_false<T>);
            }
            return interpolant;
        }

        template<size_t S>
        static constexpr auto weight_at_(
            const Vec<f64, S>& normalized_coordinate,
            const Vec<i64, S>& node_index,
            const Vec<i64, S>& resolution
        ) {
            const Vec<coord_type, S> coordinates = denormalize_coordinates_(normalized_coordinate, resolution);
            const Vec<cubic_indices_type, S> indices = coordinates_to_interp_indices_(coordinates, resolution);
            const Vec<coord_type, S> fraction = coordinates_to_interp_fraction_(coordinates, indices);
            const Vec<Vec<f64, S>, 4> weights = noa::interpolation_weights<INTERP, Vec<f64, S>>(fraction);

            if constexpr (S == 1) {
                for (i32 i = 0; i < 4; ++i)
                    if (indices[0][i] == node_index[0])
                        return weights[i][0];
            } else if constexpr (S == 2) {
                for (i32 i = 0; i < 4; ++i)
                    for (i32 j = 0; j < 4; ++j)
                        if (indices[i] == node_index[i] and indices[j] == node_index[j])
                            return weights[i][0] + weights[j][1];
            } else {
                static_assert(nt::always_false<T>);
            }

            // The node does not affect the given coordinate.
            return weight_type{};
        }
    };

    template<typename T, size_t N>
    using SplineGridCubic = SplineGrid<T, N, noa::Interp::CUBIC>;

    template<typename T, size_t N>
    using SplineGridCubicBSpline = SplineGrid<T, N, noa::Interp::CUBIC_BSPLINE>;

    // template<nt::real T, typename Op = noa::Copy>
    // void sample_cubic_bspline_1d(
    //     const SplineGrid<f64, 1>& spline,
    //     const SpanContiguous<T, 1>& output,
    //     i64 channel = 0,
    //     Op&& op = Op{}
    // ) {
    //     const f64 norm = 1 / static_cast<f64>(output.ssize() - 1);
    //     for (i64 i = 0; i < output.ssize(); ++i) {
    //         const f64 coordinate = static_cast<f64>(i) * norm; // [0,1]
    //         output[i] = static_cast<T>(op(spline.interpolate_at(coordinate, channel)));
    //     }
    // }
    //
    // template<nt::real T, typename Op = noa::Copy>
    // void sample_cubic_bsplines_1d(
    //     const SplineGrid<f64, 1>& splines,
    //     const SpanContiguous<T, 2>& output,
    //     Op&& op = Op{}
    // ) {
    //     const f64 norm = 1 / static_cast<f64>(output.ssize() - 1);
    //     for (i64 i = 0; i < output.shape()[0]; ++i) {
    //         for (i64 j = 0; j < output.shape()[1]; ++j) {
    //             const f64 coordinate = static_cast<f64>(i) * norm; // [0,1]
    //             output(i, j) = static_cast<T>(op(splines.interpolate_at(coordinate, i)));
    //         }
    //     }
    // }
    //
    // template<nt::writable_varray_of_real Output, typename Op = noa::Copy>
    // void sample_cubic_bspline_1d(
    //     const SplineGrid<f64, 1>& spline,
    //     const Output& output,
    //     i64 channel = 0,
    //     Op&& op = Op{}
    // ) {
    //     sample_cubic_bspline_1d(spline, output.span_1d_contiguous(), channel, std::forward<Op>(op));
    // }
    //
    // template<nt::writable_varray_of_real Output, typename Op = noa::Copy>
    // void sample_cubic_bsplines_1d(
    //     const SplineGrid<f64, 1>& splines,
    //     const Output& output,
    //     Op&& op = Op{}
    // ) {
    //     check(splines.n_channels() == output.shape()[0] and
    //           output.shape()[1] == 1 and
    //           output.shape()[2] == 1);
    //     sample_cubic_bsplines_1d(splines, output.span().filter(0, 3).as_contiguous(), std::forward<Op>(op));
    // }
    //
    // // TODO unused
    // inline void reset_spline_resolution(
    //     const SplineGrid<f64, 1>& input_spline,
    //     const SplineGrid<f64, 1>& output_spline
    // ) {
    //     struct FittingData {
    //         SplineGrid<f64, 1> input_spline{};
    //         SplineGrid<f64, 1> output_spline{};
    //     };
    //
    //     // Compute the least-square score between the input and output spline.
    //     auto cost = [](u32, const f64*, f64*, void* instance) {
    //         auto& opt = *static_cast<FittingData*>(instance);
    //         f64 score{};
    //         for (i64 i{}; i < 512; ++i) {
    //             const f64 coordinate = static_cast<f64>(i) / static_cast<f64>(512 - 1);
    //             const f64 input = opt.input_spline.interpolate_at(coordinate);
    //             const f64 output = opt.output_spline.interpolate_at(coordinate);
    //             const f64 diff = input - output;
    //             score += diff * diff;
    //         }
    //         return score;
    //     };
    //
    //     const auto n_channels = input_spline.n_channels();
    //     check(n_channels == output_spline.n_channels());
    //
    //     for (i64 i{}; i < n_channels; ++i) {
    //         // Extract the current channel.
    //         auto optimizer_data = FittingData{
    //             .input_spline = SplineGrid(
    //                 input_spline.resolution(), 1,
    //                 input_spline.span().subregion(i)
    //             ),
    //             .output_spline = SplineGrid(
    //                 output_spline.resolution(), 1,
    //                 output_spline.span().subregion(i)
    //             ),
    //         };
    //
    //         // Some stats about the data points.
    //         f64 mean{}, min{std::numeric_limits<f64>::max()}, max{std::numeric_limits<f64>::lowest()};
    //         for (auto node: optimizer_data.input_spline.span().as_1d()) {
    //             mean += node;
    //             min = std::min(min, node);
    //             max = std::max(max, node);
    //         }
    //         mean /= static_cast<f64>(input_spline.resolution()[0]);
    //
    //         // Initialize the spline to a line at the mean.
    //         for (auto& v: optimizer_data.output_spline.span().as_1d())
    //             v = mean;
    //
    //         // Here use a local derivative-less algorithm, since we assume
    //         // the spline resolution is <= 5, and the cost is cheap to compute.
    //         auto optimizer = Optimizer(NLOPT_LN_SBPLX, output_spline.resolution()[0]);
    //         optimizer.set_min_objective(cost, &optimizer_data);
    //
    //         // Usually min/max are large values, but the range is small, so use that to set the bounds.
    //         const auto value_range = max - min;
    //         optimizer.set_bounds(min - value_range * 1.5, max + value_range * 1.5);
    //         optimizer.optimize(optimizer_data.output_spline.span().data());
    //         // TODO tolerance
    //     }
    // }
}
