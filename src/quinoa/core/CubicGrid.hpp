#pragma once

#include "quinoa/Types.h"
#include <noa/core/geometry/Interpolate.hpp>

namespace qn {
    // Cubic spline interpolation on multidimensional grids (only 1d and 2d are currently supported).
    // Grids consist of uniformly spaced points covering the full extend of each dimension.
    //  - The resolution controls the number of points covering each dimension.
    //  - The channels controls the number of values stored on each grid point.
    //
    // We can obtain the value (interpolant) at any continuous point on the grid.
    // The grid coordinate system extends from [0, 1] along each grid dimension.
    // The interpolant is obtained by sequential application of cubic spline interpolation along each dimension.
    template<typename Value, size_t N>
    class CubicSplineGrid {
    public:
        static_assert(N == 1 || N == 2);
        static_assert(noa::traits::is_real_or_complex_v<Value>);

    public:
        CubicSplineGrid() = default;

        // Creates a grid.
        // It can be useful to manage the grid values, as such, the contiguous array with the data
        // can be passed. In this case, data() will return that array. Otherwise, a new
        // array is allocated. The data is in the C(H)W order.
        explicit CubicSplineGrid(const Vec<i64, N>& resolution, i64 channels = 1, Value* data = nullptr)
                : m_resolution(resolution), m_channels(channels) {
            if (data) {
                m_data = data;
            } else {
                m_buffer.resize(static_cast<size_t>(elements()));
                // spline.is_empty() should be true if elements() == 0, so enforce it here
                m_data = m_buffer.empty() ? nullptr : m_buffer.data();
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 1) && std::is_void_v<Void>>>
        explicit CubicSplineGrid(i64 resolution, i64 channels = 1, Value* data = nullptr)
                : CubicSplineGrid(Vec1<i64>{resolution}, channels, data) {}

        CubicSplineGrid(const CubicSplineGrid& to_copy)
                : m_resolution(to_copy.resolution()),
                  m_channels(to_copy.m_channels),
                  m_buffer(to_copy.m_buffer),
                  m_data(to_copy.m_buffer.empty() ? to_copy.m_data : m_buffer.data()) {}

        CubicSplineGrid& operator=(const CubicSplineGrid& to_copy) {
            if (this != &to_copy) {
                m_resolution = to_copy.resolution();
                m_channels = to_copy.m_channels;
                m_buffer = to_copy.m_buffer;
                m_data = to_copy.m_buffer.empty() ? to_copy.m_data : m_buffer.data();
            }
            return *this;
        }

    public:
        void set_all_points_to(Value value) const noexcept {
            if (!is_empty())
                std::fill(begin(), end(), value);
        }

        [[nodiscard]] constexpr bool is_empty() const noexcept { return m_data == nullptr; }
        [[nodiscard]] constexpr Value* data() const noexcept { return m_data; }
        [[nodiscard]] constexpr Value* begin() const noexcept { return data(); }
        [[nodiscard]] constexpr Value* end() const noexcept { return data() + elements(); }

        [[nodiscard]] constexpr const Vec<i64, N>& resolution() const noexcept { return m_resolution; }
        [[nodiscard]] constexpr i64 channels() const noexcept { return m_channels; }
        [[nodiscard]] constexpr i64 elements() const noexcept { return noa::math::product(resolution()) * channels(); }
        [[nodiscard]] constexpr i64 elements_per_channel() const noexcept { return noa::math::product(resolution()); }

    public: // Interpolation
        [[nodiscard]] constexpr Value interpolate(
                const Vec<f64, N>& normalized_coordinate,
                i64 channel = 0
        ) const noexcept {
            NOA_ASSERT(!is_empty() && channel < channels());
            // Note: normalized_coordinate will be clamped between [0,1].

            if constexpr (N == 1) {
                return fetch_1d_(
                        normalized_coordinate[0],
                        data() + channel * elements_per_channel(),
                        1,
                        resolution()[0]);
            } else if constexpr (N == 2) {
                return fetch_2d_(
                        normalized_coordinate,
                        data() + channel * elements_per_channel(),
                        Vec2<i64>{resolution()[1], 1},
                        resolution());
            } else {
                static_assert(noa::traits::always_false_v<Value>);
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 1) && std::is_void_v<Void>>>
        [[nodiscard]] constexpr Value interpolate(
                f64 normalized_coordinate,
                i64 channel = 0
        ) const noexcept {
            return interpolate(Vec1<f64>{normalized_coordinate}, channel);
        }

    public: // Weights
        // Computes the B-spline weight, i.e. how much this point affects the coordinate.
        [[nodiscard]] constexpr Value weight(
                const Vec<f64, N>& normalized_coordinate,
                const Vec<i64, N>& point_index
        ) const noexcept {
            NOA_ASSERT(!is_empty() && all(point_index >= 0 && point_index < resolution()));

            if constexpr (N == 1) {
                return fetch_weight_1d_(
                        normalized_coordinate[0],
                        point_index[0],
                        resolution()[0]);
            } else if constexpr (N == 2) {
                fetch_weight_2d_(
                        normalized_coordinate,
                        point_index,
                        resolution());
            } else {
                static_assert(noa::traits::always_false_v<Value>);
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 1) && std::is_void_v<Void>>>
        [[nodiscard]] constexpr Value weight(
                f64 normalized_coordinate,
                i64 point_index = 0
        ) const noexcept {
            return weight(Vec1<f64>{normalized_coordinate}, Vec1<i64>{point_index});
        }

    private: // Fetch functions
        [[nodiscard]] constexpr auto fetch_1d_(
                f64 normalized_coordinate,
                const Value* data,
                i64 stride, i64 size
        ) const noexcept -> f64 {
            // Special case. Interpolation is not necessary because the output value
            // is always the node value, regardless of the coordinate.
            if (size == 1)
                return data[0];

            const f64 coordinate = denormalize_coordinate_(normalized_coordinate, size);
            const Vec4<i64> indexes = coordinate_to_interp_indexes_(coordinate, size);

            // Uniformly spaced control points [p0, p1, p2, p3]. p0 and p3 can be extrapolated.
            Vec4<Value> values;
            for (i64 i = 0; i < 4; ++i)
                values[i] = get_value_or_extrapolate_(data, stride, indexes[i], size);

            const f64 fraction = coordinate_to_interp_fraction_(coordinate, indexes);
            return noa::geometry::interpolate::cubic_bspline_1d(
                    values[0], values[1], values[2], values[3], fraction);
        }

        [[nodiscard]] constexpr auto fetch_weight_1d_(
                f64 normalized_coordinate,
                i64 node_index,
                i64 size
        ) const noexcept -> f64 {
            // Special case. Interpolation is not necessary because the output weight
            // is always 1, regardless of the coordinate.
            if (size == 1)
                return Value{1};

            const f64 coordinate = denormalize_coordinate_(normalized_coordinate, size);
            const Vec4<i64> indexes = coordinate_to_interp_indexes_(coordinate, size);

            // Uniformly spaced control points [p0, p1, p2, p3]. p0 and p3 off the grid.
            // For the weight computation, set all nodes to 0, except the current node, which is set to 1.
            // The interpolated value gives us the contribution of the current node at the input coordinate.
            Vec4<Value> values{0};
            i64 i = 0;
            for (; i < 4; ++i) {
                if (indexes[i] == node_index) {
                    values[i] = 1;
                    break; // we have one node, so once it's found, stop
                }
            }
            if (i == 4)
                return 0; // all values are 0, the node has no effect at the input coordinate, no need to interpolate

            const f64 fraction = coordinate_to_interp_fraction_(coordinate, indexes);
            return noa::geometry::interpolate::cubic_bspline_1d(
                    values[0], values[1], values[2], values[3], fraction);
        }

        [[nodiscard]] constexpr f64 fetch_2d_(
                Vec2<f64> normalized_coordinate,
                const Value* data,
                Vec2<i64> strides,
                Vec2<i64> shape
        ) const noexcept {
            // Check that it cannot be simplified to 1d case.
            if (noa::any(shape == 1)) {
                if (noa::all(shape == 1))
                    return data[0];
                if (shape[0] == 1) {
                    return fetch_1d_(normalized_coordinate[1], data, strides[1], shape[1]);
                } else { // shape[1] == 1
                    return fetch_1d_(normalized_coordinate[0], data, strides[0], shape[0]);
                }
            }

            const Vec2<f64> coordinate {
                    denormalize_coordinate_(normalized_coordinate[0], shape[0]),
                    denormalize_coordinate_(normalized_coordinate[1], shape[1])
            };

            Vec4<i64> indexes[2];
            indexes[0] = coordinate_to_interp_indexes_(coordinate[0], shape[0]);
            indexes[1] = coordinate_to_interp_indexes_(coordinate[1], shape[1]);

            Value values[4][4]; // height-width
            get_values_or_extrapolate_2d_(data, strides, shape, indexes, values);

            const Vec2<f64> fraction{
                    coordinate_to_interp_fraction_(coordinate[0], indexes[0]),
                    coordinate_to_interp_fraction_(coordinate[1], indexes[1])
            };
            return noa::geometry::interpolate::cubic_bspline_2d(values, fraction[1], fraction[0]);
        }

        [[nodiscard]] constexpr f64 fetch_weight_2d_(
                Vec2<f64> normalized_coordinate,
                Vec2<i64> node_index,
                Vec2<i64> shape
        ) const noexcept {
            // Check that it cannot be simplified to 1d case.
            if (noa::any(shape == 1)) {
                if (noa::all(shape == 1))
                    return 1;
                if (shape[0] == 1) {
                    return fetch_weight_1d_(normalized_coordinate[1], node_index[1], shape[1]);
                } else { // shape[1] == 1
                    return fetch_weight_1d_(normalized_coordinate[0], node_index[0], shape[0]);
                }
            }

            const Vec2<f64> coordinate {
                    denormalize_coordinate_(normalized_coordinate[0], shape[0]),
                    denormalize_coordinate_(normalized_coordinate[1], shape[1])
            };

            Vec4<i64> indexes[2];
            indexes[0] = coordinate_to_interp_indexes_(coordinate[0], shape[0]);
            indexes[1] = coordinate_to_interp_indexes_(coordinate[1], shape[1]);

            Value values[4][4]{0}; // height-width
            bool found{false};
            for (i64 h = 0; h < 4; ++h) {
                for (i64 w = 0; w < 4; ++w) {
                    if (noa::all(node_index == Vec2<i64>{h, w})) {
                        values[h][w] = 1;
                        found = true;
                        break;
                    }
                }
            }
            if (!found)
                return 0;

            const Vec2<f64> fraction{
                    coordinate_to_interp_fraction_(coordinate[0], indexes[0]),
                    coordinate_to_interp_fraction_(coordinate[1], indexes[1])
            };
            return noa::geometry::interpolate::cubic_bspline_2d(values, fraction[1], fraction[0]);
        }

    private:
        // Switch coordinate from range [0, 1] to range [0, resolution-1].
        [[nodiscard]] static constexpr f64 denormalize_coordinate_(
                f64 normalized_coordinate,
                i64 size
        ) noexcept {
            normalized_coordinate = noa::math::clamp(normalized_coordinate, 0., 1.);
            return normalized_coordinate * static_cast<f64>(size - 1);
        }

        // Compute the interpolation indexes. The interpolation window is positioned so that
        // the indexes are always in the range [-1, size].
        [[nodiscard]] static constexpr Vec4<i64>
        coordinate_to_interp_indexes_(f64 coordinate, i64 size) noexcept {
            Vec4<i64> indexes;
            indexes[1] = static_cast<i64>(noa::math::floor(coordinate));

            // Only allow one element in the window to be out of bound.
            if (indexes[1] == -1)
                indexes[1] = 0; // [-2, -1, 0, 1] -> [-1, 0, 1, 2]
            if (indexes[1] == size - 1)
                indexes[1] -= 1; // [n-2, n-1, n, n+1] -> [n-3, n-2, n-1, n]

            indexes[0] = indexes[1] - 1;
            indexes[2] = indexes[1] + 1;
            indexes[3] = indexes[1] + 2;
            return indexes;
        }

        // Given the 4 values [p0, p1, p2, p3], the interpolation fraction is the value
        // in range [0, 1] covering the position interval between p1 and p2.
        [[nodiscard]] static constexpr f64
        coordinate_to_interp_fraction_(f64 coordinate, const Vec4<i64>& indexes) noexcept {
            return coordinate - static_cast<f64>(indexes[1]);
        }

        // Get the value at a given index. The index must be within [-1, size].
        // If the index is out of bound, i.e. equal to -1 or size, the value
        // is extrapolated using the local gradient.
        [[nodiscard]]
        static constexpr Value get_value_or_extrapolate_(
                const Value* data, i64 stride, i64 index, i64 size
        ) noexcept {
            if (index == -1) {
                // i(0)=5, i(1)=7 -> i(-1)=3
                // i(0)=7, i(1)=5 -> i(-1)=9
                const auto first = data[0 * stride];
                const auto second = data[1 * stride];
                return first - (second - first);

            } else if (index == size) {
                // Eg. i(n-2)=5, i(n-1)=7 -> i(n)=9
                // Eg. i(n-2)=7, i(n-1)=5 -> i(n)=3
                const auto last = data[(index - 1) * stride];
                const auto before_last = data[(index - 2) * stride];
                return last + (last - before_last);

            } else {
                return data[index * stride];
            }
        }

        static constexpr void get_values_or_extrapolate_2d_(
                const Value* data,
                Vec2<i64> strides,
                Vec2<i64> shape,
                Vec4<i64> indexes[2],
                Value values[4][4]
        ) noexcept {
            // Compute every possible row.
            for (i64 row = 0; row < 4; ++row) {
                const i64 index = indexes[0][row];
                if (index >= 0 && index < shape[0]) {
                    for (i64 column = 0; column < 4; ++column)
                        values[row][column] = get_value_or_extrapolate_(
                                data + index * strides[0], strides[1],
                                indexes[1][column], shape[1]);
                }
            }

            // Now fill the gaps...
            if (indexes[0][0] == -1) { // first row is out
                for (i64 column = 0; column < 4; ++column) {
                    const auto first = values[1][column];
                    const auto second = values[2][column];
                    values[0][column] = first - (second - first);
                }
            }
            if (indexes[0][3] == shape[0]) { // last row is out
                for (i64 column = 0; column < 4; ++column) {
                    const auto last = values[2][column];
                    const auto before_last = values[1][column];
                    values[3][column] = last + (last - before_last);
                }
            }
        }

    private:
        Vec<i64, N> m_resolution{};
        i64 m_channels{};
        std::vector<Value> m_buffer{};
        Value* m_data{};
    };

    template<typename Functor>
    void apply_cubic_bspline_1d(
            const View<const f32>& input,
            const View<f32>& output,
            const CubicSplineGrid<f64, 1>& spline,
            Functor functor
    ) {
        const auto input_1d = input.accessor_contiguous_1d();
        const auto output_1d = output.accessor_contiguous_1d();
        const f64 norm = 1 / static_cast<f64>(output.ssize() - 1);
        for (i64 i = 0; i < output.ssize(); ++i) {
            const f64 coordinate = static_cast<f64>(i) * norm; // [0,1]
            const auto interpolant = static_cast<f32>(spline.interpolate(coordinate));
            const auto value = functor(input_1d[i], interpolant);
            output_1d[i] = value;
        }
    }
}
