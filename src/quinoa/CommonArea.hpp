#pragma once

#include <noa/Array.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct ParallelogramMask {
        using value_type = f32;
        using vec_type = Vec<value_type, 2>;
        using mat_type = Mat<value_type, 2, 2>;

        mat_type world2local{};
        vec_type local_smoothness{};
        vec_type origin{};

        ParallelogramMask() = default;

        // Creates a parallelogram. The provided vertices a,b,(c,)d should form a CCW convex quad.
        ParallelogramMask(const vec_type& a, const vec_type& b, const vec_type& d, value_type smoothness) {
            const auto u = d - a;
            const auto v = b - a;
            world2local = mat_type::from_rows(u, v).inverse();
            local_smoothness[0] = smoothness / noa::norm(u);
            local_smoothness[1] = smoothness / noa::norm(v);
            origin = a;
        }

        NOA_HD static auto smooth_edge(value_type x, value_type width) -> value_type{
            if (x <= 0 or x >= 1)
                return 0;

            const auto edge_distance = noa::min(x, 1 - x);
            if (edge_distance >= width)
                return 1;

            constexpr auto PI = noa::Constant<value_type>::PI;
            const auto distance = noa::clamp(edge_distance / width, 0, 1);
            return (1 - noa::cos(PI * distance)) * static_cast<value_type>(0.5);
        }

        NOA_HD auto operator()(const vec_type& coordinates) const -> value_type {
            const auto local_coordinates = world2local * (coordinates - origin);
            const auto mu = smooth_edge(local_coordinates[0], local_smoothness[0]);
            const auto mv = smooth_edge(local_coordinates[1], local_smoothness[1]);
            return mu * mv;
        }
        template<nt::integer T>
        NOA_HD auto operator()(const T& h, const T& w) const -> value_type {
            return (*this)(vec_type::from_values(h, w));
        }
    };

    struct MaskFOVs {
        SpanContiguous<const f32, 3> input_images;
        SpanContiguous<f32, 3> output_images;
        SpanContiguous<const ParallelogramMask, 1> parallelograms;

        NOA_HD void operator()(i64 i, i64 h, i64 w) const {
            auto value = parallelograms[i](h, w);
            if (value > 1e-6f)
                value *= input_images(i, h, w);
            output_images(i, h, w) = value;
        }
    };

    struct MaskFOV {
        SpanContiguous<const f32, 2> input_image;
        SpanContiguous<f32, 2> output_image;
        ParallelogramMask parallelogram;

        NOA_HD void operator()(i64 h, i64 w) const {
            auto value = parallelogram(h, w);
            if (value > 1e-6f)
                value *= input_image(h, w);
            output_image(h, w) = value;
        }
    };
}

namespace qn {
    struct FOVMaskOptions {
        /// Smoothing size, relative to the image shape.
        f64 smooth_edge_percent = 0.1;

        /// Adds the shift offset to the mask.
        /// If true, the mask should be applied on the unaligned images.
        bool add_shifts{true};
    };

    class CommonFOV {
    public:
        CommonFOV() = default;
        CommonFOV(
            const Shape<i64, 2>& shape,
            const MetadataStack& metadata
        ) {
            set_geometry(shape, metadata);
        }

        void set_geometry(
            const Shape<i64, 2>& shape,
            const MetadataStack& metadata
        ) {
            m_shape = shape;

            // Initial FOV, at tilt 0, encompassing the entire image.
            const auto initial_fov = Vec{ // bottom, top, left, right
                -shape[0] / 2, (shape[0] - 1) / 2,
                -shape[1] / 2, (shape[1] - 1) / 2,
            }.as<f64>();

            Vec common_fov = initial_fov;
            for (const auto& slice: metadata) {
                // Center and stretch the image along its tilt-axis to compute its FOV.
                const auto angles = noa::deg2rad(slice.angles);
                const auto cos_scale = Vec{1 / noa::cos(angles.filter(2, 1))}; // 1 = cos(0)
                const auto to_0deg = (
                    ng::rotate<true>(angles[0]) *
                    ng::scale<true>(cos_scale) *
                    ng::rotate<true>(-angles[0]) *
                    ng::translate(-slice.shifts)
                ).pop_back();

                // Compute the FOV of the current image.
                auto image_fov = Vec{
                    (to_0deg * Vec<f64, 3>{initial_fov[0], 0, 1})[0],
                    (to_0deg * Vec<f64, 3>{initial_fov[1], 0, 1})[0],
                    (to_0deg * Vec<f64, 3>{0, initial_fov[2], 1})[1],
                    (to_0deg * Vec<f64, 3>{0, initial_fov[3], 1})[1],
                };

                // Restrain the FOV with what this image sees. Regions that this image doesn't see are removed.
                // Because the high tilts have a bigger FOV they can have large shifts perpendicular to the tilt axis
                // and still contain the entire FOV. Large shifts along the tilt-axis or in the lower tilts are more
                // likely to limit the final FOV.
                common_fov[0] = noa::max(common_fov[0], image_fov[0]); // negative values
                common_fov[1] = noa::min(common_fov[1], image_fov[1]); // positive values
                common_fov[2] = noa::max(common_fov[2], image_fov[2]); // negative values
                common_fov[3] = noa::min(common_fov[3], image_fov[3]); // positive values
            }

            // FIXME

            // Compute the center and radius of the common FOV.
            auto common_area_diameter = Vec{
                -common_fov[0] + common_fov[1] + 1,
                -common_fov[2] + common_fov[3] + 1,
            };
            m_common_area_radius = common_area_diameter / 2;
            m_common_area_center = (m_shape.vec / 2).as<f64>() + common_fov.filter(0, 2) + m_common_area_radius;
        }

        [[nodiscard]] auto set_fov(
            const MetadataSlice& metadata,
            const FOVMaskOptions& options = {}
        ) const {
            check(noa::all(m_common_area_radius >= 0), "Common area geometry is not initialized");

            const auto angles = noa::deg2rad(metadata.angles);
            const auto scale = noa::cos(angles.filter(2, 1));
            const auto shifts = options.add_shifts ? metadata.shifts : Vec<f64, 2>{};

            const auto smoothness = smooth_edge(options.smooth_edge_percent);

            // Parallelogram, such as u=a->d, v=a->b.
            auto a = m_common_area_center - m_common_area_radius;
            auto b = a + Vec{0., m_common_area_radius[1] * 2};
            auto d = a + Vec{m_common_area_radius[0] * 2, 0.};

            // Transform the basis to apply the shrinking caused by the tilt.
            const auto matrix = (
                ng::translate(m_common_area_center) *
                ng::rotate<true>(angles[0]) *
                ng::scale<true>(1 / scale) *
                ng::rotate<true>(-angles[0]) *
                ng::translate(-m_common_area_center - shifts)
            ).inverse().pop_back();
            a = matrix * a.push_back(1);
            b = matrix * b.push_back(1);
            d = matrix * d.push_back(1);

            return ParallelogramMask(a.as<f32>(), b.as<f32>(), d.as<f32>(), static_cast<f32>(smoothness));
        }

        void set_fovs(
            const MetadataStack& metadata,
            const SpanContiguous<ParallelogramMask, 1>& parallelograms,
            const FOVMaskOptions& options = {}
        ) const {
            for ( const auto& slice : metadata)
                parallelograms.at(slice.index) = set_fov(slice, options);
        }

        void apply_fovs(
            const View<const f32>& input,
            const View<f32>& output,
            const View<ParallelogramMask>& parallelograms
        ) const {
            check(vall(noa::Equal{}, m_shape, input.shape().filter(2, 3)) and
                  vall(noa::Equal{}, m_shape, output.shape().filter(2, 3)),
                  "Shapes don't match");

            noa::iwise(output.shape().filter(0, 2, 3), output.device(), MaskFOVs{
                .input_images = input.span().filter(0, 2, 3).as_contiguous(),
                .output_images = output.span().filter(0, 2, 3).as_contiguous(),
                .parallelograms = parallelograms.span_1d(),
            });
        }

        void apply_fov(
            const View<const f32>& input,
            const View<f32>& output,
            const ParallelogramMask& parallelogram
        ) const {
            check(vall(noa::Equal{}, m_shape, input.shape().filter(2, 3)) and
                  vall(noa::Equal{}, m_shape, output.shape().filter(2, 3)),
                  "Shapes don't match");

            noa::iwise(output.shape().filter(2, 3), output.device(), MaskFOV{
                .input_image = input.span().filter(2, 3).as_contiguous(),
                .output_image = output.span().filter(2, 3).as_contiguous(),
                .parallelogram = parallelogram,
            });
        }

        [[nodiscard]] constexpr auto center() const noexcept -> const Vec2<f64>& { return m_common_area_center; }
        [[nodiscard]] constexpr auto radius() const noexcept -> const Vec2<f64>& { return m_common_area_radius; }
        [[nodiscard]] constexpr auto smooth_edge(f64 smooth_edge_percent) const noexcept -> f64 {
            // The smooth-edge percent is relative to the common area size, not the original image size.
            auto smooth_edge_size = noa::max(m_common_area_radius * 2) * smooth_edge_percent;
            smooth_edge_size = std::max(10., smooth_edge_size);
            return smooth_edge_size;
        }

    private:
        Shape2<i64> m_shape{};
        Vec2<f64> m_common_area_center{};
        Vec2<f64> m_common_area_radius{-1};
    };
}
