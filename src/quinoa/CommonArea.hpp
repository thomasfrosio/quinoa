#pragma once

#include <noa/Geometry.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"

namespace qn {
    /// Compute the common field-of-view (FOV) between multiple tilt images.
    /// The area is initialized to be the entire FOV, and is progressively updated by adding the
    /// FOV of the tilt images. The common area is the region left visible in every single view.
    class CommonArea {
    public:
        using matrix_type = Mat<f32, 2, 3>;

        CommonArea() = default;
        CommonArea(
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

            // Start with the perfect area, at tilt 0, encompassing the entire image.
            const auto initial_area = Vec{ // left, right, bottom, top
                -shape[1] / 2, (shape[1] - 1) / 2,
                -shape[0] / 2, (shape[0] - 1) / 2,
            }.as<f64>();

            Vec common_area = initial_area;
            for (const MetadataSlice& slice: metadata) {
                // Cosine stretches relative to horizontal plane.
                const auto angles = noa::deg2rad(slice.angles);
                const auto cos_scale = Vec{1 / noa::cos(angles.filter(2, 1))}; // 1 = cos(0)

                // Shift and stretch the current slice area to compare it with the common area.
                const Mat<f64, 2, 3> current_tilt_to_0deg = (
                    ng::linear2affine(ng::rotate(angles[0])) *
                    ng::linear2affine(ng::scale(cos_scale)) *
                    ng::linear2affine(ng::rotate(-angles[0])) *
                    ng::translate(-slice.shifts)
                ).pop_back();

                // Compute the area of the current view.
                auto current_area = Vec{
                    (current_tilt_to_0deg * Vec<f64, 3>{0, initial_area[0], 1})[1],
                    (current_tilt_to_0deg * Vec<f64, 3>{0, initial_area[1], 1})[1],
                    (current_tilt_to_0deg * Vec<f64, 3>{initial_area[2], 0, 1})[0],
                    (current_tilt_to_0deg * Vec<f64, 3>{initial_area[3], 0, 1})[0]
                };

                // Add this view to the common area.
                // Because the high tilts have a bigger field-of-view (which is encoded by the stretching),
                // they can have large shifts perpendicular to the tilt axis and still contain the entire
                // common area. Large shifts along the tilt-axis or in the lower tilts are more likely
                // to limit the common area.
                common_area[0] = noa::max(common_area[0], current_area[0]); // negative values
                common_area[1] = noa::min(common_area[1], current_area[1]); // positive values
                common_area[2] = noa::max(common_area[2], current_area[2]); // negative values
                common_area[3] = noa::min(common_area[3], current_area[3]); // positive values
            }

            // FIXME

            // Compute the center and radius of the common area.
            auto common_area_diameter = Vec{
                -common_area[0] + common_area[1] + 1,
                -common_area[2] + common_area[3] + 1
            };
            m_common_area_radius = common_area_diameter / 2;
            m_common_area_center = (m_shape.vec / 2).as<f64>() + common_area.filter(0, 2) + m_common_area_radius;
        }

        [[nodiscard]] auto compute_inverse_transform(
            const MetadataSlice& metadata,
            bool correct_shifts = false
        ) const -> Mat23<f32> {
            check(noa::all(m_common_area_radius >= 0), "Common area geometry is not initialized");

            // The cosine shrink is relative to the 0deg pitch/tilt.
            const Vec<f64, 3> angles = noa::deg2rad(metadata.angles);
            const Vec<f64, 2> cos_scale = noa::cos(angles.filter(2, 1));

            // Unfortunately, this shrinks the smooth edge of the mask.
            // TODO add draw_shapes() that allows to batch the shape?
            auto shifts = correct_shifts ? metadata.shifts : Vec<f64, 2>{};
            return (
                ng::translate(m_common_area_center) *
                ng::linear2affine(ng::rotate(angles[0])) *
                ng::linear2affine(ng::scale(1 / cos_scale)) *
                ng::linear2affine(ng::rotate(-angles[0])) *
                ng::translate(-m_common_area_center - shifts)
            ).pop_back().as<f32>();
        }

        [[nodiscard]] auto compute_inverse_transforms(
            const MetadataStack& metadata,
            SpanContiguous<Mat23<f32>, 1> output,
            bool correct_shifts = false
        ) const {
            for (const auto& slice : metadata)
                output.at(slice.index) = compute_inverse_transform(slice, correct_shifts);
        }

        template<nt::varray_decay_or_value_of_almost_any<Mat<f32, 2, 3>> Matrices>
        void mask(
            const View<const f32>& input,
            const View<f32>& output,
            Matrices&& inverse_transforms,
            f64 smooth_edge_percent
        ) const {
            check(vall(noa::Equal{}, m_shape, input.shape().filter(2, 3)) and
                  vall(noa::Equal{}, m_shape, output.shape().filter(2, 3)),
                  "Shapes don't match");

            const auto smoothness = smooth_edge(smooth_edge_percent);
            ng::draw(input, output, ng::Rectangle{
                .center = center(),
                .radius = radius() - smoothness,
                .smoothness = smoothness,
            }.draw<f32>(), std::forward<Matrices>(inverse_transforms));
        }

        void mask(
            const View<const f32>& input,
            const View<f32>& output,
            const MetadataStack& metadata,
            bool correct_shifts,
            f64 smooth_edge_percent
        ) const {
            auto inverse_transforms = Array<Mat<f32, 2, 3>>(metadata.ssize());
            compute_inverse_transforms(metadata, inverse_transforms.span_1d_contiguous(), correct_shifts);
            if (output.device().is_gpu())
                inverse_transforms = std::move(inverse_transforms).to(
                    {.device = output.device(), .allocator = Allocator::ASYNC});

            mask(input, output, std::move(inverse_transforms), smooth_edge_percent);
        }

        void mask(
            const View<const f32>& input,
            const View<f32>& output,
            const MetadataSlice& metadata,
            bool correct_shifts,
            f64 smooth_edge_percent
        ) const {
            check(input.shape()[0] == 1);
            auto inverse_transforms = compute_inverse_transform(metadata, correct_shifts);
            mask(input, output, inverse_transforms, smooth_edge_percent);
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
