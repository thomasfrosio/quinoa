#pragma once

#include <noa/Memory.hpp>
#include <noa/Geometry.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Utilities.h"

namespace qn {
    // Compute the common area between multiple views.
    // The area is initialized to be the entire field-of-view, and is progressively updated by adding the
    // field-of-view of the views. The common area is the region that is left visible in every single view.
    class CommonArea {
    public:
        CommonArea() = default;

        CommonArea(
                const Shape2<i64>& shape,
                const MetadataStack& metadata,
                f64 max_size_loss_percent = 0.15
        ) {
            set_geometry(shape, metadata, max_size_loss_percent);
        };

        // Given a stack geometry, set the common area.
        // If the slices are too different from each other, the common area might end up being too small.
        // This function measures how much each slice restrains the common area and gets rid of the slices
        // that exceed the allowed constraint. To find the largest common area and exclude fewer views as
        // possible, it is probably best to use center_shifts first. The excluded views can still be masked
        // with the common area, but the mask will fall out of the bounds of these views.
        auto set_geometry(
                const Shape2<i64>& shape,
                const MetadataStack& metadata,
                f64 max_size_loss_percent = 0.15
        ) -> std::vector<i64> {
            // Save the shape, so we can then adjust the center when computing the masks.
            m_shape = shape;

            // Start with the perfect area, at tilt 0, encompassing the entire image.
            const auto hw = shape.vec().as<f64>();
            const Vec2<f64> center = MetadataSlice::center(shape[0], shape[1]).as<f64>();
            const auto initial_area = Vec4<f64>{ // left, right, bottom, top
                    -center[0], hw[0] - center[0],
                    -center[1], hw[1] - center[1],
            };

            // Enforce some limit on how much a view can limit the common area.
            std::vector<Vec4<f64>> areas;
            areas.reserve(metadata.size());

            // Compute the area of each view.
            for (const MetadataSlice& slice: metadata.slices()) {
                // Cosine stretch relative to 0deg elevation and 0deg tilt plane.
                const Vec3<f64> angles = noa::math::deg2rad(slice.angles);
                const Vec2<f64> cos_scale{1 / noa::math::cos(angles.filter(2, 1))}; // 1 = cos(0)

                // Shift and stretch the current slice area to compare it with the common area.
                const Double23 current_tilt_to_0deg = noa::geometry::affine2truncated(
                        noa::geometry::linear2affine(noa::geometry::rotate(angles[0])) *
                        noa::geometry::linear2affine(noa::geometry::scale(cos_scale)) *
                        noa::geometry::linear2affine(noa::geometry::rotate(-angles[0])) *
                        noa::geometry::translate(-slice.shifts));

                // Compute the area of the current view.
                // Because the high tilts have a bigger field-of-view (which is encoded by the stretching),
                // they can have large shifts perpendicular to the tilt axis and still contain the entire
                // common area. Large shifts along the tilt-axis or in the lower tilts are more likely
                // to limit the common area.
                areas.emplace_back((current_tilt_to_0deg * Vec3<f64>{0, initial_area[0], 1})[1],
                                   (current_tilt_to_0deg * Vec3<f64>{0, initial_area[1], 1})[1],
                                   (current_tilt_to_0deg * Vec3<f64>{initial_area[2], 0, 1})[0],
                                   (current_tilt_to_0deg * Vec3<f64>{initial_area[3], 0, 1})[0]);
            }

            // Get ready to remove problematic views.
            std::vector<i64> excluded_indexes;
            auto is_valid_index = [&excluded_indexes](i64 index) {
                return std::find(excluded_indexes.cbegin(), excluded_indexes.cend(), index) ==
                       excluded_indexes.cend();
            };

            // From the areas in the stack, compute the common area.
            auto get_common_area = [&, initial_area]() -> Vec4<f64> {
                Vec4<f64> common_area = initial_area;
                for (size_t i = 0; i < areas.size(); ++i) {
                    if (is_valid_index(static_cast<i64>(i))) {
                        common_area[0] = noa::math::max(common_area[0], areas[i][0]); // negative values
                        common_area[1] = noa::math::min(common_area[1], areas[i][1]); // positive values
                        common_area[2] = noa::math::max(common_area[2], areas[i][2]); // negative values
                        common_area[3] = noa::math::min(common_area[3], areas[i][3]); // positive values
                    }
                }
                return common_area;
            };

            // From an area (4 bounds [left, right, bottom, top], center at 0) to the HW diameter.
            auto get_area_diameter = [](const Vec4<f64>& area) -> Vec2<f64> {
                return {-area[0] + area[1],
                        -area[2] + area[3]};
            };

            // The goal here is to remove views that are restricting the common area (above a given threshold).
            // To do so, we check how much each view removes from the area and remove the view that removes the
            // most, until the common area becomes large enough to be above the given threshold.
            const Vec2<f64> minimum_tolerated_common_area_diameter = hw * (1 - max_size_loss_percent - 1e-6);
            Vec4<f64> common_area;
            Vec2<f64> common_area_diameter;
            while (true) {
                common_area = get_common_area();
                common_area_diameter = get_area_diameter(common_area);
                if (noa::all(minimum_tolerated_common_area_diameter <= common_area_diameter))
                    break;

                // Find the most problematic view (the view with the smallest area).
                i64 index_with_smallest_diameter{0};
                auto min_diameter = Vec2<f64>(noa::math::Limits<f64>::max());
                for (size_t i = 0; i < areas.size(); ++i) {
                    if (!is_valid_index(static_cast<i64>(i)))
                        continue;
                    auto diameter = get_area_diameter(areas[i]);
                    if (noa::any(min_diameter > diameter)) {
                        index_with_smallest_diameter = static_cast<i64>(i);
                        min_diameter = diameter;
                    }
                }
                excluded_indexes.push_back(index_with_smallest_diameter);
            }

            // Compute the center and radius of the common area.
            m_common_area_radius = common_area_diameter / 2;
            m_common_area_center = Vec2<f64>{
                    center + common_area.filter(0, 2) + m_common_area_radius};

            // At this point, the indexes are the positions of the slices in the metadata.
            // However, we want to return the positions of the slices in the stack, i.e. MetadataSlice.index.
            // Admittedly, this is a bit confusing, but it seems more robust to use MetadataSlice.index so that
            // it is independent of the order of the slices in the metadata and always matches the stack.
            for (size_t i = 0; i < excluded_indexes.size(); ++i)
                excluded_indexes[i] = metadata[i].index;

            return excluded_indexes;
        }

        void mask_views(
                const View<const f32>& input,
                const View<f32>& output,
                const MetadataStack& metadata,
                f64 smooth_edge_percent
        ) const {
            QN_CHECK(output.shape()[0] == metadata.ssize(), "The array and metadata don't match");
            const auto smooth_edge_size = static_cast<f32>(smooth_edge(smooth_edge_percent));
            noa::geometry::rectangle(
                    input, output,
                    /*center=*/ m_common_area_center.as<f32>(),
                    /*radius=*/ m_common_area_radius.as<f32>() - smooth_edge_size,
                    /*edge_size=*/ smooth_edge_size,
                    /*inv_matrix=*/ inv_transforms_(metadata, output.device()));
        }

        void mask_view(
                const View<const f32>& input,
                const View<f32>& output,
                const MetadataSlice& metadata,
                f64 smooth_edge_percent
        ) const {
            QN_CHECK(input.shape()[0] == 1, "The input must not be batched");
            const auto smooth_edge_size = static_cast<f32>(smooth_edge(smooth_edge_percent));
            noa::geometry::rectangle(
                    input, output,
                    /*center=*/ m_common_area_center.as<f32>(),
                    /*radius=*/ m_common_area_radius.as<f32>() - smooth_edge_size,
                    /*edge_size=*/ smooth_edge_size,
                    /*inv_matrix=*/ inv_transform_(metadata));
        }

        void compute(
                const View<f32>& output,
                const MetadataStack& metadata,
                f64 smooth_edge_percent
        ) const {
            QN_CHECK(output.shape()[0] == metadata.ssize(), "The array and metadata don't match");
            const auto smooth_edge_size = static_cast<f32>(smooth_edge(smooth_edge_percent));
            noa::geometry::rectangle(
                    {}, output,
                    /*center=*/ m_common_area_center.as<f32>(),
                    /*radius=*/ m_common_area_radius.as<f32>() - smooth_edge_size,
                    /*edge_size=*/ smooth_edge_size,
                    /*inv_matrix=*/ inv_transforms_(metadata, output.device()));
        }

        void compute(
                const View<f32>& output,
                const MetadataSlice& metadata,
                f64 smooth_edge_percent
        ) const {
            QN_CHECK(output.shape()[0] == 1, "The input must not be batched");
            const auto smooth_edge_size = static_cast<f32>(smooth_edge(smooth_edge_percent));
            noa::geometry::rectangle(
                    {}, output,
                    /*center=*/ adjust_center_(output.shape().filter(2,3)).as<f32>(),
                    /*radius=*/ radius().as<f32>() - smooth_edge_size,
                    /*edge_size=*/ smooth_edge_size,
                    /*inv_matrix=*/ inv_transform_(metadata));
        }

        [[nodiscard]] const Vec2<f64>& center() const noexcept { return m_common_area_center; }
        [[nodiscard]] const Vec2<f64>& radius() const noexcept { return m_common_area_radius; }

        [[nodiscard]] constexpr f64 smooth_edge(f64 smooth_edge_percent) const noexcept {
            // The smooth edge percent is relative to the common area size, not the original image size.
            auto smooth_edge_size = noa::math::max(m_common_area_radius * 2) * smooth_edge_percent;
            smooth_edge_size = std::max(10., smooth_edge_size);
            return smooth_edge_size;
        }


    private:
        [[nodiscard]] auto adjust_center_(const Shape2<i64>& new_shape) const noexcept -> Vec2<f64> {
            const auto [border_left, _] = noa::memory::shape2borders(
                    m_shape.push_front<2>({1, 1}), new_shape.push_front<2>({1, 1}));
            return m_common_area_center + border_left.filter(2, 3).as<f64>();
        }

        [[nodiscard]] auto inv_transform_(const MetadataSlice& metadata) const -> Float23 {
            QN_CHECK(noa::all(m_common_area_radius >= 0), "Common area geometry is not initialized");

            // Cosine shrink relative to the 0deg elevation/tilt.
            const Vec3<f64> angles = noa::math::deg2rad(metadata.angles);
            const Vec2<f64> cos_scale = noa::math::cos(angles.filter(2, 1));

            // Compute the transformation matrix to:
            //  - account for the view's shifts.
            //  - cosine scale to shrink the common area perpendicular to the tilt axis
            //    according to the view's tilt and rotation angle. Unfortunately,
            //    this will shrink the smooth edge of the ellipse, but this is fine
            //    if the taper is large enough, and it allows us to batch the operation.
            Float23 inv_common_area_to_view = noa::geometry::affine2truncated(
                    noa::geometry::translate(m_common_area_center) *
                    noa::geometry::linear2affine(noa::geometry::rotate(angles[0])) *
                    noa::geometry::linear2affine(noa::geometry::scale(1 / cos_scale)) *
                    noa::geometry::linear2affine(noa::geometry::rotate(-angles[0])) *
                    noa::geometry::translate(-m_common_area_center - metadata.shifts)).as<f32>();
            return inv_common_area_to_view;
        }

        [[nodiscard]] auto inv_transforms_(
                const MetadataStack& metadata,
                Device compute_device
        ) const -> Array<Float23> {
            auto inv_transforms = noa::memory::empty<Float23>(
                    metadata.ssize(), ArrayOption(compute_device, Allocator::MANAGED));

            const auto inv_transforms_span = inv_transforms.span();
            for (i64 i = 0; i < metadata.ssize(); ++i)
                inv_transforms_span[i] = inv_transform_(metadata[i]);

            return inv_transforms;
        }

    private:
        Shape2<i64> m_shape;
        Vec2<f64> m_common_area_center;
        Vec2<f64> m_common_area_radius{-1};
    };
}
