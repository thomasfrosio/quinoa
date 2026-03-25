#pragma once

#include <noa/Core.hpp>

#include "quinoa/Types.hpp"

namespace qn::ctf {
    class Grid {
    public:
        Grid() = default;
        Grid(const Shape2& slice_shape, isize patch_size, isize patch_step) :
            m_slice_shape(slice_shape),
            m_patch_size(patch_size),
            m_patch_step(patch_step)
        {
            const std::vector origins_along_y = patch_grid_1d_(m_slice_shape[0], m_patch_size, m_patch_step);
            const std::vector origins_along_x = patch_grid_1d_(m_slice_shape[1], m_patch_size, m_patch_step);

            m_grid_shape = Shape2::from_values(origins_along_y.size(), origins_along_x.size());
            m_origins.reserve(origins_along_y.size() * origins_along_x.size());
            for (isize y: origins_along_y)
                for (isize x: origins_along_x)
                    m_origins.push_back({y, x});

            m_centers.reserve(m_origins.size());
            const auto patch_center = (patch_shape() / 2).vec;
            for (const auto& patch_origin: m_origins)
                m_centers.push_back((patch_origin + patch_center).as<f64>());

            // Compute the overflow for each axis.
            const auto image_shape = slice_shape.vec;
            const auto bottom_left_corner = m_origins.front();
            const auto top_right_corner = m_origins.back() + patch_shape().vec;
            left_overflow = abs(min(0., bottom_left_corner));
            right_overflow = max(top_right_corner, image_shape) - image_shape;
        }

    public:
        [[nodiscard]] auto shape() const noexcept -> const Shape2& { return m_grid_shape; }
        [[nodiscard]] auto slice_shape() const noexcept -> const Shape2& { return m_slice_shape; }
        [[nodiscard]] auto patch_size() const noexcept -> isize { return m_patch_size; }
        [[nodiscard]] auto patch_shape() const noexcept -> Shape2 { return Shape{patch_size(), patch_size()}; }
        [[nodiscard]] auto n_patches() const noexcept -> isize { return patches_centers().ssize(); }

        /// Returns the center of each patch within the slice/grid.
        /// These coordinates are 0 at the slice origin.
        [[nodiscard]] auto patches_centers() const noexcept -> SpanContiguous<const Vec<f64, 2>> {
            return {m_centers.data(), std::ssize(m_centers)};
        }

        /// Converts the patch origins to the subregion origins, used for extraction.
        template<nt::sinteger I = i32, size_t N = 4>
        [[nodiscard]] auto compute_subregion_origins(
            isize batch_index = 0,
            const Vec<isize, 2>& origin_offset = {}
        ) const -> Array<Vec<I, N>> {
            check(N == 4 or batch_index == 0);
            auto subregion_origins = Array<Vec<I, N>>(std::ssize(m_origins));
            for (auto&& [origin, subregion_origin]: noa::zip(m_origins, subregion_origins.span_1d_contiguous())) {
                auto iorigin = (origin_offset + origin).template as<I>();
                if constexpr (N == 4)
                    subregion_origin = Vec<I, N>::from_values(batch_index, 0, iorigin[0], iorigin[1]);
                else if constexpr (N == 2)
                    subregion_origin = iorigin;
                else
                    static_assert(nt::always_false<I>);
            }
            return subregion_origins;
        }

        [[nodiscard]] static auto patch_z_offset(
            const Vec<f64, 2>& slice_center,
            const Vec<f64, 3>& slice_angles, // radians
            const Vec<f64, 2>& slice_spacing, // angstrom
            const Vec<f64, 2>& patch_center
        ) -> f64 {
            const auto plane_rotation = (
                nx::rotate_z(slice_angles[0]) *
                nx::rotate_y(slice_angles[1]) *
                nx::rotate_x(slice_angles[2])
            );
            const auto& [c, b, a] = plane_rotation * Vec{1., 0., 0.};

            const auto scale = slice_spacing * 1e-4; // pixels->micrometers
            const auto patch_center_um = (patch_center - slice_center) * scale;
            const auto patch_center_z_um = -(a * patch_center_um[1] + b * patch_center_um[0]) / c;
            return patch_center_z_um;
        }

        [[nodiscard]] auto patch_z_offset(
            const Vec<f64, 3>& slice_angles, // radians
            const Vec<f64, 2>& slice_spacing, // angstrom
            const Vec<f64, 2>& patch_center
        ) const -> f64 {
            const auto slice_center = (slice_shape() / 2).vec.as<f64>();
            return patch_z_offset(slice_center, slice_angles, slice_spacing, patch_center);
        }

        [[nodiscard]] auto overflow() const {
            return Pair{left_overflow, right_overflow};
        }

    private:
        static auto patch_grid_1d_(isize slice_size, isize patch_size, isize patch_step) -> std::vector<isize> {
            // Arange:
            const auto n_patches = noa::divide_up(slice_size, patch_step);
            check(n_patches > 1, "Only one patch along the dimension");

            std::vector<isize> patch_origin;
            patch_origin.reserve(static_cast<size_t>(n_patches));
            for (isize i{}; i < n_patches; ++i)
                patch_origin.push_back(i * patch_step);

            if (patch_origin.empty())
                return patch_origin;

            // Center:
            const isize end = patch_origin.back() + patch_size;
            const isize offset = (slice_size - end) / 2;
            for (auto& origin: patch_origin)
                origin += offset;

            return patch_origin;
        }

    private:
        Shape2 m_slice_shape{};
        Shape2 m_grid_shape{};
        isize m_patch_size{};
        isize m_patch_step{};
        std::vector<Vec<isize, 2>> m_origins{};
        std::vector<Vec<f64, 2>> m_centers{};
        Vec<isize, 2> left_overflow{};
        Vec<isize, 2> right_overflow{};
    };
}
