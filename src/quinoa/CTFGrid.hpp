#pragma once

#include <noa/Core.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Types.hpp"

namespace qn::ctf {
    class Grid {
    public:
        Grid() = default;
        Grid(const Shape<i64, 2>& slice_shape, i64 patch_size, i64 patch_step) :
            m_slice_shape(slice_shape),
            m_patch_size(patch_size),
            m_patch_step(patch_step)
        {
            const std::vector origins_along_y = patch_grid_1d_2(m_slice_shape[0], m_patch_size, m_patch_step);
            const std::vector origins_along_x = patch_grid_1d_2(m_slice_shape[1], m_patch_size, m_patch_step);

            m_origins.reserve(origins_along_y.size() * origins_along_x.size());
            for (i64 y: origins_along_y)
                for (i64 x: origins_along_x)
                    m_origins.push_back({y, x});

            m_centers.reserve(m_origins.size());
            const auto patch_center = (patch_shape() / 2).vec;
            for (const auto& patch_origin: m_origins)
                m_centers.push_back((patch_origin + patch_center).as<f64>());
        }

    public:
        [[nodiscard]] auto slice_shape() const noexcept -> const Shape<i64, 2>& { return m_slice_shape; }
        [[nodiscard]] auto patch_size() const noexcept -> i64 { return m_patch_size; }
        [[nodiscard]] auto patch_shape() const noexcept -> Shape<i64, 2> { return Shape{patch_size(), patch_size()}; }
        [[nodiscard]] auto n_patches() const noexcept -> i64 { return static_cast<i64>(patches_centers().size()); }

        /// Returns the center of each patch within the slice/grid.
        /// These coordinates are 0 at the slice origin.
        [[nodiscard]] auto patches_centers() const noexcept -> SpanContiguous<const Vec2<f64>> {
            return {m_centers.data(), static_cast<i64>(m_centers.size())};
        }

        /// Converts the patch origins to the subregion origins, used for extraction.
        template<nt::sinteger I = i32, size_t N = 4>
        [[nodiscard]] auto compute_subregion_origins(
            i64 batch_index = 0,
            const Vec<i64, 2>& origin_offset = {}
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
                    static_assert(nt::always_false_t<I>);
            }
            return subregion_origins;
        }

        [[nodiscard]] auto patch_z_offset(
            const Vec<f64, 3>& slice_angles, // radians
            const Vec<f64, 2>& slice_spacing, // angstrom
            const Vec<f64, 2>& patch_center
        ) const -> f64 {
            const Mat<f64, 1, 3> to_patch_z = (
                ng::rotate_x(slice_angles[2]) *
                ng::rotate_y(slice_angles[1]) *
                ng::rotate_z(-slice_angles[0])
            ).filter_rows(0);

            const auto slice_center = (slice_shape() / 2).vec.as<f64>();
            const auto scale = slice_spacing * 1e-4; // pixels->micrometers
            const auto patch_center_um = (patch_center - slice_center) * scale;
            const f64 patch_center_z_um = (to_patch_z * patch_center_um.push_front(0))[0];
            return patch_center_z_um;
        }

    private:
        static auto patch_grid_1d_(i64 grid_size, i64 patch_size, i64 patch_step) -> std::vector<i64> {
            // Arange:
            const auto max = grid_size - patch_size - 1;
            std::vector<i64> patch_origin;
            for (i64 i{}; i < max; i += patch_step)
                patch_origin.push_back(i);

            if (patch_origin.empty())
                return patch_origin;

            // Center:
            const i64 end = patch_origin.back() + patch_size;
            const i64 offset = (grid_size - end) / 2;
            for (auto& origin: patch_origin)
                origin += offset;

            return patch_origin;
        }

        static auto patch_grid_1d_2(i64 grid_size, i64 patch_size, i64 patch_step) -> std::vector<i64> {
            // Arange:
            const auto n_patches = noa::divide_up(grid_size, patch_step);
            std::vector<i64> patch_origin;
            patch_origin.reserve(static_cast<size_t>(n_patches));
            for (i64 i{}; i < n_patches; ++i)
                patch_origin.push_back(i * patch_step);

            if (patch_origin.empty())
                return patch_origin;

            // Center:
            const i64 end = patch_origin.back() + patch_size;
            const i64 offset = (grid_size - end) / 2;
            for (auto& origin: patch_origin)
                origin += offset;

            return patch_origin;
        }

    private:
        Shape<i64, 2> m_slice_shape;
        i64 m_patch_size;
        i64 m_patch_step;
        std::vector<Vec<i64, 2>> m_origins;
        std::vector<Vec<f64, 2>> m_centers;
    };
}
