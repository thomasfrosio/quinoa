#pragma once

#include "quinoa/Types.h"

namespace qn {
    [[nodiscard]]
    inline auto subdivide_volume_in_cubes(
            const Shape3<i64>& volume_shape,
            i64 cube_size
    ) -> std::pair<std::vector<Vec3<f32>>, Vec3<i64>> {

        const Vec3<i64> cubes_count = (volume_shape + cube_size - 1).vec() / cube_size; // divide up
        std::vector<Vec3<f32>> cubes_coords;
        cubes_coords.reserve(static_cast<size_t>(noa::math::product(cubes_count)));

        for (i64 z = 0; z < cubes_count[0]; ++z)
            for (i64 y = 0; y < cubes_count[1]; ++y)
                for (i64 x = 0; x < cubes_count[2]; ++x)
                    cubes_coords.emplace_back(Vec3<i64>{z, y, x} * cube_size + cube_size / 2); // center of the cubes
        return {cubes_coords, cubes_count};
    }

    template<typename Int>
    [[nodiscard]] constexpr bool is_consecutive_range(const View<Vec4<Int>>& sequence, Int step = 1) noexcept {
        NOA_ASSERT(noa::indexing::is_contiguous_vector(sequence));
        const auto* range = sequence.get();
        for (i64 i = 1; i < sequence.size(); ++i)
            if (range[i - 1][0] + step != range[i][0])
                return false;
        return true;
    }

    // Shift the sample by a given amount.
    inline void add_global_shift(
            MetadataStack & metadata,
            Vec2<f64> global_shift
    ) {
        for (size_t i = 0; i < metadata.size(); ++i) {
            const Vec3<f64> angles = noa::math::deg2rad(metadata[i].angles);
            const Vec2<f64> elevation_tilt = angles.filter(2, 1);
            const Double22 shrink_matrix{
                    noa::geometry::rotate(angles[0]) *
                    noa::geometry::scale(noa::math::cos(elevation_tilt)) *
                    noa::geometry::rotate(-angles[0])
            };
            metadata[i].shifts += shrink_matrix * global_shift;
        }
    }

    // Move the average untilted-shift to 0.
    inline void center_shifts(MetadataStack& metadata) {
        Vec2<f64> mean{0};
        auto mean_scale = 1 / static_cast<f64>(metadata.size());
        for (size_t i = 0; i < metadata.size(); ++i) {
            const Vec3<f64> angles = noa::math::deg2rad(metadata[i].angles);
            const Vec2<f64> elevation_tilt = angles.filter(2, 1);
            const Double22 stretch_to_0deg{
                    noa::geometry::rotate(angles[0]) *
                    noa::geometry::scale(1 / noa::math::cos(elevation_tilt)) * // 1 = cos(0deg)
                    noa::geometry::rotate(-angles[0])
            };
            const Vec2<f64> shift_at_0deg = stretch_to_0deg * metadata[i].shifts;
            mean += shift_at_0deg * mean_scale;
        }
        add_global_shift(metadata, -mean);
    }
}
