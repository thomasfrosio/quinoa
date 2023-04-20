#pragma once

#include "quinoa/Types.h"

namespace qn {
    [[nodiscard]]
    auto subdivide_volume_in_cubes(
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
}
