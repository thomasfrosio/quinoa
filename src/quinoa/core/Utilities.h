#pragma once

#include "quinoa/Types.h"

namespace qn {
    [[nodiscard]]
    auto subdivideVolumeInCubes(
            dim3_t volume_shape,
            dim_t cube_size
    ) -> std::pair<std::vector<float3_t>, dim3_t> {
        const dim3_t cubes = (volume_shape + cube_size - 1) / cube_size; // divide up
        std::vector<float3_t> cubes_coords;
        cubes_coords.reserve(noa::math::prod(cubes));
        for (dim_t z = 0; z < cubes[0]; ++z)
            for (dim_t y = 0; y < cubes[1]; ++y)
                for (dim_t x = 0; x < cubes[2]; ++x)
                    cubes_coords.emplace_back(
                            dim3_t(z, y, x) * cube_size + cube_size / 2); // center of the cubes
        return {cubes_coords, cubes};
    }
}
