#pragma once

#include "quinoa/Types.h"

namespace qn {
    struct ThirdDegreePolynomial {
        f64 a, b, c, d;

        constexpr f64 operator()(f64 x) const noexcept {
            return a * x * x * x + b * x * x + c * x + d;
        }
    };

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

    // Computes the dimensions for Fourier cropping given a target spacing.
    //  - The target spacing is aimed, but might not be obtained exactly depending on the inputs.
    //  - If the current and target spacings are within 2% of each other, or if the current spacing is larger
    //    than the target one (triggering a Fourier padding), no cropping is done.
    //  - A minimum shape can be passed and is used to ensure a minimum output shape.
    //  - During Fourier cropping, the origin (0,0) is naturally preserved (i.e. the rescaling center is 0).
    //    This function also computes the shift that can be applied to the output to instead keep the center
    //    of the images aligned.
    template<size_t N, typename = std::enable_if_t<(N < 4)>>
    auto fourier_crop_dimensions(
            const Shape<i64, N>& current_shape,
            const Vec<f64, N>& current_spacing,
            const Vec<f64, N>& target_spacing,
            const Shape<i64, N>& target_min_shape = {},
            f64 tolerance = 0.02
    ) -> std::tuple<Shape<i64, N>, Vec<f64, N>, Vec<f64, N>> {

        // Find the frequency cutoff in the current spectrum that corresponds to the desired spacing.
        const auto frequency_cutoff = 0.5 * current_spacing / target_spacing;

        // Get Fourier cropped shape.
        const auto current_shape_f64 = current_shape.vec().template as<f64>();
        const auto target_min_shape_f64 = target_min_shape.vec().template as<f64>();
        auto new_shape_f64 = noa::math::round(frequency_cutoff * current_shape_f64 / 0.5);
        if (noa::all(target_min_shape > 0) &&
            noa::any(new_shape_f64 < target_min_shape_f64)) // clamp with minimum allowed shape
            new_shape_f64 = target_min_shape_f64;
        const auto new_shape = Shape<i64, N>(new_shape_f64.template as<i64>());

        // Compute the spacing of the Fourier cropped spectrum.
        const auto new_nyquist = 0.5 * new_shape_f64 / current_shape_f64;
        const auto new_spacing = current_spacing / (2 * new_nyquist);

        // If scaling factor is close to 1, then cancel Fourier cropping.
        if (noa::all((current_spacing / new_spacing) >= (1 - tolerance)))
            return {current_shape, current_spacing, {}};

        // In order to preserve the image (real-space) center, we may need to shift the Fourier cropped image.
        const auto current_center = MetadataSlice::center<f64>(current_shape);
        const auto new_center = MetadataSlice::center<f64>(new_shape);
        const auto current_center_rescaled = current_center * (current_spacing / new_spacing);
        const auto shift_to_add = current_center_rescaled - new_center;

        return {new_shape, new_spacing, shift_to_add};
    }
}
