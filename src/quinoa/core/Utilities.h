#pragma once

#include <noa/Math.hpp>
#include <noa/IO.hpp>
#include <noa/Math.hpp>
#include <noa/Geometry.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"

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

    [[nodiscard]]
    inline auto patch_grid_1d(
            i64 grid_size,
            i64 patch_size,
            i64 patch_step
    ) {
        // Arange:
        const auto max = grid_size - patch_size - 1; // FIXME
        std::vector<i64> patch_origin;
        for (i64 i = 0; i < max; i += patch_step)
            patch_origin.push_back(i);

        // Center:
        const i64 end = patch_origin.back() + patch_size;
        const i64 offset = (grid_size - end) / 2;
        for (auto& origin: patch_origin)
            origin += offset;

        return patch_origin;
    }

    [[nodiscard]]
    inline auto patch_grid_1d_count(
            i64 grid_size,
            i64 patch_size,
            i64 patch_step
    ) {
        // Arange:
        const auto max = grid_size - patch_size - 1; // FIXME
        i64 count{};
        for (i64 i = 0; i < max; i += patch_step)
            ++count;
        return count;
    }

    [[nodiscard]]
    inline auto patch_grid_2d(
            Shape2<i64> grid_shape,
            Shape2<i64> patch_shape,
            Vec2<i64> patch_step
    ) {
        const std::vector origins_along_y = patch_grid_1d(grid_shape[0], patch_shape[0], patch_step[0]);
        const std::vector origins_along_x = patch_grid_1d(grid_shape[1], patch_shape[1], patch_step[1]);

        std::vector<Vec2<i64>> origins;
        origins.reserve(origins_along_y.size() * origins_along_x.size());
        for (auto y: origins_along_y)
            for (auto x: origins_along_x)
                origins.emplace_back(y, x);

        return origins;
    }

    template<typename Int>
    [[nodiscard]] constexpr bool is_consecutive_range(const View<Vec4<Int>>& sequence, Int step = 1) noexcept {
        NOA_ASSERT(noa::indexing::is_contiguous_vector(sequence));
        const auto* range = sequence.get();
        for (i64 i = 1; i < sequence.ssize(); ++i)
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

    inline void add_global_rotation(MetadataStack& metadata, f64 global_rotation) {
        for (size_t i = 0; i < metadata.size(); ++i)
            metadata[i].angles[0] += global_rotation;
    }

    inline void add_global_angles(MetadataStack& metadata, Vec3<f64> global_angles) {
        for (size_t i = 0; i < metadata.size(); ++i)
            metadata[i].angles += global_angles;
    }

    inline void rescale_shifts(MetadataStack& metadata, Vec2<f64> scale) {
        for (auto& slice: metadata.slices())
            slice.shifts *= scale;
    }

    inline void rescale_shifts(MetadataStack& metadata, Vec2<f64> input_spacing, Vec2<f64> output_spacing) {
        rescale_shifts(metadata, input_spacing / output_spacing);
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

    struct FourierCropDimensions {
        Shape2<i64> padded_shape; // optional padding of the input before fft and Fourier-cropping
        Shape2<i64> cropped_shape; // shape for the Fourier cropping
        Vec2<f64> cropped_spacing; // spacing after Fourier cropping.
        Vec2<f64> rescale_shifts; // shifts to add to keep the centers aligned instead of the origin
    };

    // Computes the dimensions for Fourier cropping.
    //  * Two shapes are returned. The "cropped" shape, used to Fourier crop, as well as the "padded" shape,
    //    used to pad the input before dft, effectively changing the sampling of the input to align the target
    //    cutoff frequency to an integer position, allowing to precisely crop at the target spacing.
    //    Using a large "maximum_relative_error" (e.g. 0.2 or larger) turns off this feature and the "padded" shape
    //    will be the same as the current/input shape. Note that in this case, it is possible for the target spacing
    //    to become anisotropic.
    //
    //  * If the target spacing is smaller than or equal to the current spacing, it is clamped to the current
    //    spacing, effectively resulting in zero Fourier cropping and no rescaling.
    //
    //  * A "target_min_size" can be passed and is used to ensure a minimum output shape. This is used to prevent
    //    inputs with a very small spacings to be cropped to a very small shape.
    //
    //  * During Fourier cropping, the origin (0,0) is naturally preserved (i.e. the rescaling center is 0).
    //    This function also computes the shift that can be applied to the output to instead keep the center
    //    of the images aligned.
    inline auto fourier_crop_dimensions(
            Shape2<i64> current_shape,
            Vec2<f64> current_spacing,
            Vec2<f64> target_spacing,
            f64 maximum_relative_error = 5e-4,
            i64 target_min_size = 0
    ) -> FourierCropDimensions {

        // Don't allow Fourier padding. Note that if the current spacing is anisotropic, the target is set
        // to be isotropic, since it's often simpler to handle and the caller might expect the output spacing
        // to be isotropic too.
        if (noa::any(current_spacing > target_spacing))
            target_spacing = noa::math::max(current_spacing);

        // Clamp the target spacing to the maximum spacing corresponding to the minimum allowed size.
        auto compute_new_spacing = [](i64 size, f64 spacing, i64 new_size) {
            return spacing * static_cast<f64>(size) / static_cast<f64>(new_size);
        };
        const auto target_max_spacing = noa::math::max(Vec2<f64>{
            compute_new_spacing(current_shape[0], current_spacing[0], target_min_size),
            compute_new_spacing(current_shape[1], current_spacing[1], target_min_size)});
        if (noa::any(target_max_spacing < target_spacing))
            target_spacing = target_max_spacing;

        // Possibly zero-pad in real space to place the frequency cutoff at a particular index of the spectrum.
        // This is necessary to be able to precisely crop at a frequency cutoff, and offers a way to keep
        // the target spacing isotropic within a "maximum_relative_error".
        auto pad_to_align_cutoff = [maximum_relative_error]
                (i64 i_size, f64 i_spacing, f64 o_spacing) -> i64 {
            i64 MAXIMUM_SIZE = i_size + 256; // in most case, we stop way before that (~0 to 5)
            i64 best_size = i_size;
            f64 best_error = noa::math::Limits<f64>::max();
            while (i_size < MAXIMUM_SIZE) {
                const auto new_size = std::round(static_cast<f64>(i_size) * i_spacing / o_spacing);
                const auto new_spacing = i_spacing * static_cast<f64>(i_size) / new_size;
                const auto relative_error = std::abs(new_spacing - o_spacing) / o_spacing;

                if (relative_error < maximum_relative_error) { // we found a good enough solution
                    best_size = i_size;
                    break;
                } else if (best_error > relative_error) { // we found a better solution
                    best_error = relative_error;
                    best_size = i_size;
                }
                // Try again with a larger size. Since this padded size is likely to be fft'ed,
                // keep it even sized. We could go to the next fast fft size, but this often end
                // up padding large amounts, for little performance benefits vs memory usage.
                i_size += (1 - (i_size % 2)) + 1;
            }
            return best_size;
        };
        current_shape[0] = pad_to_align_cutoff(current_shape[0], current_spacing[0], target_spacing[0]);
        current_shape[1] = pad_to_align_cutoff(current_shape[1], current_spacing[1], target_spacing[1]);

        // Get Fourier cropped shape.
        const auto current_shape_f64 = current_shape.vec().template as<f64>();
        auto new_shape_f64 = current_shape_f64 * current_spacing / target_spacing;

        // Round to nearest integer (this is where the cropping happens).
        // We'll need to recompute the actual frequency after rounding, but of course,
        // this new frequency should be within a "maximum_relative_error" from the target spacing.
        new_shape_f64 = noa::math::round(new_shape_f64);
        const auto new_shape = Shape2<i64>(new_shape_f64.template as<i64>());
        const auto new_spacing = current_spacing * current_shape_f64 / new_shape_f64;

        // In order to preserve the image (real-space) center, we may need to shift the Fourier cropped image.
        // Here "current_shape" is the padded shape, and we do assume that the initial padding keeps the centers
        // of the input and the padded input aligned.
        const auto current_center = MetadataSlice::center<f64>(current_shape);
        const auto new_center = MetadataSlice::center<f64>(new_shape);
        const auto current_center_rescaled = current_center * (current_spacing / new_spacing);
        const auto shift_to_add = current_center_rescaled - new_center;

        return {current_shape, new_shape, new_spacing, shift_to_add};
    }

//    inline ThirdDegreePolynomial poly_fit_rotation(const MetadataStack& metadata) {
//        // Exclude the first view, assuming it's the global reference.
//        const auto rows = static_cast<i64>(metadata.size());
//
//        // Find x in Ax=b. Shapes: A(M.N) * x(N.1) = b(M.1)
//        const Array<f64> A({1, 1, rows, 4});
//        const Array<f64> b({1, 1, rows, 1});
//        const auto A_ = A.accessor_contiguous<f64, 2>();
//        const auto b_ = b.accessor_contiguous_1d();
//
//        // d + cx + bx^2 + ax^3 = 0
//        for (i64 row = 0; row < rows; ++row) {
//            const MetadataSlice& slice = metadata[row];
//            const auto rotation = static_cast<f64>(slice.angles[0]);
//            const auto tilt = static_cast<f64>(slice.angles[1]);
//            A_(row, 0) = 1;
//            A_(row, 1) = tilt;
//            A_(row, 2) = tilt * tilt;
//            A_(row, 3) = tilt * tilt * tilt;
//            b_(row) = rotation;
//        }
//
//        // Least-square solution using SVD.
//        std::array<f64, 4> svd{};
//        std::array<f64, 4> x{};
//        noa::math::lstsq(
//                A.view(),
//                b.view(),
//                View<f64>(x.data(), {1, 1, 4, 1}),
//                0.,
//                View<f64>(svd.data(), 4)
//        );
//
//        return ThirdDegreePolynomial{x[3], x[2], x[1], x[0]};
//    }

    template<typename Real>
    void save_vector_to_text(View<Real> x, const Path& filename) {
        NOA_CHECK(noa::indexing::is_contiguous_vector_batched(x), "");

        // Make sure it is dereferenceable and ready to read.
        Array<std::remove_const_t<Real>> x_cpu;
        if (!x.is_dereferenceable()) {
            x_cpu = x.to_cpu();
            x = x_cpu.view();
        }

        const i64 size = x.shape().pop_front().elements();
        const i64 batches = x.shape()[0];
        x.eval();

        std::string format;
        for (i64 i = 0; i < batches; ++i) {
            const auto span = noa::Span(x.subregion(i).data(), size);
            format += fmt::format("{}\n", fmt::join(span, ","));
        }
        noa::io::save_text(format, filename);
    }
}
