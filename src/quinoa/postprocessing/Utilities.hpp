#pragma once

#include <noa/Types.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    inline auto volume2image_matrices(
        const Metadata::Stack& metadata,
        bool correct_rotation,
        const Shape2& image_shape,
        const Shape3& volume_shape
    ) -> Array<Mat<f64, 2, 4>> {
        const auto image_center = (image_shape.vec / 2).as<f64>();
        const auto volume_center = (volume_shape.vec / 2).as<f64>();
        const auto volume2image_matrices = Array<Mat<f64, 2, 4>>(metadata.ssize());
        for (auto&& [image, volume2image]: noa::zip(metadata, volume2image_matrices.span_1d())) {
            const auto angles = noa::deg2rad(image.angles);
            const auto final_rotation = correct_rotation ? 0. : angles[0];
            volume2image = (
                nx::translate((image_center + image.shifts).push_front(0)) *
                nx::rotate_z<true>(angles[0]) *
                nx::rotate_y<true>(angles[1]) *
                nx::rotate_x<true>(angles[2]) *
                nx::rotate_z<true>(-final_rotation) *
                nx::translate(-volume_center)
            ).filter_rows(1, 2);
        }

        // Matrices relating 3d positions in the tomogram to 2d positions in the images.
        return volume2image_matrices;
    }

    // Project subvolume center back to image-space and
    // extract the twice-enlarged tile origin and residual shifts.
    inline auto extract_tile_large_window(
        f64 tile_large_center,
        const Vec<f64, 3>& subvolume_center_coordinates,
        const Mat<f64, 2, 4>& volume2image) {
        const auto tile_center_coordinate = volume2image * subvolume_center_coordinates.push_back(1);
        const auto tile_large_origin_coordinate = tile_center_coordinate - tile_large_center;
        const auto tile_large_origin_truncated = noa::floor(tile_large_origin_coordinate);
        const auto tile_large_origin = tile_large_origin_truncated.as<i32>();
        const auto tile_residual_shift = tile_large_origin_coordinate - tile_large_origin_truncated;

        // tilt_large_center + tile_residual_shift points to the center of
        // the padded tile after extraction at tile_large_origin.
        return Pair{tile_large_origin, tile_residual_shift};
    }

    struct ExtractPaddedTiles {
        SpanContiguous<const f32, 3> images;

        // For the Fourier reconstruction, this is tiles_large_padded.
        // For the real-space reconstruction, this is tiles_large.
        SpanContiguous<f32, 3> output_tiles;

        SpanContiguous<const Vec<i32, 2>> tile_large_origins;
        isize tile_large_center;
        f32 taper_radius;
        f32 taper_smoothness;

        NOA_HD void operator()(isize i, isize y, isize x) const {
            const auto tile_large_indices = Vec{y, x};
            const auto tile_coordinates = tile_large_indices - tile_large_center;
            const auto image_indices = tile_large_origins[i].as<isize>() + tile_large_indices;

            f32 value{};
            if (noa::is_inbound(images.shape().pop_front(), image_indices)) {
                // Compute the smooth taper.
                f32 taper{1};
                for (i32 j{}; j < 2; ++j) {
                    const auto tile_coordinate = static_cast<f32>(noa::abs(tile_coordinates[j]));
                    if (tile_coordinate >= taper_radius + taper_smoothness) {
                        taper = 0;
                    } else if (tile_coordinate >= taper_radius) {
                        constexpr auto PI = noa::Constant<f32>::PI;
                        const auto distance = (tile_coordinate - taper_radius) / taper_smoothness;
                        taper *= noa::cos(PI * distance) * 0.5f + 0.5f;
                    }
                }
                value = taper;

                // For the Fourier reconstruction, a significant part of the output tile is after the taper
                // because the oversampling is done by zero-padding the enlarged tiles. Therefore, only read
                // the input image if it's not zeroed-out by the taper and therefore in the padding.
                if (taper > 1e-6f)
                    value *= images(image_indices.push_front(i));
            }
            output_tiles(i, y, x) = value;
        }
    };

    struct ReconstructionThickness {
        f64 z_step_nm;
        isize z_step;
        f64 z_padding;
        f64 thickness_nm;
        isize thickness;
    };

    inline auto reconstruction_thickness(
        f64 spacing_nm,
        f64 defocus_step_nm,
        f64 sample_thickness_nm,
        f64 z_padding_percent = 0
    ) {
        // For simplicity, make the defocus resolution (in pixels) an odd integer multiple of the pixel size.
        auto z_step = static_cast<isize>(std::floor(defocus_step_nm / spacing_nm));
        z_step += noa::is_even(z_step);
        const auto z_step_nm = static_cast<f64>(z_step) * spacing_nm;

        // Get volume thickness and number of z-sections (of size z_step).
        // To guarantee that the volume center is at the center of a z-section,
        // make the volume thickness an odd multiple of z_step.
        const f64 sample_thickness = sample_thickness_nm / spacing_nm;
        const f64 z_padding = sample_thickness * z_padding_percent / spacing_nm;
        auto volume_thickness = static_cast<isize>(std::round(sample_thickness + z_padding));
        volume_thickness = noa::next_multiple_of(volume_thickness, z_step);
        if (noa::is_even(volume_thickness / z_step))
            volume_thickness += z_step;
        const auto volume_thickness_nm = static_cast<f64>(volume_thickness) * spacing_nm;

        return ReconstructionThickness{
            .z_step_nm = z_step_nm,
            .z_step = z_step,
            .z_padding = z_padding,
            .thickness_nm = volume_thickness_nm,
            .thickness = volume_thickness,
        };
    }
}
