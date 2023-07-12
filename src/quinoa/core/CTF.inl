#pragma once

namespace qn {
    inline auto CTF::patch_transformed_coordinate(
            Shape2<i64> slice_shape,
            Vec2<f64> slice_shifts,
            Vec3<f64> slice_angles,
            Vec2<f64> slice_spacing,
            Vec2<f64> patch_center
    ) -> Vec3<f64> {
        slice_angles = noa::math::deg2rad(slice_angles);

        // By convention, the rotation angle is the additional rotation of the image.
        // Subtracting it aligns the tilt-axis to the y-axis.
        slice_angles[0] *= -1;

        // Switch coordinates from pixels to micrometers.
        const auto scale = slice_spacing * 1e-4;
        const auto slice_center_3d = (slice_shape.vec().as<f64>() * scale).push_front(0) / 2;
        const auto slice_shifts_3d = (slice_shifts * scale).push_front(0);

        // Place the slice into a 3d volume, with the center of the slice at the origin of the volume.
        namespace ng = noa::geometry;
        const Double44 image2microscope_matrix =
                ng::linear2affine(ng::euler2matrix(slice_angles, /*axes=*/ "zyx", /*intrinsic=*/ false)) *
                ng::translate(-slice_center_3d - slice_shifts_3d);

        const auto patch_center_3d = (patch_center * scale).push_front(0).push_back(1);
        const Vec3<f64> patch_center_transformed = (image2microscope_matrix * patch_center_3d).pop_back();
        return patch_center_transformed;
    }

    inline auto CTF::extract_patches_origins(
            const Shape2<i64>& slice_shape,
            const MetadataSlice& metadata,
            Vec2<f64> sampling_rate,
            Shape2<i64> patch_shape,
            Vec2<i64> patch_step,
            Vec2<f64> delta_z_range_nanometers
    ) -> std::vector<Vec4<i32>> {
        // Divide 2d grid in patches.
        const std::vector<Vec2<i64>> initial_patches_origins = patch_grid_2d(
                slice_shape, patch_shape, patch_step);

        std::vector<Vec4<i32>> output_patches_origins;
        const Vec2<f64>& slice_shifts = metadata.shifts;
        const Vec3<f64>& slice_angles = metadata.angles;

        for (auto patch_origin: initial_patches_origins) {
            // Get the 3d position of the patch.
            const auto patch_center = (patch_origin + patch_shape.vec() / 2).as<f64>();
            const auto patch_coordinates = patch_transformed_coordinate(
                    slice_shape, slice_shifts, slice_angles, sampling_rate, patch_center);

            // Filter based on its z position.
            // TODO Filter to remove patches at the corners?
            const auto z_nanometers = patch_coordinates[0] * 1e3; // micro -> nano
            if (z_nanometers < delta_z_range_nanometers[0] ||
                z_nanometers > delta_z_range_nanometers[1])
                continue;

            // Save the patch.
            output_patches_origins.emplace_back(0, 0, patch_origin[0], patch_origin[1]);
        }
        return output_patches_origins;
    };

    inline auto CTF::extract_patches_centers(
            const Shape2<i64>& slice_shape,
            Shape2<i64> patch_shape,
            Vec2<i64> patch_step
    ) -> std::vector<Vec2<f32>> {
        // Divide 2d grid in patches.
        const std::vector<Vec2<i64>> initial_patches_origins = patch_grid_2d(
                slice_shape, patch_shape, patch_step);

        std::vector<Vec2<f32>> output_patches_centers;
        for (auto patch_origin: initial_patches_origins) {
            const auto patch_center = (patch_origin + patch_shape.vec() / 2).as<f32>();
            output_patches_centers.push_back(patch_center);
        }
        return output_patches_centers;
    }

    inline void CTF::update_slice_patches_ctfs(
            Span<const Vec2<f32>> patches_centers,
            Span<CTFIsotropic64> patches_ctfs,
            const Shape2<i64>& slice_shape,
            const Vec2<f64>& slice_shifts,
            const Vec3<f64>& slice_angles,
            f64 slice_defocus,
            f64 additional_phase_shift
    ) {
        NOA_ASSERT(patches_centers.ssize() == patches_ctfs.ssize());

        for (i64 i = 0; i < patches_ctfs.ssize(); ++i) {
            CTFIsotropic64& patch_ctf = patches_ctfs[i];

            // Get the 3d position of the patch, in micrometers.
            const auto patch_coordinates = patch_transformed_coordinate(
                    slice_shape, slice_shifts, slice_angles,
                    Vec2<f64>{patch_ctf.pixel_size()},
                    patches_centers[i].as<f64>());

            // The defocus at the patch center is simply the slice defocus plus the z offset from the tilt axis.
            patch_ctf.set_defocus(patch_coordinates[0] + slice_defocus);
            patch_ctf.set_phase_shift(patch_ctf.phase_shift() + additional_phase_shift);
        }
    }
}
