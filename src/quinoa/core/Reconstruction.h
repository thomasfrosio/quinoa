#pragma once

#include <noa/FFT.h>
#include <noa/Geometry.h>
#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/Signal.h>

#include "quinoa/core/Utilities.h"
#include "quinoa/core/Metadata.h"

namespace qn::details {

}

namespace qn {
    struct TiledReconstructionParameters {
        dim_t volume_thickness{320};
        dim_t cube_size{64};
    };

    // Tomogram reconstruction using direct Fourier insertion of small cubes.
    // The full tomogram is subdivided into small cubes, which are reconstructed one by one.
    // Cubes are reconstructed by back-projecting their corresponding tiles in the tilt-series.
    // These tiles are twice as large as the final cubes to remove wrap-around effects from the
    // rotation in Fourier space. Zero padding can be used, but here we simply extract larger tiles.
    //
    [[nodiscard]]
    auto tiledReconstruction(
            const Array<float>& tilt_series,
            const MetadataStack& tilt_series_metadata,
            const TiledReconstructionParameters& parameters
    ) -> Array<float> {

        // TODO Load and filter stack here directly
//        Array<float> max({40, 1, 1, 1});
//        noa::math::max(tilt_series, max);
//        noa::math::ewise(tilt_series, max, tilt_series, noa::math::divide_t{});

        // The dimensions of the problem.
        const auto sub_size = parameters.cube_size;
        const auto sub_padded_size = sub_size * 2;
        const auto sub_padded_center = static_cast<double2_t>(dim2_t(sub_padded_size) / 2);
        const auto slice_shape = dim2_t{tilt_series.shape()[2], tilt_series.shape()[3]};
        const auto slice_center = static_cast<double2_t>(slice_shape / 2);
        const auto volume_shape = dim3_t{noa::math::nextMultipleOf(parameters.volume_thickness, sub_size),
                                         noa::math::nextMultipleOf(slice_shape[0], sub_size),
                                         noa::math::nextMultipleOf(slice_shape[1], sub_size)};
        const auto volume_center = static_cast<double3_t>(volume_shape / 2);

        // Prepare the tilt-series 3d rotation. This will be used later for the Fourier
        // insertion and will be copied to the compute device.
        const dim_t slice_count = tilt_series_metadata.size();
        auto slices_rotation_3d = Array<float33_t>(slice_count);
        for (size_t slice_index = 0; slice_index < slice_count; ++slice_index) {
            const MetadataSlice& slice = tilt_series_metadata[slice_index];
            auto slice_euler_angles_zyx = double3_t(noa::math::deg2rad(slice.angles));

            // The in-place rotation is the rotation of the image. To align the image onto
            // the Y axis, we need to "cancel" whatever angle is saved in the metadata.
            slice_euler_angles_zyx[0] *= -1;

            const auto rotation_3d = noa::geometry::euler2matrix(slice_euler_angles_zyx, "XYZ", false);
            slices_rotation_3d[slice_index] = static_cast<float33_t>(rotation_3d);
        }

        // Get the grid of cubes and their centers.
        const auto [cubes_coords, cubes_count] = qn::subdivideVolumeInCubes(volume_shape, sub_size);

        // For every cube, compute the set (one for each slice) of tiles origins and shifts.
        const auto tiles_origins = Array<int4_t>({cubes_coords.size(), 1, 1, slice_count});
        const auto tiles_shifts = Array<float2_t>({cubes_coords.size(), 1, 1, slice_count});
        for (size_t cube_index = 0; cube_index < cubes_coords.size(); ++cube_index) {
            const auto cube_coords = cubes_coords[cube_index];

            // Compute the projected cube coordinates, for every slice.
            // This goes from the center of the cube, to the center of the 2D tiles in the slice.
            for (size_t slice_index = 0; slice_index < slice_count; ++slice_index) {
                const MetadataSlice& slice = tilt_series_metadata[slice_index];

                // The metadata stores the "errors" of the slice, so here to point to
                // the "corrected" center of the slice, we need to add the slice shifts.
                const auto slice_correct_center = slice_center + double2_t(slice.shifts);
                const auto slice_correct_center_3d = double3_t{0, slice_correct_center[0], slice_correct_center[1]};

                // This relates 3D positions in the tomogram space to 2D positions in the tilt image.
                // The first row isn't used.
                const auto fwd_projection_matrix = double34_t{
                        noa::geometry::translate(slice_correct_center_3d) *
                        double44_t(slices_rotation_3d[slice_index]) *
                        noa::geometry::translate(-volume_center)
                };

                // Project the cube coordinates (the center of the cube) onto this slice.
                // TODO Add deformation model (i.e. 3D shifts) on the cube coords.
                const double4_t homogeneous_cube_coords = {cube_coords[0], cube_coords[1], cube_coords[2], 1};
                const double3_t projected_cube_coords = fwd_projection_matrix * homogeneous_cube_coords;
                const double2_t tile_coords = {projected_cube_coords[1], projected_cube_coords[2]};

                // Extract the tile origin and residual shifts for the extraction.
                const auto tile_origin_coords = tile_coords - sub_padded_center;
                const auto tile_origin_truncated = noa::math::floor(tile_origin_coords);
                const auto tile_origin = static_cast<int2_t>(tile_origin_truncated);
                const auto tile_residual_shift = tile_origin_coords - tile_origin_truncated;

                // These are the tile origin for memory::extract().
                tiles_origins(cube_index, 0, 0, slice_index) = {slice_index, 0, tile_origin[0], tile_origin[1]};

                // The tiles will be rotated during the Fourier insertion, so apply the
                // residual shifts and the shift for the rotation center at the same time.
                tiles_shifts(cube_index, 0, 0, slice_index) = float2_t(tile_residual_shift - sub_padded_center); // FIXME Add?
            }
        }

        // For the reconstruction itself, we can prepare some buffers.
        const auto options = noa::ArrayOption(noa::Device("gpu"), noa::Allocator::DEFAULT_ASYNC); // FIXME
        const auto tiles_shape = dim4_t{slice_count, 1, sub_padded_size, sub_padded_size};
        const auto cube_padded_shape = dim4_t{1, sub_padded_size, sub_padded_size, sub_padded_size};
        const auto [tiles, tiles_fft] = noa::fft::empty<float>(tiles_shape, options);
        const auto [cube_padded, cube_padded_fft] = noa::fft::zeros<float>(cube_padded_shape, options);
        const auto cube_padded_weight_fft = noa::memory::zeros<float>(cube_padded_shape.fft(), options);

        // We only use the central part of the cube, so we need to extract a view of this center.
        const auto get_view_of_cube = [&](const Array<float>& cube_padded_) {
            const auto central_slice = noa::indexing::slice_t{sub_size / 2, sub_size / 2 + sub_size};
            return cube_padded_.subregion(
                    noa::indexing::full_extent_t{},
                    central_slice,
                    central_slice,
                    central_slice
            );
        };

        // To limit the number of synchronization and copies between GPU<->CPU, we reconstruct
        // and accumulate a row of cubes. Once this row is reconstructed on the compute device,
        // it is copied back to the CPU and inserted in the final full reconstruction.
        const auto cubes_row_shape = dim4_t{1, sub_size, sub_size, sub_size * cubes_count[2]};
        const auto row_of_cubes = noa::memory::empty<float>(cubes_row_shape, options); // TODO pinned?

        // This returns a view of the ith cube within the row.
        auto get_view_of_cube_from_row = [&](size_t i) {
            const size_t offset = sub_size * (i % cubes_count[2]);
            return row_of_cubes.subregion(
                    noa::indexing::ellipsis_t{},
                    noa::indexing::slice_t{offset, offset + sub_size}
            );
        };

        //
        auto reconstruction = noa::memory::empty<float>({1, volume_shape[0], volume_shape[1], volume_shape[2]});
        const auto get_view_of_row_from_reconstruction = [&](size_t z, size_t y) {
            return reconstruction.subregion(
                    0,
                    noa::indexing::slice_t{sub_size * z, sub_size * z + sub_size},
                    noa::indexing::slice_t{sub_size * y, sub_size * y + sub_size},
                    noa::indexing::full_extent_t{}
            );
        };

        // Prepare the buffer for extraction of a single cube.
        auto tiles_origin = Array<int4_t>(slice_count, options);
        auto tiles_shift = Array<float2_t>(slice_count, options);

        for (size_t i = 0; i < slices_rotation_3d.size(); ++i)
            slices_rotation_3d[i] = slices_rotation_3d[i].transpose();

        if (!options.device().cpu())
            slices_rotation_3d = slices_rotation_3d.to(options.device());

        for (size_t z = 0; z < cubes_count[0]; ++z) {
            for (size_t y = 0; y < cubes_count[1]; ++y) {
                qn::Logger::trace("z={}, y={}\n", z, y);

                // Compute a row of cubes. This entire loop shouldn't require any GPU synchronization.
                for (size_t x = 0; x < cubes_count[2]; ++x) {
                    const auto cube_index = (z * cubes_count[1] + y) * cubes_count[2] + x;

                    // Extract the tiles.
                    tiles_origins.subregion(cube_index).to(tiles_origin);
//                    const auto t = tiles_origins.subregion(cube_index);
                    noa::memory::extract(tilt_series, tiles, tiles_origin);
//                    noa::signal::rectangle(tiles, tiles, float2_t{64}, float2_t{32}, 0);
//                    noa::io::save(tiles, string::format(
//                            "/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/tiles_{}_{}_{}.mrc", z,y,x));

                    // Compute the FFT of the tiles and phase-shift with residuals and tile center.
                    noa::fft::r2c(tiles, tiles_fft);
//                    noa::fft::remap(fft::H2HC, tiles_fft, tiles_fft, tiles_shape);

                    // Apply residual shifts from the extraction and the tile rotation center.
                    tiles_shifts.subregion(cube_index).to(tiles_shift);
                    noa::signal::fft::shift2D<noa::fft::H2H>(
                            tiles_fft, tiles_fft, tiles.shape(), tiles_shift);

                    // Backward projection.
                    noa::geometry::fft::insert3D<noa::fft::H2H>(
                            tiles_fft, tiles.shape(),
                            cube_padded_fft, cube_padded_shape,
                            float22_t{}, slices_rotation_3d, 0.5f);
                    noa::geometry::fft::insert3D<noa::fft::H2H>(
                            1.f, tiles.shape(), // TODO add exposure filter and CTF
                            cube_padded_weight_fft, cube_padded_shape,
                            float22_t{}, slices_rotation_3d, 0.5f);

//                    Array<float> tmp(cube_padded_fft.shape(), options);
//                    math::ewise(cube_padded_fft, tmp, math::abs_one_log_t{});
//                    fft::remap(fft::H2HC, tmp, tmp, cube_padded_shape);
//                    noa::io::save(tmp, string::format(
//                            "/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/weight_{}_{}_{}.mrc", z,y,x));

                    // Correct for multiplicity and weights.
                    noa::signal::fft::shift3D<noa::fft::H2H>(
                            cube_padded_fft, cube_padded_fft,
                            cube_padded_shape, static_cast<float3_t>(sub_padded_center[0]));
                    noa::math::ewise(cube_padded_fft, cube_padded_weight_fft, 1e-3f,
                                     cube_padded_fft, noa::math::divide_epsilon_t{});

                    // Reconstruction.
                    noa::fft::c2r(cube_padded_fft, cube_padded);
//                    noa::geometry::fft::griddingCorrection(cube_padded, cube_padded, /*post_correction=*/ true);
//                    noa::io::save(cube_padded, "/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/cube.mrc");
//                    noa::io::save(cube_padded_weight_fft, "/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/weight.mrc");

                    // Insert the central region of the cube into the row buffer.
                    const auto cube_cropped = get_view_of_cube(cube_padded);
                    const auto cube_destination = get_view_of_cube_from_row(cube_index);
                    noa::memory::copy(cube_cropped, cube_destination);

//                    noa::io::save(cube_padded, string::format(
//                            "/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/cube_padded_{}_{}_{}.mrc", z,y,x));
//                    noa::io::save(cube_cropped.copy(), string::format(
//                            "/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/cube_cropped_{}_{}_{}.mrc", z,y,x));

//                    noa::io::save(cube_cropped.copy(), "/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/cube_cropped.mrc");
//                    noa::io::save(row_of_cubes,
//                                  string::format("/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/buffer_{}{}.mrc",
//                                                 z,y));

                    // Reset the 3D Fourier volumes to 0. // TODO Move this at the beginning
                    noa::memory::fill(cube_padded_fft, cfloat_t{0});
                    noa::memory::fill(cube_padded_weight_fft, float{0});
                }

                // The row of cubes is done, we need to copy back to the CPU and insert it into the volume.
                const auto output_row_of_cubes = get_view_of_row_from_reconstruction(z, y);
                noa::memory::copy(row_of_cubes, output_row_of_cubes); // implicitly synchronizes the GPU stream
//                noa::io::save(row_of_cubes,
//                              string::format("/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/row_{}{}.mrc",
//                                             z,y));
            }

            // TODO Once a Z is done, save to the file?
        }

        return reconstruction;
    }
}
