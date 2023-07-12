#pragma once

#include <noa/FFT.h>
#include <noa/Geometry.h>
#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/Signal.h>

#include "quinoa/core/Utilities.h"
#include "quinoa/core/Metadata.h"

namespace qn::details {
    class TiledReconstruction {
    public:
        TiledReconstruction(const dim4_t& tilt_series_shape,
                            const MetadataStack& tilt_series_metadata,
                            Device compute_device,
                            dim_t volume_depth,
                            dim_t cube_size) {
            m_debug_directory = "/home/thomas/Projects/quinoa/tests/tiles";

            // Sizes.
            m_sub_size = cube_size;
            m_sub_padded_size = cube_size * 2;
            m_sub_padded_center = static_cast<double2_t>(dim2_t(m_sub_padded_size) / 2);
            m_slice_shape = dim2_t{tilt_series_shape[2], tilt_series_shape[3]};
            m_slice_center = static_cast<double2_t>(m_slice_shape / 2);
            m_volume_shape = dim3_t{volume_depth, m_slice_shape[0], m_slice_shape[1]};
            m_volume_center = static_cast<double3_t>(m_volume_shape / 2);

            const auto options = noa::ArrayOption(compute_device, noa::Allocator::DEFAULT_ASYNC);
            const auto tiles_shape = dim4_t{tilt_series_metadata.size(), 1, m_sub_padded_size, m_sub_padded_size};
            const auto cube_padded_shape = dim4_t{1, m_sub_padded_size, m_sub_padded_size, m_sub_padded_size};

            // Get the grid of cubes and their centers, and project them to get the tiles centers.
            std::vector<float3_t> cubes_coordinates;
            std::tie(cubes_coordinates, m_cubes_count) = qn::subdivideVolumeInCubes(m_volume_shape, m_sub_size);
            std::tie(m_tiles_origins, m_tiles_shifts_rotation_center, m_fwd_insertion_matrices) =
                    projectCubeCoordinates_(cubes_coordinates, tilt_series_metadata);

            m_tiles_origin = Array<int4_t>(tiles_shape[0], options);
            m_tiles_shift_rotation_center = Array<float2_t>(tiles_shape[0], options);
            if (compute_device.gpu())
                m_fwd_insertion_matrices = m_fwd_insertion_matrices.to(options);

            // For the reconstruction itself, we can prepare some buffers.
            std::tie(m_tiles, m_tiles_fft) = noa::fft::empty<float>(tiles_shape, options);
            std::tie(m_cube_padded, m_cube_padded_fft) = noa::fft::empty<float>(cube_padded_shape, options);
            m_cube_padded_weight_fft = noa::memory::empty<float>(cube_padded_shape.rfft(), options);

            if (compute_device.gpu()) {
                // To limit the number of synchronization and copies between GPU<->CPU, we reconstruct
                // and accumulate a row of cubes. Once this row is reconstructed on the compute device,
                // it is copied back to the CPU and inserted in the final full reconstruction.
                const auto cubes_row_shape = dim4_t{1, m_sub_size, m_sub_size, m_volume_shape[2]};
                m_row_of_cubes = noa::memory::empty<float>(cubes_row_shape, options); // TODO pinned?
            }
        }

        auto reconstruct(const Array<float>& tilt_series) -> noa::Array<float> {
            // Output reconstruction.
            // TODO Once a Z is done, save to the file?
            auto reconstruction = noa::memory::empty<float>(
                    {1, m_volume_shape[0], m_volume_shape[1], m_volume_shape[2]});

            const bool use_row_of_cubes_buffer = !m_row_of_cubes.empty();

            // For every cube:
            for (size_t z = 0; z < m_cubes_count[0]; ++z) {
                for (size_t y = 0; y < m_cubes_count[1]; ++y) {
                    qn::Logger::trace("Reconstructing cube: z={}, y={}", z, y);

                    // Compute a row of cubes. This entire loop shouldn't require any GPU synchronization.
                    for (size_t x = 0; x < m_cubes_count[2]; ++x) {
                        const auto cube_index = (z * m_cubes_count[1] + y) * m_cubes_count[2] + x;

                        // Reset the 3D Fourier volumes to 0.
                        noa::memory::fill(m_cube_padded_fft, cfloat_t{0});
                        noa::memory::fill(m_cube_padded_weight_fft, float{0});

                        // Extract the tiles.
                        m_tiles_origins.subregion(cube_index).to(m_tiles_origin);
                        noa::memory::extract(tilt_series, m_tiles, m_tiles_origin);
                        if (!m_debug_directory.empty()) {
                            constexpr std::string_view filename = "tiles_z{:0>2}_y{:0>2}_x{:0>2}.mrc";
                            noa::io::save(m_tiles, m_debug_directory / string::format(filename, z, y, x));
                        }

                        // Compute the FFT of the tiles and apply residual shifts from the
                        // extraction and the tile rotation center.
                        noa::fft::r2c(m_tiles, m_tiles_fft);
                        m_tiles_shifts_rotation_center.subregion(cube_index).to(m_tiles_shift_rotation_center);
                        noa::signal::fft::shift2D<noa::fft::H2H>(
                                m_tiles_fft, m_tiles_fft, m_tiles.shape(), m_tiles_shift_rotation_center);
                        noa::fft::remap(noa::fft::H2HC, m_tiles_fft, m_tiles_fft, m_tiles.shape());

                        // Backward projection, aka direct Fourier insertion.
                        noa::geometry::fft::insert3D<noa::fft::HC2H>(
                                m_tiles_fft, m_tiles.shape(),
                                m_cube_padded_fft, m_cube_padded.shape(),
                                float22_t{}, m_fwd_insertion_matrices, 0.004f, 0.5f);
                        noa::geometry::fft::insert3D<noa::fft::HC2H>(
                                1.f, m_tiles.shape(), // TODO add exposure filter and CTF
                                m_cube_padded_weight_fft, m_cube_padded.shape(),
                                float22_t{}, m_fwd_insertion_matrices, 0.004f, 0.5f);

                        // Correct for multiplicity and weights.
                        noa::signal::fft::shift3D<noa::fft::H2H>(
                                m_cube_padded_fft, m_cube_padded_fft,
                                m_cube_padded.shape(), static_cast<float3_t>(m_sub_padded_center[0]));
                        noa::math::ewise(m_cube_padded_fft, m_cube_padded_weight_fft, 1e-3f,
                                         m_cube_padded_fft, noa::math::divide_epsilon_t{});

                        // Reconstruction.
                        noa::fft::c2r(m_cube_padded_fft, m_cube_padded);
                        noa::geometry::fft::griddingCorrection(m_cube_padded, m_cube_padded, /*post_correction=*/ true);

                        // The cube is reconstructed, so save it.
                        if (use_row_of_cubes_buffer)
                            copyCentralCubeIntoRowOfCubes_(x);
                        else
                            copyCentralCubeIntoReconstruction(reconstruction, z, y, x);
                    }

                    // The row of cubes is done, we need to copy it back and insert it into the output.
                    if (use_row_of_cubes_buffer)
                        copyRowOfCubesIntoReconstruction_(reconstruction, z, y); // GPU -> CPU
                }
            }
            return reconstruction;
        }

    private:
        [[nodiscard]]
        auto projectCubeCoordinates_(const std::vector<float3_t>& cubes_coords,
                                     const MetadataStack& tilt_series_metadata)
        -> std::tuple<noa::Array<int4_t>, noa::Array<float2_t>, noa::Array<float33_t>> {

            const auto slice_count = tilt_series_metadata.size();
            auto fwd_insertion_matrices = Array<float33_t>(slice_count);
            auto fwd_projection_matrices = std::vector<double34_t>(slice_count);

            // Prepare the projection and insertion matrices.
            for (size_t slice_index = 0; slice_index < slice_count; ++slice_index) {
                const MetadataSlice& slice = tilt_series_metadata[slice_index];

                // The in-place rotation is the rotation of the image. To align the image onto
                // the Y axis, we need to "cancel" whatever angle is saved in the metadata.
                auto slice_euler_angles_zyx = double3_t(noa::math::deg2rad(slice.angles));
                slice_euler_angles_zyx[0] *= -1;

                // The metadata stores the "errors" of the slice, so here to point to
                // the "corrected" center of the slice, we need to add the slice shifts.
                const auto slice_correct_center = m_slice_center + double2_t(slice.shifts);
                const auto slice_correct_center_3d = double3_t{0, slice_correct_center[0], slice_correct_center[1]};

                // This relates 3D positions in the tomogram space to 2D positions in the tilt image.
                // These matrices are operating on the tomogram in real space. The tilt geometry is extrinsic XYZ
                // FIXME The first row isn't used.
                const auto fwd_projection_matrix = double34_t{
                        noa::geometry::translate(slice_correct_center_3d) *
                        double44_t(noa::geometry::rotateZ(noa::math::deg2rad(slice.angles[0]))) *
                        double44_t(noa::geometry::rotateY(noa::math::deg2rad(slice.angles[1]))) *
                        double44_t(noa::geometry::rotateX(noa::math::deg2rad(slice.angles[2]))) *
                        noa::geometry::translate(-m_volume_center)

                };

                // The rotation to apply to the slice is the inverse of the
                // extrinsic rotation of our tilt geometry.
                fwd_projection_matrices[slice_index] = fwd_projection_matrix;
                fwd_insertion_matrices[slice_index] = static_cast<float33_t>(fwd_projection_matrix);
//                        noa::math::inverse(double33_t(fwd_projection_matrix)));
            }

            // For every cube, compute the set (one for each slice) of tiles origins and shifts.
            auto tiles_origins = Array<int4_t>({cubes_coords.size(), 1, 1, slice_count});
            auto tiles_shifts = Array<float2_t>({cubes_coords.size(), 1, 1, slice_count});
            for (size_t cube_index = 0; cube_index < cubes_coords.size(); ++cube_index) {
                const auto cube_coords = cubes_coords[cube_index];

                // Compute the projected cube coordinates, for every slice.
                // This goes from the center of the cube, to the center of the 2D tiles in the slice.
                for (size_t slice_index = 0; slice_index < slice_count; ++slice_index) {
                    // TODO Add deformation model (i.e. 3D shifts) on the cube coords.
                    const auto fwd_projection_matrix = fwd_projection_matrices[slice_index];
                    const double4_t homogeneous_cube_coords = {cube_coords[0], cube_coords[1], cube_coords[2], 1};
                    const double3_t projected_cube_coords = fwd_projection_matrix * homogeneous_cube_coords;
                    const double2_t tile_coords = {projected_cube_coords[1], projected_cube_coords[2]};

                    // Extract the tile origin and residual shifts for the extraction.
                    const auto tile_origin_coords = tile_coords - m_sub_padded_center;
                    const auto tile_origin_truncated = noa::math::floor(tile_origin_coords);
                    const auto tile_origin = static_cast<int2_t>(tile_origin_truncated);
                    const auto tile_residual_shift = tile_origin_coords - tile_origin_truncated;

                    // These are the tile origin for memory::extract().
                    tiles_origins(cube_index, 0, 0, slice_index) = {slice_index, 0, tile_origin[0], tile_origin[1]};

                    // The tiles will be rotated during the Fourier insertion, so apply the
                    // residual shifts and the shift for the rotation center at the same time.
                    // Here we subtract to bring the rotation center at the origin.
                    const auto tile_rotation_center = float2_t(tile_residual_shift + m_sub_padded_center);
                    tiles_shifts(cube_index, 0, 0, slice_index) = -tile_rotation_center;
                }
            }
            return {tiles_origins, tiles_shifts, fwd_insertion_matrices};
        }

        // Extract a view of the central cube and copy it into the buffer.
        void copyCentralCubeIntoRowOfCubes_(size_t x) {
            const size_t size_cube_x = std::min(m_sub_size, m_row_of_cubes.shape()[3] - m_sub_size * x);

            const auto center = m_sub_size / 2;
            const auto input_central_cube = m_cube_padded.subregion(
                    0,
                    noa::indexing::Slice{center, center + m_sub_size},
                    noa::indexing::Slice{center, center + m_sub_size},
                    noa::indexing::Slice{center, center + size_cube_x});

            const auto output_central_cube = m_row_of_cubes.subregion(
                    noa::indexing::Ellipsis{},
                    noa::indexing::Slice{m_sub_size * x, m_sub_size * x + size_cube_x});

            noa::memory::copy(input_central_cube, output_central_cube);
        }

        void copyCentralCubeIntoReconstruction(const Array<float>& reconstruction,
                                               size_t z, size_t y, size_t x) {
            const size_t size_cube_y = std::min(m_sub_size, reconstruction.shape()[2] - m_sub_size * y);
            const size_t size_cube_x = std::min(m_sub_size, reconstruction.shape()[3] - m_sub_size * x);

            const auto center = m_sub_size / 2;
            const auto input_central_cube = m_cube_padded.subregion(
                    noa::indexing::FullExtent{},
                    noa::indexing::Slice{center, center + m_sub_size},
                    noa::indexing::Slice{center, center + size_cube_y},
                    noa::indexing::Slice{center, center + size_cube_x});

            // Here we could let the subregion method to clamp the size in y/x,
            // but since we already have computed the truncated size in y/x, use it instead.
            const auto output_central_cube = reconstruction.subregion(
                    0,
                    noa::indexing::Slice{m_sub_size * z, m_sub_size * z + m_sub_size},
                    noa::indexing::Slice{m_sub_size * y, m_sub_size * y + size_cube_y},
                    noa::indexing::Slice{m_sub_size * x, m_sub_size * x + size_cube_x}
            );

            noa::memory::copy(input_central_cube, output_central_cube);
        }

        void copyRowOfCubesIntoReconstruction_(const Array<float>& reconstruction, size_t z, size_t y) const {
            const size_t size_cube_y = std::min(m_sub_size, reconstruction.shape()[2] - m_sub_size * y);
            const auto valid_row_of_cubes = m_row_of_cubes.subregion(
                    0,
                    noa::indexing::FullExtent{},
                    noa::indexing::Slice{0, size_cube_y},
                    noa::indexing::FullExtent{}
            );
            const auto output_row = reconstruction.subregion(
                    0,
                    noa::indexing::Slice{m_sub_size * z, m_sub_size * z + m_sub_size},
                    noa::indexing::Slice{m_sub_size * y, m_sub_size * y + size_cube_y},
                    noa::indexing::FullExtent{}
            );
            noa::memory::copy(valid_row_of_cubes, output_row);
        };

    private:
        // Metadata
        Array<int4_t> m_tiles_origins;
        Array<int4_t> m_tiles_origin;
        Array<float2_t> m_tiles_shifts_rotation_center;
        Array<float2_t> m_tiles_shift_rotation_center;
        Array<float33_t> m_fwd_insertion_matrices;

        noa::Array<float> m_tiles;
        noa::Array<cfloat_t> m_tiles_fft;
        noa::Array<float> m_cube_padded;
        noa::Array<cfloat_t> m_cube_padded_fft;
        noa::Array<float> m_cube_padded_weight_fft;
        Array<float> m_row_of_cubes;

        dim_t m_sub_size;
        dim_t m_sub_padded_size;
        dim2_t m_slice_shape;
        dim3_t m_volume_shape;
        dim3_t m_cubes_count;
        double2_t m_sub_padded_center;
        double2_t m_slice_center;
        double3_t m_volume_center;

        path_t m_debug_directory;
    };
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
    [[nodiscard]]
    auto tiledReconstruction(
            const path_t& tilt_series_filename,
            MetadataStack& tilt_series_metadata,
            const LoadStackParameters& loading_parameters,
            const TiledReconstructionParameters& parameters
    ) -> noa::Array<float> {

        // Ensure the order in the stack is the same as the order in the metadata.
        tilt_series_metadata.sort("index");

        const auto [tilt_series, reconstruction_pixel_size, original_pixel_size] =
                loadStack(tilt_series_filename, tilt_series_metadata, loading_parameters);

        // Scale the metadata shifts to the reconstruction resolution.
        const auto pre_scale = float2_t(original_pixel_size / reconstruction_pixel_size);
        for (auto& slice: tilt_series_metadata.slices())
            slice.shifts *= pre_scale;

        auto reconstructor = details::TiledReconstruction(
                tilt_series.shape(),
                tilt_series_metadata,
                loading_parameters.compute_device,
                parameters.volume_thickness,
                parameters.cube_size);

        auto reconstruction = reconstructor.reconstruct(tilt_series);
        return reconstruction;
    }
}
