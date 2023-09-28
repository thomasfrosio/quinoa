#include "quinoa/core/Reconstruction.hpp"

namespace {
    using namespace ::qn;

    class VirtualVolume {
    private:
        noa::Array<Vec4<i32>> m_tile_origins;
        noa::Array<Vec2<f32>> m_tile_shifts;
        noa::Array<Quaternion<f32>> m_insertion_rotation;

        noa::Array<c32> m_tiles_rfft;
        noa::Array<c32> m_cubes_padded_rfft;
        noa::Array<f32> m_cubes_padded_weight_rfft;
        noa::Array<f32> m_row_of_cubes;

        i64 m_cube_size;
        i64 m_cube_padded_size;
        Vec3<i64> m_cubes_count;
        Shape2<i64> m_slice_shape;
        Shape3<i64> m_volume_shape;
        bool m_use_rasterization;

    public:
        VirtualVolume(
                const Shape2<i64>& slice_shape,
                const MetadataStack& metadata,
                Device compute_device,
                f64 sample_thickness_pix,
                i64 cube_size,
                bool use_rasterization
        ) :
                m_use_rasterization(use_rasterization)
        {
            const auto options = ArrayOption(compute_device, noa::Allocator::DEFAULT_ASYNC);
            const auto volume_thickness = static_cast<i64>(noa::math::round(sample_thickness_pix * 1.75));

            // Dimensions.
            m_cube_size = noa::fft::next_fast_size(cube_size);
            m_cube_padded_size = noa::fft::next_fast_size(m_cube_size * 4); // FIXME test smaller
            m_slice_shape = slice_shape;
            m_volume_shape = Shape3<i64>{volume_thickness, m_slice_shape[0], m_slice_shape[1]};

            // Get the grid of cubes and their centers, and project them to get the tile centers.
            std::vector<Vec3<f64>> cubes_coordinates;
            std::tie(cubes_coordinates, m_cubes_count) = subdivide_volume_in_cubes_(m_volume_shape, m_cube_size);
            std::tie(m_tile_origins, m_tile_shifts, m_insertion_rotation) = project_cube_coordinates_(
                    cubes_coordinates, metadata, m_use_rasterization,
                    volume_center_(), slice_center_(), tile_center_());

            if (compute_device.is_gpu()) {
                m_tile_origins = m_tile_origins.to(options);
                m_tile_shifts = m_tile_shifts.to(options);
                m_insertion_rotation = m_insertion_rotation.to(options);
            }

            // For the reconstruction itself, we can prepare some buffers.
            const auto tiles_shape = Shape4<i64>{metadata.ssize(), 1, m_cube_padded_size, m_cube_padded_size};
            const auto cube_padded_shape = Shape4<i64>{1, m_cube_padded_size, m_cube_padded_size, m_cube_padded_size};
            m_tiles_rfft = noa::memory::empty<c32>(tiles_shape.rfft(), options);
            m_cubes_padded_rfft = noa::memory::empty<c32>(cube_padded_shape.rfft(), options);
            m_cubes_padded_weight_rfft = noa::memory::like<f32>(m_cubes_padded_rfft);

            if (options.device().is_gpu()) {
                // To limit the number of synchronizations and copies between GPU<->CPU, we reconstruct
                // and accumulate a row of cubes. Once this row is reconstructed on the compute-device,
                // it is copied back to the CPU and inserted in the final full reconstruction.
                const auto row_shape = Shape4<i64>{1, cube_size_(), cube_size_(), volume_shape_()[2]};
                m_row_of_cubes = noa::memory::empty<f32>(row_shape, options);
            }
        }

        [[nodiscard]] auto reconstruct(const View<f32>& tilt_series, const Path& debug_directory) const -> Array<f32> {
            auto output = noa::memory::empty<f32>(volume_shape_().push_front(1));
            const auto reconstruction = output.view();

            const auto row_of_cubes = m_row_of_cubes.view();
            const bool use_row_of_cubes_buffer = !m_row_of_cubes.is_empty();

            const auto tiles_rfft = m_tiles_rfft.view();
            const auto tiles = noa::fft::alias_to_real(tiles_rfft, tiles_shape_());
            const auto cube_padded_rfft = m_cubes_padded_rfft.view();
            const auto cube_padded = noa::fft::alias_to_real(cube_padded_rfft, cube_padded_shape_().push_front(1));
            const auto cube_padded_weight_rfft = m_cubes_padded_weight_rfft.view();

            using WindowedSinc = noa::geometry::fft::WindowedSinc;
            const auto fftfreq_sinc = 1 / static_cast<f32>(cube_padded_size_());
            const auto fftfreq_window = WindowedSinc{fftfreq_sinc, fftfreq_sinc * 4};

            // For every cube:
            for (i64 z = 0; z < m_cubes_count[0]; ++z) {
                for (i64 y = 0; y < m_cubes_count[1]; ++y) {
                    qn::Logger::trace("Reconstructing cube: z={}, y={}", z, y);

                    // Compute a row of cubes. This entire loop shouldn't require any GPU synchronization.
                    for (i64 x = 0; x < m_cubes_count[2]; ++x) {
                        const auto cube_index = (z * m_cubes_count[1] + y) * m_cubes_count[2] + x;

                        // Extract the tiles.
                        noa::memory::extract_subregions(
                                tilt_series, tiles, m_tile_origins.view().subregion(cube_index));
                        if (!debug_directory.empty()) {
                            constexpr std::string_view filename = "tiles_z{:0>2}_y{:0>2}_x{:0>2}.mrc";
                            noa::io::save(tiles, debug_directory / fmt::format(filename, z, y, x));
                        }

                        // Compute the central slices of the tiles and shift their rotation center at the origin.
                        noa::fft::r2c(tiles, tiles_rfft);
                        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                                tiles_rfft, tiles_rfft, tiles.shape(),
                                m_tile_shifts.view().subregion(cube_index));

                        // Reset the volumes to 0.
                        noa::memory::fill(cube_padded_rfft, c32{0});
                        noa::memory::fill(cube_padded_weight_rfft, f32{0});

                        if (m_use_rasterization) {
                            noa::geometry::fft::insert_rasterize_3d<noa::fft::H2H>(
                                    tiles_rfft, f32{1}, tiles.shape(),
                                    cube_padded_rfft, cube_padded_weight_rfft, cube_padded.shape(),
                                    Float22{}, m_insertion_rotation); // forward rotation
                        } else {
                            // In-place fftshift, since the Fourier insertion only supported centered inputs.
                            noa::fft::remap(noa::fft::H2HC, tiles_rfft, tiles_rfft, tiles.shape());

                            // Backward projection using direct Fourier insertion.
                            noa::geometry::fft::insert_interpolate_3d<noa::fft::HC2H>(
                                    tiles_rfft, f32{1}, tiles.shape(),
                                    cube_padded_rfft, cube_padded_weight_rfft, cube_padded.shape(),
                                    Float22{}, m_insertion_rotation, fftfreq_window); // inverse rotation
                        }

                        // Shift back to the center of the padded cube.
                        noa::signal::fft::phase_shift_3d<noa::fft::H2H>(
                                cube_padded_rfft, cube_padded_rfft,
                                cube_padded.shape(), cube_padded_center_().as<f32>());

                        // Correct for the multiplicity.
                        if (m_use_rasterization) {
                            noa::ewise_binary(
                                    cube_padded_rfft, cube_padded_weight_rfft,
                                    cube_padded_rfft, qn::correct_multiplicity_rasterize_t{});
                        } else {
                            noa::ewise_binary(
                                    cube_padded_rfft, cube_padded_weight_rfft,
                                    cube_padded_rfft, qn::correct_multiplicity_t{});
                        }

                        // Reconstruction.
                        noa::fft::c2r(cube_padded_rfft, cube_padded);
                        if (m_use_rasterization) {
                            noa::geometry::fft::gridding_correction(
                                    cube_padded, cube_padded, /*post_correction=*/ true);
                        }

                        // The cube is reconstructed, so save it.
                        if (use_row_of_cubes_buffer)
                            copy_central_cube_in_row_(cube_padded, row_of_cubes, x, cube_size_());
                        else
                            copy_central_cube_in_reconstruction_(cube_padded, reconstruction, z, y, x, cube_size_());
                    }

                    // The row of cubes is done, we need to copy it back and insert it into the output.
                    if (use_row_of_cubes_buffer)
                        copy_row_into_reconstruction_(row_of_cubes, reconstruction, z, y, cube_size_());
                }
            }
            return output.eval();
        }

    private:
        [[nodiscard]] auto slice_shape_() const noexcept -> Shape2<i64> { return m_slice_shape; }
        [[nodiscard]] auto volume_shape_() const noexcept -> Shape3<i64> { return m_volume_shape; }
        [[nodiscard]] auto cube_size_() const noexcept -> i64 { return m_cube_size; }
        [[nodiscard]] auto cube_padded_size_() const noexcept -> i64 { return m_cube_padded_size; }
        [[nodiscard]] auto tile_size_() const noexcept -> i64 { return cube_padded_size_(); }

        [[nodiscard]] auto cube_padded_shape_() const noexcept -> Shape3<i64> {
            return {m_cube_padded_size, m_cube_padded_size, m_cube_padded_size};
        }

        [[nodiscard]] auto tiles_shape_() const noexcept -> Shape4<i64> {
            return {m_tiles_rfft.shape()[0], 1, tile_size_(), tile_size_()};
        }

        [[nodiscard]] auto volume_center_() const noexcept -> Vec3<f64> {
            return {
                    MetadataSlice::center<f64>(volume_shape_()[0]),
                    MetadataSlice::center<f64>(volume_shape_()[1]),
                    MetadataSlice::center<f64>(volume_shape_()[2]),
            };
        }

        [[nodiscard]] auto slice_center_() const noexcept -> Vec2<f64> {
            return MetadataSlice::center<f64>(slice_shape_());
        }

        [[nodiscard]] auto tile_center_() const noexcept -> Vec2<f64> {
            return MetadataSlice::center<f64>(tile_size_(), tile_size_());
        }

        [[nodiscard]] auto cube_padded_center_() const noexcept -> Vec3<f64> {
            auto center = MetadataSlice::center<f64>(cube_padded_size_());
            return {center, center, center};
        }

        [[nodiscard]] static auto subdivide_volume_in_cubes_(
                const Shape3<i64>& volume_shape,
                i64 cube_size
        ) -> std::pair<std::vector<Vec3<f64>>, Vec3<i64>>
        {
            const Vec3<i64> cubes_count = (volume_shape + cube_size - 1).vec() / cube_size; // divide up
            std::vector<Vec3<f64>> cubes_coordinates;
            cubes_coordinates.reserve(static_cast<size_t>(noa::math::product(cubes_count)));

            for (i64 z = 0; z < cubes_count[0]; ++z) {
                for (i64 y = 0; y < cubes_count[1]; ++y) {
                    for (i64 x = 0; x < cubes_count[2]; ++x) {
                        auto cube_coordinates = Vec3<i64>{z, y, x} * cube_size + cube_size / 2; // center of the cubes
                        cubes_coordinates.emplace_back(cube_coordinates.as<f64>());
                    }
                }
            }
            return {cubes_coordinates, cubes_count};
        }

        [[nodiscard]] static auto project_cube_coordinates_(
                const std::vector<Vec3<f64>>& cube_coordinates,
                const MetadataStack& metadata,
                bool use_rasterization,
                const Vec3<f64>& volume_center,
                const Vec2<f64>& slice_center,
                const Vec2<f64>& tile_center
        ) -> std::tuple<noa::Array<Vec4<i32>>,
                        noa::Array<Vec2<f32>>,
                        noa::Array<Quaternion<f32>>>
        {
            // Outputs.
            const i64 slice_count = metadata.ssize();
            auto insertion_rotation = Array<Quaternion<f32>>(slice_count);
            auto tiles_origins = Array<Vec4<i32>>({cube_coordinates.size(), 1, 1, slice_count});
            auto tiles_shifts = Array<Vec2<f32>>({cube_coordinates.size(), 1, 1, slice_count});

            const auto fwd_projection_matrices = Array<Double34>(slice_count);
            const auto insertion_rotation_span = insertion_rotation.span();
            const auto fwd_projection_matrices_span = fwd_projection_matrices.span();

            // Prepare the projection and insertion matrices.
            for (i64 slice_index{}; slice_index < slice_count; ++slice_index) {
                const MetadataSlice& slice = metadata[slice_index];

                // The in-place rotation is the rotation of the image. To align the image onto
                // the Y axis, we need to "cancel" whatever angle is saved in the metadata.
                auto euler_angles = noa::math::deg2rad(slice.angles);

                // Insertion matrix.
                auto insertion_rotation_matrix = noa::geometry::euler2matrix(
                        Vec3<f64>{-euler_angles[0], euler_angles[1], euler_angles[2]},
                        "zyx", /*intrinsic=*/ false);
                if (!use_rasterization)
                    insertion_rotation_matrix = insertion_rotation_matrix.transpose();
                insertion_rotation_span[slice_index] = noa::geometry::matrix2quaternion(
                        insertion_rotation_matrix).as<f32>();

                // Projection matrix.
                // 1. This relates 3d positions in the tomogram-space to 2d positions in the image-plane.
                //    These matrices are operating on the tomogram in real space.
                //    The first row of the matrix (z) isn't used for the projection.
                // 2. We usually go from the image to the aligned-volume by subtracting the image-(center+shifts),
                //    (-Z)YX rotation and add the volume-center. Here we do the opposite, and go from the
                //    aligned-volume to the image, so we need to do: subtract volume center, X(-Y)(-Z) rotation
                //    and add the image-(center+shifts).
                fwd_projection_matrices_span[slice_index] = noa::geometry::affine2truncated(
                        noa::geometry::translate((slice_center + slice.shifts).push_front(0)) *
                        noa::geometry::linear2affine(noa::geometry::rotate_z(euler_angles[0])) *
                        noa::geometry::linear2affine(noa::geometry::rotate_y(-euler_angles[1])) *
                        noa::geometry::linear2affine(noa::geometry::rotate_x(-euler_angles[2])) *
                        noa::geometry::translate(-volume_center)
                );
            }

            // For every cube, compute the set (one for each slice) of tile-origins and tilt-shifts.
            for (size_t cube_index = 0; cube_index < cube_coordinates.size(); ++cube_index) { // TODO C++20
                const auto cube_coordinate = cube_coordinates[cube_index];
                const auto tiles_origins_span = tiles_origins.view().subregion(cube_index).span();
                const auto tiles_shifts_span = tiles_shifts.view().subregion(cube_index).span();

                // Compute the projected cube coordinates, for every slice.
                for (i64 slice_index = 0; slice_index < slice_count; ++slice_index) {
                    // This goes from the center of the cube, to the center of the 2d-tiles in the slice.
                    // Note: If we had a 3|4d deformation model on the cube coords,
                    //       this is where we would add the interpolated shifts.
                    const auto& fwd_projection_matrix = fwd_projection_matrices_span[slice_index];
                    const auto tile_coordinate = (fwd_projection_matrix * cube_coordinate.push_back(1)).filter(1, 2);

                    // Extract the tile origin and residual shifts for the extraction.
                    const auto tile_origin_coords = tile_coordinate - tile_center;
                    const auto tile_origin_truncated = noa::math::floor(tile_origin_coords);
                    const auto tile_origin = tile_origin_truncated.as<i32>();
                    const auto tile_residual_shift = tile_origin_coords - tile_origin_truncated; // >0

                    // These are the origins for extract_subregions().
                    tiles_origins_span[slice_index] = {slice_index, 0, tile_origin[0], tile_origin[1]};

                    // The tiles will be rotated during the Fourier insertion, so apply the
                    // residual shifts and the shift for the rotation center at the same time.
                    // Here we subtract to bring the rotation center at the origin.
                    tiles_shifts_span[slice_index] = -(tile_center + tile_residual_shift).as<f32>();
                }
            }
            return {tiles_origins, tiles_shifts, insertion_rotation};
        }

        static void copy_central_cube_in_row_(
                const View<f32>& cube_padded,
                const View<f32>& row_of_cubes,
                i64 index_x,
                i64 cube_size
        ) {
            // Possibly truncate the last cubes along the width.
            const i64 cube_offset = cube_size * index_x;
            const i64 size_cube_x = std::min(cube_size, row_of_cubes.shape()[3] - cube_offset);

            // Take the central cube at the center of the padded cube (the center is preserved).
            const i64 center_padded = cube_padded.shape()[3] / 2;
            const i64 center = cube_size / 2;
            const i64 start = center_padded - center;
            const auto input_central_cube = cube_padded.subregion(
                    0,
                    noa::indexing::Slice{start, start + cube_size},
                    noa::indexing::Slice{start, start + cube_size},
                    noa::indexing::Slice{start, start + size_cube_x});

            // Take a view of the corresponding cube in the row of cubes.
            const auto output_central_cube = row_of_cubes.subregion(
                    noa::indexing::Ellipsis{}, noa::indexing::Slice{cube_offset, cube_offset + size_cube_x});

            noa::memory::copy(input_central_cube, output_central_cube);
        }

        static void copy_central_cube_in_reconstruction_(
                const View<f32>& cube_padded,
                const View<f32>& reconstruction,
                i64 z, i64 y, i64 x,
                i64 cube_size
        ) {
            // Possibly truncate the last cubes along the dimensions.
            const i64 size_cube_z = std::min(cube_size, reconstruction.shape()[1] - cube_size * z);
            const i64 size_cube_y = std::min(cube_size, reconstruction.shape()[2] - cube_size * y);
            const i64 size_cube_x = std::min(cube_size, reconstruction.shape()[3] - cube_size * x);

            const i64 center_padded = cube_padded.shape()[3] / 2;
            const i64 center = cube_size / 2;
            const i64 start = center_padded - center;
            const auto input_central_cube = cube_padded.subregion(
                    noa::indexing::FullExtent{},
                    noa::indexing::Slice{start, start + size_cube_z},
                    noa::indexing::Slice{start, start + size_cube_y},
                    noa::indexing::Slice{start, start + size_cube_x});

            // Here we could let the subregion method to clamp the last cube,
            // but since we already have computed the truncated sizes, use them instead.
            const auto output_central_cube = reconstruction.subregion(
                    0,
                    noa::indexing::Slice{cube_size * z, cube_size * z + size_cube_z},
                    noa::indexing::Slice{cube_size * y, cube_size * y + size_cube_y},
                    noa::indexing::Slice{cube_size * x, cube_size * x + size_cube_x}
            );

            noa::memory::copy(input_central_cube, output_central_cube);
        }

        static void copy_row_into_reconstruction_(
                const View<f32>& row_of_cubes,
                const View<f32>& reconstruction,
                i64 index_z, i64 index_y,
                i64 cube_size
        ) {
            const i64 cube_offset_z = cube_size * index_z;
            const i64 cube_offset_y = cube_size * index_y;
            const i64 size_cube_z = std::min(cube_size, reconstruction.shape()[1] - cube_offset_z);
            const i64 size_cube_y = std::min(cube_size, reconstruction.shape()[2] - cube_offset_y);

            const auto valid_row_of_cubes = row_of_cubes.subregion(
                    0,
                    noa::indexing::Slice{0, size_cube_z},
                    noa::indexing::Slice{0, size_cube_y},
                    noa::indexing::FullExtent{}
            );
            const auto output_row = reconstruction.subregion(
                    0,
                    noa::indexing::Slice{cube_offset_z, cube_offset_z + size_cube_z},
                    noa::indexing::Slice{cube_offset_y, cube_offset_y + size_cube_y},
                    noa::indexing::FullExtent{}
            );
            noa::memory::copy(valid_row_of_cubes, output_row);
        }
    };
}

namespace qn {
    void fourier_tiled_reconstruction(
            const Path& stack_filename,
            const MetadataStack& metadata,
            const Path& output_directory,
            const FourierTiledReconstructionParameters& parameters
    ) {
        noa::Timer timer;
        timer.start();
        qn::Logger::status("Reconstructing tomogram...");

        const auto loading_parameters = qn::LoadStackParameters{
                /*compute_device=*/ parameters.compute_device,
                /*allocator=*/ noa::Allocator::DEFAULT_ASYNC,
                /*precise_cutoff=*/ true,
                /*rescale_target_resolution=*/ parameters.resolution,
                /*rescale_min_size=*/ 256,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.01, 0.01},
                /*lowpass_parameters=*/ {0.5, 0.01},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.02f,
                /*zero_pad_to_fast_fft_shape=*/ false,
        };

//        if (parameters.save_aligned_stack) {
//            qn::save_stack(
//                    options.files.input_stack,
//                    options.files.output_directory / "aligned.mrc",
//                    metadata, loading_parameters);
//        }


        auto updated_metadata = metadata;
        const auto [stack, stack_spacing, file_spacing] =
                load_stack(stack_filename, updated_metadata, loading_parameters);
        updated_metadata.rescale_shifts(file_spacing, stack_spacing);

        const f64 average_spacing = noa::math::sum(stack_spacing) / 2;
        const f64 sample_thickness_pix = parameters.sample_thickness_nm / (average_spacing * 1e-1);

        auto reconstruction = VirtualVolume(
                stack.shape().pop_front<2>(), updated_metadata, stack.device(),
                sample_thickness_pix, parameters.cube_size, parameters.use_rasterization)
                .reconstruct(stack.view(), parameters.debug_directory);

        auto reconstruction_filename = output_directory / fmt::format(
                "{}_tomogram.mrc", stack_filename.stem().string());
        noa::io::save(reconstruction, reconstruction_filename);
        qn::Logger::info("{} saved", reconstruction_filename);

        // FIXME save aligned stack?

        qn::Logger::status("Reconstructing tomogram... done. Took {:.2f}s", timer.elapsed() * 1e-3);
    }
}
