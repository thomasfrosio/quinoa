#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/Reconstruction.hpp"

namespace {
    using namespace qn;

    struct CorrectMultiplicity {
        constexpr void operator()(const f32& weight, c32& value) const {
            if (weight > 1)
                value /= weight;
        }
    };

    struct SIRTWeight {
        using value_type = f32;
        f32 fake_iter;

        // With a large enough level (>1000), this is equivalent to a radial weighting.
        constexpr explicit SIRTWeight(f32 level) {
            fake_iter = level <= 15.f ? level :
                        level <= 30.f ? 15.f + 0.4f * (level - 15.f) :
                                        27.f * 0.6f * (level - 30.f);
        }

        NOA_HD auto operator()(const Vec<f32, 3>& fftfreq_3d, i32) const {
            const f32 fftfreq = noa::sqrt(dot(fftfreq_3d, fftfreq_3d));
            if (fftfreq < 1e-6f or fftfreq > 0.5f)
                return 0.f;
            const f32 max = 0.5f * (1 - noa::pow(1 - 0.00195f / 0.5f, fake_iter));
            const f32 current = fftfreq * (1 - noa::pow(1 - 0.00195f / fftfreq, fake_iter));
            return current / max;
        }
    };

    [[nodiscard]] auto subdivide_volume_in_cubes_(const Shape<i64, 3>& volume_shape, i64 cube_size) -> Vec<i64, 3> {
        return (volume_shape + cube_size - 1).vec / cube_size; // divide up
    }

    void copy_central_cube_in_row_(
        const View<f32>& cube_padded,
        const View<f32>& row_of_cubes,
        i64 x,
        i64 cube_size
    ) {
        // Possibly truncate the last cubes along the width.
        const i64 cube_offset = cube_size * x;
        const i64 size_cube_x = std::min(cube_size, row_of_cubes.shape()[3] - cube_offset);

        // Take the central cube at the center of the padded cube (the center is preserved).
        const i64 center_padded = cube_padded.shape()[3] / 2;
        const i64 center = cube_size / 2;
        const i64 start = center_padded - center;
        const auto input_central_cube = cube_padded.subregion(
            0,
            ni::Slice{start, start + cube_size},
            ni::Slice{start, start + cube_size},
            ni::Slice{start, start + size_cube_x}
        );

        // Take a view of the corresponding cube in the row of cubes.
        const auto output_central_cube = row_of_cubes.subregion(
            ni::Ellipsis{}, ni::Slice{cube_offset, cube_offset + size_cube_x});

        input_central_cube.to(output_central_cube);
    }

    void copy_cube_in_reconstruction_(
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
            ni::FullExtent{},
            ni::Slice{start, start + size_cube_z},
            ni::Slice{start, start + size_cube_y},
            ni::Slice{start, start + size_cube_x}
        );

        // Here we could let the subregion method to clamp the last cube,
        // but since we already have computed the truncated sizes, use them instead.
        const auto output_central_cube = reconstruction.subregion(
            0,
            ni::Slice{cube_size * z, cube_size * z + size_cube_z},
            ni::Slice{cube_size * y, cube_size * y + size_cube_y},
            ni::Slice{cube_size * x, cube_size * x + size_cube_x}
        );

        input_central_cube.to(output_central_cube);
    }

    void copy_row_into_reconstruction_(
        const View<f32>& row_of_cubes,
        const View<f32>& reconstruction,
        i64 z, i64 y, i64 cube_size
    ) {
        const i64 cube_offset_z = cube_size * z;
        const i64 cube_offset_y = cube_size * y;
        const i64 size_cube_z = std::min(cube_size, reconstruction.shape()[1] - cube_offset_z);
        const i64 size_cube_y = std::min(cube_size, reconstruction.shape()[2] - cube_offset_y);

        const auto valid_row_of_cubes = row_of_cubes.subregion(
            0,
            ni::Slice{0, size_cube_z},
            ni::Slice{0, size_cube_y},
            ni::FullExtent{}
        );
        const auto output_row = reconstruction.subregion(
            0,
            ni::Slice{cube_offset_z, cube_offset_z + size_cube_z},
            ni::Slice{cube_offset_y, cube_offset_y + size_cube_y},
            ni::FullExtent{}
        );
        valid_row_of_cubes.to(output_row);
    }

    // auto fourier_space_reconstruction(
    //     const View<f32>& tilt_series,
    //     const MetadataStack& metadata,
    //     i64 volume_thickness_pix,
    //     i64 cube_size,
    //     bool is_fourier_weighting,
    //     f32 weighting_level
    // ) -> Array<f32> {
    //     // Dimensions.
    //     cube_size = noa::fft::next_fast_size(cube_size);
    //     const auto cube_padded_size = noa::fft::next_fast_size(cube_size * 2);
    //     const auto slice_shape = tilt_series.shape().filter(2, 3);
    //     const auto volume_shape = Shape{volume_thickness_pix, slice_shape[0], slice_shape[1]};
    //     const auto slice_center = (slice_shape / 2).vec.as<f64>();
    //     const auto volume_center = (volume_shape / 2).vec.as<f64>();
    //     const auto tile_center = Vec<f64, 2>::from_value(cube_padded_size / 2);
    //
    //     // Get the grid of cubes and their centers, and project them to get the tile centers.
    //     const Vec grid = subdivide_volume_in_cubes_(volume_shape, cube_size);
    //     const i64 n_cubes = noa::product(grid);
    //     const i64 n_slices = metadata.ssize();
    //
    //     auto tile_inverse_rotations = Array<ng::Quaternion<f32>>(n_slices);
    //     auto forward_projection_matrices = Array<Mat<f64, 2, 4>>(n_slices);
    //     auto tile_inverse_rotations_1d = tile_inverse_rotations.span_1d_contiguous();
    //     auto forward_projection_matrices_1d = forward_projection_matrices.span_1d_contiguous();
    //
    //     // Compute the volume-image matrices.
    //     for (i64 i: noa::irange(n_slices)) {
    //         const MetadataSlice& slice = metadata[i];
    //
    //         // This relates 3d positions in the tomogram to 2d positions in the tilt image.
    //         const auto volume_to_image_matrix = (
    //             ng::translate((slice_center + slice.shifts).push_front(0)) *
    //             ng::linear2affine(ng::rotate_z(noa::deg2rad(slice.angles[0]))) *
    //             ng::linear2affine(ng::rotate_y(noa::deg2rad(slice.angles[1]))) *
    //             ng::linear2affine(ng::rotate_x(noa::deg2rad(slice.angles[2]))) *
    //             ng::translate(-volume_center)
    //         );
    //
    //         // For projecting the cube coordinates, remove the z-row (and the homogeneous row while we are at it).
    //         forward_projection_matrices_1d[i] = volume_to_image_matrix.filter_rows(1, 2);
    //
    //         // For the Fourier insertion of the central slices, noa::fourier_insert_interpolate_3d needs the
    //         // inverse of the image to volume transform, i.e. the volume to image transform, which we just computed.
    //         // Of course, this is Fourier space, so only use the 3d rotation (the shifts are computed below).
    //         // Use quaternions for faster loading on GPU.
    //         tile_inverse_rotations_1d[i] = ng::matrix2quaternion(ng::affine2linear(volume_to_image_matrix)).as<f32>();
    //     }
    //
    //     // For the tile extraction, we need to know the origin of every tile.
    //     // The extraction leaves residual shifts that we'll need to add before the backward projection.
    //     // As such, for every cube, compute the set (one for each slice) of tile origins and residual shifts.
    //     auto tile_origins = Array<Vec<i32, 4>>({n_cubes, 1, 1, n_slices});
    //     auto tile_shifts = Array<Vec<f32, 2>>({n_cubes, 1, 1, n_slices});
    //     auto tiles_origins_2d = tile_origins.span().filter(0, 3).as_contiguous();
    //     auto tiles_shifts_2d = tile_shifts.span().filter(0, 3).as_contiguous();
    //
    //     for (i64 z = 0; z < grid[0]; ++z) {
    //         for (i64 y = 0; y < grid[1]; ++y) {
    //             for (i64 x = 0; x < grid[2]; ++x) {
    //                 const i64 i = (z * grid[1] + y) * grid[2] + x;
    //                 const auto cube_coordinates = (Vec{z, y, x} * cube_size + cube_size / 2).as<f64>();
    //
    //                 for (i64 j: noa::irange(n_slices)) {
    //                     // Compute the projected cube coordinates.
    //                     // This goes from the center of the cube, to the center of the 2d tiles in the slice.
    //                     const Vec tile_coordinate = forward_projection_matrices_1d[j] * cube_coordinates.push_back(1);
    //
    //                     // Extract the tile origin and residual shifts for the extraction.
    //                     const Vec tile_origin_coordinate = tile_coordinate - tile_center;
    //                     const Vec tile_origin_truncated = noa::floor(tile_origin_coordinate);
    //                     const Vec tile_origin = tile_origin_truncated.as<i32>();
    //                     const Vec tile_residual_shift = tile_origin_coordinate - tile_origin_truncated;
    //
    //                     // These are the tile origin for noa::extract_subregion.
    //                     tiles_origins_2d(i, j) = {static_cast<i32>(j), 0, tile_origin[0], tile_origin[1]};
    //
    //                     // The tiles will be rotated during the Fourier insertion, so apply the
    //                     // residual shifts and the shift for the rotation center at the same time.
    //                     // Here we subtract to bring the rotation center at the origin.
    //                     const auto tile_rotation_center = tile_center + tile_residual_shift;
    //                     tiles_shifts_2d(i, j) = -tile_rotation_center.as<f32>();
    //                 }
    //             }
    //         }
    //     }
    //
    //     const auto options = ArrayOption{tilt_series.device(), Allocator::MANAGED};
    //
    //     // For the reconstruction itself, we can prepare some buffers.
    //     const auto tiles_shape = Shape4<i64>{metadata.ssize(), 1, cube_padded_size, cube_padded_size};
    //     const auto tiles_buffer = noa::fft::empty<f32>(tiles_shape, options);
    //     const auto tiles = tiles_buffer.first.view();
    //     const auto tiles_rfft = tiles_buffer.second.view();
    //
    //     const auto cube_padded_shape = Shape4<i64>{1, cube_padded_size, cube_padded_size, cube_padded_size};
    //     const auto cube_padded_center = (cube_padded_shape.pop_front() / 2).vec.as<f32>();
    //     const auto cube_padded_buffer = noa::fft::empty<f32>(cube_padded_shape, options);
    //     const auto cube_padded = cube_padded_buffer.first.view();
    //     const auto cube_padded_rfft = cube_padded_buffer.second.view();
    //     const auto cube_padded_weight_rfft = noa::like<f32>(cube_padded_rfft);
    //
    //     auto row_of_cubes = noa::Array<f32>{};
    //     if (options.device.is_gpu()) {
    //         tile_origins = std::move(tile_origins).to(options);
    //         tile_shifts = std::move(tile_shifts).to(options);
    //         tile_inverse_rotations = std::move(tile_inverse_rotations).to(options);
    //
    //         // To limit the number of synchronizations and copies between GPU and CPU, we reconstruct
    //         // and accumulate a row of cubes. Once this row is reconstructed on the compute-device,
    //         // it is copied back to the CPU and inserted into the final full reconstruction.
    //         row_of_cubes = noa::Array<f32>({1, cube_size, cube_size, volume_shape[2]}, options);
    //     }
    //
    //     auto output = Array<f32>(volume_shape.push_front(1));
    //     const bool use_row_of_cubes_buffer = not row_of_cubes.is_empty();
    //
    //     const auto zero_mask = ng::Sphere{
    //         .center = tile_center,
    //         .radius = (static_cast<f64>(cube_size) * std::sqrt(2) + 10) / 2,
    //         .smoothness = 20,
    //     };
    //     Logger::trace("Tile masking: center={}, radius={}, smoothness",
    //                   zero_mask.center, zero_mask.radius, zero_mask.smoothness);
    //
    //     // Windowed-sinc.
    //     const auto fftfreq_sinc = 1 / static_cast<f64>(cube_padded_size);
    //     const auto windowed_sinc = ng::WindowedSinc{fftfreq_sinc, fftfreq_sinc * 10};
    //     const auto fourier_insertion_options = ng::FourierInsertInterpolateOptions{
    //         .interp = noa::Interp::LANCZOS8,
    //         .windowed_sinc = windowed_sinc
    //     };
    //     Logger::trace("Windowed-sinc: fftfreq_sinc={}, fftfreq_blackman={}",
    //                   windowed_sinc.fftfreq_sinc, windowed_sinc.fftfreq_blackman);
    //
    //     // Compute a row of cubes. This entire loop shouldn't require any GPU synchronization.
    //     for (i64 z = 0; z < grid[0]; ++z) {
    //         for (i64 y = 0; y < grid[1]; ++y) {
    //             auto tt = Logger::trace_scope_time("row_of_cubes");
    //             for (i64 x = 0; x < grid[2]; ++x) {
    //                 auto ttt = Logger::trace_scope_time(fmt::format("Reconstructing cubes: z={}, y={}, x={}", z, y, x));
    //                 const i64 cube_index = (z * grid[1] + y) * grid[2] + x;
    //
    //                 // Extract and mask the tiles.
    //                 noa::extract_subregions(tilt_series, tiles.view(), tile_origins.view().subregion(cube_index));
    //                 ng::draw_shape(tiles.view(), tiles.view(), zero_mask);
    //                 // noa::write(tiles, "/home/thomas/Projects/quinoa/tests/test02/tiles.mrc");
    //
    //                 // Compute the FFT of the tiles.
    //                 // Shift to the tile rotation center, plus correct for the residual extraction shifts.
    //                 // Only the center of the twice-as-large cubes is used, so no need to zero-pad here.
    //                 noa::fft::r2c(tiles, tiles_rfft);
    //                 ns::phase_shift_2d<"h2h">(
    //                     tiles_rfft, tiles_rfft, tiles.shape(),
    //                     tile_shifts.view().subregion(cube_index)
    //                 );
    //
    //                 // Backward project by inserting the central slices in the Fourier volume.
    //                 // TODO add exposure filter and CTF
    //                 noa::fill(cube_padded_rfft, {});
    //                 if (is_fourier_weighting) {
    //                     noa::fill(cube_padded_weight_rfft, {});
    //                     ng::fourier_insert_interpolate_3d<"h2h">(
    //                         tiles_rfft, {}, tiles.shape(),
    //                         cube_padded_rfft, cube_padded_weight_rfft, cube_padded.shape(),
    //                         {}, tile_inverse_rotations, fourier_insertion_options);
    //                 } else {
    //                     ng::fourier_insert_interpolate_3d<"h2h">(
    //                         tiles_rfft, {}, tiles.shape(),
    //                         cube_padded_rfft, {}, cube_padded.shape(),
    //                         {}, tile_inverse_rotations, fourier_insertion_options);
    //                 }
    //
    //                 // Shift back to the origin.
    //                 ns::phase_shift_3d<"h2h">(
    //                     cube_padded_rfft, cube_padded_rfft, cube_padded.shape(), cube_padded_center
    //                 );
    //
    //                 // Correct for the multiplicity.
    //                 if (is_fourier_weighting) {
    //                     noa::ewise(cube_padded_weight_rfft, cube_padded_rfft, CorrectMultiplicity{});
    //                 } else {
    //                     ns::filter_spectrum_3d<"h2h">(
    //                         cube_padded_rfft, cube_padded_rfft, cube_padded.shape(), SIRTWeight(weighting_level));
    //                 }
    //
    //                 // Reconstruction. Since we used a windowed-sinc interpolation, there's no need for a
    //                 // correction since the apodization window is pretty much a perfect step function.
    //                 // If we had used something like a linear interpolation, things would be different.
    //                 noa::fft::c2r(cube_padded_rfft, cube_padded);
    //                 // noa::write(cube_padded, "/home/thomas/Projects/quinoa/tests/test02/cube_padded.mrc");
    //
    //                 // The cube is reconstructed, so save it.
    //                 use_row_of_cubes_buffer ?
    //                     copy_central_cube_in_row_(cube_padded, row_of_cubes.view(), x, cube_size) :
    //                     copy_cube_in_reconstruction_(cube_padded, output.view(), z, y, x, cube_size);
    //             }
    //
    //             // The row of cubes is done, we need to copy it back and insert it into the output.
    //             if (use_row_of_cubes_buffer)
    //                 copy_row_into_reconstruction_(row_of_cubes.view(), output.view(), z, y, cube_size); // GPU -> CPU
    //         }
    //     }
    //     return output;
    // }

    auto real_space_reconstruction(
        const View<f32>& tilt_series,
        const MetadataStack& metadata,
        i64 volume_thickness_pix,
        i64 cube_size,
        f32 sirt_like_level
    ) -> Array<f32> {
        // Dimensions.
        cube_size = noa::fft::next_fast_size(cube_size);
        const auto cube_padded_size = noa::fft::next_fast_size(cube_size * 2);
        const auto slice_shape = tilt_series.shape().filter(2, 3);
        const auto volume_shape = Shape{volume_thickness_pix, slice_shape[0], slice_shape[1]};

        // Grid of cubes.
        const Vec slice_center = (slice_shape / 2).vec.as<f64>();
        const Vec volume_center = (volume_shape / 2).vec.as<f64>();
        const Vec grid = subdivide_volume_in_cubes_(volume_shape, cube_size);
        const i64 n_slices = metadata.ssize();

        // Prepare the tilt-series for Cubic B-Spline interpolation.
        // Note: we "consume" the tilt-series here... I don't want to copy it.
        noa::cubic_bspline_prefilter(tilt_series, tilt_series);

        // Backward projection matrices, for ng::backward_project_3d.
        const auto device = tilt_series.device();
        auto projection_matrices = Array<Mat<f32, 2, 4>>({grid[0], grid[1], grid[2], n_slices});
        for (i64 z = 0; z < grid[0]; ++z) {
            for (i64 y = 0; y < grid[1]; ++y) {
                for (i64 x = 0; x < grid[2]; ++x) {
                    const auto cube_offset = (Vec{z, y, x} * cube_size - cube_size / 2).as<f64>();

                    for (i64 i: noa::irange(n_slices)) {
                        const MetadataSlice& slice = metadata[i];

                        projection_matrices(z, y, x, i) = (
                            ng::translate((slice_center + slice.shifts).push_front(0)) *
                            ng::linear2affine(ng::rotate_z(noa::deg2rad(slice.angles[0]))) *
                            ng::linear2affine(ng::rotate_y(noa::deg2rad(slice.angles[1]))) *
                            ng::linear2affine(ng::rotate_x(noa::deg2rad(slice.angles[2]))) *
                            ng::translate(-volume_center + cube_offset)
                        ).filter_rows(1, 2).as<f32>();
                    }
                }
            }
        }
        if (device.is_gpu())
            projection_matrices = std::move(projection_matrices).to({device, Allocator::ASYNC});

        // Allocate the main buffers for the reconstruction.
        // Use unified memory; let the driver manage device access.
        const auto options = ArrayOption{device, Allocator::ASYNC};
        const auto cube_buffer = noa::fft::empty<f32>({1, cube_padded_size, cube_padded_size, cube_padded_size}, options);
        const auto cube = cube_buffer.first.view();
        const auto cube_rfft = cube_buffer.second.view();
        auto output = Array<f32>(volume_shape.push_front(1));

        auto row_of_cubes = noa::Array<f32>{};
        if (options.device.is_gpu()) {
            // To limit the number of synchronizations and copies between GPU and CPU, we reconstruct
            // and accumulate a row of cubes. Once this row is reconstructed on the compute-device,
            // it is copied back to the CPU and inserted into the final full reconstruction.
            row_of_cubes = noa::Array<f32>({1, cube_size, cube_size, volume_shape[2]}, options);
        }

        const auto tiles_shape = Shape4<i64>{metadata.ssize(), 1, cube_padded_size, cube_padded_size};
        const auto tiles_buffer = noa::fft::empty<f32>(tiles_shape, options);
        const auto tiles = tiles_buffer.first.view();
        const auto tiles_rfft = tiles_buffer.second.view();

        auto tile_origins = Array<Vec<i32, 2>>(metadata.ssize(), options);
        tilt_series.eval();

        // Reconstruct.
        for (i64 z = 0; z < grid[0]; ++z) {
            for (i64 y = 0; y < grid[1]; ++y) {
                auto tt = Logger::trace_scope_time("row_of_cubes");

                for (i64 x = 0; x < grid[2]; ++x) {
                    auto ttt = Logger::trace_scope_time("Reconstructing cubes: z={}, y={}, x={}", z, y, x);

                    // noa::extract_subregions(tilt_series, tiles.view(), tile_origins.view());
                    // ng::draw_shape(tiles.view(), tiles.view(), zero_mask);
                    // noa::fft::r2c(tiles, tiles_rfft);

                    // Backprojection.
                    ng::backward_project_3d(
                        tiles, cube,
                        projection_matrices.view().subregion(z, y, x),
                        {.interp = noa::Interp::CUBIC_BSPLINE}
                    );

                    // Filtering.
                    noa::fft::r2c(cube, cube_rfft);
                    ns::filter_spectrum_3d<"h2h">(cube_rfft, cube_rfft, cube.shape(), SIRTWeight(sirt_like_level));
                    noa::fft::c2r(cube_rfft, cube);

                    // The cube is reconstructed, so save it.
                    options.device.is_gpu() ?
                        copy_central_cube_in_row_(cube, row_of_cubes.view(), x, cube_size) :
                        copy_cube_in_reconstruction_(cube, output.view(), z, y, x, cube_size);
                }

                // The row of cubes is done, we need to copy it back and insert it into the output.
                if (options.device.is_gpu())
                    copy_row_into_reconstruction_(row_of_cubes.view(), output.view(), z, y, cube_size); // GPU -> CPU
            }
        }

        return output;
    }
}

namespace qn {
    auto tomogram_reconstruction(
        const View<f32>& stack,
        const Vec<f64, 2>& stack_spacing,
        const MetadataStack& metadata,
        const TomogramReconstructionParameters& parameters
    ) -> Array<f32> {
        auto timer = Logger::info_scope_time("Reconstructing tomogram");

        // Reconstruction mode.
        const auto is_mode_fourier = parameters.mode == "fourier";
        const auto is_weighting_fourier = parameters.weighting == "fourier";
        const auto weighting =
            is_weighting_fourier ? ReconstructionWeighting::FOURIER :
            parameters.weighting == "radial" ? ReconstructionWeighting::RADIAL :
            ReconstructionWeighting::SIRT;
        f32 sirt_level{1000};
        if (weighting == ReconstructionWeighting::SIRT) {
            auto level_str = noa::string::offset_by(parameters.weighting, 5);
            auto level = noa::string::parse<i32>(level_str);
            if (level.has_value())
                sirt_level = static_cast<f32>(level.value());
        }

        // Volume thickness.
        const f64 spacing = noa::mean(stack_spacing);
        const f64 resolution = spacing * 2;
        const f64 sample_thickness_pix = std::round(parameters.sample_thickness_nm / (spacing * 1e-1));
        const f64 z_padding = std::round(sample_thickness_pix * parameters.z_padding_percent / 100) * 2;
        const f64 volume_thickness_pix = sample_thickness_pix + z_padding;
        const f64 volume_thickness_nm = volume_thickness_pix * (spacing * 1e-1);

        Logger::info(
            "Reconstruction:\n"
            "  compute_device={}\n"
            "  mode={} (interp={})\n"
            "  weighting={}\n"
            "  cube_size={}\n"
            "  resolution={:.3f}A\n"
            "  sample_thickness={}pixels ({:.2f}nm)\n"
            "  volume_thickness={}pixels ({:.2f}nm)",
            stack.device(),
            parameters.mode, is_mode_fourier ? noa::Interp::LANCZOS8 : noa::Interp::CUBIC_BSPLINE,
            parameters.weighting,
            parameters.cube_size, resolution,
            sample_thickness_pix, parameters.sample_thickness_nm,
            volume_thickness_pix, volume_thickness_nm
        );

        if (is_mode_fourier) {
            // return fourier_space_reconstruction(
            //     stack.view(), metadata,
            //     static_cast<i64>(volume_thickness_pix),
            //     parameters.cube_size, is_weighting_fourier, sirt_level
            // );
        } else {
            return real_space_reconstruction(
                stack.view(), metadata,
                static_cast<i64>(volume_thickness_pix),
                parameters.cube_size, sirt_level
            );
        }
    }
}
