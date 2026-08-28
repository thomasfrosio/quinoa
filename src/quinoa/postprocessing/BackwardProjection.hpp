#pragma once

#include <noa/FFT.hpp>
#include <noa/Xform.hpp>

#include "quinoa/postprocessing/Utilities.hpp"

namespace qn {
    template<typename Input, typename Matrices, typename Output>
    struct BackwardProjection {
        Input tiles_large{}; // (n,h,w)
        Matrices projection_matrices{}; // (n)
        Output subvolume_large{}; // (d,h,w)

        constexpr void operator()(const Vec<isize, 3>& indices) const {
            const auto volume_coordinates = indices.as<f32>().push_back(1);

            f32 value{};
            for (isize i{}; i < projection_matrices.ssize(); ++i) {
                const auto image_coordinates = projection_matrices[i] * volume_coordinates;
                value += tiles_large.interpolate_at(image_coordinates, i);
            }
            subvolume_large(indices) = value;
        }
    };

    struct BackwardProjectionMaker {
        static constexpr auto BORDER = noa::Border::ZERO;
        using input_span_t = SpanContiguous<const f32, 3>;
        using matrices_span_t = SpanContiguous<const Mat<f32, 2, 4>>;
        using output_span_t = SpanContiguous<f32, 3>;

        input_span_t tiles_large{}; // (n,h,w)
        matrices_span_t projection_matrices{}; // (n)
        output_span_t subvolume_large{}; // (d,h,w)

        template<nx::Interp INTERP>
        [[nodiscard]] auto get() const {
            using interpolator_t = nx::Interpolator<2, INTERP, BORDER, input_span_t>;
            return BackwardProjection<interpolator_t, matrices_span_t, output_span_t>{
                .tiles_large = interpolator_t(tiles_large, tiles_large.shape().pop_front()),
                .projection_matrices = projection_matrices,
                .subvolume_large = subvolume_large,
            };
        }
    };

    class TiledBackwardProjection {
    public:
        nx::Interp interp;
        i32 n_threads{};

        isize actual_oversampling_factor{};
        Shape3 subvolume_shape{};
        Shape3 subvolume_large_shape{};
        Shape3 subvolume_large_padded_shape{};
        Shape4 grid_shape{};

        f64 tile_large_center;
        Array<Vec<i32, 2>> tile_large_origins;
        Array<Mat<f32, 2, 4>> tile_large_padded_matrices;

        Array<f32> subvolume_row;
        Array<c32> tiles_large_buffer;
        Array<c32> tiles_large_padded_buffer;
        Array<c32> subvolume_large_buffer;
        Array<c32> subvolume_large_padded_buffer;

    public:
        TiledBackwardProjection() = default;
        TiledBackwardProjection(
            const Shape2& image_shape,
            const Shape3& volume_shape,
            const Metadata::Stack& metadata,
            const Device& device,
            isize oversampling_factor,
            bool correct_rotation,
            nx::Interp interpolation,
            f64 spacing_nm,
            f64 z_step_nm
        ) :
            interp{interpolation}
        {
            const auto z_step = static_cast<isize>(z_step_nm / spacing_nm);
            const auto n_sections = volume_shape[0] / z_step;
            check(noa::is_odd(z_step) and noa::is_odd(n_sections));

            // 1. To support large reconstructions with oversampling, the volume is divided into subvolumes.
            // To backproject a subvolume, the input tiles should be large enough to map all voxels of the subvolume
            // from any angle. More specifically, if the tilt-axis is along Y, tiles should be sqrt(2)=1.41 times
            // larger than the largest dimension of the subvolume. If the tilt-axis is not aligned, tiles should be
            // sqrt(3)=1.73 times larger.
            //
            // 2. To prevent aliasing, we oversample both the tiles and subvolume. Oversampling real-space tiles is
            // done by zero-padding in Fourier space and thus requires the real-space tiles to have smoothed edges to
            // remove/reduce the Gibbs phenomenon.
            //
            // As such, we extract tiles twice as large and apply a smooth zero-taper to keep edges at zero. Then,
            // we oversample the padded tiles and backproject them. The resulting subvolume is then downsampled,
            // and the central subvolume is extracted and placed back into the volume. In other words, we backproject
            // subvolumes 4 times larger (x2 padding, x2 oversampling) than the final subvolume.
            const auto tile_size = nf::next_fast_size(std::max(z_step, isize{64}));
            const auto tile_large_size = tile_size * 2;
            actual_oversampling_factor = oversampling_factor;
            const auto tile_large_padded_size = tile_large_size * actual_oversampling_factor;

            subvolume_shape = Shape{z_step, tile_size, tile_size};
            subvolume_large_shape = Shape{z_step * 2, tile_large_size, tile_large_size};
            subvolume_large_padded_shape = Shape{z_step * 2 * actual_oversampling_factor, tile_large_padded_size, tile_large_padded_size};

            grid_shape = Shape{
                n_sections,
                noa::divide_up(image_shape[0], tile_size),
                noa::divide_up(image_shape[1], tile_size),
                metadata.ssize()
            };

            // Compute the transformation for each subvolume.
            const auto subvolume_center = (subvolume_shape.vec / 2).as<f64>();
            const auto subvolume_large_center = (subvolume_large_shape.vec / 2).as<f64>();
            tile_large_center = static_cast<f64>(tile_large_size / 2);
            tile_large_origins = Array<Vec<i32, 2>>(grid_shape);
            tile_large_padded_matrices = Array<Mat<f32, 2, 4>>(grid_shape);

            const auto volume2image = volume2image_matrices(metadata, correct_rotation, image_shape, volume_shape);
            const auto volume2image_1d = volume2image.span_1d();

            for (isize z{}; z < grid_shape[0]; ++z) {
                for (isize y{}; y < grid_shape[1]; ++y) {
                    for (isize x{}; x < grid_shape[2]; ++x) {
                        const auto subvolume_origin = Vec{z, y, x} * subvolume_shape.vec;
                        const auto subvolume_center_coordinates = subvolume_origin.as<f64>() + subvolume_center;

                        for (isize t{}; t < grid_shape[3]; ++t) {
                            const auto [tile_large_origin, tile_residual_shift] = extract_tile_large_window(
                                tile_large_center, subvolume_center_coordinates, volume2image_1d[t]);

                            // Compute the backward projection matrix (this is done on the enlarged and oversampled tiles).
                            // Note that the enlarged-tiles/subvolumes are even-sized, so the center is preserved
                            // during oversampling, meaning we can just scale the center with the oversampling factor.
                            const auto angles = noa::deg2rad(metadata[t].angles);
                            const auto final_rotation = correct_rotation ? 0. : angles[0];
                            const auto scale = static_cast<f64>(actual_oversampling_factor);

                            tile_large_origins.span()(z, y, x, t) = tile_large_origin;
                            tile_large_padded_matrices.span()(z, y, x, t) = ( // volume->image
                                nx::translate((tile_large_center + tile_residual_shift).push_front(0) * scale) *
                                nx::rotate_z<true>(angles[0]) *
                                nx::rotate_y<true>(angles[1]) *
                                nx::rotate_x<true>(angles[2]) *
                                nx::rotate_z<true>(-final_rotation) *
                                nx::translate(-subvolume_large_center * scale)
                            ).filter_rows(1, 2).as<f32>(); // (y, x)
                        }
                    }
                }
            }
            if (device.is_gpu()) {
                const auto options_async = ArrayOption{.device = device, .allocator = Allocator::ASYNC};
                tile_large_origins = std::move(tile_large_origins).to(options_async);
                tile_large_padded_matrices = std::move(tile_large_padded_matrices).to(options_async);
            }

            // Compute device.
            // On the GPU, use the tmp row buffer.
            // On the CPU, distribute subvolumes to threads. Since each thread needs its own buffers,
            // limit the number of threads to keep the memory usage reasonable. 10 threads need about 0.5GB.
            n_threads = device.is_gpu() ? 1 : std::max(Stream::current(device).thread_limit(), 8);

            // Allocate buffers.
            // Note that for the CPU mode, each thread needs its own buffer.
            // To retrieve the buffer (as a real and complex view), use the *_pair(tid) functions.
            const auto bd = Vec<isize, 2>::from_values(n_threads, grid_shape[3]);
            const auto tile_large_shape = Shape{tile_large_size, tile_large_size};
            const auto tile_large_padded_shape = Shape{tile_large_padded_size, tile_large_padded_size};
            const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};

            tiles_large_buffer = Array<c32>(tile_large_shape.rfft().push_front(bd), options);
            subvolume_large_buffer = Array<c32>(subvolume_large_shape.rfft().push_front(n_threads), options);
            if (actual_oversampling_factor > 1) {
                tiles_large_padded_buffer = Array<c32>(tile_large_padded_shape.rfft().push_front(bd), options);
                subvolume_large_padded_buffer = Array<c32>(subvolume_large_padded_shape.rfft().push_front(n_threads), options);
            }
            if (device.is_gpu())
                subvolume_row = Array<f32>(subvolume_shape.set<2>(volume_shape[2]).push_front(1), options);
        }

        [[nodiscard]] auto tiles_large_pair(isize tid) const {
            const auto& [td, th, tw] = subvolume_large_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = tiles_large_buffer.view().subregion(tid).permute({1, 0, 2, 3});
            pair.first = nf::alias_to_real(pair.second, Shape{grid_shape[3], isize{1}, th, tw});
            return pair;
        }

        [[nodiscard]] auto tiles_large_padded_pair(isize tid) const {
            if (actual_oversampling_factor == 1)
                return tiles_large_pair(tid);
            const auto& [td, th, tw] = subvolume_large_padded_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = tiles_large_padded_buffer.view().subregion(tid).permute({1, 0, 2, 3});
            pair.first = nf::alias_to_real(pair.second, Shape{grid_shape[3], isize{1}, th, tw});
            return pair;
        }

        [[nodiscard]] auto subvolume_large_pair(isize tid) const {
            const auto& [td, th, tw] = subvolume_large_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = subvolume_large_buffer.view().subregion(tid);
            pair.first = nf::alias_to_real(pair.second, Shape{isize{1}, td, th, tw});
            return pair;
        }

        [[nodiscard]] auto subvolume_large_padded_pair(isize tid) const {
            if (actual_oversampling_factor == 1)
                return subvolume_large_pair(tid);
            const auto& [td, th, tw] = subvolume_large_padded_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = subvolume_large_padded_buffer.view().subregion(tid);
            pair.first = nf::alias_to_real(pair.second, Shape{isize{1}, td, th, tw});
            return pair;
        }

        void prepare_rffts() const {
            if (actual_oversampling_factor > 1) {
                auto [tiles_large, tiles_large_rfft] = tiles_large_pair(0);
                auto [tiles_large_padded, tiles_large_padded_rfft] = tiles_large_padded_pair(0);
                auto [subvolume_large, subvolume_large_rfft] = subvolume_large_pair(0);
                auto [subvolume_large_padded, subvolume_large_padded_rfft] = subvolume_large_padded_pair(0);

                nf::r2c(tiles_large, tiles_large_rfft, {.record_and_share_workspace = true});
                nf::c2r(tiles_large_padded_rfft, tiles_large_padded, {.record_and_share_workspace = true});
                nf::r2c(subvolume_large_padded, subvolume_large_padded_rfft, {.record_and_share_workspace = true});
                nf::c2r(subvolume_large_rfft, subvolume_large, {.record_and_share_workspace = true});
            }
        }

        NOA_NOINLINE auto reconstruct_subvolume(
            const View<const f32>& input_stack,
            isize z, isize y, isize x, isize tid = 0
        ) const -> View<f32> {
            const auto [tiles_large, tiles_large_rfft] = tiles_large_pair(tid);
            const auto [tiles_large_padded, tiles_large_padded_rfft] = tiles_large_padded_pair(tid);
            const auto [subvolume_large, subvolume_large_rfft] = subvolume_large_pair(tid);
            const auto [subvolume_large_padded, subvolume_large_padded_rfft] = subvolume_large_padded_pair(tid);

            // Extract the twice-enlarged tiles and apply the zero-taper at the same time.
            // The backprojected region of the tile is, at most, sqrt(3)=1.73, so each edges has
            // an extra 6.5% of padding that isn't backprojected. We use the last 5% for the taper,
            // which should be enough to remove oversampling artifacts.
            const auto output_tiles = tiles_large.span_contiguous<f32, 3>();
            noa::iwise(output_tiles.shape(), tiles_large.device(), ExtractPaddedTiles{
                .images = input_stack.span_contiguous<const f32, 3>(),
                .output_tiles = output_tiles,
                .tile_large_origins = tile_large_origins.span().subregion(z, y, x).as_1d(),
                .tile_large_center = static_cast<isize>(tile_large_center),
                .taper_radius = static_cast<f32>(tile_large_center * 0.95),
                .taper_smoothness = static_cast<f32>(tile_large_center * 0.05),
            });

            // Oversample, if necessary.
            if (actual_oversampling_factor > 1) {
                nf::r2c(tiles_large, tiles_large_rfft);
                nf::resize<"h">(
                    tiles_large_rfft, tiles_large.shape(),
                    tiles_large_padded_rfft, tiles_large_padded.shape(), {
                        // Correcting for Nyquist/Hermitian symmetry isn't necessary
                        // because we are padding, not cropping.
                        .correct_nyquist = false,
                    }
                );
                nf::c2r(tiles_large_padded_rfft, tiles_large_padded);
            }

            // Prefilter, if necessary.
            // TODO We could also prefilter before the oversampling.
            if (interp == nx::Interp::CUBIC_BSPLINE)
                nx::cubic_bspline_prefilter(tiles_large_padded, tiles_large_padded);

            // Backward project.
            const auto device = subvolume_large_padded.device();
            const auto bp = BackwardProjectionMaker{
                .tiles_large = tiles_large_padded.span().filter(0, 2, 3).as_contiguous(),
                .projection_matrices = tile_large_padded_matrices.span().subregion(z, y, x).as_1d(),
                .subvolume_large = subvolume_large_padded.span().filter(1, 2, 3).as_contiguous(),
            };
            if (interp == nx::Interp::CUBIC_BSPLINE) {
                noa::iwise(bp.subvolume_large.shape(), device, bp.get<nx::Interp::CUBIC_BSPLINE>());
            } else if (interp == nx::Interp::LINEAR) {
                noa::iwise(bp.subvolume_large.shape(), device, bp.get<nx::Interp::LINEAR>());
            } else {
                panic("Unsupported interpolation mode");
            }

            // Downsample, if necessary.
            if (actual_oversampling_factor > 1) {
                nf::r2c(subvolume_large_padded, subvolume_large_padded_rfft);
                nf::resize<"h">(
                    subvolume_large_padded_rfft, subvolume_large_padded.shape(),
                    subvolume_large_rfft, subvolume_large.shape(), {
                        // Correcting for Nyquist is necessary here. We don't bandpass and
                        // the Hermitian symmetry is likely broken from the cropping.
                        .correct_nyquist = true,
                    }
                );
                nf::c2r(subvolume_large_rfft, subvolume_large);
            }

            // Return a view of the subvolume (excluding the padding).
            const auto left_padding = subvolume_large_shape.vec / 2 - subvolume_shape.vec / 2;
            auto subvolume = subvolume_large.view().subregion(0,
                Slice{left_padding[0], left_padding[0] + subvolume_shape[0]},
                Slice{left_padding[1], left_padding[1] + subvolume_shape[1]},
                Slice{left_padding[2], left_padding[2] + subvolume_shape[2]}
            );
            return subvolume;
        }
    };
}
