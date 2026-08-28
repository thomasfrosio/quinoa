#pragma once

#include <noa/Xform.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>

#include "quinoa/postprocessing/Utilities.hpp"

namespace qn {
    template<typename Input, typename Rotation, typename Output>
    struct FourierInsertion {
        Input tiles_large_padded_rfft;
        Rotation tile_rotations;
        Output subvolume_large_padded_rfft;

        f32 fftfreq_step;
        f32 fftfreq_sinc;
        f32 fftfreq_blackman;

        NOA_FHD auto volume2slice(const Vec<f32, 3>& fftfreq_3d, i32 slice_index) const {
            const auto fftfreq_3d_ = tile_rotations[slice_index].rotate(fftfreq_3d);
            return Pair{fftfreq_3d_[0], fftfreq_3d_.pop_front()};
        }

        NOA_HD void operator()(i32 oz, i32 oy, i32 ox) const noexcept {
            const auto frequency = nf::index2frequency<false, true>(Vec{oz, oy, ox}, subvolume_large_padded_rfft.shape());
            const auto fftfreq_3d = frequency.template as<f32>() * fftfreq_step;
            if (noa::dot(fftfreq_3d, fftfreq_3d) > 0.25f) {
                subvolume_large_padded_rfft(oz, oy, ox) = 0;
                return;
            }

            c32 value{};
            f32 weights{};
            for (i32 i{}; i < tile_rotations.shape()[0]; ++i) {
                const auto [fftfreq_z, fftfreq_2d] = volume2slice(fftfreq_3d, i);

                if (abs(fftfreq_z) <= fftfreq_blackman) { // the slice affects the voxel
                    const auto window = nx::details::windowed_sinc(fftfreq_z, fftfreq_sinc, fftfreq_blackman);
                    const auto frequency_2d = fftfreq_2d / fftfreq_step;
                    value += tiles_large_padded_rfft.interpolate_spectrum_at(frequency_2d, i) * window;
                    weights += window; // TODO should it be abs(window)?
                }
            }
            subvolume_large_padded_rfft(oz, oy, ox) = value / noa::max(noa::abs(weights), 1.f);
        }
    };

    struct FourierInsertionMaker {
        using input_span_t = SpanContiguous<const c32, 3, i32>;
        using output_span_t = SpanContiguous<c32, 3, i32>;
        using rotation_span_t = SpanContiguous<const nx::Quaternion<f32>, 1, i32>;

        input_span_t tiles_large_padded_rfft;
        Shape<i32, 2> tiles_large_padded_shape_2d;
        rotation_span_t tile_rotations;
        output_span_t subvolume_large_padded_rfft;

        f32 fftfreq_step;
        f32 fftfreq_sinc;
        f32 fftfreq_blackman;

        template<nx::Interp INTERP>
        [[nodiscard]] auto get() const {
            using interpolator_t = nx::InterpolatorSpectrum<2, nf::Layout::H2H, INTERP, input_span_t>;
            return FourierInsertion<interpolator_t, rotation_span_t, output_span_t>{
                .tiles_large_padded_rfft = interpolator_t(tiles_large_padded_rfft, tiles_large_padded_shape_2d),
                .tile_rotations = tile_rotations,
                .subvolume_large_padded_rfft = subvolume_large_padded_rfft,
                .fftfreq_step = fftfreq_step,
                .fftfreq_sinc = fftfreq_sinc,
                .fftfreq_blackman = fftfreq_blackman,
            };
        }
    };

    template<nx::Interp INTERP>
    struct FourierInterpolationCorrection {
        using volume_span_t = SpanContiguous<f32, 3, i32>;

        volume_span_t subvolume;
        Vec<f32, 3> subvolume_center;
        f32 fftfreq_step;

        NOA_HD void operator()(i32 z, i32 y, i32 x) const noexcept {
            // Shift spatial coordinates relative to the subvolume center
            const Vec<f32, 3> r = Vec<f32, 3>{static_cast<f32>(z), static_cast<f32>(y), static_cast<f32>(x)} - subvolume_center;

            // Compute 1D kernel response along each axis in real space
            const f32 wx = kernel_real_space(r[2]);
            const f32 wy = kernel_real_space(r[1]);
            const f32 wz = kernel_real_space(r[0]);

            // Separable 3D weight
            const f32 total_weight = wz * wy * wx;

            // Avoid division by zero/tiny values at extreme boundaries
            constexpr f32 min_weight = 1e-4f;
            if (noa::abs(total_weight) > min_weight) {
                subvolume(z, y, x) /= total_weight;
            }
        }

    private:
        NOA_HD static auto sinc(f32 x) noexcept -> f32 {
            if (noa::abs(x) < 1e-5f) return 1.0f;
            const f32 px = noa::Constant<f32>::PI * x;
            return noa::sin(px) / px;
        }

        NOA_HD auto kernel_real_space(f32 dist) const noexcept -> f32 {
            if constexpr (INTERP == nx::Interp::LINEAR) {
                // Linear interpolation in Fourier space corresponds to a sinc^2(x) envelope in real space.
                // Dist is normalized by the grid spacing step.
                const f32 x = dist * fftfreq_step;
                const f32 s = sinc(x);
                return s * s;
            }
            else if constexpr (INTERP == nx::Interp::LANCZOS6) {
                // Lanczos-a in Fourier space (a=6) maps to a convolution envelope in real space.
                // Analytical FT of a windowed sinc kernel (Lanczos 6):
                // Approximated by the main lobe response w(r) = sinc(r / a_scale)
                // For Lanczos 6, kernel half-width in Fourier space is 6 * fftfreq_step.
                const f32 x = dist * fftfreq_step;
                const f32 a = 6.0f;

                // Standard Fourier transform envelope of Lanczos kernel
                const f32 s_main = sinc(x);
                const f32 s_window = sinc(x / a);
                return s_main * s_window;
            }
        }
    };

    class TiledFourierInsertion {
    public:
        nx::Interp interp;
        i32 n_threads{};

        isize actual_oversampling_factor{};
        isize tile_large_padded_size{};
        f64 tile_large_center{};
        Shape3 subvolume_shape{};
        Vec<f64, 3> subvolume_center{};
        Shape3 subvolume_large_padded_shape{};

        Shape4 grid_shape{};

        Array<Vec<i32, 2>> tile_large_origins;
        Array<nx::Quaternion<f32>> tile_rotations;
        Array<Vec<f32, 2>> tile_large_shifts;

        Array<f32> subvolume_row;
        Array<c32> tiles_large_padded_buffer;
        Array<c32> subvolume_large_padded_buffer;

    public:
        TiledFourierInsertion() = default;
        TiledFourierInsertion(
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

            // 1. Similar to the real-space reconstruction, the reconstruction is divided into subvolumes and
            // to backproject a subvolume, the input tiles should be twice as large as the subvolume.
            //
            // 2. Then, the twice-enlarged tiles are zero-tapered and zero-padded to match the desired oversampling
            // factor. This zero-padding is done on the right-side to simplify the phase-shifts, and a minimum of 3x
            // oversampling is enforced (3x the subvolume size) to ensure that the edges of the enlarged tiles don't
            // warp around when rotating in Fourier space.
            const auto tile_size = nf::next_fast_size(std::max(z_step, isize{64}));
            const auto tile_large_size = tile_size * 2;
            actual_oversampling_factor = std::max(isize{3}, oversampling_factor);
            tile_large_padded_size = tile_size * actual_oversampling_factor;
            subvolume_large_padded_shape = Shape3::from_value(tile_large_padded_size);

            // Importantly, while we reconstruct twice-enlarged zero-padded cubes, we center them in z
            // so that the center of the cubes matches the original subvolume z-center, aka z_step/2.
            subvolume_shape = Shape{z_step, tile_size, tile_size};
            subvolume_center = (subvolume_shape.vec / 2).as<f64>();

            grid_shape = Shape{
                n_sections,
                noa::divide_up(image_shape[0], tile_size),
                noa::divide_up(image_shape[1], tile_size),
                metadata.ssize()
            };

            // Compute the rotation of the central-slices.
            tile_rotations = Array<nx::Quaternion<f32>>(grid_shape[3]);
            for (auto&& [image, rotation]: noa::zip(metadata, tile_rotations.span_1d())) {
                const auto angles = noa::deg2rad(image.angles);
                const auto final_rotation = correct_rotation ? 0. : angles[0];
                rotation = nx::matrix2quaternion( // volume->slice
                    nx::rotate_z(angles[0]) *
                    nx::rotate_y(angles[1]) *
                    nx::rotate_x(angles[2]) *
                    nx::rotate_z(-final_rotation)
                ).as<f32>();
            }

            // Compute the shifts for every subvolume.
            tile_large_center = static_cast<f64>(tile_large_size / 2);
            tile_large_origins = Array<Vec<i32, 2>>(grid_shape);
            tile_large_shifts = Array<Vec<f32, 2>>(grid_shape);

            const auto volume2image = volume2image_matrices(metadata, correct_rotation, image_shape, volume_shape);
            const auto volume2image_1d = volume2image.span_1d();

            for (isize z{}; z < grid_shape[0]; ++z) {
                for (isize y{}; y < grid_shape[1]; ++y) {
                    for (isize x{}; x < grid_shape[2]; ++x) {
                        const auto subvolume_origin = Vec{z, y, x} * subvolume_shape.vec;
                        const auto subvolume_center_coordinates = subvolume_origin.as<f64>() + subvolume_center;

                        for (isize t{}; t < grid_shape[3]; ++t) {
                            const auto& [tile_padded_origin, tile_residual_shift] = extract_tile_large_window(
                                tile_large_center, subvolume_center_coordinates, volume2image_1d[t]);

                            tile_large_origins.span()(z, y, x, t) = tile_padded_origin;
                            tile_large_shifts.span()(z, y, x, t) = -(tile_large_center + tile_residual_shift).as<f32>();
                        }
                    }
                }
            }
            if (device.is_gpu()) {
                const auto options_async = ArrayOption{.device = device, .allocator = Allocator::ASYNC};
                tile_large_origins = std::move(tile_large_origins).to(options_async);
                tile_large_shifts = std::move(tile_large_shifts).to(options_async);
                tile_rotations = std::move(tile_rotations).to(options_async);
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
            const auto tile_large_padded_shape = Shape2::from_value(tile_large_padded_size);
            const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};

            tiles_large_padded_buffer = Array<c32>(tile_large_padded_shape.rfft().push_front(bd), options);
            subvolume_large_padded_buffer = Array<c32>(subvolume_large_padded_shape.rfft().push_front(n_threads), options);
            if (device.is_gpu())
                subvolume_row = Array<f32>(subvolume_shape.set<2>(volume_shape[2]).push_front(1), options);
        }

        [[nodiscard]] auto tiles_large_padded_pair(isize tid) const {
            const auto& ts = tile_large_padded_size;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = tiles_large_padded_buffer.view().subregion(tid).permute({1, 0, 2, 3});
            pair.first = nf::alias_to_real(pair.second, Shape{grid_shape[3], isize{1}, ts, ts});
            return pair;
        }

        [[nodiscard]] auto subvolume_large_padded_pair(isize tid) const {
            const auto& ts = tile_large_padded_size;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = subvolume_large_padded_buffer.view().subregion(tid);
            pair.first = nf::alias_to_real(pair.second, Shape{isize{1}, ts, ts, ts});
            return pair;
        }

        void prepare_rffts() const {
            auto [tiles_large_padded, tiles_large_padded_rfft] = tiles_large_padded_pair(0);
            auto [subvolume_large_padded, subvolume_large_padded_rfft] = subvolume_large_padded_pair(0);

            nf::r2c(tiles_large_padded, tiles_large_padded_rfft, {.record_and_share_workspace = true});
            nf::c2r(subvolume_large_padded_rfft, subvolume_large_padded, {.record_and_share_workspace = true});
        }

        NOA_NOINLINE auto reconstruct_subvolume(
            const View<const f32>& input_stack,
            isize z, isize y, isize x, isize tid = 0
        ) const -> View<f32> {
            const auto [tiles_large_padded, tiles_large_padded_rfft] = tiles_large_padded_pair(tid);
            const auto [subvolume_large_padded, subvolume_large_padded_rfft] = subvolume_large_padded_pair(tid);
            const auto device = tiles_large_padded.device();

            // Extract and taper the padded tiles.
            // For the Fourier reconstruction, the tiles are padded by twice for the backprojection
            // and an additional zero-padding is done on the right for the interpolation in Fourier space.
            const auto output_tiles = tiles_large_padded.span_contiguous<f32, 3>();
            noa::iwise(output_tiles.shape(), device, ExtractPaddedTiles{
                .images = input_stack.span_contiguous<const f32, 3>(),
                .output_tiles = output_tiles,
                .tile_large_origins = tile_large_origins.span().subregion(z, y, x).as_1d(),
                .tile_large_center = static_cast<isize>(tile_large_center),
                .taper_radius = static_cast<f32>(tile_large_center * 0.95),
                .taper_smoothness = static_cast<f32>(tile_large_center * 0.05), // TODO is it too sharp?
            });

            // Compute the rotation-centered central-slices.
            nf::r2c(tiles_large_padded, tiles_large_padded_rfft);
            ns::phase_shift_2d<"h">(
                tiles_large_padded_rfft, tiles_large_padded_rfft,
                tiles_large_padded.shape(), tile_large_shifts.view().subregion(z, y, x)
            );

            // Insert the central-slices.
            const auto iwise_shape = subvolume_large_padded_rfft.shape().pop_front().as<i32>();
            const auto fi = FourierInsertionMaker{
                .tiles_large_padded_rfft = tiles_large_padded_rfft.span_contiguous<const c32, 3, i32>(),
                .tiles_large_padded_shape_2d = tiles_large_padded.shape().filter(2, 3).as<i32>(),
                .tile_rotations = tile_rotations.span_1d().as_index<i32>(),
                .subvolume_large_padded_rfft = subvolume_large_padded_rfft.span_contiguous<c32, 3, i32>(),
                .fftfreq_step = 1 / static_cast<f32>(tile_large_padded_size),
                .fftfreq_sinc = 1 / static_cast<f32>(tile_large_padded_size),
                .fftfreq_blackman = 8 / static_cast<f32>(tile_large_padded_size),
            };
            if (interp == nx::Interp::LINEAR) {
                noa::iwise(iwise_shape, device, fi.get<nx::Interp::LINEAR>());
            } else if (interp == nx::Interp::LANCZOS6) {
                noa::iwise(iwise_shape, device, fi.get<nx::Interp::LANCZOS6>());
            } else {
                panic("Unsupported interpolation mode");
            }

            // Compute the reconstructed subvolume.
            ns::phase_shift_3d<"h">(
                subvolume_large_padded_rfft, subvolume_large_padded_rfft,
                subvolume_large_padded.shape(), subvolume_center.as<f32>()
            );
            nf::c2r(subvolume_large_padded_rfft, subvolume_large_padded);

            // TODO interpolation correction
            // Apply real-space deconvolution correction before cropping
            // nx::fourier_interpolation_correction(subvolume_large_padded, subvolume_large_padded, interp, true);
            // if (interp == nx::Interp::LINEAR) {
            //     using deconv_t = FourierInterpolationCorrection<nx::Interp::LINEAR>;
            //     noa::iwise(subvolume_large_padded.shape().pop_front().as<i32>(), device, deconv_t{
            //         .subvolume = subvolume_large_padded.span_contiguous<f32, 3, i32>(),
            //         .subvolume_center = subvolume_center.as<f32>(),
            //         .fftfreq_step = 1.0f / static_cast<f32>(tile_large_padded_size),
            //     });
            // } else if (interp == nx::Interp::LANCZOS6) {
            //     using deconv_t = FourierInterpolationCorrection<nx::Interp::LANCZOS6>;
            //     noa::iwise(subvolume_large_padded.shape().pop_front().as<i32>(), device, deconv_t{
            //         .subvolume = subvolume_large_padded.span_contiguous<f32, 3, i32>(),
            //         .subvolume_center = subvolume_center.as<f32>(),
            //         .fftfreq_step = 1.0f / static_cast<f32>(tile_large_padded_size),
            //     });
            // }

            // Return a view of the subvolume.
            auto subvolume = subvolume_large_padded.view().subregion(0,
                Slice{0, subvolume_shape[0]},
                Slice{0, subvolume_shape[1]},
                Slice{0, subvolume_shape[2]}
            );
            return subvolume;
        }
    };
}
