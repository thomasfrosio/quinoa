#include "quinoa/Logger.hpp"
#include "quinoa/postprocessing/FilterStack.hpp"
#include "quinoa/postprocessing/Utilities.hpp"
#include "quinoa/postprocessing/BackwardProjection.hpp"
#include "quinoa/postprocessing/FourierInsertion.hpp"
#include "quinoa/postprocessing/Run.hpp"

namespace {
    struct FakeSIRT {
        static constexpr f32 ALPHA = 0.00195f;
        static constexpr f32 MATCH_ADD = 0.3f;
        f32 iter_use;
        f32 exponent;

        explicit FakeSIRT(i32 n_iterations) {
            iter_use = static_cast<f32>(n_iterations);
            if (n_iterations > 15)
                iter_use = 15.f + 0.8f * static_cast<f32>(n_iterations - 15);
            if (n_iterations > 30)
                iter_use = 27.f + 0.6f * static_cast<f32>(n_iterations - 30);
            exponent = iter_use + MATCH_ADD;
        }

        NOA_HD auto operator()(const Vec<f32, 2>& fftfreq_2d, isize) const -> f32 {
            const auto fftfreq = noa::sqrt(noa::dot(fftfreq_2d, fftfreq_2d));
            if (fftfreq <= ALPHA)
                return 1.0;
            return 1.f - noa::pow(1.f - ALPHA / fftfreq, exponent);
        }
    };

    template<typename T>
    void reconstruct_z_section(const T& reconstructor, const View<const f32>& stack, const View<f32>& z_section, isize z) {
        const auto [sy, sx] = reconstructor.subvolume_shape.filter(1, 2);
        const auto [ny, nx] = reconstructor.grid_shape.filter(1, 2);

        if (stack.device().is_cpu()) {
            // The subvolumes are relatively small, so to better distribute resources,
            // distribute subvolumes to threads directly, and set each subvolume to one thread.
            auto& stream = Stream::current({});
            auto n_threads = stream.thread_limit();
            stream.set_thread_limit(1);
            parallel_for(reconstructor.n_threads, Shape{ny, nx}, [&](isize tid, isize y, isize x) {
                const auto subvolume = reconstructor.reconstruct_subvolume(stack, z, y, x, tid);
                auto dst = z_section.subregion(
                    Ellipsis{},
                    Slice{y * sy, y * sy + sy},
                    Slice{x * sx, x * sx + sx}
                );
                auto src = subvolume.subregion(Ellipsis{}, Slice{0, dst.shape()[2]}, Slice{0, dst.shape()[3]});
                src.to(dst);
            });
            stream.set_thread_limit(n_threads);
        } else {
            // Store the subvolumes into the row buffer.
            // Once the row of subvolumes is computed, transfer to the host.
            for (isize y{}; y < ny; ++y) {
                for (isize x{}; x < nx; ++x) {
                    const auto subvolume = reconstructor.reconstruct_subvolume(stack, z, y, x);
                    const auto dst = reconstructor.subvolume_row.view().subregion(Ellipsis{}, Slice{x * sx, x * sx + sx});

                    auto src = subvolume.subregion(Ellipsis{}, Slice{0, dst.shape()[2]}, Slice{0, dst.shape()[3]});
                    src.to(dst);
                }
                auto dst = z_section.subregion(Ellipsis{}, Slice{y * sy, y * sy + sy}, Full{});
                reconstructor.subvolume_row.view().subregion(Ellipsis{}, Slice{0, dst.shape()[2]}, Full{}).to(dst);
            }
        }
    }
}

namespace qn {
    void save_stack(
        StackLoader& stack,
        const Path& filename,
        const Metadata::Stack& metadata,
        bool cache_loader,
        const Settings::PostProcessing::Stack& settings
    ) {
        auto timer = Logger::trace_scope_time("Saving stack");

        // Output buffer.
        const auto center = (stack.slice_shape().vec / 2).as<f64>();
        const auto output = Array<f32>(stack.slice_shape().push_front(Vec<isize, 2>{2, 1}), {
            .device = stack.compute_device(),
            .allocator = Allocator::MANAGED
        });
        const auto buffer_io = output.view().subregion(0);
        const auto buffer_xform = output.view().subregion(1);
        const auto buffer_rfft =
            settings.fake_sirt_iterations ?
            Array<c32>(stack.slice_shape().push_front<2>(1).rfft(), output.options()) : Array<c32>{};


        // Set up the output file.
        auto output_file = noa::io::ImageFile(filename, {.write = true}, {
            .shape = stack.slice_shape().push_front(Vec{metadata.ssize(), isize{1}}),
            .spacing = stack.stack_spacing().push_front(1),
            .dtype = settings.dtype,
        });

        // Slices will be saved in the same order as in the metadata.
        for (isize i{}; const auto& image: metadata) {
            const auto rotation = settings.correct_rotation ? noa::deg2rad(image.angles[0]) : 0;
            const auto shifts = settings.correct_shift ? Vec{0., 0.} : noa::deg2rad(image.shifts);
            const auto inverse_transform = (
                nx::translate(center + shifts) *
                nx::rotate<true>(-rotation) *
                nx::translate(-center - image.shifts)
            ).inverse().as<f32>();

            stack.read_slice(buffer_io, image.index_file, cache_loader);
            nx::transform_2d(buffer_io, buffer_xform, inverse_transform, {
                .interp = settings.interpolation,
                .border = noa::Border::ZERO,
            });

            if (settings.fake_sirt_iterations) {
                nf::r2c(buffer_xform, buffer_rfft);
                ns::filter_spectrum_2d<"h">(
                    buffer_rfft, buffer_rfft, buffer_xform.shape(),
                    FakeSIRT(settings.fake_sirt_iterations)
                );
                nf::c2r(buffer_rfft, buffer_xform);
            }

            output_file.write_slice(
                buffer_xform.reinterpret_as_cpu().span<const f32>(),
                {.bd_offset = {i++, 0}}
            );
        }
        Logger::trace("{} saved", filename);
    }

    auto reconstruct_tomogram(
        StackLoader&& stack,
        Metadata& metadata,
        const Settings::PostProcessing::Tomogram& settings
    ) -> Array<f32> {
        auto timer = Logger::info_scope_time("Reconstructing tomogram");

        auto run = [&]<typename T>() {
            const auto spacing = mean(stack.stack_spacing());
            const auto spacing_nm = spacing * 1e-1;
            const auto ctf = CTFIsotropic64({
                .pixel_size = spacing,
                .defocus = 0.,
                .voltage = metadata.sample.voltage,
                .amplitude = metadata.sample.amplitude,
                .cs = metadata.sample.cs,
                .phase_shift = 0,
                .bfactor = settings.ctf_bfactor,
                .scale = 1.,
            });

            const auto setup = reconstruction_thickness(
                spacing_nm, settings.ctf_defocus_step_nm,
                metadata.sample.thickness, settings.z_padding_percent
            );
            const auto image_shape = stack.slice_shape();
            const auto volume_shape = image_shape.push_front(setup.thickness);
            const auto device = stack.compute_device();

            // For the real-space backward projection, use the ramp filter to weight the projection.
            // For the fourier-space insertion, the ramp filter does not make sense since
            // the exact weights are applied directly during the insertion.
            constexpr bool IS_FOURIER = std::same_as<TiledFourierInsertion, T>;
            const bool use_ramp_filter = IS_FOURIER ? false : settings.real.ramp_filter;

            // Aligned the reconstruction is not rotated, always use the unrotated stack.
            const bool prealign_stack = settings.correct_rotation and
                IS_FOURIER ? settings.fourier.prealign_stack : settings.real.prealign_stack;
            const auto prealign_stack_interp =
                IS_FOURIER ? settings.fourier.prealign_stack_interpolation : settings.real.prealign_stack_interpolation;

            auto stack_filterer = StackFilterer{};
            if (settings.correct_ctf) {
                stack_filterer = StackFilterer(
                    std::move(stack), metadata.stack,
                    prealign_stack, prealign_stack_interp,
                    use_ramp_filter, settings.fake_sirt_iterations,
                    ctf, setup.thickness_nm, setup.z_step_nm, settings.ctf_phase_flip_strength
                );
            } else {
                stack_filterer = StackFilterer(
                    std::move(stack), metadata.stack,
                    prealign_stack, prealign_stack_interp,
                    use_ramp_filter, settings.fake_sirt_iterations
                );
            }

            const auto oversampling = IS_FOURIER ? settings.fourier.oversampling_factor : settings.real.oversampling_factor;
            const auto interpolation = IS_FOURIER ? settings.fourier.interpolation : settings.real.interpolation;
            const auto reconstructor = T(
                image_shape, volume_shape, metadata.stack, device,
                oversampling, settings.correct_rotation, interpolation,
                spacing_nm, setup.z_step_nm
            );

            Logger::trace(
                "Tiled reconstruction:\n"
                "  shape={}\n"
                "  spacing={:.1f}A (resolution={:.1f}A)\n"
                "  correct_rotation={}\n"
                "  thickness={:.1f}nm (specimen={:.1f}nm, z_padding={:.1f}nm|{:.1f}pix)\n"
                "  algorithm={} (interp={}, oversampling_factor={})\n"
                "  subvolume_shape={} (actual_subvolume_shape={})\n"
                "  grid_shape={} (n_subvolumes={})\n"
                "  device={}{}",
                volume_shape,
                spacing_nm * 10, spacing_nm * 20, settings.correct_rotation,
                setup.thickness_nm, metadata.sample.thickness, setup.z_padding * spacing_nm, setup.z_padding,
                settings.algorithm, reconstructor.interp, reconstructor.actual_oversampling_factor,
                reconstructor.subvolume_shape, reconstructor.subvolume_large_padded_shape,
                reconstructor.grid_shape.pop_back(), reconstructor.grid_shape.pop_back().n_elements(),
                device, device.is_cpu() ? fmt::format(" (n_threads={})", reconstructor.n_threads) : ""
            );

            if (device.is_gpu()) {
                nf::clear_cache(device);
                nf::set_cache_limit(12, device);
            }

            // Prepare for FFTs with various batch sizes.
            // In CPU, this only precomputes the plans and isn't necessary.
            // In CUDA, while this is also optional, it allows sharing the workspace
            // across all transforms, possibly saving a lot of memory.
            auto b0 = Allocator::bytes_currently_allocated(device);
            reconstructor.prepare_rffts();
            stack_filterer.prepare_irffts(); // allocates the workspace
            auto b1 = Allocator::bytes_currently_allocated(device);
            Logger::trace("FFT workspace allocated: {}={:.2f}GB", device, static_cast<f64>(b1 - b0) * 1e-9);

            // Reconstruct the (possibly CTF-corrected) tomogram.
            auto tomogram = Array<f32>(volume_shape.push_front(1));
            const auto sz = reconstructor.subvolume_shape[0];
            const auto nz = reconstructor.grid_shape[0];

            auto t = noa::Timer{};
            t.start();
            Logger::trace("Reconstructing z-sections");
            for (isize z{}; z < nz; ++z) {
                Logger::trace("z={:02}/{:02}", z + 1, nz);
                const auto filtered_stack = stack_filterer.compute_filtered_stack(z);
                const auto volume_z_section = tomogram.view().subregion(0, Slice{z * sz, z * sz + sz});
                reconstruct_z_section(reconstructor, filtered_stack.view(), volume_z_section, z);
            }
            Logger::trace("Average time per z-section: {}", t.elapsed() / nz);
            nf::clear_cache(device);
            return tomogram;
        };

        if (settings.algorithm == "fourier")
            return run.operator()<TiledFourierInsertion>();
        if (settings.algorithm == "real")
            return run.operator()<TiledBackwardProjection>();
        panic("Unknown reconstruction algorithm: {}", settings.algorithm);
    }

    void postprocess(
        const Path& input_stack,
        const Metadata& metadata,
        Device device,
        const Settings::PostProcessing& settings,
        const Path& output_directory
    ) {
        auto loader = StackLoader(input_stack, {
            .compute_device = device,
            .allocator = Allocator::DEFAULT_ASYNC,
            .precise_cutoff = true,
            .rescale_target_resolution = settings.resolution,
            .rescale_min_size = settings.min_size_pix,
            .rescale_max_size = settings.max_size_pix,
            .bandpass{
                // TODO note behavior when lowpass filtering
                .highpass_cutoff = settings.bandpass.highpass_cutoff,
                .highpass_width = settings.bandpass.highpass_width,
                .lowpass_cutoff = settings.bandpass.lowpass_cutoff,
                .lowpass_width = settings.bandpass.lowpass_width,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .exposure_filter_voltage = metadata.sample.voltage,
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.02,
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });

        const auto spacing = mean(loader.stack_spacing());
        const auto basename = input_stack.stem().string();

        auto meta = metadata;
        meta.set_spacing(spacing);
        meta.stack.sort("tilt").reset_indices();

        if (settings.stack.run) {
            const auto filename = output_directory / fmt::format("{}_stack.mrc", basename);
            const auto cache_loader = settings.tomogram.run; // stack is reloaded in reconstruct_tomogram
            save_stack(loader, filename, meta.stack, cache_loader, settings.stack);
        }

        if (settings.tomogram.run) {
            const auto filename = output_directory / fmt::format("{}_tomogram.mrc", basename);
            const auto tomogram = reconstruct_tomogram(std::move(loader), meta, settings.tomogram);
            noa::write_image(tomogram, filename, {
                .spacing = Vec<f64, 3>::from_value(spacing),
                .dtype = settings.tomogram.dtype,
            });
            Logger::trace("{} saved", filename);
        }
    }
}
