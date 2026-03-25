#include <noa/Runtime.hpp>
#include <noa/FFT.hpp>
#include <noa/Xform.hpp>

#include "quinoa/Logger.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Utilities.hpp"

namespace {
    using namespace noa::types;

    /// Fourier cropping parameters.
    /// \details Fourier cropping at the exact target spacing isn't always possible because we can only crop at an
    ///          integer multiple of the input's fftfreq. However, we can pad the real-space input, effectively
    ///          stretching its spectrum, to minimize the difference between the target fftfreq and an integer multiple
    ///          of the stretched input's fftfreq. Note that the stretching is centered on the origin (0,0) of the
    ///          input, which may lead to a small shift between the input and the Fourier cropped output. As such,
    ///          we also return this shift for the caller to apply to keep the centers aligned.
    struct FourierCropDimensions {
        /// Shape of the padded input. If no padding was required, padded_shape == input_shape.
        Shape2 padded_shape;

        /// Logical shape of the Fourier-cropped spectrum.
        Shape2 cropped_shape;

        /// Actual spacing after Fourier cropping the (possibly padded) input.
        Vec<f64, 2> cropped_spacing;

        /// Shifts to add to the Fourier-cropped output to keep its (real-space) center aligned with the input.
        /// If no padding was required (padded_shape == input_shape), this is zero.
        Vec<f64, 2> rescale_shifts;
    };

    /// Computes the dimensions for Fourier cropping.
    /// \param current_shape            HW shape of the input.
    /// \param current_spacing          HW spacing of the input.
    /// \param target_spacing           Desired HW spacing.
    ///                                 If it is less than to the current_spacing, it is clamped to the current_spacing,
    ///                                 effectively cancelling the Fourier cropping. In other words, Fourier padding
    ///                                 is never allowed.
    /// \param maximum_relative_error   Tolerable error between the target_spacing and the output spacing.
    ///                                 Using a large error (e.g. 0.2 or larger) effectively disallowing to
    ///                                 pad the input. Note that in this case and for non-squared input shapes,
    ///                                 it is possible for the target spacing to become anisotropic.
    /// \param target_min_size          Minimum tolerable size. It is used to ensure a minimum output.cropped_shape.
    /// \param target_max_size          Maximum tolerable size. It is used to ensure a maximum output.cropped_shape.
    auto fourier_crop_dimensions(
        Shape2 current_shape,
        Vec<f64, 2> current_spacing,
        Vec<f64, 2> target_spacing,
        f64 maximum_relative_error = 5e-4,
        isize target_min_size = 0,
        isize target_max_size = 0
    ) -> FourierCropDimensions {
        check(current_spacing > 0 and target_spacing >= 0);

        // Disallow Fourier padding.
        // Note that if the current spacing is anisotropic, the target is set to be isotropic, since it is
        // often simpler to handle and the caller might expect the output spacing to be isotropic too.
        if (current_spacing.any_gt(target_spacing))
            target_spacing = noa::max(current_spacing);

        // Clamp the target spacing to the maximum spacing corresponding to the minimum allowed size.
        if (target_min_size > 0 and target_min_size < min(current_shape)) {
            const auto target_max_spacing = noa::min(
                current_spacing * current_shape.vec.as<f64>() / static_cast<f64>(target_min_size));
            if (target_spacing.any_gt(target_max_spacing))
                target_spacing = target_max_spacing;
        }

        // Clamp the target spacing to the minimum spacing corresponding to the maximum allowed size.
        if (target_max_size > 0 and target_max_size < min(current_shape)) {
            const auto target_min_spacing = noa::max(
                current_spacing * current_shape.vec.as<f64>() / static_cast<f64>(target_max_size));
            if (target_spacing.any_lt(target_min_spacing))
                target_spacing = target_min_spacing;
        }

        // Possibly pad in real space to place the frequency cutoff at a particular index of the spectrum.
        // This is necessary to be able to precisely crop at a frequency cutoff, and offers a way to keep
        // the target spacing isotropic within a maximum_relative_error.
        auto pad_to_align_cutoff = [maximum_relative_error](i64 i_size, f64 i_spacing, f64 o_spacing) -> i64
        {
            const isize maximum_size = i_size + 256; // in all tested cases, we stop way before that (~0 to 6)
            isize best_size = i_size;
            f64 best_error = std::numeric_limits<f64>::max();
            while (i_size < maximum_size) {
                const auto new_size = std::round(static_cast<f64>(i_size) * i_spacing / o_spacing);
                const auto new_spacing = i_spacing * static_cast<f64>(i_size) / new_size;
                const auto relative_error = std::abs(new_spacing - o_spacing) / o_spacing;

                if (relative_error < maximum_relative_error) {
                    // We found a good enough solution.
                    best_size = i_size;
                    break;
                } else if (relative_error < best_error) {
                    // We found a better solution.
                    best_error = relative_error;
                    best_size = i_size;
                }
                // Try again with a larger size. Since this padded size is likely to be FFTed,
                // keep it even sized. We could go to the next fast fft size, but this often ends
                // up padding large amounts, for little performance benefits vs. memory usage.
                i_size += 1 + noa::is_even(i_size);
            }
            return best_size;
        };
        current_shape[0] = pad_to_align_cutoff(current_shape[0], current_spacing[0], target_spacing[0]);
        current_shape[1] = pad_to_align_cutoff(current_shape[1], current_spacing[1], target_spacing[1]);

        // Get Fourier cropped shape.
        const auto current_shape_f64 = current_shape.vec.as<f64>();
        auto new_shape_f64 = current_shape_f64 * current_spacing / target_spacing;

        // Round to the nearest integer (this is where we crop).
        // We'll need to recompute the actual frequency after rounding, but of course,
        // this new frequency should be within a "maximum_relative_error" from the target spacing.
        new_shape_f64 = noa::round(new_shape_f64);
        const auto new_shape = Shape{new_shape_f64.as<isize>()};
        const auto new_spacing = current_spacing * current_shape_f64 / new_shape_f64;

        // To preserve the image center, we may need to shift the Fourier-cropped image.
        const auto current_center = (current_shape / 2).vec.as<f64>();
        const auto new_center = (new_shape / 2).vec.as<f64>();
        const auto current_center_rescaled = current_center * (current_spacing / new_spacing);
        const auto shift_to_add = new_center - current_center_rescaled;

        return {
            .padded_shape = current_shape,
            .cropped_shape = new_shape,
            .cropped_spacing = new_spacing,
            .rescale_shifts = shift_to_add,
        };
    }

    struct Filter {
        SpanContiguous<c32, 2, i32> spectrum;
        Shape<i32, 2> logical_shape;

        f32 highpass_cutoff;
        f32 highpass_width;
        f32 lowpass_cutoff;
        f32 lowpass_width;

        f32 exposure;
        f32 spacing;
        f32 k;

        constexpr void operator()(i32 u, i32 v) const {
            const auto frequency = nf::index2frequency<false, true>(Vec{u, v}, logical_shape);
            const auto fftfreq_2d = frequency.as<f32>() / logical_shape.vec.as<f32>();
            const auto fftfreq = sqrt(dot(fftfreq_2d, fftfreq_2d));

            auto filter = f32{1};

            // Bandpass.
            constexpr auto PI = noa::Constant<f32>::PI;
            filter *=
                fftfreq <= lowpass_cutoff ? 1 :
                lowpass_cutoff + lowpass_width <= fftfreq ? 0 :
                (1.f + cos(PI * (lowpass_cutoff - fftfreq) / lowpass_width)) * 0.5f;
            filter *=
                highpass_cutoff <= fftfreq ? 1 :
                fftfreq <= highpass_cutoff - highpass_width ? 0 :
                (1.f + cos(PI * (fftfreq - highpass_cutoff) / highpass_width)) * 0.5f;

            // Exposure filter.
            if (exposure > 0) {
                // DOI:10.7554/eLife.06980
                constexpr f32 A = +0.24499f;
                constexpr f32 B = -1.6649f;
                constexpr f32 C = +2.8141f;
                const f32 c0 = k * (A * pow(fftfreq * fftfreq / spacing, B) + C);
                filter *= exp(-0.5f * exposure / c0);
            }

            spectrum(u, v) *= filter;
        }
    };
}

namespace qn {
    Array<std::byte> StackLoader::s_input_stack{};
    noa::io::DataType StackLoader::s_input_stack_dtype{};

    void StackLoader::register_input_stack(const Path& filename) {
        auto timer = Logger::info_scope_time("Loading and decoding the input stack");

        using namespace noa::io;
        auto file = ImageFile(filename, {.read = true});
        const auto file_dtype = file.dtype();
        s_input_stack_dtype = file_dtype.closest_static_type();

        const auto file_shape = file.shape();
        const auto n_elements = file_shape.n_elements();
        const auto encoded_size = static_cast<f32>(file_dtype.n_bytes(n_elements));
        const auto decoded_size = static_cast<f32>(s_input_stack_dtype.n_bytes(n_elements));

        // Adding more threads is only useful when the file is compressed. Without compression, we're just
        // waiting for the filesystem, and having multiple threads in the mix seems to make it worse.
        const bool is_compressed = file.is_compressed();
        const i32 n_threads = is_compressed ? 4 : 1;

        Logger::trace(
            "Stack registry:\n"
            "  path={}\n"
            "  shape={} (compressed={})\n"
            "  dtype={}->{}\n"
            "  size={:.2f}GB->{:.2f}GB\n"
            "  n_threads={}",
            filename, file_shape, is_compressed,
            file_dtype, s_input_stack_dtype,
            encoded_size / 1e9f, decoded_size / 1e9f,
            n_threads
        );

        const auto type_erased_shape = file_shape.set<3>(file_shape[3] * s_input_stack_dtype.n_bytes(1));
        s_input_stack = Array<std::byte>(type_erased_shape);
        file.read_all(s_input_stack.span_1d(), s_input_stack_dtype, {.n_threads = n_threads});

        // Some files are not encoded correctly; reinterpret a volume as a stack of images.
        if (s_input_stack.shape()[0] == 1 and s_input_stack.shape()[1] > 1) {
            Logger::trace("Input stack encoded as a 3d volume... reinterpreting it to a stack of 2d images");
            s_input_stack = std::move(s_input_stack).permute({1, 0, 2, 3});
        }
    }

    StackLoader::StackLoader(ni::ImageFile&& file, const LoadStackParameters& parameters)
        : m_file(std::move(file)), m_parameters(parameters) { init_(); }

    StackLoader::StackLoader(const Path& filename, const LoadStackParameters& parameters) : m_parameters(parameters) {
        m_file.open(filename, {.read = true});
        init_();
    }

    void StackLoader::record_fft() const {
        const bool needs_mirror_pad = not m_bandpass_slice_rfft.is_empty();
        if (m_has_cropping or (m_has_filter and not needs_mirror_pad)) {
            const auto [padded_slice, padded_slice_rfft] = padded_slice_();
            const auto [cropped_slice, cropped_slice_rfft] = cropped_slice_();
            nf::r2c(padded_slice, padded_slice_rfft, {.record_and_share_workspace = true});
            nf::c2r(cropped_slice_rfft, cropped_slice, {.record_and_share_workspace = true});
        }
        if (m_has_filter and needs_mirror_pad) {
            const auto [bandpass_slice, bandpass_slice_rfft] = bandpass_slice_();
            nf::r2c(bandpass_slice, bandpass_slice_rfft, {.record_and_share_workspace = true});
            nf::c2r(bandpass_slice_rfft, bandpass_slice, {.record_and_share_workspace = true});
        }
    }

    void StackLoader::read_slice(const View<f32>& output_slice, isize file_slice_index, bool cache, f64 exposure) {
        check(output_slice.device() == compute_device());
        check(output_slice.shape() == m_output_slice_shape.push_front<2>(1));
        check(file_slice_index < m_file_slice_count,
              "Slice index is invalid. This happened because the file and the metadata don't match. "
              "Trying to access slice index {}, but the file stack has a total of {} slices",
              file_slice_index, m_file_slice_count);

        // Use the cached slice if it exists.
        for (const auto& [index, buffer]: m_cache) {
            if (index == file_slice_index) {
                buffer.to(output_slice);
                return;
            }
        }

        const bool needs_mirror_pad = not m_bandpass_slice_rfft.is_empty();
        const bool needs_smooth_edge = m_parameters.smooth_edge_percent > 0;
        const bool needs_final_zero_pad = m_cropped_slice_shape != m_output_slice_shape;
        const bool needs_normalized = m_parameters.normalize_and_standardize;
        const bool has_preprocessing =
            m_has_cropping or m_has_filter or needs_smooth_edge or
            needs_final_zero_pad or needs_normalized;

        // Bandpass and exposure filter.
        // The exposure filter is essentially a very smooth lowpass that increases with the exposure.
        // For 10A resolution, the filter is ~0.24 at Nyquist.
        auto filter = Filter{
            .spectrum = {}, // initialize below
            .logical_shape = {}, // initialize below
            .highpass_cutoff = static_cast<f32>(m_parameters.bandpass.highpass_cutoff),
            .highpass_width = static_cast<f32>(m_parameters.bandpass.highpass_width),
            .lowpass_cutoff = static_cast<f32>(m_parameters.bandpass.lowpass_cutoff),
            .lowpass_width = static_cast<f32>(m_parameters.bandpass.lowpass_width),
            .exposure = static_cast<f32>(exposure),
            .spacing = static_cast<f32>(mean(m_output_spacing)), // isotropic, small deviations would have any significance
            .k =
                noa::allclose(m_parameters.exposure_filter_voltage, 300.) ? 1 :
                noa::allclose(m_parameters.exposure_filter_voltage, 200.) ? 0.8f :
                noa::allclose(m_parameters.exposure_filter_voltage, 120.) ? 0.45f :
                1,
        };

        //
        auto [input_slice, input_slice_rfft] = input_slice_();
        auto [padded_slice, padded_slice_rfft] = padded_slice_();
        auto [cropped_slice, cropped_slice_rfft] = cropped_slice_();

        // Synchronize to not overwrite the io buffer in GPU case.
        input_slice.eval();

        // Read the slice.
        // We use an intermediary buffer, creating an extra copy, but this is to keep things contiguous
        // when reading from the file or register, and to not have to rely on unified memory.
        if (not s_input_stack.is_empty())
            ni::cast(s_input_stack.view().subregion(file_slice_index), s_input_stack_dtype, m_io_slice.view());
        else
            m_file.read_slice(m_io_slice.span(), {.bd_offset = {file_slice_index, 0}, .clamp = false});

        // If no preprocessing, we can copy directly to the output slice and return;
        m_io_slice.view().to(has_preprocessing ? input_slice : output_slice);

        // Optional padding of the input slice for accurate Fourier cropping cutoff.
        if (m_has_padding) {
            const auto padding_right = (padded_slice.shape() - input_slice.shape()).vec;
            noa::resize(input_slice, padded_slice, {}, padding_right, noa::Border::REFLECT);
        }

        // Fourier cropping and/or filtering if there's no mirror-padding needed for the highpass.
        if (m_has_cropping or (m_has_filter and not needs_mirror_pad)) {
            nf::r2c(padded_slice, padded_slice_rfft);
            if (m_has_cropping) {
                nf::resize<"h">(
                    padded_slice_rfft, padded_slice.shape(),
                    cropped_slice_rfft, cropped_slice.shape());
                ns::phase_shift_2d<"h">(
                    cropped_slice_rfft, cropped_slice_rfft,
                    cropped_slice.shape(), m_rescale_shift.as<f32>());
            }
            if (m_has_filter and not needs_mirror_pad) {
                // No mirror-padding was required for the highpass, so we can filter here.
                const auto iwise_shape = cropped_slice.shape().filter(2, 3).as<i32>();
                filter.spectrum = cropped_slice_rfft.span_contiguous<c32, 2, i32>();
                filter.logical_shape = iwise_shape;
                noa::iwise(iwise_shape.rfft(), cropped_slice_rfft.device(), filter);
            }
            nf::c2r(cropped_slice_rfft, cropped_slice);
        }

        // Optimize resizes and transfers as much as possible.
        const bool direct_bandpass_to_output =
            (needs_mirror_pad and needs_smooth_edge) or
            (needs_mirror_pad and not needs_final_zero_pad);
        const bool direct_taper_to_output =
            not needs_mirror_pad and needs_smooth_edge and not needs_final_zero_pad;

        // Highpass filtering is very likely to benefit from mirror padding the image first. Indeed, not padding the
        // images creates edge artifacts, which can end up dominating the cross-correlation function. In principle,
        // x2 mirror padding would give close to ideal results since we make the input cyclic and edges are effectively
        // removed. In practice, with smooth passes, similar results can be achieved with just a fraction of that
        // (10%-50%). However, with sharper passes, I would recommend at least 50%. Since this depends on the bandpass
        // and affects (runtime) performance, this was made a parameter we can change depending on the context.
        if (m_has_filter and needs_mirror_pad) {
            auto [bandpass_slice, bandpass_slice_rfft] = bandpass_slice_();
            noa::resize(cropped_slice, bandpass_slice, noa::Border::REFLECT);
            nf::r2c(bandpass_slice, bandpass_slice_rfft);

            const auto iwise_shape = bandpass_slice.shape().filter(2, 3).as<i32>();
            filter.spectrum = bandpass_slice_rfft.span_contiguous<c32, 2, i32>();
            filter.logical_shape = iwise_shape;
            noa::iwise(iwise_shape.rfft(), bandpass_slice_rfft.device(), filter);

            nf::c2r(bandpass_slice_rfft, bandpass_slice);
            noa::resize(bandpass_slice, direct_bandpass_to_output ? output_slice : cropped_slice);
        }

        // Smooth edges to zero.
        // We assume there's always at least a small highpass that sets the mean to zero and
        // removes the large contrast gradients. Otherwise, this mask will not look good.
        if (needs_smooth_edge) {
            const auto untapered_slice = direct_bandpass_to_output ? output_slice : cropped_slice;
            const auto tapered_slice = direct_taper_to_output ? output_slice : untapered_slice;

            const auto center = (untapered_slice.shape().filter(2, 3).vec / 2).as<f64>();
            const auto radius = (cropped_slice.shape().filter(2, 3).vec / 2).as<f64>();
            const auto smooth_edge_size =
                static_cast<f64>(noa::max(cropped_slice.shape().filter(2, 3))) *
                m_parameters.smooth_edge_percent;

            nx::draw(untapered_slice, tapered_slice, nx::Rectangle{
                .center = center,
                .radius = radius - smooth_edge_size,
                .smoothness = smooth_edge_size,
            }.draw());
        }

        // Copy to the output slice, or final zero-padding to the output shape.
        if (has_preprocessing and (not direct_bandpass_to_output and not direct_taper_to_output))
            noa::resize(cropped_slice, output_slice);

        // Final normalization (mean=0, stddev=1).
        // If a highpass is applied, the mean should be close to zero at this point,
        // but the zero-taper and zero-padding can slightly offset the mean.
        if (needs_normalized)
            noa::normalize(output_slice, output_slice, {.mode = noa::Norm::MEAN_STD});

        // Cache the output. Since we only get a view, make sure to synchronize before quitting (.eval())
        // in case the caller gets rid of the corresponding array while the copy runs asynchronously.
        if (cache)
            m_cache.emplace_back(file_slice_index, output_slice.to_cpu());
    }

    void StackLoader::init_() {
        auto file_shape = m_file.shape();
        if (file_shape[0] == 1 and file_shape[1] > 1) {
            Logger::warn(
                "{}. A tilt-series was expected, but the image file encodes a volume. To continue, we will assume "
                "the file metadata is not encoded properly and will interpret this volume as a stack of 2d images",
                m_file.path()
            );
            std::swap(file_shape[0], file_shape[1]);
        }
        check(file_shape[1] == 1, "{}. A tilt-series was expected, but got image file with shape {}", m_file.path(), file_shape);
        m_file_slice_count = file_shape[0];
        m_input_slice_shape = file_shape.filter(2, 3);

        // Fourier cropping parameters.
        m_input_spacing = m_file.spacing().pop_front().as<f64>();
        const auto target_spacing = Vec<f64, 2>::from_value(m_parameters.rescale_target_resolution / 2);
        const auto relative_freq_error = m_parameters.precise_cutoff ? 2.5e-4 : 1.;
        const auto fourier_crop = fourier_crop_dimensions(
            m_input_slice_shape, m_input_spacing,
            target_spacing, relative_freq_error,
            m_parameters.rescale_min_size, m_parameters.rescale_max_size
        );
        m_padded_slice_shape = fourier_crop.padded_shape;
        m_cropped_slice_shape = fourier_crop.cropped_shape;
        m_output_spacing = fourier_crop.cropped_spacing;
        m_rescale_shift = fourier_crop.rescale_shifts;

        // Bypass flags.
        m_has_padding = m_input_slice_shape != m_padded_slice_shape;
        m_has_cropping = m_padded_slice_shape != m_cropped_slice_shape;
        m_has_filter =
            m_parameters.exposure_filter_voltage > 0 or
            m_parameters.bandpass.highpass_cutoff > 0 or
            m_parameters.bandpass.lowpass_cutoff > 0;

        const auto options = ArrayOption(m_parameters.compute_device, m_parameters.allocator);
        const auto bytes_before = Allocator::bytes_currently_allocated(options.device);

        m_io_slice = Array<f32>(m_input_slice_shape.push_front<2>(1));
        m_input_slice_rfft = Array<c32>(m_input_slice_shape.push_front<2>(1).rfft(), options);
        m_padded_slice_rfft = m_has_padding ? Array<c32>(m_padded_slice_shape.push_front<2>(1).rfft(), options) : m_input_slice_rfft;
        m_cropped_slice_rfft = m_has_cropping ? Array<c32>(m_cropped_slice_shape.push_front<2>(1).rfft(), options) : m_padded_slice_rfft;

        // Mirror padding for bandpass.
        m_bandpass_slice_shape = m_cropped_slice_shape;
        if (m_has_filter and m_parameters.bandpass_mirror_padding_factor > 0) {
            const auto padding = m_bandpass_slice_shape.vec.as<f64>() * m_parameters.bandpass_mirror_padding_factor;
            m_bandpass_slice_shape += Shape{noa::round(padding).as<isize>()};
            m_bandpass_slice_shape = nf::next_fast_shape(m_bandpass_slice_shape);
            // m_bandpass_slice_shape = noa::max(2 * m_cropped_slice_shape, nf::next_fast_shape(m_bandpass_slice_shape)); // FIXME
            m_bandpass_slice_rfft = Array<c32>(m_bandpass_slice_shape.push_front<2>(1).rfft(), options);
        }

        // Final zero-padding.
        m_output_slice_shape = m_cropped_slice_shape;
        if (m_parameters.zero_pad_to_square_shape)
            m_output_slice_shape = noa::max(m_output_slice_shape);
        if (m_parameters.zero_pad_to_fast_fft_shape) {
            m_output_slice_shape[0] = noa::fft::next_fast_size(m_output_slice_shape[0]);
            m_output_slice_shape[1] = noa::fft::next_fast_size(m_output_slice_shape[1]);
        }

        // To save as much memory as possible, share the FFT workspace.
        if (m_parameters.allocate_fft_workspace and options.device.is_gpu()) {
            record_fft();
            if (const auto workspace_size = nf::workspace_left_to_allocate(options.device)) {
                const auto n_plans_set = nf::set_workspace(options.device, Array<std::byte>(workspace_size, options));
                if (n_plans_set == 0)
                    Logger::warn("FFT workspace couldn't be set, please report this");
            }
        }

        const auto bytes_after = Allocator::bytes_currently_allocated(options.device);
        const auto bytes_allocated = static_cast<f64>(bytes_after - bytes_before) * 1e-6;
        const bool has_register = not s_input_stack.is_empty();
        Logger::trace(
            "Stack loader:\n"
            "  device={} (allocated={:.1f}MB, {})\n"
            "  exposure_filter={}\n"
            "  normalize={} (mean=0, stddev=1)\n"
            "  zero_taper={:.1f}%\n"
            "  n_slices={}\n"
            "  input_shape={}   (spacing={::.3f}, registered={})\n"
            "  padded_shape={}  (precise_cutoff={})\n"
            "  cropped_shape={} (rescale_shift={::.3f})\n"
            "  bandpass_shape={} (mirror_padding_factor={:.2f})\n"
            "  output_shape={}  (spacing={::.3f}, fast_shape={})",
            m_parameters.compute_device, bytes_allocated, options.allocator,
            m_parameters.exposure_filter_voltage > 0,
            m_parameters.normalize_and_standardize,
            m_parameters.smooth_edge_percent * 100.,
            file_shape[0],
            m_input_slice_shape, m_input_spacing, has_register,
            m_padded_slice_shape, m_parameters.precise_cutoff,
            m_cropped_slice_shape, m_rescale_shift,
            m_bandpass_slice_shape, m_parameters.bandpass_mirror_padding_factor,
            m_output_slice_shape, m_output_spacing,
            m_parameters.zero_pad_to_fast_fft_shape
        );
    }

    auto StackLoader::input_slice_() const -> Pair<View<f32>, View<c32>> {
        const auto input_shape = m_input_slice_shape.push_front<2>(1);
        const auto input_slice_rfft = m_input_slice_rfft.view();
        const auto input_slice = nf::alias_to_real(input_slice_rfft, input_shape);
        return {input_slice, input_slice_rfft};
    }
    auto StackLoader::padded_slice_() const -> Pair<View<f32>, View<c32>> {
        const auto padded_shape = m_padded_slice_shape.push_front<2>(1);
        const auto padded_slice_rfft = m_padded_slice_rfft.view();
        const auto padded_slice = nf::alias_to_real(padded_slice_rfft, padded_shape);
        return {padded_slice, padded_slice_rfft};
    }
    auto StackLoader::cropped_slice_() const -> Pair<View<f32>, View<c32>> {
        const auto cropped_shape = m_cropped_slice_shape.push_front<2>(1);
        const auto cropped_slice_rfft = m_cropped_slice_rfft.view();
        const auto cropped_slice = nf::alias_to_real(cropped_slice_rfft, cropped_shape);
        return {cropped_slice, cropped_slice_rfft};
    }
    auto StackLoader::bandpass_slice_() const -> Pair<View<f32>, View<c32>> {
        const auto bandpass_shape = m_bandpass_slice_shape.push_front<2>(1);
        const auto bandpass_slice_rfft = m_bandpass_slice_rfft.view();
        const auto bandpass_slice = nf::alias_to_real(bandpass_slice_rfft, bandpass_shape);
        return {bandpass_slice, bandpass_slice_rfft};
    }

    void StackLoader::read_stack(Metadata::Stack& metadata, const View<f32>& stack) {
        auto timer = Logger::trace_scope_time("Loading the stack");
        for (i32 batch{}; auto& image: metadata) {
            read_slice(stack.subregion(batch), image.index_file, false, image.exposure[1]);
            image.index = batch; // reset order of the slices in the stack.
            ++batch;
        }
    }

    auto StackLoader::read_stack(Metadata::Stack& metadata) -> Array<f32> {
        const auto shape = slice_shape().push_front(Vec{metadata.ssize(), isize{1}});
        auto stack = noa::Array<f32>(shape, {.device = compute_device(), .allocator = Allocator::DEFAULT_ASYNC});
        read_stack(metadata, stack.view());
        return stack;
    }

    void save_stack(
        StackLoader& stack,
        const Path& filename,
        const Metadata::Stack& metadata,
        const SaveStackParameters& saving_parameters
    ) {
        auto timer = Logger::trace_scope_time("Saving stack");

        // Output buffer.
        const auto center = (stack.slice_shape().vec / 2).as<f64>();
        auto output = Array<f32>(stack.slice_shape().push_front(Vec<isize, 2>{2, 1}), {
            .device = stack.compute_device(),
            .allocator = Allocator::MANAGED
        });

        // Set up the output file.
        auto output_file = noa::io::ImageFile(filename, {.write = true}, {
            .shape = stack.slice_shape().push_front(Vec{metadata.ssize(), isize{1}}),
            .spacing = stack.stack_spacing().push_front(1),
            .dtype = saving_parameters.dtype,
        });

        // Slices will be saved in the same order as in the metadata.
        for (isize i{}; const auto& image: metadata) {
            const auto rotation = saving_parameters.correct_rotation ? noa::deg2rad(image.angles[0]) : 0;
            const auto inverse_transform = (
                nx::translate(center) *
                nx::rotate<true>(-rotation) *
                nx::translate(-center - image.shifts)
            ).inverse().as<f32>();

            stack.read_slice(output.view().subregion(0), image.index_file, saving_parameters.cache_loader);
            nx::transform_2d(output.view().subregion(0), output.view().subregion(1), inverse_transform, {
                .interp = saving_parameters.interp,
                .border = saving_parameters.border,
            });

            output_file.write_slice(
                output.view().subregion(1).reinterpret_as_cpu().span<const f32>(),
                {.bd_offset = {i++, 0}}
            );
        }
        Logger::trace("{} saved", filename);
    }

    void save_stack(
        const View<const f32>& stack,
        const Vec<f64, 2>& spacing,
        const Metadata::Stack& metadata,
        const Path& filename,
        const SaveStackParameters& saving_parameters
    ) {
        auto timer = Logger::trace_scope_time("Saving stack");

        // Output buffer.
        const auto slice_shape = stack.shape().set<0>(1);
        const auto center = (slice_shape.filter(2, 3).vec / 2).as<f64>();
        auto output = noa::Array<f32>(slice_shape, {
            .device = stack.device(),
            .allocator = Allocator::MANAGED
        });

        // Set up the output file.
        auto output_file = noa::io::ImageFile(filename, {.write = true}, {
            .shape = stack.shape(),
            .spacing = spacing.push_front(1),
            .dtype = saving_parameters.dtype,
        });

        // Slices will be saved in the same order as in the metadata.
        for (i64 i{}; const auto& image: metadata) {
            const auto rotation = saving_parameters.correct_rotation ? noa::deg2rad(image.angles[0]) : 0;
            const auto inverse_transform = (
                nx::translate(center) *
                nx::rotate<true>(-rotation) *
                nx::translate(-center - image.shifts)
            ).inverse().as<f32>();

            nx::transform_2d(stack.subregion(image.index), output.view(), inverse_transform, {
                .interp = saving_parameters.interp,
                .border = saving_parameters.border,
            });

            output_file.write_slice(
                output.view().reinterpret_as_cpu().span<const f32>(),
                {.bd_offset = {i++, 0}}
            );
        }
        Logger::trace("{} saved", filename);
    }
}
