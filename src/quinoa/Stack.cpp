#include <noa/Array.hpp>
#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/Utils.hpp>

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
        Shape<i64, 2> padded_shape;

        /// Logical shape of the Fourier-cropped spectrum.
        Shape<i64, 2> cropped_shape;

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
        Shape<i64, 2> current_shape,
        Vec<f64, 2> current_spacing,
        Vec<f64, 2> target_spacing,
        f64 maximum_relative_error = 5e-4,
        i64 target_min_size = 0,
        i64 target_max_size = 0
    ) -> FourierCropDimensions {
        qn::check(noa::all(current_spacing > 0 and target_spacing >= 0));

        // Disallow Fourier padding.
        // Note that if the current spacing is anisotropic, the target is set to be isotropic, since it is
        // often simpler to handle and the caller might expect the output spacing to be isotropic too.
        if (noa::any(current_spacing > target_spacing))
            target_spacing = noa::max(current_spacing);

        // Clamp the target spacing to the maximum spacing corresponding to the minimum allowed size.
        if (target_min_size > 0 and target_min_size < min(current_shape)) {
            const auto target_max_spacing = noa::min(
                current_spacing * current_shape.vec.as<f64>() / static_cast<f64>(target_min_size));
            if (noa::any(target_max_spacing < target_spacing))
                target_spacing = target_max_spacing;
        }

        // Clamp the target spacing to the minimum spacing corresponding to the maximum allowed size.
        if (target_max_size > 0 and target_max_size < min(current_shape)) {
            const auto target_min_spacing = noa::max(
                current_spacing * current_shape.vec.as<f64>() / static_cast<f64>(target_max_size));
            if (noa::any(target_spacing < target_min_spacing))
                target_spacing = target_min_spacing;
        }

        // Possibly pad in real space to place the frequency cutoff at a particular index of the spectrum.
        // This is necessary to be able to precisely crop at a frequency cutoff, and offers a way to keep
        // the target spacing isotropic within a maximum_relative_error.
        auto pad_to_align_cutoff = [maximum_relative_error](i64 i_size, f64 i_spacing, f64 o_spacing) -> i64
        {
            i64 MAXIMUM_SIZE = i_size + 256; // in all tested cases, we stop way before that (~0 to 6)
            i64 best_size = i_size;
            f64 best_error = std::numeric_limits<f64>::max();
            while (i_size < MAXIMUM_SIZE) {
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
        const auto new_shape = Shape{new_shape_f64.as<i64>()};
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
}

namespace qn {
    Array<f32> StackLoader::s_input_stack{};

    void StackLoader::register_input_stack(const Path& filename) {
        auto timer = Logger::info_scope_time("Loading and decoding the input stack");

        using namespace noa::io;
        auto file = ImageFile(filename, {.read = true});
        const auto n_elements = file.shape().n_elements();
        const auto encoded_size = static_cast<f32>(Encoding::encoded_size(file.dtype(), n_elements));
        const auto decoded_size = static_cast<f32>(Encoding::encoded_size(Encoding::F32, n_elements));

        // Adding more threads is only useful when the file is compressed. Without compression, we're just
        // waiting for the filesystem, and having multiple threads in the mix seems to make it worse.
        const i32 n_threads = file.is_compressed() ? 4 : 1;

        Logger::trace(
            "Stack registry:\n"
            "  path={}\n"
            "  shape={}\n"
            "  dtype={}->f32\n"
            "  size={:.2f}GB->{:.2f}GB\n"
            "  n_threads={}",
            filename, file.shape(), file.dtype(),
            encoded_size / 1e9f, decoded_size / 1e9f,
            n_threads
        );

        // TODO Use managed memory? This makes reading slices faster, but may increase pressure when loading patches.
        s_input_stack = Array<f32>(file.shape());
        file.read_all(s_input_stack.span(), {.n_threads = n_threads});

        // Some files are not encoded correctly; reinterpret a volume as a stack of images.
        if (s_input_stack.shape()[0] == 1 and s_input_stack.shape()[1] > 1)
            s_input_stack = std::move(s_input_stack).permute({1, 0, 2, 3});
    }

    StackLoader::StackLoader(const Path& filename, const LoadStackParameters& parameters) : m_parameters(parameters) {
        m_file.open(filename, {.read = true});
        auto file_shape = m_file.shape();
        if (file_shape[0] == 1 and file_shape[1] > 1) {
            Logger::warn(
                "{}. A tilt-series was expected, but the image file encodes a volume. To continue, we will assume "
                "the file metadata is not encoded properly and will interpret this volume as a stack of 2d images",
                filename
            );
            std::swap(file_shape[0], file_shape[1]);
        }
        check(file_shape[1] == 1, "{}. A tilt-series was expected, but got image file with shape {}", filename, file_shape);
        m_file_slice_count = file_shape[0];
        m_input_slice_shape = file_shape.filter(2, 3);

        const auto options = ArrayOption(parameters.compute_device, parameters.allocator);

        // Reading the file.
        const auto input_shape = m_input_slice_shape.push_front<2>(1);
        const bool use_gpu = parameters.compute_device.is_gpu();
        const bool has_initial_padding = noa::any(m_input_slice_shape != m_padded_slice_shape);
        const bool has_register = not s_input_stack.is_empty();
        if (not has_register and use_gpu)
            m_input_slice_io = Array<f32>(input_shape);
        if (has_initial_padding and ((has_register and use_gpu) or not has_register))
            m_input_slice = Array<f32>(input_shape, options);

        // Fourier cropping.
        m_input_spacing = m_file.spacing().pop_front().as<f64>();
        const auto target_spacing = Vec<f64, 2>::from_value(parameters.rescale_target_resolution / 2);
        const auto relative_freq_error = parameters.precise_cutoff ? 2.5e-4 : 1.;
        const FourierCropDimensions fourier_crop = fourier_crop_dimensions(
            m_input_slice_shape, m_input_spacing,
            target_spacing, relative_freq_error,
            parameters.rescale_min_size, parameters.rescale_max_size
        );
        m_padded_slice_shape = fourier_crop.padded_shape;
        m_cropped_slice_shape = fourier_crop.cropped_shape;
        m_output_spacing = fourier_crop.cropped_spacing;
        m_rescale_shift = fourier_crop.rescale_shifts;
        m_padded_slice_rfft = Array<c32>(m_padded_slice_shape.push_front<2>(1).rfft(), options);
        m_cropped_slice_rfft = Array<c32>(m_cropped_slice_shape.push_front<2>(1).rfft(), options);

        // Mirror padding for bandpass.
        m_bandpass_slice_shape = m_cropped_slice_shape;
        if (parameters.bandpass_mirror_padding_factor > 0) {
            auto padding = m_bandpass_slice_shape.vec.as<f64>() * parameters.bandpass_mirror_padding_factor;
            m_bandpass_slice_shape += Shape{noa::round(padding).as<i64>()};
            m_bandpass_slice_shape = noa::max(2 * m_cropped_slice_shape, nf::next_fast_shape(m_bandpass_slice_shape));
            m_bandpass_slice_rfft = Array<c32>(m_bandpass_slice_shape.push_front<2>(1).rfft(), options);
        }

        // Final zero-padding.
        m_output_slice_shape = m_cropped_slice_shape;
        if (parameters.zero_pad_to_square_shape)
            m_output_slice_shape = noa::max(m_output_slice_shape);
        if (parameters.zero_pad_to_fast_fft_shape) {
            m_output_slice_shape[0] = noa::fft::next_fast_size(m_output_slice_shape[0]);
            m_output_slice_shape[1] = noa::fft::next_fast_size(m_output_slice_shape[1]);
        }

        Logger::trace(
            "Stack loader:\n"
            "  compute_device={}\n"
            "  exposure_filter={}\n"
            "  normalize={} (mean=0, stddev=1)\n"
            "  zero_taper={:.1f}%\n"
            "  n_slices={}\n"
            "  input_shape={}   (spacing={::.3f}, registered={})\n"
            "  padded_shape={}  (precise_cutoff={})\n"
            "  cropped_shape={} (rescale_shift={::.3f})\n"
            "  bandpass_shape={} (mirror_padding_factor={:.2f})\n"
            "  output_shape={}  (spacing={::.3f}, fast_shape={})\n",
            parameters.compute_device,
            parameters.exposure_filter,
            parameters.normalize_and_standardize,
            parameters.smooth_edge_percent * 100.,
            file_shape[0],
            m_input_slice_shape, m_input_spacing, has_register,
            m_padded_slice_shape, parameters.precise_cutoff,
            m_cropped_slice_shape, m_rescale_shift,
            m_bandpass_slice_shape, parameters.bandpass_mirror_padding_factor,
            m_output_slice_shape, m_output_spacing,
            parameters.zero_pad_to_fast_fft_shape
        );
    }

    void StackLoader::read_slice(const View<f32>& output_slice, i64 file_slice_index, bool cache) {
        check(output_slice.device() == compute_device());
        check(noa::all(output_slice.shape() == m_output_slice_shape.push_front<2>(1)));
        check(file_slice_index < m_file_slice_count,
              "Slice index is invalid. This happened because the file and the metadata don't match. "
              "Trying to access slice index {}, but the file stack has a total of {} slices",
              file_slice_index, m_file_slice_count);

        // Check if it's in the cache.
        for (const auto& [index, buffer]: m_cache) {
            if (index == file_slice_index) {
                buffer.to(output_slice);
                return;
            }
        }

        const auto padded_shape = m_padded_slice_shape.push_front<2>(1);
        const auto cropped_shape = m_cropped_slice_shape.push_front<2>(1);

        const auto padded_slice_rfft = m_padded_slice_rfft.view();
        const auto padded_slice = nf::alias_to_real(padded_slice_rfft, padded_shape);
        const auto cropped_slice_rfft = m_cropped_slice_rfft.view();
        const auto cropped_slice = nf::alias_to_real(cropped_slice_rfft, cropped_shape);

        const bool needs_mirror_pad = not m_bandpass_slice_rfft.is_empty();
        const bool needs_smooth_edge = m_parameters.smooth_edge_percent > 0;
        const bool needs_final_zero_pad = noa::any(m_cropped_slice_shape != m_output_slice_shape);

        // Read the slice, transfer to the compute device, and precision pad if necessary.
        read_slice_and_precision_pad_(file_slice_index, padded_slice);

        // Fourier-space cropping.
        nf::r2c(padded_slice, padded_slice_rfft);
        nf::resize<"h2h">(padded_slice_rfft, padded_shape, cropped_slice_rfft, cropped_shape);
        ns::phase_shift_2d<"h2h">(cropped_slice_rfft, cropped_slice_rfft, cropped_shape, m_rescale_shift.as<f32>());
        if (not needs_mirror_pad) // no padding was asked for the bandpass, so we can do it here
            ns::bandpass<"h2h">(cropped_slice_rfft, cropped_slice_rfft, cropped_shape, m_parameters.bandpass);
        nf::c2r(cropped_slice_rfft, cropped_slice);

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
        if (needs_mirror_pad) {
            const auto bandpass_shape = m_bandpass_slice_shape.push_front<2>(1);
            const auto bandpass_slice_rfft = m_bandpass_slice_rfft.view();
            const auto bandpass_slice = nf::alias_to_real(bandpass_slice_rfft, bandpass_shape);

            noa::resize(cropped_slice, bandpass_slice, noa::Border::REFLECT);
            nf::r2c(bandpass_slice, bandpass_slice_rfft);
            ns::bandpass<"h2h">(bandpass_slice_rfft, bandpass_slice_rfft, bandpass_shape, m_parameters.bandpass);
            nf::c2r(bandpass_slice_rfft, bandpass_slice);
            noa::resize(bandpass_slice, direct_bandpass_to_output ? output_slice : cropped_slice);
        }

        // Smooth edges to zero.
        // We assume there's always at least a small highpass that sets the mean to zero and
        // removes the large contrast gradients. Otherwise, this mask will not look good.
        if (needs_smooth_edge) {
            const auto input_slice = direct_bandpass_to_output ? output_slice : cropped_slice;
            const auto tapered_slice = direct_taper_to_output ? output_slice : input_slice;

            const auto center = (input_slice.shape().filter(2, 3).vec / 2).as<f64>();
            const auto radius = (cropped_slice.shape().filter(2, 3).vec / 2).as<f64>();
            const auto smooth_edge_size =
                static_cast<f64>(noa::max(cropped_shape.filter(2, 3))) *
                m_parameters.smooth_edge_percent;

            ng::draw(input_slice, tapered_slice, ng::Rectangle{
                .center = center,
                .radius = radius - smooth_edge_size,
                .smoothness = smooth_edge_size,
            }.draw());
        }

        // Copy to the output slice, or final zero-padding to the output shape.
        if (not direct_bandpass_to_output and not direct_taper_to_output)
            noa::resize(cropped_slice, output_slice);

        // Final normalization (mean=0, stddev=1).
        // If a highpass is applied, the mean should be close to zero at this point,
        // but the zero-taper and zero-padding can slightly offset the mean.
        if (m_parameters.normalize_and_standardize)
            noa::normalize(output_slice, output_slice, {.mode = noa::Norm::MEAN_STD});

        // Cache the output. Since we only get a view, make sure to synchronize before quitting (.eval())
        // in case the caller gets rid of the corresponding array while the copy runs asynchronously.
        if (cache)
            m_cache.emplace_back(file_slice_index, output_slice.to_cpu());
    }

    // This function could be simpler, but here I'm willing to pay the price of complexity
    // to gain performance and reduce the memory usage.
    void StackLoader::read_slice_and_precision_pad_(i64 file_slice_index, const View<f32>& padded_slice) {
        const bool has_initial_padding = noa::any(m_input_slice_shape != m_padded_slice_shape);
        const bool is_gpu = compute_device().is_gpu();

        // Load the input slice to the compute device.
        View<f32> input_slice;
        if (not s_input_stack.is_empty()) { // has_register
            const auto registered_slice = s_input_stack.view().subregion(file_slice_index);
            if (not has_initial_padding) {
                registered_slice.to(padded_slice);
                return;
            }
            if (is_gpu) {
                input_slice = m_input_slice.view();
                registered_slice.to(input_slice);
            } else {
                input_slice = registered_slice;
            }
        } else {
            input_slice = has_initial_padding ? m_input_slice.view() : padded_slice;
            auto input_slice_io = is_gpu ? m_input_slice_io.view() : input_slice;

            m_file.read_slice(input_slice_io.eval().span(), {.bd_offset = {file_slice_index, 0}, .clamp = false});
            if (is_gpu)
                noa::copy(input_slice_io, input_slice);
        }

        // Optional adding of the input slice for accurate Fourier cropping cutoff.
        if (has_initial_padding) {
            const auto padding_right = (padded_slice.shape() - input_slice.shape()).vec;
            noa::resize(input_slice, padded_slice, {}, padding_right, noa::Border::REFLECT); // TODO Zero?
        }
    }

    void StackLoader::read_stack(MetadataStack& metadata, const View<f32>& stack) {
        auto timer = Logger::trace_scope_time("Loading the stack");
        // noa::Session::set_fft_cache_limit(4); // FIXME
        for (i32 batch{}; auto& slice_metadata: metadata) {
            read_slice(stack.subregion(batch), slice_metadata.index_file);
            slice_metadata.index = batch; // reset order of the slices in the stack.
            ++batch;
        }
    }

    auto StackLoader::read_stack(MetadataStack& metadata) -> Array<f32> {
        const auto shape = slice_shape().push_front(Vec{metadata.ssize(), i64{1}});
        auto stack = noa::Array<f32>(shape, {.device = compute_device(), .allocator = Allocator::DEFAULT_ASYNC});
        read_stack(metadata, stack.view());
        return stack;
    }

    void save_stack(
        StackLoader& stack,
        const Path& filename,
        const MetadataStack& metadata,
        const SaveStackParameters& saving_parameters
    ) {
        auto timer = Logger::trace_scope_time("Saving stack");

        // Output buffer.
        const auto center = (stack.slice_shape().vec / 2).as<f64>();
        auto output = Array<f32>(stack.slice_shape().push_front(Vec<i64, 2>{1, 1}), {
            .device = stack.compute_device(),
            .allocator = Allocator::MANAGED
        });

        // Set up the output file.
        auto output_file = noa::io::ImageFile(filename, {.write = true}, {
            .shape = stack.slice_shape().push_front(Vec{metadata.ssize(), i64{1}}),
            .spacing = stack.stack_spacing().push_front(1),
            .dtype = saving_parameters.dtype,
        });

        // Slices will be saved in the same order as in the metadata.
        for (i64 i{}; const auto& slice: metadata) {
            const auto rotation = saving_parameters.correct_rotation ? noa::deg2rad(slice.angles[0]) : 0;
            const auto inverse_transform = (
                ng::translate(center) *
                ng::rotate<true>(-rotation) *
                ng::translate(-center - slice.shifts)
            ).inverse().as<f32>();

            stack.read_slice(output.view().subregion(0), slice.index_file, saving_parameters.cache_loader);
            ng::transform_2d(output.view().subregion(0), output.view().subregion(1), inverse_transform, {
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

    void save_stack(
        const View<f32>& stack,
        const Vec<f64, 2>& spacing,
        const MetadataStack& metadata,
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
        for (i64 i{}; const auto& slice: metadata) {
            const auto rotation = saving_parameters.correct_rotation ? noa::deg2rad(slice.angles[0]) : 0;
            const auto inverse_transform = (
                ng::translate(center) *
                ng::rotate<true>(-rotation) *
                ng::translate(-center - slice.shifts)
            ).inverse().as<f32>();

            ng::transform_2d(stack.subregion(slice.index), output.view(), inverse_transform, {
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
