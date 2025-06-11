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
    ///          integer multiple of the input's fftfreq. However, we can zero pad the real-space input, effectively
    ///          stretching its spectrum, to minimize the difference between the target fftfreq and an integer multiple
    ///          of the stretched input's fftfreq. Note that the stretching is centered on the origin (0,0) of the
    ///          input, which may lead to a +/- 0.5 shift between the input and the Fourier cropped output. As such,
    ///          we also return this shift for the caller to apply to keep the centers aligned.
    struct FourierCropDimensions {
        /// Shape for the (optional) zero padded input. Padding the input, effectively changing its sampling to
        /// align the target cutoff frequency to an integer position, may be necessary to precisely crop at the
        /// target spacing.
        Shape<i64, 2> padded_shape;

        /// Logical shape of the Fourier-cropped spectrum.
        Shape<i64, 2> cropped_shape;

        /// Actual spacing after Fourier cropping the (possibly zero-padded) input.
        Vec<f64, 2> cropped_spacing;

        /// Shifts to add to the Fourier-cropped output to keep its (real-space) center aligned with the input.
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
    ///                                 zero pad the input. Note that in this case and for non-squared input shapes,
    ///                                 it is possible for the target spacing to become anisotropic.
    /// \param target_min_size          Minimum tolerable size.
    ///                                 It is used to ensure a minimum output.cropped_shape. This is used to prevent
    ///                                 inputs with a very small pixel sizes to be cropped to a very small shape.
    /// \param target_max_size          Maximum tolerable size.
    ///                                 It is used to ensure a maximum output.cropped_shape. This is used to prevent
    ///                                 inputs with large pixel sizes to be kept at a size that is too big.
    auto fourier_crop_dimensions(
        Shape<i64, 2> current_shape,
        Vec<f64, 2> current_spacing,
        Vec<f64, 2> target_spacing,
        f64 maximum_relative_error = 5e-4,
        i64 target_min_size = 0,
        i64 target_max_size = 0
    ) -> FourierCropDimensions {
        qn::check(noa::all(current_spacing > 0 and target_spacing >= 0));

        // Disallow Fourier padding. Note that if the current spacing is anisotropic, the target is set
        // to be isotropic, since it's often simpler to handle and the caller might expect the output spacing
        // to be isotropic too.
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

        // Possibly zero-pad in real space to place the frequency cutoff at a particular index of the spectrum.
        // This is necessary to be able to precisely crop at a frequency cutoff, and offers a way to keep
        // the target spacing isotropic within a maximum_relative_error.
        auto pad_to_align_cutoff = [maximum_relative_error](i64 i_size, f64 i_spacing, f64 o_spacing) -> i64
        {
            i64 MAXIMUM_SIZE = i_size + 256; // in most case, we stop way before that (~0 to 6)
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

        // In order to preserve the image center, we may need to shift the Fourier-cropped image.
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
        constexpr i32 n_threads = 4;
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

        s_input_stack = Array<f32>(file.shape());
        file.read_all(s_input_stack.span(), {.n_threads = n_threads});

        // Some files are not encoded correctly; reinterpret a volume as a stack of images.
        if (s_input_stack.shape()[0] == 1 and s_input_stack.shape()[1] > 1)
            s_input_stack = std::move(s_input_stack).permute({1, 0, 2, 3});
    }

    StackLoader::StackLoader(const Path& filename, const LoadStackParameters& parameters) : m_parameters(parameters) {
        // Some files may not be encoded properly, so if the file encodes a single volume,
        // still interpret it as a stack of 2d images.
        m_file.open(filename, {.read = true});
        auto file_shape = m_file.shape();
        if (file_shape[0] == 1 and file_shape[1] > 1)
            std::swap(file_shape[0], file_shape[1]);
        check(file_shape[1] == 1,
              "File: {}. A tilt-series was expected, but got image file with shape {}",
              filename, file_shape);
        m_file_slice_count = file_shape[0];
        m_input_slice_shape = file_shape.filter(2, 3);

        // Fourier crop.
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
        m_cropped_slice_center = (m_cropped_slice_shape.vec / 2).as<f64>();

        // Zero-padding in real-space after cropping.
        m_output_slice_shape = m_cropped_slice_shape;
        if (parameters.zero_pad_to_square_shape)
            m_output_slice_shape = noa::max(m_output_slice_shape);
        if (parameters.zero_pad_to_fast_fft_shape) {
            m_output_slice_shape[0] = noa::fft::next_fast_size(m_output_slice_shape[0]);
            m_output_slice_shape[1] = noa::fft::next_fast_size(m_output_slice_shape[1]);
        }

        const auto options = noa::ArrayOption(parameters.compute_device, parameters.allocator);
        const auto input_shape = m_input_slice_shape.push_front<2>(1);
        const auto padded_shape = m_padded_slice_shape.push_front<2>(1);
        const auto cropped_shape = m_cropped_slice_shape.push_front<2>(1);
        const bool use_gpu = parameters.compute_device.is_gpu();
        const bool has_initial_padding = noa::any(m_input_slice_shape != m_padded_slice_shape);
        const bool has_register = not s_input_stack.is_empty();

        // Main buffers.
        m_padded_slice_rfft = noa::empty<c32>(padded_shape.rfft(), options);
        m_cropped_slice_rfft = noa::empty<c32>(cropped_shape.rfft(), options);

        // Optional buffers for reading the slice.
        if (not has_register and use_gpu)
            m_input_slice_io = noa::empty<f32>(input_shape);
        if (has_initial_padding and ((has_register and use_gpu) or not has_register))
            m_input_slice = noa::empty<f32>(input_shape, options);

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
            "  output_shape={}  (spacing={::.3f}, fast_shape={})\n",
            parameters.compute_device,
            parameters.exposure_filter,
            parameters.normalize_and_standardize,
            parameters.smooth_edge_percent * 100.,
            file_shape[0],
            m_input_slice_shape, m_input_spacing, has_register,
            m_padded_slice_shape, parameters.precise_cutoff,
            m_cropped_slice_shape, m_rescale_shift,
            m_output_slice_shape, m_output_spacing,
            parameters.zero_pad_to_fast_fft_shape
        );
    }

    void StackLoader::read_slice(const View<f32>& output_slice, i64 file_slice_index, bool cache) {
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

        auto padded_slice_rfft = m_padded_slice_rfft.view();
        auto padded_slice = noa::fft::alias_to_real(padded_slice_rfft, padded_shape);
        auto cropped_slice_rfft = m_cropped_slice_rfft.view();
        auto cropped_slice = noa::fft::alias_to_real(cropped_slice_rfft, cropped_shape);

        // Initial step. There's a bit of logic here, so abstract this away from this function.
        read_slice_and_precision_pad_(file_slice_index, padded_slice);

        // Fourier-space cropping and filtering:
        noa::fft::r2c(padded_slice, padded_slice_rfft);
        noa::fft::resize<"h2h">(
            padded_slice_rfft, padded_shape,
            cropped_slice_rfft, cropped_shape);
        noa::signal::phase_shift_2d<"h2h">(
            cropped_slice_rfft, cropped_slice_rfft,
            cropped_shape, m_rescale_shift.as<f32>());
        // TODO Add exposure filter.
        noa::signal::bandpass<"h2h">(cropped_slice_rfft, cropped_slice_rfft, cropped_shape, m_parameters.bandpass);
        noa::fft::c2r(cropped_slice_rfft, cropped_slice);

        // Smooth edges - zero taper.
        // We assume there's always at least a small highpass that 1) sets the mean to 0
        // and 2) removes the large contrast gradients. If not, this mask will not look good.
        const auto smooth_edge_size =
            static_cast<f64>(std::max(cropped_shape[2], cropped_shape[3])) *
            m_parameters.smooth_edge_percent;
        ng::draw_shape(cropped_slice, cropped_slice, ng::Rectangle{
            .center = m_cropped_slice_center,
            .radius = m_cropped_slice_center - smooth_edge_size,
            .smoothness = smooth_edge_size,
        });

        /// Get the final slice view. If the output slice is on the compute-device,
        /// we can do everything in-place. Otherwise use a new contiguous buffer and
        /// do the final copy after the normalization.
        /// TODO If no padding and same device, we could draw_shape directly into the
        ///      output_slice and save a copy...
        View<f32> final_slice;
        const bool is_same_device = cropped_slice.device() == output_slice.device();
        if (is_same_device) {
            final_slice = output_slice;
        } else {
            if (m_output_buffer.is_empty())
                m_output_buffer = Array<f32>(m_output_slice_shape.push_front<2>(1), cropped_slice.options());
            final_slice = m_output_buffer.view();
        }

        // Copy, or pad to square and/or fast FFT shape.
        noa::resize(cropped_slice, final_slice);

        // Final normalization (mean=0, stddev=1). The mean is likely very close to zero
        // at this point, but the zero-taper and padding can slightly offset the mean,
        // so normalize here.
        if (m_parameters.normalize_and_standardize)
            noa::normalize(final_slice, final_slice, {.mode = noa::Norm::MEAN_STD});

        // By this point, if output is on the compute-device, final_slice == output_slice
        // and we can just return. Otherwise, copy from the compute-device to the output.
        if (not is_same_device)
            final_slice.to(output_slice);

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

            m_file.read_slice(input_slice_io.span(), {.bd_offset = {file_slice_index, 0}, .clamp = false});
            if (is_gpu)
                noa::copy(input_slice_io, input_slice);
        }

        // Optional zero-padding for accurate Fourier cropping cutoff.
        if (has_initial_padding) {
            const auto padding_right = (padded_slice.shape() - input_slice.shape()).vec;
            noa::resize(input_slice, padded_slice, {}, padding_right);
        }
    }

    void StackLoader::read_stack(MetadataStack& metadata, const View<f32>& stack) {
        auto timer = Logger::trace_scope_time("Loading the stack");
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

    // void save_stack(
    //     const Path& input_stack_path,
    //     const Path& output_stack_path,
    //     const MetadataStack& metadata,
    //     const LoadStackParameters& loading_parameters,
    //     const SaveStackParameters& saving_parameters
    // ) {
    //     auto timer = Logger::trace_scope_time("Saving aligned stack");
    //
    //     auto stack_loader = StackLoader(input_stack_path, loading_parameters);
    //     const auto output_slice_shape = stack_loader.slice_shape().push_front<2>(1);
    //     const auto output_slice_center = (stack_loader.slice_shape().vec / 2).as<f64>();
    //     const auto options = ArrayOption{stack_loader.compute_device(), stack_loader.allocator()};
    //
    //     // Output buffers.
    //     auto output_slice = noa::Array<f32>(output_slice_shape, options);
    //     auto output_slice_aligned = noa::Array<f32>(output_slice_shape, options);
    //     auto output_slice_io = options.is_dereferenceable() ? noa::Array<f32>{} : noa::Array<f32>(output_slice_shape);
    //
    //     // Set up the output file.
    //     auto output_file = noa::io::ImageFile(output_stack_path, {.write = true}, {
    //         .shape = stack_loader.slice_shape().push_front(Vec{metadata.ssize(), i64{1}}),
    //         .spacing = stack_loader.stack_spacing().push_front(1),
    //         .dtype = noa::io::Encoding::F32
    //     });
    //
    //     // As always, the metadata should be unscaled, i.e. shifts are at the original spacing as
    //     // encoded in the input file. Here we apply the shifts on the rescaled images returned by
    //     // the stack loader, so we need to scale the shifts from the metadata down before applying them.
    //     const auto shift_scale_factor = stack_loader.file_spacing() / stack_loader.stack_spacing();
    //
    //     // Slices are saved in the same order as specified in the metadata.
    //     i64 output_file_index{};
    //     for (const MetadataSlice& slice_metadata: metadata) {
    //         stack_loader.read_slice(output_slice.view(), slice_metadata.index_file);
    //
    //         const auto slice_shifts = slice_metadata.shifts * shift_scale_factor;
    //         const auto slice_rotation =
    //             saving_parameters.correct_rotation ?
    //             noa::deg2rad(slice_metadata.angles[0]) : 0;
    //
    //         const auto inverse_transform = (
    //             ng::translate(output_slice_center) *
    //             ng::linear2affine(ng::rotate(-slice_rotation)) *
    //             ng::translate(-output_slice_center - slice_shifts)
    //         ).inverse().as<f32>();
    //
    //         ng::transform_2d(output_slice.view(), output_slice_aligned.view(), inverse_transform, {
    //             .interp = saving_parameters.interp,
    //             .border = saving_parameters.border,
    //         });
    //
    //         View<const f32> to_write = options.is_dereferenceable() ?
    //             output_slice_aligned.view().reinterpret_as_cpu() :
    //             output_slice_aligned.view().to(output_slice_io.view());
    //
    //         output_file.write_slice(to_write.span(), {.bd_offset = {output_file_index, 0}});
    //         ++output_file_index;
    //     }
    // }

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
            const auto slice_rotation =
                saving_parameters.correct_rotation ?
                noa::deg2rad(-slice.angles[0]) : 0;

            const auto inverse_transform = (
                ng::translate(center) *
                ng::linear2affine(ng::rotate(-slice_rotation)) *
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
