#include "quinoa/core/Stack.hpp"

namespace qn {
    Array<f32> StackLoader::s_input_stack{};

    void StackLoader::register_input_stack(const Path& filename) {
        s_input_stack = noa::io::load_data<f32>(filename, /*enforce_2d_stack=*/ true);
    }

    StackLoader::StackLoader(const Path& filename, const LoadStackParameters& parameters) : m_parameters(parameters) {
        // Some files may not be encoded properly, so if the file encodes a single volume,
        // still interpret it as a stack of 2d images.
        m_file.open(filename, noa::io::READ);
        auto file_shape = m_file.shape();
        if (file_shape[0] == 1 && file_shape[1] > 1)
            std::swap(file_shape[0], file_shape[1]);
        QN_CHECK(file_shape[1] == 1,
                 "File: {}. A tilt-series was expected, but got image file with shape {}",
                 filename, file_shape);
        m_file_slice_count = file_shape[0];
        m_input_slice_shape = file_shape.filter(2, 3);

        // Fourier crop.
        m_input_spacing = m_file.pixel_size().pop_front().as<f64>();
        const auto target_spacing = Vec2<f64>(parameters.rescale_target_resolution / 2);
        const auto relative_freq_error = parameters.precise_cutoff ? 2.5e4 : 1.;
        const FourierCropDimensions fourier_crop = fourier_crop_dimensions(
                m_input_slice_shape, m_input_spacing,
                target_spacing, relative_freq_error, parameters.rescale_min_size);
        m_padded_slice_shape = fourier_crop.padded_shape;
        m_cropped_slice_shape = fourier_crop.cropped_shape;
        m_output_spacing = fourier_crop.cropped_spacing;
        m_rescale_shift = fourier_crop.rescale_shifts;
        m_cropped_slice_center = (m_cropped_slice_shape.vec() / 2).as<f32>();

        // Zero-padding in real-space after cropping.
        m_output_slice_shape = m_cropped_slice_shape;
        if (parameters.zero_pad_to_square_shape)
            m_output_slice_shape = noa::math::max(m_output_slice_shape);
        if (parameters.zero_pad_to_fast_fft_shape) {
            m_output_slice_shape[0] = noa::fft::next_fast_size(m_output_slice_shape[0]);
            m_output_slice_shape[1] = noa::fft::next_fast_size(m_output_slice_shape[1]);
        }

        const auto options = noa::ArrayOption(parameters.compute_device, parameters.allocator);
        const auto input_shape = m_input_slice_shape.push_front<2>({1, 1});
        const auto padded_shape = m_padded_slice_shape.push_front<2>({1, 1});
        const auto cropped_shape = m_cropped_slice_shape.push_front<2>({1, 1});
        const bool use_gpu = parameters.compute_device.is_gpu();
        const bool has_initial_padding = noa::any(m_input_slice_shape != m_padded_slice_shape);
        const bool has_register = !s_input_stack.is_empty();

        // Main buffers.
        m_padded_slice_rfft = noa::memory::empty<c32>(padded_shape.rfft(), options);
        m_cropped_slice_rfft = noa::memory::empty<c32>(cropped_shape.rfft(), options);

        // Optional buffers for reading the slice.
        if (!has_register && use_gpu)
            m_input_slice_io = noa::memory::empty<f32>(input_shape);
        if (has_initial_padding && ((has_register && use_gpu) || !has_register))
            m_input_slice = noa::memory::empty<f32>(input_shape, options);

        qn::Logger::trace("Stack loader:\n"
                          "  compute_device={}\n"
                          "  exposure_filter={}\n"
                          "  normalize={} (mean=0, stddev=1)\n"
                          "  zero_taper={:.1f}%\n"
                          "  n_slices={}\n"
                          "  input_shape={}   (spacing={::.2f}, registered={})\n"
                          "  padded_shape={}  (precise_cutoff={})\n"
                          "  cropped_shape={} (rescale_shift={::.3f})\n"
                          "  output_shape={}  (spacing={::.2f}, fast_shape={})\n",
                          parameters.compute_device,
                          parameters.exposure_filter,
                          parameters.normalize_and_standardize,
                          parameters.smooth_edge_percent * 100.f,
                          file_shape[0],
                          m_input_slice_shape, m_input_spacing, has_register,
                          m_padded_slice_shape, parameters.precise_cutoff,
                          m_cropped_slice_shape, m_rescale_shift,
                          m_output_slice_shape, m_output_spacing,
                          parameters.zero_pad_to_fast_fft_shape);
    }

    void StackLoader::read_slice(const View<f32>& output_slice, i64 file_slice_index, bool cache) {
        QN_CHECK(file_slice_index < m_file_slice_count,
                 "Slice index is invalid. This happened because the file and the metadata don't match. "
                 "Trying to access slice index {}, but the file stack has a total of {} slices",
                 file_slice_index, m_file_slice_count);

        // Check if it's in the cache.
        for (const auto& slice: m_cache) {
            if (slice.first == file_slice_index) {
                slice.second.to(output_slice);
                return;
            }
        }

        const auto padded_shape = m_padded_slice_shape.push_front<2>({1, 1});
        const auto cropped_shape = m_cropped_slice_shape.push_front<2>({1, 1});

        auto padded_slice_rfft = m_padded_slice_rfft.view();
        auto padded_slice = noa::fft::alias_to_real(padded_slice_rfft, padded_shape);
        auto cropped_slice_rfft = m_cropped_slice_rfft.view();
        auto cropped_slice = noa::fft::alias_to_real(cropped_slice_rfft, cropped_shape);

        // Initial step. There's a bit of logic here, so abstract this away from this function.
        read_slice_and_precision_pad_(file_slice_index, padded_slice);

        // Fourier-space cropping and filtering:
        noa::fft::r2c(padded_slice, padded_slice_rfft);
        noa::fft::resize<noa::fft::H2H>(
                padded_slice_rfft, padded_shape,
                cropped_slice_rfft, cropped_shape);
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                cropped_slice_rfft, cropped_slice_rfft,
                cropped_shape, m_rescale_shift.as<f32>());
        // TODO Add exposure filter.
        noa::signal::fft::bandpass<noa::fft::H2H>(
                cropped_slice_rfft, cropped_slice_rfft, cropped_shape,
                m_parameters.highpass_parameters[0], m_parameters.lowpass_parameters[0],
                m_parameters.highpass_parameters[1], m_parameters.lowpass_parameters[1]);
        noa::fft::c2r(cropped_slice_rfft, cropped_slice);

        // Smooth edges - zero taper.
        // We assume there's always at least a small highpass that 1) sets the mean to 0
        // and 2) removes the large contrast gradients. If not, this mask will not look good.
        const auto smooth_edge_size =
                static_cast<f32>(std::max(cropped_shape[2], cropped_shape[3])) *
                m_parameters.smooth_edge_percent;
        noa::geometry::rectangle(
                cropped_slice, cropped_slice, m_cropped_slice_center,
                m_cropped_slice_center - smooth_edge_size, smooth_edge_size);

        // Final normalization (mean=0, stddev=1). The mean is likely very close to zero
        // at this point, but the zero-taper can slightly offset the mean, so normalize here.
        if (m_parameters.normalize_and_standardize)
            noa::math::normalize(cropped_slice, cropped_slice);

        // Allow the output slice to be on a difference device.
        if (output_slice.device() == cropped_slice.device()) {
            noa::memory::resize(cropped_slice, output_slice);
        } else {
            if (m_output_buffer.is_empty())
                m_output_buffer = noa::memory::like(cropped_slice);
            noa::memory::resize(cropped_slice, m_output_buffer);
            m_output_buffer.to(output_slice);
        }

        if (cache)
            m_cache.emplace_back(file_slice_index, output_slice.to_cpu());
    }

    // This function could be simpler, but here I'm willing to pay the price of complexity
    // to gain performance and reduce the memory usage.
    void StackLoader::read_slice_and_precision_pad_(i64 file_slice_index, const View<f32>& padded_slice) {
        const bool has_initial_padding = noa::any(m_input_slice_shape != m_padded_slice_shape);
        const bool is_gpu = compute_device().is_gpu();

        View<f32> input_slice;
        if (!s_input_stack.is_empty()) { // has_register
            const auto registered_slice = s_input_stack.view().subregion(file_slice_index);
            if (!has_initial_padding) {
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
            input_slice = !has_initial_padding ? padded_slice : m_input_slice.view();
            auto input_slice_io = !is_gpu ? input_slice : m_input_slice_io.view();

            m_file.read_slice(input_slice_io, file_slice_index, false);
            if (is_gpu)
                noa::memory::copy(input_slice_io, input_slice);
        }

        // Optional zero-padding for accurate Fourier cropping cutoff.
        if (has_initial_padding)
            noa::memory::resize(input_slice, padded_slice); // TODO check that zero-padding is ok here
    }

    void StackLoader::read_stack(MetadataStack& metadata, const View<f32>& stack) {
        qn::Logger::trace("Loading the stack...");
        Timer timer;
        timer.start();

        i64 batch{0};
        for (auto& slice_metadata: metadata.slices()) {
            read_slice(stack.subregion(batch), slice_metadata.index_file);
            slice_metadata.index = static_cast<i32>(batch); // reset order of the slices in the stack.
            ++batch;
        }
        qn::Logger::trace("Loading the stack... done. Took {:.2f}ms\n", timer.elapsed());
    }

    Array<f32> StackLoader::read_stack(MetadataStack& metadata) {
        const auto options = ArrayOption(compute_device(), Allocator::DEFAULT_ASYNC);
        auto stack = noa::memory::empty<f32>(slice_shape().push_front<2>({metadata.size(), 1}), options);
        read_stack(metadata, stack.view());
        return stack;
    }

    void save_stack(
            const Path& input_stack_path,
            const Path& output_stack_path,
            const MetadataStack& metadata,
            const LoadStackParameters& parameters,
            InterpMode interpolation_mode ,
            BorderMode border_mode
    ) {
        qn::Logger::info("Saving the tilt-series...");
        noa::Timer timer;
        timer.start();

        auto stack_loader = StackLoader(input_stack_path, parameters);
        const auto output_slice_shape = stack_loader.slice_shape().push_front<2>({1, 1});
        const auto output_slice_center = MetadataSlice::center(output_slice_shape);
        const auto options = ArrayOption(stack_loader.compute_device(), stack_loader.allocator());
        const auto use_gpu = stack_loader.compute_device().is_gpu();

        // Output buffers.
        auto output_slice = noa::memory::empty<f32>(output_slice_shape, options);
        auto output_slice_buffer = noa::Array<f32>(output_slice_shape, options);
        auto output_slice_buffer_io = use_gpu ? noa::memory::empty<f32>(output_slice_shape) : output_slice_buffer;
        auto output_slice_texture = noa::Texture<f32>(
                output_slice_shape, options.device(), interpolation_mode, border_mode);

        // Set up the output file.
        auto output_file = noa::io::ImageFile(output_stack_path, noa::io::WRITE);
        output_file.set_dtype(noa::io::DataType::F32);
        output_file.set_shape(stack_loader.slice_shape().push_front<2>({metadata.size(), 1}));
        output_file.set_pixel_size(stack_loader.stack_spacing().push_front(1).as<f32>());

        // The metadata should be unscaled, so shifts are at the original pixel size.
        // Here we apply the shifts on the fourier cropped slices, so we need to scale
        // the shifts from the metadata down before applying them.
        const auto shift_scale_factor = stack_loader.file_spacing() / stack_loader.stack_spacing();

        // Slices are saved in the same order as specified in the metadata.
        i64 output_file_index{0};
        for (const MetadataSlice& slice_metadata: metadata.slices()) {
            // Get the slice.
            stack_loader.read_slice(output_slice.view(), slice_metadata.index_file);
            output_slice_texture.update(output_slice.view());

            const auto slice_shifts = slice_metadata.shifts * shift_scale_factor;
            const auto inv_transform = noa::math::inverse(
                    noa::geometry::translate(output_slice_center.as<f64>()) *
                    noa::geometry::linear2affine(noa::geometry::rotate(noa::math::deg2rad(-slice_metadata.angles[0]))) *
                    noa::geometry::translate(-output_slice_center.as<f64>() - slice_shifts)
            );
            noa::geometry::transform_2d(output_slice_texture, output_slice_buffer, inv_transform.as<f32>());

            if (use_gpu)
                noa::memory::copy(output_slice_buffer, output_slice_buffer_io);
            output_file.write_slice(output_slice_buffer_io, output_file_index);
            ++output_file_index;
        }

        qn::Logger::info("Saving the tilt-series... done. Took {:.2f}ms\n", timer.elapsed());
    }
}
