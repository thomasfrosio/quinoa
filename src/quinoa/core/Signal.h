#pragma once

#include <noa/FFT.h>
#include <noa/Signal.h>
#include <noa/IO.h>

#include "quinoa/Types.h"

namespace qn::signal {
    /// Computes the dimension sizes for Fourier cropping at a target resolution.
    /// \details The target pixel size is aimed, but might not be obtained exactly, depending on the input shape.
    ///          The actual pixel size of the output array is returned and is usually a bit lower than the target.
    /// \param shape                Logical BDHW shape.
    /// \param current_pixel_size   (D)HW current pixel size of the input, in A/pixel.
    /// \param target_pixel_size    (D)HW target (aimed) pixel size, in A/pixel.
    /// \param fit_to_fast_shape    Fits to a "fast" shape of FFTs.
    ///                             This will almost very slightly increase the size of the output
    ///                             and decrease its pixel size.
    template<typename floatX_t, typename = std::enable_if_t<traits::is_any_v<floatX_t, float2_t, float3_t>>>
    std::pair<size4_t, floatX_t>
    fourierCrop(size4_t shape,
                floatX_t current_pixel_size,
                floatX_t target_pixel_size,
                bool fit_to_fast_shape = true) {

        constexpr bool IS_3D = floatX_t::COUNT == 3;
        const floatX_t current_shape(shape.get(2 - IS_3D));
        floatX_t target_nyquist = current_pixel_size * 0.5f / target_pixel_size;
        floatX_t target_shape = target_nyquist * current_shape / 0.5f;

        target_shape = noa::math::floor(target_shape);
        if (fit_to_fast_shape) {
            for (size_t i = 0; i < floatX_t::COUNT; ++i) {
                const auto size = clamp_cast<size_t>(target_shape[i]);
                if (size > 1)
                    target_shape[i] = static_cast<float>(noa::fft::nextFastSize(size));
            }
        }

        target_nyquist = target_shape * 0.5 / current_shape;
        target_pixel_size = current_pixel_size / (2 * target_nyquist);

        int4_t new_shape{
                shape[0],
                IS_3D ? target_shape[0] : 1,
                target_shape[0 + IS_3D],
                target_shape[1 + IS_3D]
        };

        return std::pair<size4_t, floatX_t>(new_shape, target_pixel_size);
    }

    /// Fourier crops a stack.
    /// \param[in] input_filename   Path of the input stack to Fourier crop.
    /// \param[out] output_filename Path where to save the cropped stack.
    /// \param target_pixel_size    HW target (aimed) pixel size, in A/pixel.
    /// \param compute_device       Device on which the computation is done.
    /// \param fit_to_fast_shape    Fits to a "fast" shape of FFTs. This should very slightly increase the size
    ///                             of the output and decrease its pixel size.
    /// \param highpass_cutoff      Cutoff, in cycle/pix, where the pass is fully recovered.
    /// \param lowpass_cutoff       Cutoff, in cycle/pix, where the pass starts to roll-off.
    /// \param highpass_width       Frequency width, in cycle/pix, of the Hann window between 0 and \p highpass_cutoff.
    /// \param lowpass_width        Frequency width, in cycle/pix, of the Hann window, from \p lowpass_cutoff.
    /// \return The input and output pixel sizes are returned.
    ///         The output pixel size is usually a bit lower than the target.
    template<typename Real = float, typename = std::enable_if_t<traits::is_any_v<Real, float, double>>>
    std::pair<float2_t, float2_t>
    fourierCrop(const path_t& input_filename, const path_t& output_filename,
                float2_t target_pixel_size, Device compute_device,
                bool fit_to_fast_shape = true, float smooth_edge_percent = 0.1f,
                float highpass_cutoff = 0.075f, float lowpass_cutoff = 0.5f,
                float highpass_width = 0.075f, float lowpass_width = 0.05f) {
        using namespace ::noa;

        // Some files are not encoded properly, so if file is a volume, still interpret it as stack of 2D images.
        io::ImageFile input_file(input_filename, io::READ);
        dim4_t shape = input_file.shape();
        if (shape[0] == 1 && shape[1] > 1)
            std::swap(shape[0], shape[1]);
        QN_CHECK(shape[1] == 1, "File: {}. A tilt-series was expected, but got image file with shape {}",
                 input_filename, shape);

        const auto current_pixel_size = float2_t(input_file.pixelSize().get(1));
        auto[new_shape, new_pixel_size] = fourierCrop(shape, current_pixel_size, target_pixel_size, fit_to_fast_shape);
        // TODO If new_shape == shape, just copy?

        const auto slice_shape = dim4_t{1, 1, shape[2], shape[3]};
        const auto new_slice_shape = dim4_t{1, 1, new_shape[2], new_shape[3]};
        const auto new_slice_center = float2_t{new_shape[2] / 2, new_shape[3] / 2};
        const auto smooth_edge_size = static_cast<float>(std::max(new_shape[2], new_shape[3])) * smooth_edge_percent;

        io::ImageFile output_file(output_filename, io::WRITE);
        output_file.shape(new_shape);
        output_file.pixelSize(float3_t{1, new_pixel_size[0], new_pixel_size[1]});
        Array input_buffer_io = compute_device.gpu() ? memory::empty<Real>(slice_shape) : Array<Real>{};
        Array output_buffer_io = compute_device.gpu() ? memory::empty<Real>(new_slice_shape) : Array<Real>{};

        const ArrayOption options(compute_device, Allocator::DEFAULT_ASYNC);
        auto[input, input_fft] = noa::fft::empty<Real>(slice_shape, options);
        auto[output, output_fft] = noa::fft::empty<Real>(new_slice_shape, options);

        // TODO Add async support. On the GPU, use two streams to overlap memory copy and kernel.
        //      The IO can be done on another CPU thread.
        for (size_t i = 0; i < shape[0]; ++i) {
            // Get the current slice.
            if (compute_device.gpu()) {
                input_file.readSlice(input_buffer_io, i, false);
                memory::copy(input_buffer_io, input);
            } else {
                input_file.readSlice(input, i, false);
            }

            // Fourier crop.
            noa::fft::r2c(input, input_fft);
            noa::fft::resize<fft::H2H>(input_fft, slice_shape, output_fft, new_slice_shape);
            noa::signal::fft::bandpass<fft::H2H>(
                    output_fft, output_fft, new_slice_shape,
                    highpass_cutoff, lowpass_cutoff,
                    highpass_width, lowpass_width);
            noa::signal::fft::standardize<fft::H2H>(output_fft, output_fft, new_slice_shape);
            noa::fft::c2r(output_fft, output);
            noa::signal::rectangle(
                    output, output, new_slice_center,
                    new_slice_center - smooth_edge_size, smooth_edge_size);

            // Save.
            if (compute_device.gpu()) {
                memory::copy(output, output_buffer_io);
                output_file.writeSlice(output_buffer_io, i, false);
            } else {
                output_file.writeSlice(output, i, false);
            }
        }

        return {current_pixel_size, new_pixel_size};
    }

    // Removes images from a stack of images based on
    template<typename T>
    Array<T> massNormalization(const Array<T>& images, const Array<T>& output, float threshold = 0.4f) {
        const size_t batches = images.shape()[0];
        Array<T> means(noa::size4_t{batches, 1, 1, 1});
        noa::math::mean(images, means);

        const T median = noa::math::median(means.flat());
        threshold *= median;
        if (means.device().gpu())
            means = means.to(Device{});

        const T* means_ = means.eval().get();
        std::vector<int4_t> origins;
        for (size_t i = 0; i < batches; ++i)
            if (means_[i] >= threshold)
                origins.emplace_back(i, 0, images.shape()[2], images.shape()[3]);

        output = Array<T>(size4_t{origins.size(), 1, images.shape()[2], images.shape()[3]}, images.options());
        noa::memory::extract(images, output, Array<int4_t>(origins.data(), origins.size()), noa::BORDER_NOTHING);
        return output;
    }
}
