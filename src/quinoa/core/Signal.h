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
        const floatX_t shape_(shape.get(2 - IS_3D));
        const floatX_t frequency_cutoff = current_pixel_size / (2 * target_pixel_size);
        const floatX_t sample_cutoff = frequency_cutoff * shape_;
        const floatX_t ceil_sample_cutoff = noa::math::ceil(sample_cutoff);
        const floatX_t ceil_frequency_cutoff = ceil_sample_cutoff / shape_;
        const floatX_t actual_pixel_size = current_pixel_size / ceil_frequency_cutoff;

        int4_t new_shape{
                shape[0],
                IS_3D ? sample_cutoff[0] * 2 : 1,
                sample_cutoff[0 + IS_3D] * 2,
                sample_cutoff[1 + IS_3D] * 2
        };
        if (fit_to_fast_shape)
            new_shape = noa::fft::nextFastShape(new_shape);

        return std::pair<size4_t, floatX_t>(new_shape, actual_pixel_size);
    }

    template<typename real_t = float, typename = std::enable_if_t<traits::is_any_v<real_t, float, double>>>
    void fourierCrop(const path_t& input_filename, const path_t& output_filename,
                     float2_t target_pixel_size, Device compute_device,
                     bool fit_to_fast_shape = true, bool standardize = true, float lowpass_cutoff = 0.5f) {
        using namespace ::noa;

        // Some files are not encoded properly, so if file is a volume, still interpret it as stack of 2D images.
        io::ImageFile input_file(input_filename, io::READ);
        size4_t shape = input_file.shape();
        if (shape[0] == 1 && shape[1] > 1)
            std::swap(shape[0], shape[1]);
        QN_CHECK(shape[1] == 1, "File: {}. A tilt-series was expected, but got image file with shape {}",
                 input_filename, shape);

        const float2_t current_pixel_size(input_file.pixelSize().get(1));
        auto[new_shape, new_pixel_size] = fourierCrop(shape, current_pixel_size, target_pixel_size, fit_to_fast_shape);
        const size4_t slice_shape{1, 1, shape[2], shape[3]};
        const size4_t new_slice_shape{1, 1, new_shape[2], new_shape[3]};

        io::ImageFile output_file(output_filename, io::WRITE);
        output_file.shape(new_shape);
        output_file.pixelSize(float3_t{1, new_pixel_size[0], new_pixel_size[1]});
        Array<real_t> buffer_io = compute_device.gpu() ? memory::empty<real_t>(new_slice_shape) : Array<real_t>{};

        const ArrayOption options(compute_device, Allocator::DEFAULT_ASYNC);
        auto[input, input_fft] = fft::empty<real_t>(slice_shape, options);
        auto[output, output_fft] = fft::empty<real_t>(new_slice_shape, options);

        // TODO Add async support. On the GPU, use two streams to overlap memory copy and kernel.
        //      The IO can be done on another CPU thread.
        for (size_t i = 0; i < shape[0]; ++i) {
            // Get the current slice.
            input_file.readSlice(input, i, false);
            fft::r2c(input, input_fft);

            // Fourier crop.
            fft::resize<fft::H2H>(input_fft, slice_shape, output_fft, new_slice_shape);
            noa::signal::fft::lowpass<fft::H2H>(output_fft, output_fft, new_slice_shape, lowpass_cutoff, 0.05f);
            if (standardize)
                noa::signal::fft::standardize<fft::H2H>(output_fft, output_fft, new_slice_shape);
            fft::c2r(output_fft, output);

            // Save.
            if (compute_device.gpu()) {
                memory::copy(output, buffer_io);
                output_file.writeSlice(buffer_io, i, false);
            } else {
                output_file.writeSlice(output, i, false);
            }
        }
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
