#pragma once

#include <noa/FFT.h>
#include <noa/Geometry.h>
#include <noa/IO.h>
#include <noa/Memory.h>
#include <noa/Signal.h>

#include "quinoa/core/Metadata.h"
#include "quinoa/io/Logging.h"

namespace qn::details {
    /// Computes the dimension sizes for Fourier cropping at a target resolution.
    /// \details The target pixel size is aimed, but might not be obtained exactly, depending on the input shape.
    ///          The actual pixel size of the output array is returned.
    template<typename FloatN>
    auto fourierCropDimensions(
            dim4_t shape,
            FloatN current_pixel_size,
            FloatN target_pixel_size
    ) -> std::pair<dim4_t, FloatN> {

        using coord_t = noa::traits::value_type_t<FloatN>;
        constexpr bool IS_3D = FloatN::COUNT == 3;

        // Get the initial target shape for that resolution.
        const auto current_shape = static_cast<FloatN>(shape.get(2 - IS_3D));
        const auto target_nyquist = current_pixel_size * coord_t{0.5} / target_pixel_size;
        const auto target_shape = noa::math::round(target_nyquist * current_shape / coord_t{0.5});

        // Compute new pixel size and new shape.
        const auto new_nyquist = target_shape * 0.5 / current_shape;
        const auto new_pixel_size = current_pixel_size / (2 * new_nyquist);
        const auto new_shape = dim4_t{
                shape[0],
                IS_3D ? target_shape[0] : 1,
                target_shape[0 + IS_3D],
                target_shape[1 + IS_3D]
        };

        return {new_shape, new_pixel_size};
    }
}

namespace qn {
    struct LoadStackParameters {
        Device compute_device;

        // Initial filtering on original images:
        int32_t median_filter_window{0};

        // Fourier cropping:
        double target_resolution;

        // Signal processing after cropping:
        bool exposure_filter{false};
        float2_t highpass_parameters{0.10, 0.10};
        float2_t lowpass_parameters{0.45, 0.05};

        // Image processing after cropping:
        bool normalize_and_standardize{true};
        float smooth_edge_percent{0.01f};
        bool zero_pad_to_fast_fft_shape{true};
    };

    struct LoadStackOutputs {
        Array<float> output_stack;
        double2_t output_pixel_size;
        double2_t input_pixel_size;
    };

    /// Loads a tilt-series and does some preprocessing.
    [[nodiscard]]
    auto loadStack(
            const path_t& tilt_series_path,
            const MetadataStack& tilt_series_metadata,
            const LoadStackParameters& parameters
    ) -> LoadStackOutputs {
        qn::Logger::info("Loading the tilt-series...");
        Timer timer;
        timer.start();

        // Some files are not encoded properly, so if file encodes a volume,
        // still interpret it as stack of 2D images.
        auto input_file = noa::io::ImageFile(tilt_series_path, noa::io::READ);
        dim4_t input_shape = input_file.shape();
        if (input_shape[0] == 1 && input_shape[1] > 1)
            std::swap(input_shape[0], input_shape[1]);
        QN_CHECK(input_shape[1] == 1,
                 "File: {}. A tilt-series was expected, but got image file with shape {}",
                 tilt_series_path, input_shape);

        // Fourier crop setup.
        const auto input_pixel_size = double2_t(input_file.pixelSize().get(1));
        const auto target_pixel_size = double2_t(parameters.target_resolution / 2);
        const auto [cropped_shape, output_pixel_size] = details::fourierCropDimensions(
                input_shape, input_pixel_size, target_pixel_size);

        // Zero-padding in real-space.
        auto output_shape = cropped_shape;
        if (parameters.zero_pad_to_fast_fft_shape)
            output_shape = noa::fft::nextFastShape(output_shape);

        const auto input_slice_shape = dim4_t{1, 1, input_shape[2], input_shape[3]};
        const auto cropped_slice_shape = dim4_t{1, 1, cropped_shape[2], cropped_shape[3]};
        const auto cropped_slice_center = float2_t{cropped_shape[2] / 2, cropped_shape[3] / 2};
        const auto output_stack_shape = dim4_t{tilt_series_metadata.size(), 1, output_shape[2], output_shape[3]};
        const auto smooth_edge_size = static_cast<float>(std::max(cropped_shape[2], cropped_shape[3])) *
                                      parameters.smooth_edge_percent;

        const bool use_gpu = parameters.compute_device.gpu();
        const bool do_median_filter = parameters.median_filter_window > 1;
        const auto median_window = static_cast<dim_t>(parameters.median_filter_window);
        const auto options = noa::ArrayOption(parameters.compute_device, noa::Allocator::DEFAULT_ASYNC);

        qn::Logger::trace("Compute device: {}\n"
                          "Median filter on input: {}\n"
                          "Exposure filter: {}\n"
                          "Normalize and Standardize: {}\n"
                          "Zero-taper: {:.1f}%\n"
                          "Zero-padding to fast shape: {}\n"
                          "Input:  shape={}, pixel_size={:.2f}\n"
                          "Output: shape={}, pixel_size={:.2f}",
                          parameters.compute_device,
                          do_median_filter ? noa::string::format("true (window={})", median_window) : "false",
                          parameters.exposure_filter,
                          parameters.normalize_and_standardize,
                          parameters.smooth_edge_percent * 100.f,
                          parameters.zero_pad_to_fast_fft_shape,
                          input_shape, input_pixel_size,
                          output_shape, output_pixel_size);

        // Input buffers. If compute device is a GPU, we need a stage buffer for the IO.
        // This could be pinned for faster copy, but for now, keep it to pageable memory.
        // For median filtering, we need another buffer because it is an out-of-place operation.
        auto [input_slice, input_slice_fft] = noa::fft::empty<float>(input_slice_shape, options);
        auto input_slice_io = use_gpu ? noa::memory::empty<float>(input_slice_shape) : input_slice;
        auto input_slice_median = do_median_filter ? noa::memory::like(input_slice) : noa::Array<float>{};

        // Fourier-crop and output buffers.
        const auto [cropped_slice, cropped_slice_fft] = noa::fft::empty<float>(cropped_slice_shape, options);
        const auto output_stack = noa::memory::empty<float>(output_stack_shape, options);

        // Normalization.
        // Allocate the mean and stddev on the compute device, in order
        // to not have to synchronize and transfer back and forth to the GPU.
        const auto do_normalization = parameters.normalize_and_standardize;
        const auto mean = noa::memory::empty<float>(do_normalization ? dim4_t{1} : dim4_t{}, options);
        const auto stddev = noa::memory::empty<float>(do_normalization ? dim4_t{1} : dim4_t{}, options);

        // Process one input slice at a time.
        for (const auto& slice_metadata: tilt_series_metadata.slices()) {

            // Just make sure the image file matches the metadata.
            const auto input_slice_index = static_cast<dim_t>(slice_metadata.index_file);
            const auto output_slice_index = slice_metadata.index;
            QN_CHECK(input_slice_index < input_shape[0],
                     "Slice index is invalid. This happened because the file stack and the metadata don't match. "
                     "Trying to access slice index {}, but the file stack has a total of {} slices",
                     input_slice_index, input_shape[0]);

            // Read the current slice from the file.
            // If CPU, input_slice_io is just an alias for input_slice.
            // If GPU, we need an extra copy.
            input_file.readSlice(input_slice_io, input_slice_index, false);
            if (use_gpu)
                noa::memory::copy(input_slice_io, input_slice);

            if (do_median_filter) {
                noa::signal::median2(input_slice, input_slice_median, median_window);
                std::swap(input_slice, input_slice_median); // interchangeable arrays
            }

            // Fourier-space cropping and filtering:
            noa::fft::r2c(input_slice, input_slice_fft);
            noa::fft::resize<noa::fft::H2H>(
                    input_slice_fft, input_slice_shape,
                    cropped_slice_fft, cropped_slice_shape);
            // TODO Add exposure filter.
            noa::signal::fft::bandpass<noa::fft::H2H>(
                    cropped_slice_fft, cropped_slice_fft, cropped_slice_shape,
                    parameters.highpass_parameters[0], parameters.lowpass_parameters[0],
                    parameters.highpass_parameters[1], parameters.lowpass_parameters[1]);
            noa::fft::c2r(cropped_slice_fft, cropped_slice);

            // Unfortunately, because of the zero-padding, we have to compute
            // the stats and normalize one slice at a time.
            if (do_normalization) {
                 noa::math::mean(cropped_slice, mean);
                 noa::math::std(cropped_slice, stddev);
                 noa::math::ewise(cropped_slice, mean, stddev,
                                  cropped_slice, noa::math::minus_divide_t{});
            }

            // Zero-padding and save to output stack.
            noa::signal::rectangle(
                    cropped_slice, cropped_slice, cropped_slice_center,
                    cropped_slice_center - smooth_edge_size, smooth_edge_size);
            noa::memory::resize(cropped_slice, output_stack.subregion(output_slice_index));
        }

        qn::Logger::info("Loading the tilt-series... done. Took {:.2f}ms\n", timer.elapsed());
        return {output_stack, output_pixel_size, input_pixel_size};
    }
}

namespace qn {
    struct SaveStackParameters{
        Device compute_device;

        // Initial filtering on original images:
        int32_t median_filter_window{0};

        // Fourier cropping:
        double target_resolution;

        // Signal processing after cropping:
        bool exposure_filter{false};
        float2_t highpass_parameters{0.10, 0.10};
        float2_t lowpass_parameters{0.45, 0.05};

        // Image processing after cropping:
        bool normalize_and_standardize{true};
        float smooth_edge_percent{0.01f};

        // Transformation:
        InterpMode interpolation_mode = InterpMode::INTERP_LINEAR;
        BorderMode border_mode = BorderMode::BORDER_ZERO;
    };

    /// Corrects for the in-plane rotation and shifts, as encoded in the metadata, and save the transformed stack.
    void saveStack(
            const path_t& input_tilt_series_path,
            const MetadataStack& input_tilt_series_metadata,
            const path_t& output_tilt_series_path,
            const SaveStackParameters& parameters
    ) {
        qn::Logger::info("Saving the tilt-series...");
        Timer timer;
        timer.start();

        // Some files are not encoded properly, so if file encodes a volume,
        // still interpret it as stack of 2D images.
        auto input_file = noa::io::ImageFile(input_tilt_series_path, noa::io::READ);
        dim4_t input_shape = input_file.shape();
        if (input_shape[0] == 1 && input_shape[1] > 1)
            std::swap(input_shape[0], input_shape[1]);
        QN_CHECK(input_shape[1] == 1,
                 "File: {}. A tilt-series was expected, but got image file with shape {}",
                 input_tilt_series_path, input_shape);

        const auto input_pixel_size = double2_t(input_file.pixelSize().get(1));
        const auto target_pixel_size = double2_t(parameters.target_resolution / 2);
        const auto [output_shape, output_pixel_size] = details::fourierCropDimensions(
                input_shape, input_pixel_size, target_pixel_size);

        const auto input_slice_shape = dim4_t{1, 1, input_shape[2], input_shape[3]};
        const auto output_slice_shape = dim4_t{1, 1, output_shape[2], output_shape[3]};
        const auto output_stack_shape = dim4_t{input_tilt_series_metadata.size(), 1, output_shape[2], output_shape[3]};
        const auto output_slice_center = float2_t{output_shape[2] / 2, output_shape[3] / 2};
        const auto smooth_edge_size = static_cast<float>(std::max(output_shape[2], output_shape[3])) *
                                      parameters.smooth_edge_percent;

        const bool use_gpu = parameters.compute_device.gpu();
        const bool do_median_filter = parameters.median_filter_window > 1;
        const auto median_window = static_cast<dim_t>(parameters.median_filter_window);
        const auto options = noa::ArrayOption(parameters.compute_device, noa::Allocator::DEFAULT_ASYNC);

        qn::Logger::trace("Compute device: {}\n"
                          "Median filter on input: {}\n"
                          "Exposure filter: {}\n"
                          "Normalize and Standardize: {}\n"
                          "Zero-taper: {:.1f}%\n"
                          "Input:  shape={}, pixel_size={:.2f}\n"
                          "Output: shape={}, pixel_size={:.2f}",
                          parameters.compute_device,
                          do_median_filter ? noa::string::format("true (window={})", median_window) : "false",
                          parameters.exposure_filter,
                          parameters.normalize_and_standardize,
                          parameters.smooth_edge_percent * 100.f,
                          input_shape, input_pixel_size,
                          output_shape, output_pixel_size);

        // Input buffers. If compute device is a GPU, we need a stage buffer for the IO.
        // This could be pinned for faster copy, but for now, keep it to pageable memory.
        // For median filtering, we need another buffer because it is an out-of-place operation.
        auto [input_slice, input_slice_fft] = noa::fft::empty<float>(input_slice_shape, options);
        auto input_slice_io = use_gpu ? noa::memory::empty<float>(input_slice_shape) : input_slice;
        auto input_slice_median = do_median_filter ? noa::memory::like(input_slice) : noa::Array<float>{};

        // Output buffers.
        auto [output_slice, output_slice_fft] = noa::fft::empty<float>(output_slice_shape, options);
        auto output_slice_buffer = noa::Array<float>(output_slice_shape, options);
        auto output_slice_buffer_io = use_gpu ? noa::memory::empty<float>(output_slice_shape) : output_slice_buffer;
        auto output_slice_texture = noa::Texture<float>(output_slice_shape, options.device(),
                                                        parameters.interpolation_mode,
                                                        parameters.border_mode);

        auto output_file = noa::io::ImageFile(output_tilt_series_path, noa::io::WRITE);
        output_file.dtype(noa::io::FLOAT32);
        output_file.shape(output_stack_shape);
        output_file.pixelSize(float3_t{1, output_pixel_size[0], output_pixel_size[1]});

        // Normalization.
        // Allocate the mean and stddev on the compute device, in order
        // to not have to synchronize and transfer back and forth to the GPU.
        const auto do_normalization = parameters.normalize_and_standardize;
        const auto mean = noa::memory::empty<float>(do_normalization ? dim4_t{1} : dim4_t{}, options);
        const auto stddev = noa::memory::empty<float>(do_normalization ? dim4_t{1} : dim4_t{}, options);

        // The metadata should be unscaled, so shifts are at the original pixel size.
        // Here we apply the shifts on the fourier cropped slices, so we need to scale
        // the shifts from the metadata down before applying them.
        const auto shift_scale_factor = float2_t(input_pixel_size / output_pixel_size);

        for (const MetadataSlice& slice_metadata: input_tilt_series_metadata.slices()) {
            const auto input_slice_index = static_cast<dim_t>(slice_metadata.index_file);
            const auto output_slice_index = static_cast<dim_t>(slice_metadata.index);
            QN_CHECK(input_slice_index < input_shape[0],
                     "Slice index is invalid. This happened because the file stack and the metadata don't match. "
                     "Trying to access slice index {}, but the file stack has a total of {} slices",
                     input_slice_index, input_shape[0]);

            // Read the current slice from the file.
            // If CPU, input_slice_io is just an alias for input_slice.
            // If GPU, we need an extra copy.
            input_file.readSlice(input_slice_io, input_slice_index);
            if (use_gpu)
                noa::memory::copy(input_slice_io, input_slice);

            // At this point, input_slice points to the data on the compute device.
            // If median filter is required, use the buffer allocated for that and
            // update for it to point to input_slice.
            if (do_median_filter) {
                noa::signal::median2(input_slice, input_slice_median, median_window);
                std::swap(input_slice, input_slice_median); // alike arrays
            }

            // At this point, this can be an in-place or out-of-place FFT,
            // whether the median filter was applied or not.
            noa::fft::r2c(input_slice, input_slice_fft);
            noa::fft::resize<noa::fft::H2H>(input_slice_fft, input_slice_shape,
                                            output_slice_fft, output_slice_shape);

            // TODO Add exposure filter and update the lowpass cutoff.
            noa::signal::fft::bandpass<noa::fft::H2H>(
                    output_slice_fft, output_slice_fft, output_slice_shape,
                    parameters.highpass_parameters[0], parameters.lowpass_parameters[0],
                    parameters.highpass_parameters[1], parameters.lowpass_parameters[1]);

            noa::fft::c2r(output_slice_fft, output_slice);
            noa::signal::rectangle(
                    output_slice, output_slice, output_slice_center,
                    output_slice_center - smooth_edge_size, smooth_edge_size);

            // Apply the transformation encoded in the metadata.
            if (do_normalization) {
                noa::math::mean(output_slice, mean);
                noa::math::std(output_slice, stddev);
                noa::math::ewise(output_slice, mean, stddev,
                                 output_slice, noa::math::minus_divide_t{});
            }

            output_slice_texture.update(output_slice);
            const auto slice_shifts = slice_metadata.shifts * shift_scale_factor;
            const auto inv_transform = noa::math::inverse(
                    noa::geometry::translate(output_slice_center) *
                    float33_t(noa::geometry::rotate(noa::math::deg2rad(-slice_metadata.angles[0]))) *
                    noa::geometry::translate(-output_slice_center - slice_shifts)
            );
            noa::geometry::transform2D(output_slice_texture, output_slice_buffer, inv_transform);

            if (use_gpu)
                noa::memory::copy(output_slice_buffer, output_slice_buffer_io);
            output_file.writeSlice(output_slice_buffer_io, output_slice_index);
        }

        qn::Logger::info("Saving the tilt-series... done. Took {:.2f}ms\n", timer.elapsed());
    }
}
