#pragma once

#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Math.hpp>
#include <noa/Memory.hpp>
#include <noa/Signal.hpp>
#include <noa/core/utils/Timer.hpp>
#include <noa/algorithms/Utilities.hpp>

#include "quinoa/core/Metadata.h"
#include "quinoa/io/Logging.h"

namespace qn::details {
    /// Computes the dimension sizes for Fourier cropping at a target resolution.
    /// \details The target pixel size is aimed, but might not be obtained exactly, depending on the input shape.
    ///          The actual pixel size of the output array is returned.
    template<typename Real, size_t N>
    auto fourier_crop_dimensions(
            const Shape4<i64>& shape,
            const Vec<Real, N>& current_pixel_size,
            const Vec<Real, N>& target_pixel_size
    ) -> std::tuple<Shape4<i64>, Vec<Real, N>, Vec<Real, N>> {

        // Find the frequency cutoff in the current spectrum that corresponds to the desired spacing.
        const auto current_shape = Shape<i64, N>(shape.data() + 4 - N);
        const auto current_shape_f = current_shape.vec().template as<f64>();
        const auto frequency_cutoff = current_pixel_size * Real{0.5} / target_pixel_size;

        // Get Fourier cropped shape and the actual new spacing.
        const auto new_shape_f = noa::math::round(frequency_cutoff * current_shape_f / Real{0.5});
        const auto new_nyquist = new_shape_f * 0.5 / current_shape_f;
        const auto new_pixel_size = current_pixel_size / (2 * new_nyquist);

        // Output shape.
        const auto new_shape = Shape<i64, N>(new_shape_f.template as<i64>());
        const auto new_shape_4d = Shape4<i64>(Vec4<i64>{
                shape[0],
                N == 3 ? new_shape[0] : 1,
                new_shape[0 + N == 3],
                new_shape[1 + N == 3]
        });

        // In order to preserve the center, we may need to shift.
        // Adding this shift to the cropped dft moves the dc_original_index to dc_new_index.
        const auto dc_original_index = noa::algorithm::frequency2index<true>(Vec<i64, N>{0}, current_shape);
        const auto dc_new_index = noa::algorithm::frequency2index<true>(Vec<i64, N>{0}, new_shape);
        const auto center = dc_original_index.template as<f64>() * (current_pixel_size / new_pixel_size);
        const auto shift = dc_new_index.template as<f64>() - center;

        return {new_shape_4d, new_pixel_size, shift};
    }
}

namespace qn {
    struct LoadStackParameters {
        Device compute_device;

        // Initial filtering on original images:
        i32 median_filter_window{0};

        // Fourier cropping:
        f64 target_resolution;

        // Signal processing after cropping:
        bool exposure_filter{false};
        Vec2<f32> highpass_parameters{0.10, 0.10};
        Vec2<f32> lowpass_parameters{0.45, 0.05};

        // Image processing after cropping:
        bool normalize_and_standardize{true};
        f32 smooth_edge_percent{0.01f};
        bool zero_pad_to_fast_fft_shape{true};
    };

    struct LoadStackOutputs {
        Array<f32> output_stack;
        Vec2<f64> output_pixel_size;
        Vec2<f64> input_pixel_size;
    };

    /// Loads a tilt-series and does some preprocessing.
    [[nodiscard]]
    auto load_stack(
            const Path& tilt_series_path,
            const MetadataStack& tilt_series_metadata,
            const LoadStackParameters& parameters
    ) -> LoadStackOutputs {
        qn::Logger::info("Loading the tilt-series...");
        Timer timer;
        timer.start();

        // Some files are not encoded properly, so if file encodes a single volume,
        // still interpret it as stack of 2D images.
        auto input_file = noa::io::ImageFile(tilt_series_path, noa::io::READ);
        auto input_shape = input_file.shape();
        if (input_shape[0] == 1 && input_shape[1] > 1)
            std::swap(input_shape[0], input_shape[1]);
        QN_CHECK(input_shape[1] == 1,
                 "File: {}. A tilt-series was expected, but got image file with shape {}",
                 tilt_series_path, input_shape);

        // Fourier crop setup.
        const auto input_pixel_size = input_file.pixel_size().pop_front().as<f64>();
        const auto target_pixel_size = Vec2<f64>(parameters.target_resolution / 2);
        const auto [cropped_shape, output_pixel_size, rescale_shift] = details::fourier_crop_dimensions(
                input_shape, input_pixel_size, target_pixel_size);

        // Zero-padding in real-space.
        auto output_shape = cropped_shape;
        if (parameters.zero_pad_to_fast_fft_shape)
            output_shape = noa::fft::next_fast_shape(output_shape);

        const auto input_slice_shape = Shape4<i64>{1, 1, input_shape[2], input_shape[3]};
        const auto cropped_slice_shape = Shape4<i64>{1, 1, cropped_shape[2], cropped_shape[3]};
        const auto cropped_slice_center = Vec2<f32>{cropped_shape[2] / 2, cropped_shape[3] / 2};
        const auto output_stack_shape = Shape4<i64>{tilt_series_metadata.size(), 1, output_shape[2], output_shape[3]};
        const auto smooth_edge_size = static_cast<f32>(std::max(cropped_shape[2], cropped_shape[3])) *
                                      parameters.smooth_edge_percent;

        const bool use_gpu = parameters.compute_device.is_gpu();
        const bool do_median_filter = parameters.median_filter_window > 1;
        const auto median_window = parameters.median_filter_window;
        const auto options = noa::ArrayOption(parameters.compute_device, noa::Allocator::DEFAULT_ASYNC);

        qn::Logger::trace("Compute device: {}\n"
                          "Median filter on input: {}\n"
                          "Exposure filter: {}\n"
                          "Normalize and standardize: {}\n"
                          "Zero-taper: {:.1f}%\n"
                          "Zero-padding to fast shape: {}\n"
                          "Input:  shape={}, pixel_size={::.2f}\n"
                          "Output: shape={}, pixel_size={::.2f}",
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
        auto [input_slice, input_slice_fft] = noa::fft::empty<f32>(input_slice_shape, options);
        auto input_slice_io = use_gpu ? noa::memory::empty<f32>(input_slice_shape) : input_slice;
        auto input_slice_median = do_median_filter ? noa::memory::like(input_slice) : noa::Array<f32>{};

        // Fourier-crop and output buffers.
        const auto [cropped_slice, cropped_slice_fft] = noa::fft::empty<f32>(cropped_slice_shape, options);
        const auto output_stack = noa::memory::empty<f32>(output_stack_shape, options);

        // Process one input slice at a time.
        for (const auto& slice_metadata: tilt_series_metadata.slices()) {

            // Just make sure the image file matches the metadata.
            const i64 input_slice_index = slice_metadata.index_file;
            const i64 output_slice_index = slice_metadata.index;
            QN_CHECK(input_slice_index < input_shape[0],
                     "Slice index is invalid. This happened because the file stack and the metadata don't match. "
                     "Trying to access slice index {}, but the file stack has a total of {} slices",
                     input_slice_index, input_shape[0]);

            // Read the current slice from the file.
            // If CPU, input_slice_io is just an alias for input_slice.
            // If GPU, we need an extra copy.
            input_file.read_slice(input_slice_io, input_slice_index, false);
            if (use_gpu)
                noa::memory::copy(input_slice_io, input_slice);

            if (do_median_filter) {
                noa::signal::median_filter_2d(input_slice, input_slice_median, median_window);
                std::swap(input_slice, input_slice_median); // interchangeable arrays
            }

            // Fourier-space cropping and filtering:
            noa::fft::r2c(input_slice, input_slice_fft);
            noa::fft::resize<noa::fft::H2H>(
                    input_slice_fft, input_slice_shape,
                    cropped_slice_fft, cropped_slice_shape);
            noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                    cropped_slice_fft, cropped_slice_fft,
                    cropped_slice_shape, rescale_shift.as<f32>());
            // TODO Add exposure filter.
            noa::signal::fft::bandpass<noa::fft::H2H>(
                    cropped_slice_fft, cropped_slice_fft, cropped_slice_shape,
                    parameters.highpass_parameters[0], parameters.lowpass_parameters[0],
                    parameters.highpass_parameters[1], parameters.lowpass_parameters[1]);
            noa::fft::c2r(cropped_slice_fft, cropped_slice);

            // Unfortunately, because of the zero-padding, we have to compute
            // the stats and normalize one slice at a time.
            if (parameters.normalize_and_standardize)
                noa::math::normalize(cropped_slice, cropped_slice);

            // Zero-padding and save to output stack.
            noa::geometry::rectangle(
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
        i32 median_filter_window{0};

        // Fourier cropping:
        f64 target_resolution;

        // Signal processing after cropping:
        bool exposure_filter{false};
        Vec2<f32> highpass_parameters{0.10, 0.10};
        Vec2<f32> lowpass_parameters{0.45, 0.05};

        // Image processing after cropping:
        bool normalize_and_standardize{true};
        f32 smooth_edge_percent{0.01f};

        // Transformation:
        InterpMode interpolation_mode = InterpMode::LINEAR;
        BorderMode border_mode = BorderMode::ZERO;
    };

    /// Corrects for the in-plane rotation and shifts, as encoded in the metadata, and save the transformed stack.
    void save_stack(
            const Path& input_tilt_series_path,
            const MetadataStack& input_tilt_series_metadata,
            const Path& output_tilt_series_path,
            const SaveStackParameters& parameters
    ) {
        qn::Logger::info("Saving the tilt-series...");
        noa::Timer timer;
        timer.start();

        // Some files are not encoded properly, so if file encodes a volume,
        // still interpret it as stack of 2D images.
        auto input_file = noa::io::ImageFile(input_tilt_series_path, noa::io::READ);
        auto input_shape = input_file.shape();
        if (input_shape[0] == 1 && input_shape[1] > 1)
            std::swap(input_shape[0], input_shape[1]);
        QN_CHECK(input_shape[1] == 1,
                 "File: {}. A tilt-series was expected, but got image file with shape {}",
                 input_tilt_series_path, input_shape);

        const auto input_pixel_size = input_file.pixel_size().pop_front().as<f64>();
        const auto target_pixel_size = Vec2<f64>(parameters.target_resolution / 2);
        const auto [output_shape, output_pixel_size, rescale_shift] = details::fourier_crop_dimensions(
                input_shape, input_pixel_size, target_pixel_size);

        const auto input_slice_shape = Shape4<i64>{1, 1, input_shape[2], input_shape[3]};
        const auto output_slice_shape = Shape4<i64>{1, 1, output_shape[2], output_shape[3]};
        const auto output_stack_shape = Shape4<i64>{input_tilt_series_metadata.size(), 1, output_shape[2], output_shape[3]};
        const auto output_slice_center = Vec2<f32>{output_shape[2] / 2, output_shape[3] / 2};
        const auto smooth_edge_size = static_cast<f32>(std::max(output_shape[2], output_shape[3])) *
                                      parameters.smooth_edge_percent;

        const bool use_gpu = parameters.compute_device.is_gpu();
        const bool do_median_filter = parameters.median_filter_window > 1;
        const auto median_window = parameters.median_filter_window;
        const auto options = noa::ArrayOption(parameters.compute_device, noa::Allocator::DEFAULT_ASYNC);

        qn::Logger::trace("Compute device: {}\n"
                          "Median filter on input: {}\n"
                          "Exposure filter: {}\n"
                          "Normalize and standardize: {}\n"
                          "Zero-taper: {:.1f}%\n"
                          "Input:  shape={}, pixel_size={::.2f}\n"
                          "Output: shape={}, pixel_size={::.2f}",
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
        auto [input_slice, input_slice_fft] = noa::fft::empty<f32>(input_slice_shape, options);
        auto input_slice_io = use_gpu ? noa::memory::empty<f32>(input_slice_shape) : input_slice;
        auto input_slice_median = do_median_filter ? noa::memory::like(input_slice) : noa::Array<f32>{};

        // Output buffers.
        auto [output_slice, output_slice_fft] = noa::fft::empty<f32>(output_slice_shape, options);
        auto output_slice_buffer = noa::Array<f32>(output_slice_shape, options);
        auto output_slice_buffer_io = use_gpu ? noa::memory::empty<f32>(output_slice_shape) : output_slice_buffer;
        auto output_slice_texture = noa::Texture<f32>(output_slice_shape, options.device(),
                                                      parameters.interpolation_mode,
                                                      parameters.border_mode);

        auto output_file = noa::io::ImageFile(output_tilt_series_path, noa::io::WRITE);
        output_file.set_dtype(noa::io::DataType::F32);
        output_file.set_shape(output_stack_shape);
        output_file.set_pixel_size({1, output_pixel_size[0], output_pixel_size[1]});

        // The metadata should be unscaled, so shifts are at the original pixel size.
        // Here we apply the shifts on the fourier cropped slices, so we need to scale
        // the shifts from the metadata down before applying them.
        const auto shift_scale_factor = input_pixel_size / output_pixel_size;

        for (const MetadataSlice& slice_metadata: input_tilt_series_metadata.slices()) {
            const i64 input_slice_index = slice_metadata.index_file;
            const i64 output_slice_index = slice_metadata.index;
            QN_CHECK(input_slice_index < input_shape[0],
                     "Slice index is invalid. This happened because the file stack and the metadata don't match. "
                     "Trying to access slice index {}, but the file stack has a total of {} slices",
                     input_slice_index, input_shape[0]);

            // Read the current slice from the file.
            // If CPU, input_slice_io is just an alias for input_slice.
            // If GPU, we need an extra copy.
            input_file.read_slice(input_slice_io, input_slice_index);
            if (use_gpu)
                noa::memory::copy(input_slice_io, input_slice);

            // At this point, input_slice points to the data on the compute device.
            // If median filter is required, use the buffer allocated for that and
            // update for it to point to input_slice.
            if (do_median_filter) {
                noa::signal::median_filter_2d(input_slice, input_slice_median, median_window);
                std::swap(input_slice, input_slice_median); // alike arrays
            }

            // At this point, this can be an in-place or out-of-place FFT,
            // whether the median filter was applied or not.
            noa::fft::r2c(input_slice, input_slice_fft);
            noa::fft::resize<noa::fft::H2H>(
                    input_slice_fft, input_slice_shape,
                    output_slice_fft, output_slice_shape);
            // TODO We could apply the rescale-shifts in real-space with the other shifts
            noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                    output_slice_fft, output_slice_fft,
                    output_slice_shape, rescale_shift.as<f32>());
            // TODO Add exposure filter and update the lowpass cutoff.
            noa::signal::fft::bandpass<noa::fft::H2H>(
                    output_slice_fft, output_slice_fft, output_slice_shape,
                    parameters.highpass_parameters[0], parameters.lowpass_parameters[0],
                    parameters.highpass_parameters[1], parameters.lowpass_parameters[1]);

            noa::fft::c2r(output_slice_fft, output_slice);
            noa::geometry::rectangle(
                    output_slice, output_slice, output_slice_center,
                    output_slice_center - smooth_edge_size, smooth_edge_size);

            // Apply the transformation encoded in the metadata.
            if (parameters.normalize_and_standardize)
                noa::math::normalize(output_slice, output_slice);

            output_slice_texture.update(output_slice);
            const auto slice_shifts = slice_metadata.shifts * shift_scale_factor;
            const auto inv_transform = noa::math::inverse(
                    noa::geometry::translate(output_slice_center.as<f64>()) *
                    noa::geometry::linear2affine(noa::geometry::rotate(noa::math::deg2rad(-slice_metadata.angles[0]))) *
                    noa::geometry::translate(-output_slice_center.as<f64>() - slice_shifts)
            );
            noa::geometry::transform_2d(output_slice_texture, output_slice_buffer, inv_transform.as<f32>());

            if (use_gpu)
                noa::memory::copy(output_slice_buffer, output_slice_buffer_io);
            output_file.write_slice(output_slice_buffer_io, output_slice_index);
        }

        qn::Logger::info("Saving the tilt-series... done. Took {:.2f}ms\n", timer.elapsed());
    }
}
