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
    ///          The actual pixel size of the output array is returned and is usually a bit lower than the target.
    /// \param shape                BDHW shape.
    /// \param current_pixel_size   (D)HW current pixel size of the input, in A/pixel.
    /// \param target_pixel_size    (D)HW target (aimed) pixel size, in A/pixel.
    /// \param fit_to_fast_shape    Fits to a "fast" shape for FFTs.
    ///                             This often very slightly increases the size of the output
    ///                             and decrease its pixel size.
    template<typename FloatN>
    auto fourierCropDimensions(
            dim4_t shape,
            FloatN current_pixel_size,
            FloatN target_pixel_size,
            bool fit_to_fast_shape = true
    ) -> std::pair<dim4_t, FloatN> {

        using coord_t = noa::traits::value_type_t<FloatN>;
        constexpr bool IS_3D = FloatN::COUNT == 3;

        // Get the initial target shape for that resolution.
        const auto current_shape = static_cast<FloatN>(shape.get(2 - IS_3D));
        const auto target_nyquist = current_pixel_size * coord_t{0.5} / target_pixel_size;
        auto target_shape = noa::math::floor(target_nyquist * current_shape / coord_t{0.5});

        // Update the target shape for faster FFTs.
        if (fit_to_fast_shape) {
            for (size_t i = 0; i < FloatN::COUNT; ++i) {
                const auto size = noa::clamp_cast<dim_t>(target_shape[i]);
                if (size > 1)
                    target_shape[i] = static_cast<coord_t>(noa::fft::nextFastSize(size));
            }
        }

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
    struct PreProcessStackParameters {
        Device compute_device;

        // Fourier cropping:
        double target_resolution;
        bool fit_to_fast_fft_shape{true};

        // Image processing:
        int32_t median_filter_window{0};
        bool exposure_filter{false};
        float2_t highpass_parameters{0.10, 0.10};
        float2_t lowpass_parameters{0.45, 0.05};
        bool center_and_standardize{true};
        float smooth_edge_percent{0.01f};
    };

    struct PreProcessStackOutputs {
        Array<float> output_stack;
        double2_t output_pixel_size;
        double2_t input_pixel_size;
    };

    /// Preprocesses the input tilt-series.
    /// \details The input tilt-series is loaded one slice at a time. Slices are median filtered, fourier cropped,
    ///          bandpass filtered, standardized and masked to have smooth edges. Then, slices can be saved to a file
    ///          (one slice at a time) and the output stack is then returned.
    [[nodiscard]]
    auto preProcessStack(
            const path_t& tilt_series_path,
            const PreProcessStackParameters& parameters
    ) -> PreProcessStackOutputs {
        qn::Logger::info("Preprocessing tilt-series...");
        qn::Logger::trace("Compute device: {}", parameters.compute_device);
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
        const auto [output_shape, output_pixel_size] = details::fourierCropDimensions(
                input_shape, input_pixel_size, target_pixel_size, parameters.fit_to_fast_fft_shape);

        qn::Logger::trace("Fourier cropping parameters:\n"
                          "  Fit to fast shape: {}\n"
                          "  Input:  shape={}, pixel_size={:.2f}\n"
                          "  Output: shape={}, pixel_size={:.2f}",
                          parameters.fit_to_fast_fft_shape,
                          input_shape, input_pixel_size, output_shape, output_pixel_size);

        // Setting up dimensions.
        const auto slice_count = input_shape[0];
        const auto slice_shape = dim4_t{1, 1, input_shape[2], input_shape[3]};
        const auto new_slice_shape = dim4_t{1, 1, output_shape[2], output_shape[3]};
        const auto new_stack_shape = dim4_t{slice_count, 1, output_shape[2], output_shape[3]};
        const auto new_slice_center = float2_t{output_shape[2] / 2, output_shape[3] / 2};
        const auto smooth_edge_size = static_cast<float>(std::max(output_shape[2], output_shape[3])) *
                parameters.smooth_edge_percent;

        const bool use_gpu = parameters.compute_device.gpu();
        const bool do_median_filter = parameters.median_filter_window > 1;
        const auto median_window = static_cast<dim_t>(parameters.median_filter_window);
        const auto options = noa::ArrayOption(parameters.compute_device, noa::Allocator::DEFAULT_ASYNC);

        qn::Logger::trace("Median filter: {}\n"
                          "Center & standardize: {}\n"
                          "Smooth edge: {:.1f}%\n"
                          "Exposure filter: {}",
                          do_median_filter ? noa::string::format("true (window={})", median_window) : "false",
                          parameters.center_and_standardize,
                          parameters.smooth_edge_percent * 100.f,
                          parameters.exposure_filter);

        // Input buffers. If compute device is a GPU, we need a transition buffer for the IO.
        // This could be pinned for faster copy, but for now, keep it to pageable memory.
        auto [input_slice, input_slice_fft] = noa::fft::empty<float>(slice_shape, options);
        auto input_slice_io = use_gpu ? noa::memory::empty<float>(slice_shape) : input_slice;

        // Output buffers.
        // For median filtering, we need another buffer because it is an out-of-place operation.
        const auto output_slice_fft = noa::memory::empty<cfloat_t>(new_slice_shape.fft(), options);
        const auto output_stack = noa::memory::empty<float>(new_stack_shape, options);
        auto output_slice_median =
                do_median_filter ? noa::memory::empty<float>(new_slice_shape, options) : noa::Array<float>{};

        // Process one input slice at a time.
        for (dim_t slice_index = 0; slice_index < slice_count; ++slice_index) {
            // Read the current slice from the file.
            // If CPU, input_slice_io is just an alias for input_slice.
            // If GPU, we need an extra copy.
            input_file.readSlice(input_slice_io, slice_index, false);
            if (use_gpu)
                noa::memory::copy(input_slice_io, input_slice);

            noa::fft::r2c(input_slice, input_slice_fft);
            noa::fft::resize<noa::fft::H2H>(input_slice_fft, slice_shape, output_slice_fft, new_slice_shape);

            // We need to do the median filter after the Fourier cropping, but before the bandpass.
            const auto output_slice = output_stack.subregion(slice_index);
            if (do_median_filter) {
                noa::fft::c2r(output_slice_fft, output_slice);
                noa::signal::median2(output_slice, output_slice_median, median_window);
                noa::fft::r2c(output_slice_median, output_slice_fft);
            }

            // TODO Add exposure filter and update the lowpass cutoff.
            noa::signal::fft::bandpass<noa::fft::H2H>(
                    output_slice_fft, output_slice_fft, new_slice_shape,
                    parameters.highpass_parameters[0], parameters.lowpass_parameters[0],
                    parameters.highpass_parameters[1], parameters.lowpass_parameters[1]);

            if (parameters.center_and_standardize)
                noa::signal::fft::standardize<noa::fft::H2H>(output_slice_fft, output_slice_fft, new_slice_shape);

            // Save the output slice directly into the output stack.
            noa::fft::c2r(output_slice_fft, output_slice);
            noa::signal::rectangle(
                    output_slice, output_slice, new_slice_center,
                    new_slice_center - smooth_edge_size, smooth_edge_size);
        }

        qn::Logger::info("Preprocessing tilt-series... done. Took {:.2f}ms\n", timer.elapsed());
        return {output_stack, output_pixel_size, input_pixel_size};
    }

    struct PostProcessStackParameters{
        Device compute_device;

        // Fourier cropping:
        double target_resolution;
        bool fit_to_fast_fft_shape{false};

        // Image processing:
        int32_t median_filter_window{0};
        bool exposure_filter{false};
        float2_t highpass_parameters{0.05, 0.05};
        float2_t lowpass_parameters{0.45, 0.05};
        bool center_and_standardize{false};
        float smooth_edge_percent{0.008f};

        // Transformation:
        InterpMode interpolation_mode = InterpMode::INTERP_LINEAR_FAST;
        BorderMode border_mode = BorderMode::BORDER_ZERO;
    };

    /// Corrects for the in-plane rotation and shifts, as encoded in the metadata, and save the transformed stack.
    /// \details The slices in the output file are saved in the order as specified in the metadata.
    ///          Excluded views are still saved in the output file, but they are not transformed.
    ///          To remove them from the file, simply remove them from the metadata (see MetadataStack::squeeze()).
    void postProcessStack(
            const path_t& input_tilt_series_path,
            const MetadataStack& input_tilt_series_metadata,
            const path_t& output_tilt_series_path,
            const PostProcessStackParameters& parameters
    ) {
        qn::Logger::info("Postprocessing tilt-series...");
        qn::Logger::trace("Compute device: {}", parameters.compute_device);
        Timer timer;
        timer.start();

        // Some files are not encoded properly, so if file encodes a volume, still interpret it as stack of 2D images.
        auto input_file = noa::io::ImageFile(input_tilt_series_path, noa::io::READ);
        dim4_t input_shape = input_file.shape();
        if (input_shape[0] == 1 && input_shape[1] > 1)
            std::swap(input_shape[0], input_shape[1]);
        const auto input_slice_shape = dim4_t{1, 1, input_shape[2], input_shape[3]};

        QN_CHECK(input_shape[1] == 1,
                 "File: {}. A tilt-series was expected, but got image file with shape {}",
                 input_tilt_series_path, input_shape);

        // Fourier crop setup.
        const auto input_pixel_size = double2_t(input_file.pixelSize().get(1));
        const auto target_pixel_size = double2_t(parameters.target_resolution / 2);
        const auto [output_shape, output_pixel_size] = details::fourierCropDimensions(
                input_shape, input_pixel_size, target_pixel_size, parameters.fit_to_fast_fft_shape);
        const auto output_slice_shape = dim4_t{1, 1, output_shape[2], output_shape[3]};
        const auto output_stack_shape = dim4_t{input_tilt_series_metadata.size(), 1, output_shape[2], output_shape[3]};
        const auto output_slice_center = float2_t{output_shape[2] / 2, output_shape[3] / 2};

        qn::Logger::trace("Fourier cropping parameters:\n"
                          "  Fit to fast shape: {}\n"
                          "  Input:  shape={}, pixel_size={:.2f}\n"
                          "  Output: shape={}, pixel_size={:.2f}",
                          parameters.fit_to_fast_fft_shape,
                          input_shape, input_pixel_size, output_shape, output_pixel_size);

        // The metadata should be unscaled, so shifts are at the original pixel size.
        // Here we apply the shifts on the fourier cropped slices, so we need to scale
        // the shifts from the metadata down before applying them.
        const auto shift_scale_factor = float2_t(input_pixel_size / output_pixel_size);

        const bool use_gpu = parameters.compute_device.gpu();
        const bool do_median_filter = parameters.median_filter_window > 1;
        const auto median_window = static_cast<dim_t>(parameters.median_filter_window);
        const auto options = noa::ArrayOption(parameters.compute_device, noa::Allocator::DEFAULT_ASYNC);
        const auto smooth_edge_size = static_cast<float>(std::max(output_shape[2], output_shape[3])) *
                                      parameters.smooth_edge_percent;

        qn::Logger::trace("Median filter: {}\n"
                          "Center & standardize: {}\n"
                          "Smooth edge: {:.1f}%\n"
                          "Exposure filter: {}\n"
                          "Interpolation method: {}\n"
                          "Border mode: {}",
                          do_median_filter ? noa::string::format("true (window={})", median_window) : "false",
                          parameters.center_and_standardize,
                          parameters.smooth_edge_percent * 100.f,
                          parameters.exposure_filter,
                          parameters.interpolation_mode,
                          parameters.border_mode);

        // Input buffers. If compute device is a GPU, we need a transition buffer for the IO.
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

        for (size_t slice_index = 0; slice_index < input_tilt_series_metadata.size(); ++slice_index) {
            const MetadataSlice& slice_metadata = input_tilt_series_metadata[slice_index];
            QN_CHECK(static_cast<dim_t>(slice_metadata.index) < input_shape[0],
                     "Slice index is invalid. This happened because the stack and the metadata don't match");

            // Read the current slice from the file.
            // If CPU, input_slice_io is just an alias for input_slice.
            // If GPU, we need an extra copy.
            input_file.readSlice(input_slice_io, static_cast<dim_t>(slice_metadata.index));
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

            if (parameters.center_and_standardize) {
                noa::signal::fft::standardize<noa::fft::H2H>(
                        output_slice_fft, output_slice_fft, output_slice_shape);
            }

            noa::fft::c2r(output_slice_fft, output_slice);
            noa::signal::rectangle(
                    output_slice, output_slice, output_slice_center,
                    output_slice_center - smooth_edge_size, smooth_edge_size);

            // Apply the transformation encoded in the metadata.
            if (!slice_metadata.excluded) {
                output_slice_texture.update(output_slice);
                const auto slice_shifts = slice_metadata.shifts * shift_scale_factor;
                const auto inv_transform = noa::math::inverse(
                        noa::geometry::translate(output_slice_center) *
                        float33_t(noa::geometry::rotate(noa::math::deg2rad(-slice_metadata.angles[0]))) *
                        noa::geometry::translate(-output_slice_center - slice_shifts)
                );
                noa::geometry::transform2D(output_slice_texture, output_slice_buffer, inv_transform);
            } else {
                std::swap(output_slice, output_slice_buffer); // alike arrays
            }

            if (use_gpu)
                noa::memory::copy(output_slice_buffer, output_slice_buffer_io);
            output_file.writeSlice(output_slice_buffer_io, slice_index);
        }

        qn::Logger::info("Postprocessing tilt-series... done. Took {:.2f}ms\n", timer.elapsed());
    }
}
