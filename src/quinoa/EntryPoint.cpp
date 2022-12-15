#include <noa/Session.h>
#include <noa/IO.h>

#include "quinoa/Exception.h"
#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"
#include "quinoa/core/Alignment.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Reconstruction.h"

namespace qn {
    void tiltSeriesAlignment(const qn::Options& options) {
        // Parses options early, to quickly throw if there's an invalid option.
        const auto threads = options["compute_cpu_threads"].as<size_t>(size_t{0});
        const auto device_name = options["compute_device"].as<std::string>("gpu");
        const auto original_stack_filename = options["stack_file"].as<path_t>();
        const auto output_directory = options["output_directory"].as<path_t>();
        const auto alignment_resolution = options["alignment_resolution"].as<double>();
        MetadataStack metadata(options);

        // Setting up the workers.
        noa::Session::threads(threads);
        auto compute_device = noa::Device(device_name);
        if (compute_device.gpu()) {
            // If left unspecified, use the GPU with the most amount of unused memory.
            const size_t pos = device_name.find(':');
            if (pos != std::string::npos && pos != device_name.size() - 1)
                compute_device = noa::Device::mostFree(noa::Device::GPU);
            auto stream = noa::Stream(compute_device, noa::Stream::ASYNC);
            noa::Stream::current(stream);
        }

        // Preprocessing.
        const auto preprocessing_parameters = PreProcessStackParameters{
                compute_device,

                // Fourier cropping:
                alignment_resolution, // target_resolution
                true, // fit_to_fast_fft_shape

                // Image processing:
                0, // median_filter_window
                false, // exposure_filter
                {0.10, 0.10}, // highpass_parameters
                {0.45, 0.05}, // lowpass_parameters
                true, // center_and_standardize
                0.1f // smooth_edge_percent
        };
        const auto [preprocessed_stack, preprocessed_pixel_size, original_pixel_size] =
                preProcessStack(original_stack_filename, preprocessing_parameters);

        if (options["save_preprocessed_stack"].as<bool>(false)) {
            const auto cropped_stack_filename =
                    output_directory /
                    noa::string::format("{}_preprocessed{}",
                                        original_stack_filename.stem().string(),
                                        original_stack_filename.extension().string());
            noa::io::save(preprocessed_stack, float2_t(preprocessed_pixel_size), cropped_stack_filename);
        }

        // TODO Initial tilt-axis angle
        // TODO Tilt-angles offset
        // TODO Exclude views using mass normalization?

        // Initial alignment.
        {
            // Focus in the center of the tilt-series. This seems to be the safest approach.
            // The edges can be (more) problematic, especially for thick samples. Indeed, they
            // are the regions that vary the most with the tilt, making them difficult to track.
            // For this reason, the stack is filtered with (very) smooth mask located at the
            // center of the FOV. This should focus the FOV in the center of the stack, but still
            // include enough information for the cross-correlation, even with large shifts.

            // Pairwise cosine:
            const auto pairwise_cosine_parameters = PairwiseCosineParameters {
                    {}, // max_shift
                    0.35f, // smooth_edge_percent

                    {0.03, 0.03}, // highpass_filter
                    {0.40, 0.05}, // lowpass_filter

                    true, // center_tilt_axis
                    noa::InterpMode::INTERP_LINEAR_FAST,
                    {}//output_directory / "stretch_alignment" // debug_directory
            };

            // Projection matching:
            const auto project_matching_parameters = ProjectionParameters{

            };

            const auto global_alignment_parameters = InitialGlobalAlignmentParameters{
                    true, // do_pairwise_cosine_alignment
                    false, // do_projection_matching_alignment
                    output_directory / "debug_initial_alignment" // debug_directory
            };

            const auto pre_scale = float2_t(original_pixel_size / preprocessed_pixel_size);
            for (auto& slice: metadata.slices())
                if (!slice.excluded)
                    slice.shifts *= pre_scale;

            initialGlobalAlignment(preprocessed_stack, metadata,
                                   global_alignment_parameters,
                                   pairwise_cosine_parameters,
                                   project_matching_parameters);

            const auto post_scale = 1 / pre_scale;
            for (auto& slice: metadata.slices())
                if (!slice.excluded)
                    slice.shifts *= post_scale;
        }

        // Refinement and deformations.
        {
            // TODO
        }

        if (options["save_aligned_stack"].as<bool>(false)) {
            const auto post_process_stack_parameters = PostProcessStackParameters {
                    compute_device,

                    // Fourier cropping:
                    alignment_resolution, // target_resolution
                    true, // fit_to_fast_fft_shape

                    // Image processing:
                    0, // median_filter_window
                    false, // exposure_filter
                    {0.05, 0.05}, // highpass_parameters
                    {0.45, 0.05}, // lowpass_parameters
                    true, // center_and_standardize
                    0.01f, // smooth_edge_percent

                    // Transformation:
                    noa::InterpMode::INTERP_LINEAR_FAST,
                    noa::BorderMode::BORDER_ZERO
            };

            const auto aligned_stack_filename =
                    options["output_directory"].as<path_t>() /
                    noa::string::format("{}_aligned{}",
                                        original_stack_filename.stem().string(),
                                        original_stack_filename.extension().string());

            qn::postProcessStack(original_stack_filename, metadata.squeeze(),
                                 aligned_stack_filename,
                                 post_process_stack_parameters);
        }

        if (options["save_tomogram"].as<bool>(true)) {
//            const auto reconstruction_stack_parameters = PreProcessStackParameters{
//                    compute_device,
//
//                    // Fourier cropping:
//                    alignment_resolution, // target_resolution
//                    true, // fit_to_fast_fft_shape
//
//                    // Image processing:
//                    0, // median_filter_window
//                    false, // exposure_filter
//                    {0.05, 0.05}, // highpass_parameters
//                    {0.48, 0.05}, // lowpass_parameters
//                    true, // center_and_standardize
//                    0.08f // smooth_edge_percent
//            };
//            auto [reconstruction_stack, reconstruction_pixel_size, _] =
//                    preProcessStack(original_stack_filename, reconstruction_stack_parameters);
//            reconstruction_stack = reconstruction_stack.subregion(noa::indexing::slice_t{1, 41});

            const auto reconstruction_stack = noa::io::load<float>(
                    "/home/thomas/Projects/quinoa/tests/ribo3_reconstruct/tilt1_aligned.mrc", true, compute_device);
            MetadataStack aligned_metadata(options);
            aligned_metadata.squeeze();
            for (auto& slice: aligned_metadata.slices())
                slice.angles[0] = 0;

            const auto reconstruction_parameters = TiledReconstructionParameters{
                320, // volume thickness
                64 // cube size
            };

            // Scale the shifts to the reconstruction stack.
//            const auto pre_scale = float2_t(original_pixel_size / reconstruction_pixel_size);
//            for (auto& slice: metadata.slices())
//                if (!slice.excluded)
//                    slice.shifts *= pre_scale;

            const auto reconstruction = qn::tiledReconstruction(
                    reconstruction_stack, aligned_metadata, reconstruction_parameters);
//            const auto average_pixel_size = noa::math::sum(reconstruction_pixel_size) / 2;
//            const auto tomogram_pixel_size = float3_t(average_pixel_size);

            const auto tomogram_filename =
                    output_directory /
                    noa::string::format("{}_tomogram.mrc", original_stack_filename.stem().string());
            noa::io::save(reconstruction, tomogram_filename);
        }

        // TODO Reconstruction
        // TODO Once reconstructed, molecular mask and forward project to remove noise, then start alignment again?
        // TODO CTF (maybe one day...)
    }
}

int main(int argc, char* argv[]) {
    // Initialize everything that needs to be initialized.
    noa::Session session("quinoa", "quinoa.log", Logger::VERBOSE); // TODO set verbosity
    // TODO Set CUDA memory buffer
    // TODO Set number of cached cufft plans.

    try {
        qn::Options options(argc, argv);
        qn::tiltSeriesAlignment(options);

    } catch (...) {
        qn::Logger::error(qn::Exception::backtrace());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
