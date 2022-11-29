#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/FFT.h>
#include <noa/IO.h>

#include "quinoa/Types.h"
#include "quinoa/Exception.h"

#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"

#include "quinoa/core/PairwiseCosine.h"
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Geometry.h"
#include "quinoa/core/Signal.h"
#include "quinoa/core/Metadata.h"

namespace qn {
    void tiltSeriesAlignment(const qn::Options& options) {
        // Parses options early, to quickly throw if there's an invalid option.
        MetadataStack metadata(options);

        // Number of (OpenMP) threads:
        const auto threads = options["compute_cpu_threads"].as<size_t>(size_t{0});
        noa::Session::threads(threads);

        // Array options. Use the "most free" device if the ID is not specified.
        const auto device_name = options["compute_device"].as<std::string>("gpu");
        Device device(device_name);
        if (device.type() == Device::GPU) {
            const size_t pos = device_name.find(':');
            if (pos != std::string::npos && pos != device_name.size() - 1)
                device = Device::mostFree(Device::GPU);
            Stream stream(device, Stream::ASYNC);
            Stream::current(stream);
        }

        std::vector<float> alignment_resolutions; // TODO Use flat_vector instead
        if (options["alignment_resolution"].IsScalar())
            alignment_resolutions.emplace_back(options["alignment_resolution"].as<float>());
        else if (options["alignment_resolution"].IsSequence())
            alignment_resolutions = options["alignment_resolution"].as<std::vector<float>>();
        else
            alignment_resolutions = {16.f, 12.f}; // TODO Use pixel size and image size
        std::sort(alignment_resolutions.begin(), alignment_resolutions.end(), std::greater{});

        // Alignment resolution loop. Progressively increase the resolution of the input stack.
        const MetadataStack metadata_before_alignment = metadata;
        for (size_t alignment_resolution_iter = 0;
             alignment_resolution_iter < alignment_resolutions.size();
             ++alignment_resolution_iter) {

            // Fourier crop to target resolution and preprocess.
            auto original_stack_filename = options["stack_file"].as<path_t>();
            auto cropped_stack_filename =
                    options["output_directory"].as<path_t>() /
                    string::format("{}_cropped_iter{:>02}{}",
                                   original_stack_filename.stem().string(),
                                   alignment_resolution_iter,
                                   original_stack_filename.extension().string());
            const auto target_resolution = alignment_resolutions[alignment_resolution_iter];
            const auto [original_pixel_size, cropped_pixel_size] =
                    qn::signal::fourierCrop(original_stack_filename, cropped_stack_filename,
                                            float2_t(target_resolution / 2), device);
            // TODO Exposure filter.

            // Exclude bad images.
            if (alignment_resolution_iter == 0) {
                if (options["exclude_blank_views"].as<bool>(false)) {
                    // TODO Mass normalization
                }

                if (options["exclude_views_from_stack"].as<bool>(false))
                    metadata.squeeze();
            }

            // TODO Check GPU memory to see if we push the entire stack there.
            //      It prevents having to copy the slices to the GPU every time.
            //      Instead of loading again, couldn't we return the stack in fourierCrop?
            Array cropped_stack = noa::io::load<float>(cropped_stack_filename);

            // Scale the metadata to the current pixel size.
            for (auto& slice: metadata.slices())
                if (!slice.excluded)
                    slice.shifts *= original_pixel_size / cropped_pixel_size;

            if (alignment_resolution_iter == 0) {
                // TODO Find the initial rotation angle.
            }
            // TODO Find the tilt offset.

            // Cosine stretching alignment:
            // For the first iteration, run the alignment again with the new shifts.
            // At this point, the stack should be preprocessed, which includes the highpass
            // to remove the density gradients and the zero-taper.
            {
                qn::alignment::PairwiseCosine pairwise_cosine(cropped_stack.shape(), device);
                pairwise_cosine.updateShifts(cropped_stack, metadata);
                if (alignment_resolution_iter == 0)
                    pairwise_cosine.updateShifts(cropped_stack, metadata);
            }

            // Projection matching alignment:
            {
                qn::alignment::ProjectionMatching projection_matching(cropped_stack.shape(), device);
                const size_t max_iterations = 1; // TODO Find convergence
                for (size_t i = 0; i < max_iterations; ++i) {
                    projection_matching.updateShift(cropped_stack, metadata);
                    // TODO Rotation alignment
                }
            }

            auto cropped_aligned_stack_filename =
                    options["output_directory"].as<path_t>() /
                    string::format("{}_iter{:0>2}{}",
                                   cropped_stack_filename.stem().string(),
                                   alignment_resolution_iter,
                                   cropped_stack_filename.extension().string());
            qn::geometry::transform(cropped_stack, metadata,
                                    cropped_aligned_stack_filename,
                                    device, cropped_pixel_size);

            // Scale shifts back to original pixel size.
            for (auto& slice: metadata.slices())
                if (!slice.excluded)
                    slice.shifts *= cropped_pixel_size / original_pixel_size;

            MetadataStack::logUpdate(metadata_before_alignment, metadata);
        }

        // TODO Reconstruction
        // TODO Once reconstructed, molecular mask and forward project to remove noise, then start alignment again?
        // TODO Patch local alignment with BSpline grid?
        // TODO CTF (maybe one day...)
    }
}

int main(int argc, char* argv[]) {
    // Initialize everything that needs to be initialized.
    noa::Session session("quinoa", "quinoa.log", Logger::VERBOSE); // TODO set verbosity

    try {
        qn::Options options(argc, argv);
        qn::tiltSeriesAlignment(options);

    } catch (...) {
        qn::Logger::error(qn::Exception::backtrace());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
