#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/FFT.h>
#include <noa/IO.h>

#include "quinoa/Types.h"
#include "quinoa/Exception.h"

#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"

#include "quinoa/core/Alignment.h"
#include "quinoa/core/Geometry.h"
#include "quinoa/core/Signal.h"
#include "quinoa/core/Metadata.h"

namespace qn {
    void tiltSeriesAlignment(const qn::Options& options) {
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

        // Parses options early, to quickly throw if there's an invalid option.
        MetadataStack metadata(options);

        // Fourier crop to target resolution.
        auto original_stack_filename = options["stack_file"].as<path_t>();
        auto cropped_stack_filename =
                options["output_directory"].as<path_t>() /
                string::format("{}_cropped{}",
                               original_stack_filename.stem().string(),
                               original_stack_filename.extension().string());
        const auto target_resolution = options["alignment_resolution"].as<float>(16.f);
        const auto [original_pixel_size, cropped_pixel_size] =
                qn::signal::fourierCrop(original_stack_filename, cropped_stack_filename,
                                        float2_t(target_resolution / 2), device);

        // Exclude bad images.
        const auto exclude_blank_views = options["exclude_blank_views"].as<bool>(false);
        if (exclude_blank_views) {
            // TODO
        }

        // Preprocess images.
        // Exposure, Remove gradient, center and standardize.

        // TODO Find the initial rotation angle.

        // Initial translation alignment using cosine stretched as input.

        metadata.sort("tilt");
        if (options["exclude_views_from_stack"].as<bool>(false))
            metadata.squeeze();
        MetadataStack new_metadata = metadata;

        // TODO From this point, we could iterate for multiple resolutions.
        //      Start very low res and increase up to resolution target.

        const Array<float> cropped_stack = noa::io::load<float>(cropped_stack_filename);
        qn::align::shiftPairwiseCosine(cropped_stack, new_metadata, device, {}, true);

        { // Logging
            MetadataStack::logUpdate(metadata, new_metadata, cropped_pixel_size / original_pixel_size);
            auto cropped_coarse_stack_filename =
                    options["output_directory"].as<path_t>() /
                    string::format("{}_coarse{}",
                                   cropped_stack_filename.stem().string(),
                                   cropped_stack_filename.extension().string());
            qn::geometry::transform(cropped_stack, new_metadata, cropped_pixel_size,
                                    cropped_coarse_stack_filename, device);
        }


        // TODO Check GPU memory to see if we push the entire stack there.
        //      It prevents having to copy the slices to the GPU every time.

        // Preprocess for projection matching:
        qn::align::preprocessProjectionMatching(cropped_stack, new_metadata, device, 0.08f, 0.05f);

        const size_t max_iterations = 3;
        for (size_t i = 0; i < max_iterations; ++i) {
            metadata = new_metadata;

            qn::align::shiftProjectionMatching(cropped_stack, new_metadata, device);
            // qn::align::rotationProjectionMatching(stack, new_metadata, device);

            { // Logging
                MetadataStack::logUpdate(metadata, new_metadata, cropped_pixel_size / original_pixel_size);
                auto cropped_coarse_stack_filename =
                        options["output_directory"].as<path_t>() /
                        string::format("{}_pm{:0>2}{}",
                                       cropped_stack_filename.stem().string(), i,
                                       cropped_stack_filename.extension().string());
                qn::geometry::transform(cropped_stack, new_metadata, cropped_pixel_size,
                                        cropped_coarse_stack_filename, device);
            }
        }

        // TODO Reconstruction
        // TODO Once reconstructed, molecular mask and forward project to remove noise, then start alignment again?
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
