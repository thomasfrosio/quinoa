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
        auto stack_file = options["stack_file"].as<path_t>();

        // Fourier crop to target resolution.
        const auto target_resolution = options["alignment_resolution"].as<float>(16.f);
        float2_t target_pixel_size(target_resolution / 2);
        path_t output_filename =
                options["output_directory"].as<path_t>() /
                string::format("{}_cropped{}", stack_file.stem().string(), stack_file.extension().string());
        qn::signal::fourierCrop(stack_file, output_filename, target_pixel_size, device);
        stack_file = std::move(output_filename);

        // Exclude bad images.
        const auto exclude_blank_views = options["exclude_blank_views"].as<bool>(false);
        if (exclude_blank_views) {
            // TODO Implement, with ability to update tilt-scheme as well
        }

        // Preprocess images.
        // Exposure, Remove gradient, center and standardize.

        // TODO Find the initial rotation angle.

        // Initial translation alignment using cosine stretched as input.
        io::ImageFile file(stack_file, io::READ);
        const Array<float> stack = file.read<float>();
        const float2_t pixe_size(file.pixelSize().get(1));
        file.close();

        metadata.sort("tilt");
        if (options["exclude_views_from_stack"].as<bool>(false))
            metadata.squeeze();
        MetadataStack new_metadata = metadata;

        // TODO From this point, we could iterate for multiple resolutions.
        //      Start very low res and increase up to resolution target.

        qn::align::shiftPairwiseCosine(stack, new_metadata, device);
        MetadataStack::logUpdate(metadata, new_metadata);

        output_filename =
                options["output_directory"].as<path_t>() /
                string::format("{}_coarse{}", stack_file.stem().string(), stack_file.extension().string());
        qn::geometry::transform(stack, new_metadata, pixe_size, output_filename, device);

        // TODO Check GPU memory to see if we push the entire stack there.
        //      It prevents having to copy the slices to the GPU every time.

        // Preprocess for projection matching:
        qn::align::preprocessProjectionMatching(stack, new_metadata, device);

        const size_t max_iterations = 1;
        for (size_t i = 0; i < max_iterations; ++i) {
            qn::align::shiftProjectionMatching(stack, new_metadata, device);
            // qn::align::rotationProjectionMatching(stack, new_metadata, device);

            // Log results.
            MetadataStack::logUpdate(metadata, new_metadata);
            metadata = new_metadata;

            output_filename =
                    options["output_directory"].as<path_t>() /
                    string::format("{}_iter{:0<1}{}", stack_file.stem().string(), i, stack_file.extension().string());
            qn::geometry::transform(stack, metadata, pixe_size, output_filename, device);
        }

        // TODO Reconstruction
        // TODO Once reconstructed, molecular mask and backproject to remove noise, then start alignment again?
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
