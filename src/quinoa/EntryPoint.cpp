#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/FFT.h>
#include <noa/IO.h>

#include "quinoa/Types.h"
#include "quinoa/Exception.h"

#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"

#include "quinoa/core/Alignment.h"
#include "quinoa/core/Signal.h"
#include "quinoa/core/Metadata.h"

namespace qn {
    void tiltSeriesAlignment(const qn::Options& options) {
        // Number of (OpenMP) threads:
        const auto threads = options["compute_cpu_threads"].as<size_t>(size_t{0});
        Stream::current(Device{}).cpu().threads(threads);

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

        MetadataStack stack_metadata(options);
        auto stack_file = options["stack_file"].as<path_t>();

        // Fourier crop to target resolution.
        const auto target_resolution = options["alignment_resolution"].as<float>(16.f);
        float2_t target_pixel_size(target_resolution / 2);
        const path_t output_filename =
                options["output_directory"].as<path_t>() /
                string::format("{}_cropped{}", stack_file.stem().string(), stack_file.extension().string());
        qn::signal::fourierCrop(stack_file, output_filename, target_pixel_size, device);
        stack_file = output_filename;

        // Exclude bad images.
        const auto exclude_blank_views = options["exclude_blank_views"].as<bool>(false);
        if (exclude_blank_views) {
            // TODO Implement, with ability to update tilt-scheme as well
        }

        // Preprocess images.
        // Exposure, Remove gradient, center and standardize.


        // TODO Find the initial rotation angle.

        // Initial translation alignment using cosine stretched as input.
        const Array<float> stack = io::load<float>(stack_file);
        std::vector<float2_t> shifts = qn::align::shiftPairwiseCosine(stack, stack_metadata, device, {});
        stack_metadata.shifts(shifts).centerShifts();


        // see AreTomo...
    }
}

int main(int argc, char* argv[]) {
    // Initialize everything that needs to be initialized.
    noa::Session session("quinoa", "quinoa.log"); // TODO set verbosity

    try {
        qn::Options options(argc, argv);
        qn::tiltSeriesAlignment(options);

    } catch (...) {
        qn::Logger::error(qn::Exception::backtrace());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
