#include <noa/Session.hpp>

#include "quinoa/core/Alignment.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/Exception.h"
#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"

int main(int argc, char* argv[]) {
    try {
        // Initialize logger before doing anything else.
        qn::Logger::initialize("quinoa.log");

        // Parse the options.
        qn::Options options(argc, argv);

        // Adjust global settings.
        qn::Logger::set_level(options.compute.log_level);
        noa::Session::set_cuda_lazy_loading();
        noa::Session::set_thread_limit(options.compute.n_cpu_threads);

        // Create a user-async stream for the GPU, but ensure that the CPU stream is synchronous.
        if (options.compute.device.is_gpu())
            noa::Stream::set_current(noa::Stream(options.compute.device, noa::StreamMode::ASYNC));
        noa::Stream::set_current(noa::Stream(noa::Device{}, noa::StreamMode::DEFAULT));

        // Initialize the metadata early in case the parsing fails.
        auto metadata = qn::MetadataStack(options);

        // Preprocessing.
        if (options.preprocessing.run) {
            if (options.preprocessing.exclude_blank_views) {
                // TODO Exclude views using mass normalization?
            }

            if (!options.preprocessing.exclude_view_indexes.empty())
                metadata.exclude(options.preprocessing.exclude_view_indexes);
        }

        // Alignment.
        qn::CTFAnisotropic64 average_ctf;
        std::vector<noa::f64> per_view_defocus;
        if (options.alignment.run) {
            std::tie(metadata, average_ctf, per_view_defocus) = align(options, metadata);
        }

        // Reconstruction.


    } catch (...) {
        qn::Logger::error(qn::Exception::backtrace());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
