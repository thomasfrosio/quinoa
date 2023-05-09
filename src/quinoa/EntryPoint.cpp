#include <noa/Session.hpp>
#include <noa/IO.hpp>

#include "quinoa/Exception.h"
#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"
#include "quinoa/core/Alignment.h"
#include "quinoa/core/Metadata.h"
//#include "quinoa/core/Reconstruction.h"

namespace qn {
    void tilt_series_alignment(const qn::Options& options) {
        // Parses options early, to quickly throw if there's an invalid option.
        const auto threads = options["compute_cpu_threads"].as<i32>(i32{0});
        const auto device_name = options["compute_device"].as<std::string>("gpu");
        const auto original_stack_filename = options["stack_file"].as<Path>();
        const auto output_directory = options["output_directory"].as<Path>();
        auto metadata = MetadataStack(options);

        // Setting up the workers.
        noa::Session::set_threads(threads);
        auto compute_device = noa::Device(device_name);
        if (compute_device.is_gpu()) {
            // If left unspecified, use the GPU with the most amount of unused memory.
            const size_t pos = device_name.find(':');
            if (pos != std::string::npos && pos != device_name.size() - 1)
                compute_device = noa::Device::most_free(noa::DeviceType::GPU);
            auto stream = noa::Stream(compute_device, noa::StreamMode::ASYNC);
            noa::Stream::set_current(stream);
        }

        // Preprocessing.
        if (options["preprocessing_run"].as<bool>(true)) {
            // TODO Exclude views using mass normalization?

            constexpr std::string_view exclude_key = "preprocessing_exclude_view_indexes";
            const auto exclude_node = options[exclude_key.data()];
            if (!exclude_node.IsNull()) {
                std::vector<i32> exclude_views_idx;
                if (exclude_node.IsSequence())
                    exclude_views_idx = exclude_node.as<std::vector<i32>>();
                else if (exclude_node.IsScalar())
                    exclude_views_idx.emplace_back(exclude_node.as<i32>());
                else
                    QN_THROW("The value of \"{}\" is not recognized", exclude_key);
                metadata.exclude(exclude_views_idx);
            }
        }

        // Alignment.
        if (options["alignment_run"].as<bool>(true)) {
            const auto alignment_max_resolution = options["alignment_max_resolution"].as<f64>(10);

            const auto pairwise_alignment_parameters = PairwiseAlignmentParameters {
                /*compute_device=*/ compute_device,
                /*maximum_resolution=*/ alignment_max_resolution,
                /*search_rotation_offset=*/ options["alignment_rotation_offset"].as<bool>(true),
                /*search_tilt_offset=*/ options["alignment_tilt_offset"].as<bool>(true),
                /*output_directory=*/ output_directory,
                /*debug_directory=*/ output_directory / "debug_pairwise",
            };
            pairwise_alignment(original_stack_filename, metadata,
                               pairwise_alignment_parameters);

//            const auto project_matching_parameters = ProjectionMatchingParameters{
//                    {}, // max_shift
//                    0.1f, // smooth_edge_percent
//
//                    /*projection_slice_z_radius=*/ 0.0005f,
//                    /*projection_cutoff=*/ 0.5f,
//                    /*projection_max_tilt_angle=*/ 45,
//
//                    /*highpass_filter=*/ {0.1, 0.05},
//                    /*lowpass_filter=*/ {0.35, 0.1},
//
//                    /*max_iterations=*/ 2,
//                    true, // center_shifts
//                    output_directory / "debug_pm" // debug_directory
//            };

        }

        // TODO CTF, update tilt and elevation offset.
        // TODO Another initial global alignment

        // TODO Projection global alignment

//        if (options["reconstruction_run"].as<bool>(true)) {
//            const auto reconstruction_resolution = options["reconstruction_resolution"].as<double>(14);
//            const auto loading_parameters = LoadStackParameters{
//                    compute_device, // Device compute_device
//                    int32_t{1}, // int32_t median_filter_window
//                    reconstruction_resolution, // double target_resolution
//                    false, // bool exposure_filter
//                    {0.05, 0.04}, // float2_t highpass_parameters
//                    {0.45, 0.05}, // float2_t lowpass_parameters
//                    true, // bool normalize_and_standardize
//                    0.08f, // float smooth_edge_percent
//                    true, // bool zero_pad_to_fast_fft_shape
//            };
//
//            const auto reconstruction_thickness = options["reconstruction_thickness"].as<double>(320);
//            const auto volume_thickness = static_cast<const dim_t>(reconstruction_thickness);
//            // TODO Convert the μm to pixels
//            const auto reconstruction_parameters = TiledReconstructionParameters{
//                    volume_thickness, // volume thickness
//                    options["reconstruction_cube_size"].as<dim_t>(dim_t{64}) // cube size
//            };
//
//            const auto reconstruction = qn::tiledReconstruction(
//                    original_stack_filename, metadata,
//                    loading_parameters,
//                    reconstruction_parameters);
//
//            const auto tomogram_filename =
//                    output_directory /
//                    noa::string::format("{}_tomogram.mrc", original_stack_filename.stem().string());
//            noa::io::save(reconstruction, tomogram_filename);
//        }

        // TODO Once reconstructed, molecular mask and forward project to remove noise, then start alignment again?
        // TODO CTF (maybe one day...)
    }
}

int main(int argc, char* argv[]) {
    // Initialize everything that needs to be initialized.
    noa::Session session("quinoa", "quinoa.log", noa::Logger::VERBOSE); // TODO set verbosity
    // TODO Set CUDA memory buffer
    // TODO Set number of cached cufft plans.

    try {
        qn::Options options(argc, argv);
        qn::tilt_series_alignment(options);

    } catch (...) {
        qn::Logger::error(qn::Exception::backtrace());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
