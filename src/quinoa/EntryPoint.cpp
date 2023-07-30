#include <noa/Session.hpp>

#include "quinoa/core/Alignment.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/Exception.h"
#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"

#include <noa/Geometry.hpp>

namespace {
    void defocus_field() {
        using namespace noa;

        f64 slice_spacing = 2.1; // A/pix
        Vec2<f64> slice_shape{4000, 4000}; // pix;
        Vec3<f64> slice_angles{175, -45, 0}; // deg

        // preprocess:
        slice_spacing *= 1e-4;
        slice_angles = noa::math::deg2rad(slice_angles);
        fmt::print("{}\n", slice_angles);
        Vec2<f64> slice_center = slice_shape / 2;

        auto compute_z_offset_micrometers = [&](Vec2<f64> point){
            const auto slice_center_3d = (slice_center * slice_spacing).push_front(0);

            // Apply the tilt and elevation.
            const Double44 image2microscope_matrix = // FIXME
                    noa::geometry::translate(slice_center_3d) * // 6. shift back
                    noa::geometry::linear2affine(noa::geometry::rotate_z(slice_angles[0])) * // 5. rotate back
                    noa::geometry::linear2affine(noa::geometry::rotate_x(slice_angles[2])) * // 4. elevation
                    noa::geometry::linear2affine(noa::geometry::rotate_y(slice_angles[1])) * // 3. tilt
                    noa::geometry::linear2affine(noa::geometry::rotate_z(-slice_angles[0])) * // 2. align tilt-axis
                    noa::geometry::translate(-slice_center_3d); // 1. slice rotation center

            const auto point_3d = (point * slice_spacing).push_front(0).push_back(1);
            const Vec3<f64> patch_center_transformed = (image2microscope_matrix * point_3d).pop_back();
            return patch_center_transformed[0];
        };

        // 4 points at the edges.
        Vec2<f64> a00{0, 0};
        Vec2<f64> a01{0, slice_shape[0] - 1};
        Vec2<f64> a10{slice_shape[1] - 1, 0};
        Vec2<f64> a11{slice_shape[1] - 1, slice_shape[0] - 1};

        // Compute the z-offset in micrometers.
        fmt::print("{}\n", compute_z_offset_micrometers(a00));
        fmt::print("{}\n", compute_z_offset_micrometers(a01));
        fmt::print("{}\n", compute_z_offset_micrometers(a10));
        fmt::print("{}\n", compute_z_offset_micrometers(a11));
    }
}

auto main(int argc, char* argv[]) -> int {
    try {
        // Initialize logger before doing anything else.
        qn::Logger::initialize();

        // Parse the options.
        qn::Options options(argc, argv);

        // Adjust global settings.
        qn::Logger::add_logfile(options.files.output_directory / "quinoa.log");
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
        if (options.alignment.run) {
            std::tie(metadata, average_ctf) = tilt_series_alignment(options, metadata);
        }

        // Reconstruction.


    } catch (...) {
        qn::Logger::error(qn::Exception::backtrace());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
