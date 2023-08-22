#include <noa/Session.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/core/Alignment.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Stack.hpp"
#include "quinoa/Exception.h"
#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"

#include <noa/Geometry.hpp>
#include <quinoa/core/Ewise.hpp>

// random tests
namespace {
    using namespace noa;
    using namespace qn;

    auto pad_to_square(const Array<f32>& stack) {
        auto square_shape = stack.shape();
        const auto padded_size = noa::math::max(square_shape[2], square_shape[3]) * 2;
        square_shape[2] = padded_size;
        square_shape[3] = padded_size;
        return noa::memory::resize(stack, square_shape);
    }

    auto fourier_crop() {
        noa::Session::set_thread_limit(20);
        const auto input_filename = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/aligned.mrc");
        const auto output_filename = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/aligned_cropped.mrc");

        const auto options = ArrayOption(Device("gpu"), Allocator::MANAGED);
        const auto stack = noa::io::load_data<f32>(input_filename, true, options);
        const auto stack_output_shape = Shape4<i64>{stack.shape()[0], 1, 1024, 1024};
        const auto slice_output_shape = stack_output_shape.set<0>(1);
        const auto stack_cropped = noa::memory::empty<f32>(stack_output_shape, options);

        for (i64 i = 0; i < stack.shape()[0]; ++i) {
            const auto slice = pad_to_square(stack.subregion(i));
            const auto slice_rfft = noa::fft::r2c(slice);
            const auto slice_rfft_cropped = noa::fft::resize<fft::H2H>(slice_rfft, slice.shape(), slice_output_shape);
            noa::fft::c2r(slice_rfft_cropped, slice_output_shape).to(stack_cropped.subregion(i));
        }

        noa::io::save(stack_cropped, output_filename);
    }

    void test(int argc, char* argv[]) {
        noa::Session::set_thread_limit(16);
        const auto directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/test");
        const auto stack_filename = directory / Path("aligned_cropped.mrc");
        const auto recons_filename = directory / Path("recons.mrc");
        const auto weights_filename = directory / Path("recons_weights.mrc");
        const auto projection_filename = directory / Path("projection.mrc");

        qn::Logger::initialize();
        qn::Options options(argc, argv);
        qn::Logger::add_logfile(options.files.output_directory / "quinoa.log");
        qn::Logger::set_level(options.compute.log_level);
        options.files.input_stack = stack_filename;
        auto metadata = qn::MetadataStack(options);

        const auto array_options = ArrayOption(Device("gpu"), Allocator::MANAGED);

        auto stack = noa::io::load_data<f32>(stack_filename, true, array_options);
        const auto stack_shape = stack.shape();
        const auto slice_shape = stack_shape.set<0>(1);
        const auto stack_rfft = noa::fft::r2c(stack.release(), noa::fft::NORM_DEFAULT, false);
        noa::fft::remap(fft::H2HC, stack_rfft, stack_rfft, stack_shape);
        stack_rfft.eval();

        const auto size = stack_shape[3];
        const auto volume_shape = Shape4<i64>{1, size, size, size};
        auto volume_rfft = noa::memory::zeros<c32>(volume_shape.rfft(), array_options);
        auto volume_rfft_weight = noa::memory::zeros<f32>(volume_shape.rfft(), array_options);

        // Phase-shift to the center of rotation.
        noa::signal::fft::phase_shift_2d<fft::HC2HC>(
                stack_rfft, stack_rfft, stack_shape, -Vec2<f32>(size / 2));

        // Filter.
        auto metadata_insert = metadata;
        metadata_insert.exclude([](const MetadataSlice& slice) {
            return std::abs(slice.angles[1] - 0) < 1e-2;
        });

        // 0: 0.000976563f, 0.000976563f
        // 1: 5*0.000976563f, 0.000976563f
        // 2: 0.000976563f, 2*0.000976563f
        // 3: 0.000976563f, 4*0.000976563f
        // 4: 0.000976563f, 20*0.000976563f
        // 4: 0.000976563f, 20
        const noa::geometry::fft::WindowedSinc insert_windowed_sinc{
                0.000976563f, 20}; // 0.000976563
        const noa::geometry::fft::WindowedSinc z_windowed_sinc{
                0.01f, 0.5f};

        for (const MetadataSlice& slice: metadata_insert.slices()) {
            fmt::print("insert {} degrees\n", slice.angles[1]);
            const auto inv_rotation_matrix = noa::geometry::rotate_y(noa::math::deg2rad(slice.angles[1]))
                    .inverse().as<f32>();

            noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                    stack_rfft.subregion(slice.index_file), slice_shape,
                    volume_rfft, volume_shape,
                    Float22{}, inv_rotation_matrix,
                    insert_windowed_sinc
            );
            noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                    1.f, slice_shape,
                    volume_rfft_weight, volume_shape,
                    Float22{}, inv_rotation_matrix,
                    insert_windowed_sinc
            );
        }

        noa::ewise_binary(volume_rfft, volume_rfft_weight, volume_rfft, qn::correct_multiplicity_t{});
//        noa::signal::fft::phase_shift_3d<fft::HC2HC>(
//                volume_rfft, volume_rfft, volume_shape, Vec3<f32>(size / 2));
//        volume_rfft.eval();

//        noa::io::save(volume_rfft_weight, weights_filename);

        // Filter to account for the specimen thickness.
//        const auto filter = noa::io::load_data<f32>(directory / "filter1.mrc").reshape({1, -1, 1, 1});
//        noa::ewise_binary(filter, noa::math::sum(filter), filter, noa::divide_t{});
//        save_vector_to_text(filter.view(), directory / "filter1.txt");

//        const auto filter_z = noa::indexing::broadcast(filter.to(volume_rfft.options()), volume_rfft.shape());
//        noa::ewise_binary(volume_rfft, filter_z, volume_rfft, noa::multiply_t{});

//        noa::signal::fft::phase_shift_3d<fft::HC2HC>(
//                volume_rfft, volume_rfft, volume_shape, Vec3<f32>(size / 2));
//        volume_rfft.eval();

//        auto slice_rfft = noa::math::sum(volume_rfft, Vec4<bool>{0, 1, 0, 0});
//        noa::io::save(slice_rfft, directory / "sum.mrc");

//        noa::signal::fft::phase_shift_2d<fft::HC2HC>(
//                slice_rfft, slice_rfft, slice_shape, Vec2<f32>(size / 2));
//
//        auto projection = noa::fft::c2r(
//                noa::fft::remap(fft::HC2H, slice_rfft.release(), slice_shape),
//                slice_shape, noa::fft::NORM_DEFAULT, false);
//        noa::io::save(projection, projection_filename);

//        auto chunk = volume_rfft0.subregion(0, noa::indexing::Slice{482, 543});
//        auto slice_rfft = noa::math::sum(chunk, slice_shape.rfft());
//        auto projection = noa::fft::c2r(
//                noa::fft::remap(fft::HC2H, slice_rfft.release(), slice_shape),
//                slice_shape, noa::fft::NORM_DEFAULT, false);
//        noa::io::save(projection, projection_filename);

//        // Forward project.
//        auto projection_rfft = noa::memory::zeros<c32>(slice_shape.rfft(), array_options);
//        noa::geometry::fft::extract_3d<fft::HC2HC>(
//                volume_rfft, volume_shape,
//                projection_rfft, slice_shape,
//                Float22{}, Float33{},
//                z_windowed_sinc
//        );
//        noa::signal::fft::phase_shift_2d<fft::HC2HC>(
//                projection_rfft, projection_rfft, slice_shape, Vec2<f32>(size / 2));
//        projection_rfft.eval();
//
//        noa::io::save(noa::ewise_unary(projection_rfft, noa::abs_one_log_t{}), directory / "projection_weights.mrc");
//        const auto projection2 = noa::fft::c2r(
//                noa::fft::remap(fft::HC2H, projection_rfft.release(), slice_shape),
//                slice_shape, noa::fft::NORM_DEFAULT, false);
//        noa::io::save(projection2, projection_filename);

        noa::signal::fft::phase_shift_3d<fft::HC2HC>(
                volume_rfft, volume_rfft, volume_shape, Vec3<f32>(size / 2));
        const auto volume = noa::fft::c2r(
                noa::fft::remap(fft::HC2H, volume_rfft.release(), volume_shape),
                volume_shape, noa::fft::NORM_DEFAULT, false);
        noa::io::save(volume, recons_filename);
    }

    void test_mask() {
        const auto mask = noa::memory::ones<f32>({1, 1024, 1024, 1024});
        noa::geometry::rectangle(mask, mask, Vec3<f32>{512, 512, 512}, Vec3<f32>{50, 512, 512}, 50);
        const auto mask_rfft = noa::fft::remap(fft::H2HC, noa::fft::r2c(mask), mask.shape());
        noa::io::save(mask, "/home/thomas/Projects/quinoa/tests/ribo_ctf/mask.mrc");
        noa::io::save(noa::ewise_unary(mask_rfft, noa::abs_one_log_t{}),
                      "/home/thomas/Projects/quinoa/tests/ribo_ctf/mask_ps.mrc");
    }

    void test_filter() {
        auto filter = noa::io::load_data<f32>("/home/thomas/Projects/quinoa/tests/ribo_ctf/filter.mrc").flat();
        filter = noa::fft::remap(fft::F2FC, filter.copy(), filter.shape());
        save_vector_to_text(filter.view(), "/home/thomas/Projects/quinoa/tests/ribo_ctf/filter.txt");
    }

    void test_add_to_output() {
        const auto matrices = noa::memory::empty<Float33>(3);
        matrices.span()[0] = noa::geometry::rotate_y(noa::math::deg2rad(6.f));
        matrices.span()[1] = noa::geometry::rotate_y(noa::math::deg2rad(9.f));
        matrices.span()[2] = noa::geometry::rotate_y(noa::math::deg2rad(-3.f));

        const auto shape = Shape4<i64>{1, 1, 1024, 1024};
        const auto input = noa::memory::ones<f32>(shape.rfft().set<0>(3));
        const auto output = noa::memory::zeros<f32>(shape.rfft());
        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                input, shape.set<0>(3), output, shape,
                Float22{}, matrices,
                Float22{}, Float33{},
                {-1, 1}, {0.05f, 0.2f}, false
                );
        noa::io::save(output, "/home/thomas/Projects/quinoa/tests/ribo_ctf/test_add_to_output0.mrc");

        noa::memory::fill(output, 0.f);
        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                input.subregion(0), shape, output, shape,
                Float22{}, matrices.span()[0],
                Float22{}, Float33{},
                {-1, 1}, {0.05f, 0.2f}, true
        );
        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                input.subregion(1), shape, output, shape,
                Float22{}, matrices.span()[1],
                Float22{}, Float33{},
                {-1, 1}, {0.05f, 0.2f}, true
        );
        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                input.subregion(2), shape, output, shape,
                Float22{}, matrices.span()[2],
                Float22{}, Float33{},
                {-1, 1}, {0.05f, 0.2f}, true
        );
        noa::io::save(output, "/home/thomas/Projects/quinoa/tests/ribo_ctf/test_add_to_output1.mrc");
    }
}


auto main(int argc, char* argv[]) -> int {
//    fourier_crop();
//    test_mask();
//    test_filter();
//    test(argc, argv);
//    test_add_to_output();
//    return 0;

    try {
        noa::Timer timer;
        timer.start();

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

        // Register the input stack. The application loads the input stack many times. To save computation,
        // load the input (and unspoiled) stack once and save it inside a static array. The StackLoader will
        // check for it when needed. This is optional as the user may want to save memory...
        if (options.compute.register_input_stack)
            StackLoader::register_input_stack(options.files.input_stack);

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

            // FIXME
            const auto loading_parameters = qn::LoadStackParameters{
                    /*compute_device=*/ options.compute.device,
                    /*allocator=*/ noa::Allocator::DEFAULT_ASYNC,
                    /*precise_cutoff=*/ false,
                    /*rescale_target_resolution=*/ -1,
                    /*rescale_min_size=*/ 1024,
                    /*exposure_filter=*/ false,
                    /*highpass_parameters=*/ {0.01, 0.01},
                    /*lowpass_parameters=*/ {0.8, 0.00},
                    /*normalize_and_standardize=*/ true,
                    /*smooth_edge_percent=*/ 0.02f,
                    /*zero_pad_to_fast_fft_shape=*/ false,
            };
            qn::save_stack(
                    options.files.input_stack,
                    options.files.output_directory / "aligned.mrc",
                    metadata, loading_parameters);
        }

        // Reconstruction.

        qn::Logger::status("Done... took {:.2}s", timer.elapsed() * 1e-3);

    } catch (...) {
        qn::Logger::error(qn::Exception::backtrace());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
