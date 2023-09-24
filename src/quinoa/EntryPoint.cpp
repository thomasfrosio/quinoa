#include <noa/Session.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/core/Alignment.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Stack.hpp"
#include "quinoa/core/CommonArea.hpp"
#include "quinoa/core/ExcludeViews.hpp"
#include "quinoa/core/Reconstruction.hpp"

#include "quinoa/Exception.h"
#include "quinoa/io/Logging.h"
#include "quinoa/io/Options.h"


#include <noa/Geometry.hpp>
#include <quinoa/core/Ewise.hpp>

// random tests
namespace {
    using namespace noa;
    using namespace qn;
//
//    auto pad_to_square(const Array<f32>& stack) {
//        auto square_shape = stack.shape();
//        const auto padded_size = noa::math::max(square_shape[2], square_shape[3]) * 2;
//        square_shape[2] = padded_size;
//        square_shape[3] = padded_size;
//        return noa::memory::resize(stack, square_shape);
//    }
//
//    auto fourier_crop() {
//        noa::Session::set_thread_limit(20);
//        const auto input_filename = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/aligned.mrc");
//        const auto output_filename = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/aligned_cropped.mrc");
//
//        const auto options = ArrayOption(Device("gpu"), Allocator::MANAGED);
//        const auto stack = noa::io::load_data<f32>(input_filename, true, options);
//        const auto stack_output_shape = Shape4<i64>{stack.shape()[0], 1, 1024, 1024};
//        const auto slice_output_shape = stack_output_shape.set<0>(1);
//        const auto stack_cropped = noa::memory::empty<f32>(stack_output_shape, options);
//
//        for (i64 i = 0; i < stack.shape()[0]; ++i) {
//            const auto slice = pad_to_square(stack.subregion(i));
//            const auto slice_rfft = noa::fft::r2c(slice);
//            const auto slice_rfft_cropped = noa::fft::resize<fft::H2H>(slice_rfft, slice.shape(), slice_output_shape);
//            noa::fft::c2r(slice_rfft_cropped, slice_output_shape).to(stack_cropped.subregion(i));
//        }
//
//        noa::io::save(stack_cropped, output_filename);
//    }
//
//    void test(const Path& stack_filename, MetadataStack& metadata) {
//        auto meta = metadata;
//        const auto directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/test2");
//        const auto options = ArrayOption(Device("gpu"), Allocator::MANAGED);
//
//        const auto loading_parameters = LoadStackParameters{
//                /*compute_device=*/ Device(),
//                /*allocator=*/ Allocator::DEFAULT,
//                /*precise_cutoff=*/ true,
//                /*rescale_target_resolution=*/ 40.,
//                /*rescale_min_size=*/ 512,
//                /*exposure_filter=*/ false,
//                /*highpass_parameters=*/ {0.01, 0.01},
//                /*lowpass_parameters=*/ {0.4, 0.1},
//                /*normalize_and_standardize=*/ true,
//                /*smooth_edge_percent=*/ 0.1f,
//                /*zero_pad_to_fast_fft_shape=*/ true,
//        };
//        auto [stack, stack_spacing, file_spacing] =
//                load_stack(stack_filename, meta, loading_parameters);
//        meta.rescale_shifts(file_spacing, stack_spacing);
//
//        // Common-area.
//        {
//            auto common_area = CommonArea(stack.shape().filter(2, 3), meta);
//            common_area.mask_views(stack.view(), stack.view(), meta, 0.05);
//        }
//
//        const auto stack_shape = stack.shape();
//        const auto slice_shape = stack_shape.set<0>(1);
//        const auto slice_center = MetadataSlice::center(slice_shape);
//
//        const auto size_padded = noa::math::max(stack_shape.filter(2, 3)) * 2;
//        const auto size_padded_center = MetadataSlice::center(Shape2<i64>{size_padded});
//        const auto slice_padded_shape = Shape4<i64>{1, 1, size_padded, size_padded};
//        const auto volume_shape = Shape4<i64>{1, size_padded, size_padded, size_padded};
//
//        auto volume_rfft = noa::memory::zeros<c32>(volume_shape.rfft()).to(options);
//        auto volume_rfft_weight = noa::memory::zeros<f32>(volume_shape.rfft()).to(options);
//
//        // Options
//        auto metadata_insert = meta;
//        metadata_insert.exclude([](const MetadataSlice& slice) {
//            return std::abs(slice.angles[1]) > 29;
//        }, /*reset_index_field=*/ false);
//        const noa::geometry::fft::WindowedSinc insert_windowed_sinc{
//                0.001587302f, 0.05f}; // 0.000976563
//        const noa::geometry::fft::WindowedSinc z_windowed_sinc{
//                0.067f, 0.3f};// 0.02f, 0.5f
//
//        for (const MetadataSlice& slice: metadata_insert.slices()) {
//            fmt::print("insert {} degrees\n", slice.angles[1]);
//            const auto inv_rotation_matrix = noa::geometry::euler2matrix(noa::math::deg2rad(
//                    Vec3<f64>{-slice.angles[0], slice.angles[1], slice.angles[2]}), "zyx", false)
//                    .transpose().as<f32>();
//
////            const auto slice_padded = noa::memory::resize(stack.subregion(slice.index), slice_padded_shape);
//            const auto slice_padded = stack.subregion(slice.index);
//            noa::io::save(slice_padded, directory / "slice_padded.mrc");
//            const auto slice_padded_rfft = noa::fft::r2c(slice_padded).to(options);
//            noa::fft::remap(fft::H2HC, slice_padded_rfft, slice_padded_rfft, slice_shape);
//
//            noa::signal::fft::phase_shift_2d<fft::HC2HC>(
//                    slice_padded_rfft, slice_padded_rfft, slice_shape,
//                    -slice_center - slice.shifts.as<f32>());
//
//            noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
//                    slice_padded_rfft, slice_shape,
//                    volume_rfft, volume_shape,
//                    Float22{}, inv_rotation_matrix,
//                    insert_windowed_sinc
//            );
//            noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
//                    1.f, slice_shape,
//                    volume_rfft_weight, volume_shape,
//                    Float22{}, inv_rotation_matrix,
//                    insert_windowed_sinc
//            );
//        }
//
////        noa::io::save(volume_rfft_weight, directory / "weights.mrc");
//        noa::ewise_binary(volume_rfft, volume_rfft_weight.release(), volume_rfft, qn::correct_multiplicity_t{});
//        volume_rfft = volume_rfft.to_cpu();
//
//        // Forward project.
//        auto metadata_extract = meta;
//        metadata_extract.exclude([](const MetadataSlice& slice) {
//            return std::abs(slice.angles[1] - 30) > 1e-2;
//        }, /*reset_index_field=*/ false);
//        const auto m_eslice = metadata_extract[0];
//        fmt::print("extract {} degrees\n", m_eslice.angles[1]);
//        const auto fwd_rotation_matrix = noa::geometry::euler2matrix(noa::math::deg2rad(
//                Vec3<f64>{-m_eslice.angles[0], m_eslice.angles[1], m_eslice.angles[2]}), "zyx", false);
//
//        auto projection_rfft = noa::memory::empty<c32>(slice_shape.rfft());
//        noa::geometry::fft::extract_3d<fft::HC2HC>(
//                volume_rfft, volume_shape,
//                projection_rfft, slice_shape,
//                Float22{}, fwd_rotation_matrix.as<f32>(),
//                z_windowed_sinc
//        );
//        noa::signal::fft::phase_shift_2d<fft::HC2HC>(
//                projection_rfft, projection_rfft, slice_shape,
//                Vec2<f32>(slice_shape.vec().filter(2, 3) / 2));
//
////        noa::io::save(noa::ewise_unary(projection_rfft, noa::abs_one_log_t{}), directory / "projection_weights.mrc");
//        const auto projection2 = noa::fft::c2r(
//                noa::fft::remap(fft::HC2H, projection_rfft.release(), slice_shape),
//                slice_shape, noa::fft::NORM_DEFAULT, false);
//        noa::io::save(projection2, directory / "extracted.mrc");
//
//        noa::signal::fft::phase_shift_3d<fft::HC2HC>(
//                volume_rfft, volume_rfft, volume_shape,
//                Vec3<f32>(volume_shape.filter(1, 2, 3).vec() / 2));
//        const auto volume = noa::fft::c2r(
//                noa::fft::remap(fft::HC2H, volume_rfft.release(), volume_shape),
//                volume_shape, noa::fft::NORM_DEFAULT, false);
//        noa::io::save(volume, directory / "recons.mrc");
//    }
//
//    void test_fast(int argc, char* argv[]) {
//        noa::Session::set_thread_limit(16);
//        const auto directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/test");
//        const auto stack_filename = directory / Path("aligned_cropped.mrc");
//
//        qn::Logger::initialize();
//        qn::Options options(argc, argv);
//        qn::Logger::add_logfile(options.files.output_directory / "quinoa.log");
//        qn::Logger::set_level(options.compute.log_level);
//        options.files.input_stack = stack_filename;
//        auto metadata = qn::MetadataStack(options);
//
//        const auto array_options = ArrayOption(Device("gpu"), Allocator::MANAGED);
//
//        auto stack = noa::io::load_data<f32>(stack_filename, true, array_options);
//
//        {
//            using namespace noa::indexing;
//            auto original_stack = stack.view().subregion(Ellipsis{}, Slice{256, 769}, Slice{256, 769});
//
//            constexpr f64 maximum_size_loss = 0.2;
//            auto common_area = CommonArea(original_stack.shape().filter(2, 3), metadata, maximum_size_loss);
//            common_area.mask_views(original_stack, original_stack, metadata, 0.05);
////            noa::io::save(stack, directory / "stack_masked.mrc");
//            stack.eval();
//        }
//
//        const auto stack_shape = stack.shape();
//        const auto slice_shape = stack_shape.set<0>(1);
//        const auto stack_rfft = noa::fft::r2c(stack, noa::fft::NORM_DEFAULT, false);
//        noa::fft::remap(fft::H2HC, stack_rfft, stack_rfft, stack_shape);
//        stack_rfft.eval();
//
//        const auto extracted_rfft = noa::memory::zeros<c32>(slice_shape.rfft(), array_options);
//        const auto extracted_weight_rfft = noa::memory::zeros<f32>(slice_shape.rfft(), array_options);
//
//        // Filter.
////        metadata.add_global_angles({0, -60, 0});
//        auto metadata_insert = metadata;
//        metadata_insert.exclude([](const MetadataSlice& slice) {
//            return std::abs(slice.angles[1]) > 40;
//        });
//        auto metadata_extract = metadata;
//        metadata_extract.exclude([](const MetadataSlice& slice) {
//            return std::abs(slice.angles[1] - 42) > 1e-2;
//        });
//        fmt::print("extract {} degrees\n", metadata_extract[0].angles[1]);
//        const auto fwd_rotation_matrix = noa::geometry::rotate_y(noa::math::deg2rad(metadata_extract[0].angles[1])).as<f32>();
//
//        const noa::geometry::fft::WindowedSinc insert_windowed_sinc{
//                1./1024, 0.5}; // 0.000976563
//        const noa::geometry::fft::WindowedSinc z_windowed_sinc{
//                0.02f, 0.3f};// 0.02f, 0.5f
//
//        // Phase-shift to the center of rotation.
//        const auto size = stack_shape[3];
//        noa::signal::fft::phase_shift_2d<fft::HC2HC>(
//                stack_rfft, stack_rfft, stack_shape, -Vec2<f32>(size / 2));
//
//        for (i64 i = 0; i < metadata_insert.ssize(); ++i) {
//            const MetadataSlice& slice = metadata_insert[i];
//            fmt::print("insert {} degrees\n", slice.angles[1]);
//            const auto inv_rotation_matrix = noa::geometry::rotate_y(noa::math::deg2rad(slice.angles[1]))
//                    .inverse().as<f32>();
//
//            noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
//                    stack_rfft.subregion(slice.index_file), slice_shape,
//                    extracted_rfft, slice_shape,
//                    Float22{}, inv_rotation_matrix,
//                    Float22{}, fwd_rotation_matrix,
//                    insert_windowed_sinc,
//                    z_windowed_sinc,
//                    /*add_to_output=*/ true
//            );
//            noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
//                    1.f, slice_shape,
//                    extracted_weight_rfft, slice_shape,
//                    Float22{}, inv_rotation_matrix,
//                    Float22{}, fwd_rotation_matrix,
//                    insert_windowed_sinc,
//                    z_windowed_sinc,
//                    /*add_to_output=*/ true
//            );
//        }
//        noa::ewise_binary(extracted_rfft, extracted_weight_rfft, extracted_rfft, qn::correct_multiplicity_t{});
//        noa::signal::fft::phase_shift_2d<fft::H2H>(
//                extracted_rfft, extracted_rfft, slice_shape, Vec2<f32>(size / 2));
//
//        noa::io::save(noa::ewise_unary(extracted_rfft, noa::abs_one_log_t{}), directory / "extracted_rfft.mrc");
//        const auto extracted = noa::fft::c2r(extracted_rfft, slice_shape);
//        noa::io::save(extracted, directory / "extracted.mrc");
//
//        auto target = stack.subregion(metadata_extract[0].index_file).to_cpu();
//        auto target_rfft = noa::fft::r2c(target);
//        noa::ewise_binary(target_rfft, extracted_weight_rfft.to_cpu(), target_rfft,
//                          [](c32 value, f32 weight) {
//                              if (weight < 1.f)
//                                  value *= weight;
//                              return value;
//                          });
//        noa::io::save(noa::ewise_unary(target_rfft, noa::abs_one_log_t{}), directory / "target_rfft.mrc");
//        noa::fft::c2r(target_rfft, target);
//        noa::io::save(target, directory / "target.mrc");
//    }
//
//    void test_mask() {
//        const auto mask = noa::memory::ones<f32>({1, 1024, 1024, 1024});
//        noa::geometry::rectangle(mask, mask, Vec3<f32>{512, 512, 512}, Vec3<f32>{50, 512, 512}, 50);
//        const auto mask_rfft = noa::fft::remap(fft::H2HC, noa::fft::r2c(mask), mask.shape());
//        noa::io::save(mask, "/home/thomas/Projects/quinoa/tests/ribo_ctf/mask.mrc");
//        noa::io::save(noa::ewise_unary(mask_rfft, noa::abs_one_log_t{}),
//                      "/home/thomas/Projects/quinoa/tests/ribo_ctf/mask_ps.mrc");
//    }
//
//    void test_filter() {
//        auto filter = noa::io::load_data<f32>("/home/thomas/Projects/quinoa/tests/ribo_ctf/filter.mrc").flat();
//        filter = noa::fft::remap(fft::F2FC, filter.copy(), filter.shape());
//        save_vector_to_text(filter.view(), "/home/thomas/Projects/quinoa/tests/ribo_ctf/filter.txt");
//    }
//
//    void test_add_to_output() {
//        const auto matrices = noa::memory::empty<Float33>(3);
//        matrices.span()[0] = noa::geometry::rotate_y(noa::math::deg2rad(6.f));
//        matrices.span()[1] = noa::geometry::rotate_y(noa::math::deg2rad(9.f));
//        matrices.span()[2] = noa::geometry::rotate_y(noa::math::deg2rad(-3.f));
//
//        const auto shape = Shape4<i64>{1, 1, 1024, 1024};
//        const auto input = noa::memory::ones<f32>(shape.rfft().set<0>(3));
//        const auto output = noa::memory::zeros<f32>(shape.rfft());
//        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
//                input, shape.set<0>(3), output, shape,
//                Float22{}, matrices,
//                Float22{}, Float33{},
//                {-1, 1}, {0.05f, 0.2f}, false
//                );
//        noa::io::save(output, "/home/thomas/Projects/quinoa/tests/ribo_ctf/test_add_to_output0.mrc");
//
//        noa::memory::fill(output, 0.f);
//        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
//                input.subregion(0), shape, output, shape,
//                Float22{}, matrices.span()[0],
//                Float22{}, Float33{},
//                {-1, 1}, {0.05f, 0.2f}, true
//        );
//        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
//                input.subregion(1), shape, output, shape,
//                Float22{}, matrices.span()[1],
//                Float22{}, Float33{},
//                {-1, 1}, {0.05f, 0.2f}, true
//        );
//        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
//                input.subregion(2), shape, output, shape,
//                Float22{}, matrices.span()[2],
//                Float22{}, Float33{},
//                {-1, 1}, {0.05f, 0.2f}, true
//        );
//        noa::io::save(output, "/home/thomas/Projects/quinoa/tests/ribo_ctf/test_add_to_output1.mrc");
//    }
//
//    void test_fast2(const Path& stack_filename, MetadataStack& metadata) {
//        auto meta = metadata;
//        const auto directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/test2");
//        const auto options = ArrayOption(Device("gpu"), Allocator::MANAGED);
//
//        const auto loading_parameters = LoadStackParameters{
//                /*compute_device=*/ options.device(),
//                /*allocator=*/ options.allocator(),
//                /*precise_cutoff=*/ true,
//                /*rescale_target_resolution=*/ 20,
//                /*rescale_min_size=*/ 512,
//                /*exposure_filter=*/ false,
//                /*highpass_parameters=*/ {0.01, 0.01},
//                /*lowpass_parameters=*/ {0.4, 0.1},
//                /*normalize_and_standardize=*/ true,
//                /*smooth_edge_percent=*/ 0.1f,
//                /*zero_pad_to_fast_fft_shape=*/ false,
//        };
//        const auto [stack, stack_spacing, file_spacing] =
//                load_stack(stack_filename, meta, loading_parameters);
//        meta.rescale_shifts(file_spacing, stack_spacing);
//
//        // Common-area.
//        {
//            auto common_area = CommonArea(stack.shape().filter(2,3), meta);
//            common_area.mask_views(stack.view(), stack.view(), meta, 0.05);
//        }
//
//        // Parameters.
//        auto metadata_insert = meta;
//        metadata_insert.exclude([](const MetadataSlice& slice) {
//            return std::abs(slice.angles[1]) > 50;
//        }, /*reset_index_field=*/ false);
//        auto metadata_extract = meta;
//        metadata_extract.exclude([](const MetadataSlice& slice) {
//            return std::abs(slice.angles[1] - 57) > 1e-2;
//        }, /*reset_index_field=*/ false);
//        const noa::geometry::fft::WindowedSinc insert_windowed_sinc{
//                -1, 0.001f}; // 0.000976563
//        const noa::geometry::fft::WindowedSinc z_windowed_sinc{
//                0.0167f, 0.2f};//0.0167f, 0.3f 0.02f, 0.5f
//
//        // Zero-pad.
//        i64 size_padded = noa::math::max(stack.shape().filter(2, 3));
//        size_padded = static_cast<i64>(std::sqrt(static_cast<f64>(size_padded * size_padded + size_padded * size_padded)));
//        size_padded = noa::fft::next_fast_size(size_padded);
//        auto stack_padded = noa::memory::empty<f32>({stack.shape()[0], 1, size_padded, size_padded}, options);
//        noa::memory::resize(stack, stack_padded);
//        noa::io::save(stack_padded, directory / "input_stack_padded.mrc");
//
//        // Go to real-space.
//        const auto stack_padded_shape = stack_padded.shape();
//        const auto slice_padded_shape = stack_padded_shape.set<0>(1);
//        const auto stack_padded_rfft = noa::fft::r2c(stack_padded);
//        noa::fft::remap(fft::H2HC, stack_padded_rfft, stack_padded_rfft, stack_padded_shape);
//
//        const auto extracted_rfft = noa::memory::zeros<c32>(slice_padded_shape.rfft(), options);
//        const auto extracted_weight_rfft = noa::memory::zeros<f32>(slice_padded_shape.rfft(), options);
//
//        const auto& extracted_slice_metadata = metadata_extract[0];
//        Float33 fwd_rotation_matrix;
//        {
//            fmt::print("extract {} degrees\n", extracted_slice_metadata.angles[1]);
//            const Vec3<f64> extraction_angles = noa::math::deg2rad(extracted_slice_metadata.angles);
//            fwd_rotation_matrix = noa::geometry::euler2matrix(
//                    Vec3<f64>{-extraction_angles[0], extraction_angles[1], extraction_angles[2]},
//                    "zyx", /*intrinsic=*/ false).as<f32>();
//        }
//
//        for (i64 i = 0; i < metadata_insert.ssize(); ++i) {
//            // Get the slice to insert.
//            const MetadataSlice& slice_metadata = metadata_insert[i];
//            const View<c32> slice_padded_rfft = stack_padded_rfft.view().subregion(slice_metadata.index);
//            fmt::print("insert {} degrees\n", slice_metadata.angles[1]);
//
//            const Vec3<f64> insertion_angles = noa::math::deg2rad(slice_metadata.angles);
//            const auto inv_rotation_matrix = noa::geometry::euler2matrix(
//                    Vec3<f64>{-insertion_angles[0], insertion_angles[1], insertion_angles[2]},
//                    "zyx", /*intrinsic=*/ false).transpose().as<f32>();
//
//            // Center the central slice.
//            noa::signal::fft::phase_shift_2d<fft::HC2HC>(
//                    slice_padded_rfft, slice_padded_rfft, slice_padded_shape,
//                    -Vec2<f32>(size_padded / 2) - slice_metadata.shifts.as<f32>());
//
//            noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
//                    slice_padded_rfft, slice_padded_shape,
//                    extracted_rfft, slice_padded_shape,
//                    Float22{}, inv_rotation_matrix,
//                    Float22{}, fwd_rotation_matrix,
//                    insert_windowed_sinc,
//                    z_windowed_sinc,
//                    /*add_to_output=*/ true
//            );
//            noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
//                    1.f, slice_padded_shape,
//                    extracted_weight_rfft, slice_padded_shape,
//                    Float22{}, inv_rotation_matrix,
//                    Float22{}, fwd_rotation_matrix,
//                    insert_windowed_sinc,
//                    z_windowed_sinc,
//                    /*add_to_output=*/ true
//            );
//        }
//
//        //
//        noa::ewise_binary(extracted_rfft, extracted_weight_rfft, extracted_rfft, qn::correct_multiplicity_t{});
//        noa::signal::fft::phase_shift_2d<fft::H2H>(
//                extracted_rfft, extracted_rfft, slice_padded_shape,
//                Vec2<f32>(size_padded / 2));
//
//        noa::io::save(noa::ewise_unary(extracted_rfft, noa::abs_one_log_t{}), directory / "extracted_rfft.mrc");
//        const auto extracted = noa::fft::c2r(extracted_rfft, slice_padded_shape);
//        noa::io::save(extracted, directory / "extracted.mrc");
//
//        auto target_padded = stack_padded.subregion(extracted_slice_metadata.index).to_cpu();
//        auto target_padded_rfft = noa::fft::r2c(target_padded);
//        noa::ewise_binary(target_padded_rfft, extracted_weight_rfft.to_cpu(), target_padded_rfft,
//                          [](c32 value, f32 weight) {
//                              if (weight < 1.f)
//                                  value *= weight;
//                              return value;
//                          });
//        noa::signal::fft::phase_shift_2d<fft::H2H>(
//                target_padded_rfft, target_padded_rfft, slice_padded_shape,
//                -extracted_slice_metadata.shifts.as<f32>()
//        );
//        noa::io::save(noa::ewise_unary(target_padded_rfft, noa::abs_one_log_t{}), directory / "target_padded_rfft.mrc");
//        noa::fft::c2r(target_padded_rfft, target_padded);
//        noa::io::save(target_padded, directory / "target_padded.mrc");
//    }
//
//    void test_fast3(const Path& stack_filename, MetadataStack& metadata) {
//        auto meta = metadata;
//        const auto directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/test2");
//        const auto options = ArrayOption(Device("gpu"), Allocator::MANAGED);
//
//        const auto loading_parameters = LoadStackParameters{
//                /*compute_device=*/ options.device(),
//                /*allocator=*/ options.allocator(),
//                /*precise_cutoff=*/ true,
//                /*rescale_target_resolution=*/ 20,
//                /*rescale_min_size=*/ 512,
//                /*exposure_filter=*/ false,
//                /*highpass_parameters=*/ {0.01, 0.01},
//                /*lowpass_parameters=*/ {0.4, 0.1},
//                /*normalize_and_standardize=*/ true,
//                /*smooth_edge_percent=*/ 0.1f,
//                /*zero_pad_to_fast_fft_shape=*/ false,
//                /*zero_pad_to_square_shape=*/ false
//        };
//        auto [stack, stack_spacing, file_spacing] =
//                load_stack(stack_filename, meta, loading_parameters);
//        meta.rescale_shifts(file_spacing, stack_spacing);
//
//        auto common_area = CommonArea(stack.shape().filter(2, 3), meta);
//        common_area.mask_views(stack.view(), stack.view(), meta, 0.05);
//
//        // Parameters.
//        auto metadata_insert = meta;
//        metadata_insert.exclude([](const MetadataSlice& slice) {
//            return std::abs(slice.angles[1]) > 30;
//        }, /*reset_index_field=*/ false);
//        auto metadata_extract = meta;
//        metadata_extract.exclude([](const MetadataSlice& slice) {
//            return std::abs(slice.angles[1] - 42) > 1e-2;
//        }, /*reset_index_field=*/ false);
//        const noa::geometry::fft::WindowedSinc insert_windowed_sinc{
//                -1, 0.001f}; // 0.000976563
//        const noa::geometry::fft::WindowedSinc z_windowed_sinc{
//                0.0167f, 0.2f};//0.0167f, 0.3f 0.02f, 0.5f
//
//        //
////        const auto size_padded = noa::math::max(stack.shape().filter(2, 3)) * 2;
//        const i64 size_padded = 1484;
//        stack = noa::memory::resize(stack, {stack.shape()[0], 1, size_padded, size_padded});
//        const auto stack_rfft = noa::fft::r2c(stack);
//        noa::fft::remap(fft::H2HC, stack_rfft, stack_rfft, stack.shape());
//
//        const auto slice_shape = stack.shape().set<0>(1);
//        const auto extracted_rfft = noa::memory::zeros<c32>(slice_shape.rfft(), options);
//
//        const auto& extracted_slice_metadata = metadata_extract[0];
//        Float33 fwd_rotation_matrix;
//        {
//            fmt::print("extract {} degrees\n", extracted_slice_metadata.angles[1]);
//            const Vec3<f64> extraction_angles = noa::math::deg2rad(extracted_slice_metadata.angles);
//            fwd_rotation_matrix = noa::geometry::euler2matrix(
//                    Vec3<f64>{-extraction_angles[0], extraction_angles[1], extraction_angles[2]},
//                    "zyx", /*intrinsic=*/ false).as<f32>();
//        }
//
//        for (i64 i = 0; i < metadata_insert.ssize(); ++i) {
//            // Get the slice to insert.
//            const MetadataSlice& slice_metadata = metadata_insert[i];
//            const View<c32> slice_rfft = stack_rfft.view().subregion(slice_metadata.index);
//            fmt::print("insert {} degrees\n", slice_metadata.angles[1]);
//
//            const Vec3<f64> insertion_angles = noa::math::deg2rad(slice_metadata.angles);
//            const auto inv_rotation_matrix = noa::geometry::euler2matrix(
//                    Vec3<f64>{-insertion_angles[0], insertion_angles[1], insertion_angles[2]},
//                    "zyx", /*intrinsic=*/ false).transpose().as<f32>();
//
//            // Center the central slice.
//            noa::signal::fft::phase_shift_2d<fft::HC2HC>(
//                    slice_rfft, slice_rfft, slice_shape,
//                    -Vec2<f32>(slice_shape.filter(2, 3).vec() / 2) - slice_metadata.shifts.as<f32>());
//
//            noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
//                    slice_rfft, slice_shape,
//                    extracted_rfft, slice_shape,
//                    Float22{}, inv_rotation_matrix,
//                    Float22{}, fwd_rotation_matrix,
//                    insert_windowed_sinc,
//                    z_windowed_sinc,
//                    /*add_to_output=*/ true,
//                    /*correct_multiplicity=*/ true
//            );
//        }
//
//        //
//        noa::signal::fft::phase_shift_2d<fft::H2H>(
//                extracted_rfft, extracted_rfft, slice_shape,
//                Vec2<f32>(slice_shape.filter(2, 3).vec() / 2));
//
//        noa::io::save(noa::ewise_unary(extracted_rfft, noa::abs_one_log_t{}), directory / "extracted_rfft.mrc");
//        const auto extracted = noa::fft::c2r(extracted_rfft, slice_shape);
//        noa::io::save(extracted, directory / "extracted.mrc");
//        fmt::print("done");
//
////        auto target = stack.subregion(extracted_slice_metadata.index).to_cpu();
////        auto target_rfft = noa::fft::r2c(target);
////        noa::ewise_binary(target_rfft, extracted_weight_rfft.to_cpu(), target_rfft,
////                          [](c32 value, f32 weight) {
////                              if (weight < 1.f)
////                                  value *= weight;
////                              return value;
////                          });
////        noa::signal::fft::phase_shift_2d<fft::H2H>(
////                target_rfft, target_rfft, slice_shape,
////                -extracted_slice_metadata.shifts.as<f32>()
////        );
////        noa::io::save(noa::ewise_unary(target_rfft, noa::abs_one_log_t{}), directory / "target_rfft.mrc");
////        noa::fft::c2r(target_rfft, target);
////        noa::io::save(target, directory / "target_padded.mrc");
//    }

//    void benchmark_projection() {
//        const auto options = ArrayOption(Device("gpu"), Allocator::DEFAULT_ASYNC);
//        const auto options_managed = ArrayOption(Device("gpu"), Allocator::MANAGED);
//
//        const i64 n_slices = 60;
//        const auto size = noa::fft::next_fast_size(2048);
//        const auto insertion_shape = Shape4<i64>{n_slices, 1, size, size};
//        auto slices_to_insert = noa::memory::zeros<c32>(insertion_shape.rfft(), options_managed);
//        auto slice_to_extract = noa::memory::zeros<c32>(insertion_shape.set<0>(1).rfft(), options);
//        auto slices_to_insert_weight = noa::memory::like<f32>(slices_to_insert);
//        auto slice_to_extract_weight = noa::memory::like<f32>(slice_to_extract);
//
////        noa::Texture<c32> texture(
////                slices_to_insert.release(), options.device(), InterpMode::LINEAR_FAST, BorderMode::ZERO,
////                c32{0}, true, false);
//
//        slice_to_extract.eval();
//
//        auto insertion_inv_rotations = noa::memory::empty<Float33>(n_slices);
//        for (i64 i: irange(n_slices)) {
//            insertion_inv_rotations.span()[i] = noa::geometry::euler2matrix(
//                    Vec3<f64>{-2.3, 0.1 * static_cast<f64>(i), 0},
//                    "zyx", /*intrinsic=*/ false).transpose().as<f32>();
//        }
//        insertion_inv_rotations = insertion_inv_rotations.to(options);
//
//        const Float33 extraction_fwd_rotation = noa::geometry::euler2matrix(
//                Vec3<f64>{-2.3, 0.45, 0},
//                "zyx", /*intrinsic=*/ false).as<f32>();
//
//        using WindowedSinc = noa::geometry::fft::WindowedSinc;
//        const auto i_windowed_sinc = WindowedSinc{0.000488281f, 0.000488281f * 4};
//        const auto o_windowed_sinc = WindowedSinc{0.01f, 0.04f};
//
//        noa::Timer timer;
//        timer.start();
//
//        constexpr i64 N = 10;
//        for (i64 i = 0; i < N; ++i) {
////            Float33 matrix = noa::geometry::euler2matrix(
////                        Vec3<f64>{-2.3, 0.1 * static_cast<f64>(i), 0},
////                        "zyx", /*intrinsic=*/ false).transpose().as<f32>();
//
//            noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
//                    slices_to_insert, slices_to_insert_weight, insertion_shape,
//                    slice_to_extract, slice_to_extract_weight, insertion_shape.set<0>(1),
//                    Float22{}, insertion_inv_rotations, //insertion_inv_rotations, //Float33{0.94},
//                    Float22{}, extraction_fwd_rotation,
//                    i_windowed_sinc, o_windowed_sinc,
//                    /*add_to_output=*/ false,
//                    /*correct_multiplicity=*/ true);
//
////            noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
////                    slices_to_insert_weight, {}, insertion_shape,
////                    slice_to_extract_weight, {}, insertion_shape.set<0>(1),
////                    Float22{}, insertion_inv_rotations, //insertion_inv_rotations, //Float33{0.94},
////                    Float22{}, extraction_fwd_rotation,
////                    i_windowed_sinc, o_windowed_sinc,
////                    /*add_to_output=*/ false,
////                    /*correct_multiplicity=*/ true);
//        }
//
//        slice_to_extract.eval();
//        fmt::print("projection took {}ms\n", timer.elapsed() / N);
//    }
}

auto main(int argc, char* argv[]) -> int {
//    fourier_crop();
//    test_mask();
//    test_filter();
//    test_add_to_output();
//    benchmark_projection();
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
            if (!options.preprocessing.exclude_view_indexes.empty())
                metadata.exclude(options.preprocessing.exclude_view_indexes);

            // TODO Remove hot pixels?

            if (options.preprocessing.exclude_blank_views) {
                qn::detect_and_exclude_blank_views(
                        options.files.input_stack, metadata, {
                                /*compute_device=*/ options.compute.device,
                                /*allocator=*/ Allocator::DEFAULT_ASYNC,
                                /*resolution=*/ 20.,
                        });
            }
        }

        // Alignment.
        qn::CTFAnisotropic64 average_ctf;
        if (options.alignment.run) {
            std::tie(metadata, average_ctf) = tilt_series_alignment(options, metadata);

            const auto loading_parameters = qn::LoadStackParameters{
                    /*compute_device=*/ options.compute.device,
                    /*allocator=*/ noa::Allocator::DEFAULT_ASYNC,
                    /*precise_cutoff=*/ true,
                    /*rescale_target_resolution=*/ 20,
                    /*rescale_min_size=*/ 1024,
                    /*exposure_filter=*/ false,
                    /*highpass_parameters=*/ {0.01, 0.01},
                    /*lowpass_parameters=*/ {0.5, 0.01},
                    /*normalize_and_standardize=*/ true,
                    /*smooth_edge_percent=*/ 0.02f,
                    /*zero_pad_to_fast_fft_shape=*/ false,
            };
            qn::save_stack(
                    options.files.input_stack,
                    options.files.output_directory / "aligned.mrc",
                    metadata, loading_parameters);

//            qn::full_reconstruction(
//                    options.files.input_stack, metadata, options.files.output_directory / "reconstruction.mrc");
        }

        // Reconstruction.

        qn::Logger::status("Done... took {:.2}s", timer.elapsed() * 1e-3);

    } catch (...) {
        qn::Logger::error(qn::Exception::backtrace());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
