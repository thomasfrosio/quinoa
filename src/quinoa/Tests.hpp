#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Options.hpp"
#include "quinoa/Stack.hpp"

#include <noa/Geometry.hpp>

namespace {
    using namespace qn;

    void test_stack_loader_rescale_shift_() {
        Logger::initialize();
        Logger::set_level("trace");
        const Path test_data = "/home/thomas/Projects/quinoa/tests/tilt1/stack_loader_input.mrc";

        const auto shape = Shape<i64, 4>{1, 1, 5760, 4092};
        const auto shape_hw = shape.filter(2, 3).vec.as<f64>();
        auto output = Array<f32>(shape);
        const auto horizontal_line = ng::Rectangle{
            .center = shape_hw / 2,
            .radius = Vec{5., shape_hw[1]},
            .smoothness = 20.,
        };
        ng::draw_shape({}, output, horizontal_line);

        const auto vertical_line = ng::Rectangle{
            .center = shape_hw / 2,
            .radius = Vec{shape_hw[0], 5.},
            .smoothness = 20.,
        };
        ng::draw_shape(output, output, vertical_line, {}, noa::Plus{});
        noa::write(output, Vec{2.1, 2.1}, test_data);


        std::vector<MetadataSlice> slices;
        slices.emplace_back(MetadataSlice{});
        auto metadata = MetadataStack(std::move(slices));

        auto [stack, stack_spacing, file_spacing] = qn::load_stack(test_data, metadata, {
            .compute_device = "cpu",
            .allocator = noa::Allocator::DEFAULT_ASYNC,

            // Fourier cropping:
            .precise_cutoff = true,
            .rescale_target_resolution = 20.,
            .rescale_min_size = 0,

            // Signal processing after cropping:
            .exposure_filter = false,
            .bandpass = noa::signal::Bandpass{
                .highpass_cutoff = 0.10,
                .highpass_width = 0.10,
                .lowpass_cutoff = 0.45,
                .lowpass_width = 0.05,
            },

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.01,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        noa::write(stack, test_data.parent_path() / "cropped.mrc");
    }

    void test_pairwise_shift_data() {
        const Path test_dir = "/home/thomas/Projects/quinoa/tests/test_pairwise_shift";
        const Path test_data = "/home/thomas/Projects/simtilt/simulated_volume.mrc";

        // Tilt-series geometry.
        constexpr size_t n_images = 41;
        constexpr f64 rotation = noa::deg2rad(-45.);
        auto randomizer = noa::random_generator(noa::Uniform{-100., 100.});
        std::array<f64, n_images> tilts;
        std::array<Vec<f64, 2>, n_images> shifts;
        auto shift_file = noa::io::OutputTextFile(test_dir / "shifts.txt", {.write = true});
        for (f64 i{-60}; auto&& [tilt, shift]: noa::zip(tilts, shifts)) {
            tilt = noa::deg2rad(i);
            shift = Vec{randomizer(), randomizer()};
            i += 3;
            shift_file.write(fmt::format("{:+.4f}, {:+.4f}\n", shift[0], shift[1]));
        }

        // {
        //     // Compute the cosine-stretched tilt-series
        //     auto volume = noa::read_data<f32>(test_data, {.n_threads = 4});
        //     auto projection = noa::sum(volume.release(), {.depth = true});
        //     auto center = MetadataSlice::center<f64>(projection.shape());
        //     std::array<Mat23<f32>, n_images> inverse_matrices_aligned;
        //     std::array<Mat23<f32>, n_images> inverse_matrices;
        //     for (auto&& [tilt, shift, m0, m1]: noa::zip(tilts, shifts, inverse_matrices, inverse_matrices_aligned)) {
        //         m0 = (
        //             ng::translate(center + shift) *
        //             ng::linear2affine(ng::rotate(rotation)) *
        //             ng::linear2affine(ng::scale(Vec{1., std::cos(tilt)})) *
        //             ng::linear2affine(ng::rotate(-rotation)) *
        //             ng::translate(-center)
        //         ).inverse().pop_back().as<f32>();
        //
        //         m1 = (
        //             ng::translate(center) *
        //             ng::linear2affine(ng::rotate(rotation)) *
        //             ng::linear2affine(ng::scale(Vec{1., std::cos(tilt)})) *
        //             ng::linear2affine(ng::rotate(-rotation)) *
        //             ng::translate(-center)
        //         ).inverse().pop_back().as<f32>();
        //     }
        //
        //     auto tilt_series = noa::Array<f32>(projection.shape().set<0>(n_images));
        //     ng::transform_2d(projection, tilt_series, View(inverse_matrices_aligned.data(), n_images));
        //     noa::write(tilt_series, Vec{1., 1.}, test_dir / "cosine_stretched_aligned.mrc");
        //
        //     ng::transform_2d(projection, tilt_series, View(inverse_matrices.data(), n_images));
        //     noa::write(tilt_series, Vec{1., 1.}, test_dir / "cosine_stretched.mrc");
        // }

        {
            // Project the tilt-series.
            auto volume = noa::read_data<f32>(test_data, {.n_threads = 4}, {.device = "gpu"});
            auto center = (volume.shape().vec.pop_front() / 2).as<f64>();
            auto projection_matrices_aligned = Array<Mat<f32, 3, 4>>(n_images);
            auto projection_matrices = Array<Mat<f32, 3, 4>>(n_images);

            i64 projection_window{0};
            for (auto&& [tilt, shift, m0, m1]: noa::zip(
                tilts, shifts,
                projection_matrices.span_1d(),
                projection_matrices_aligned.span_1d())
            ) {
                m0 = (
                    ng::translate(center + shift.push_front(0)) *
                    ng::linear2affine(ng::rotate_z(rotation)) *
                    ng::linear2affine(ng::rotate_y(tilt)) *
                    ng::linear2affine(ng::rotate_z(-rotation)) *
                    ng::translate(-center)
                ).inverse().pop_back<1>().as<f32>();
                m1 = (
                    ng::translate(center) *
                    ng::linear2affine(ng::rotate_z(rotation)) *
                    ng::linear2affine(ng::rotate_y(tilt)) *
                    ng::linear2affine(ng::rotate_z(-rotation)) *
                    ng::translate(-center)
                ).inverse().pop_back<1>().as<f32>();

                auto window = ng::forward_projection_window_size(volume.shape().pop_front(), m0);
                fmt::println("tilt={:.3f}, window={}", noa::rad2deg(tilt), window);
                projection_window = std::max(projection_window, window);
            }

            projection_matrices = std::move(projection_matrices).to(volume.options());
            projection_matrices_aligned = std::move(projection_matrices_aligned).to(volume.options());

            auto tilt_series = noa::Array<f32>(volume.shape().set<0>(n_images).set<1>(1), volume.options());
            ng::forward_project_3d(volume, tilt_series, projection_matrices, projection_window);
            noa::write(tilt_series, Vec{1., 1.}, test_dir / "tilt_series.mrc");

            ng::forward_project_3d(volume, tilt_series, projection_matrices_aligned, projection_window);
            noa::write(tilt_series, Vec{1., 1.}, test_dir / "tilt_series_aligned.mrc");
        }
    }

    void test_reconstruct() {
        const Path test_dir = "/home/thomas/Projects/quinoa/tests/test_pairwise_shift";

        constexpr size_t n_images = 41;
        constexpr f64 volume_center = 350.;

        // Project the tilt-series.
        auto tilt_series = noa::read_data<f32>(test_dir / "tilt_series_aligned.mrc",
                                               {.n_threads = 4}, {.device = "gpu"});
        auto slice_shape = tilt_series.shape().filter(2, 3);
        auto center = (slice_shape.vec.push_front(volume_center) / 2).as<f64>();

        auto projection_matrices = Array<Mat<f64, 2, 4>>(n_images);
        for (f64 tilt{-60 + 20}; auto& matrix: projection_matrices.span_1d()) {
            // volume to image, we add the angle offset to go to image space.
            matrix = (
                ng::translate(center) *
                ng::linear2affine(ng::rotate_z(noa::deg2rad(-45.))) *
                ng::linear2affine(ng::rotate_y(noa::deg2rad(tilt))) *
                ng::translate(-center)
            ).filter_rows(1, 2);
            tilt += 3;
        }
        projection_matrices = std::move(projection_matrices).to(tilt_series.options());

        auto volume = noa::Array<f32>(tilt_series.shape().set<0>(1).set<1>(volume_center), tilt_series.options());
        ng::backward_project_3d(tilt_series, volume, projection_matrices);
        noa::write(volume, Vec{1., 1.}, test_dir / "tilt_series_reconstructed.mrc");

        auto patch_z_offset = [&](
            const Vec<f64, 3>& slice_angles, // radians
            const Vec<f64, 2>& patch_center
        ) {
            // Switch coordinates from pixels to micrometers.
            const auto slice_center = (slice_shape / 2).vec.as<f64>();
            const auto slice_center_3d = slice_center.push_front(0);

            // Apply the tilt and pitch.
            Mat<f64, 3, 4> image_to_microscope = (
                ng::translate(slice_center_3d) *                    // 6. shift back (optional, z is not affected)
                ng::linear2affine(ng::rotate_z(slice_angles[0])) *  // 5. rotate back (optional, z is not affected)
                ng::linear2affine(ng::rotate_x(-slice_angles[2])) * // 4. align pitch
                ng::linear2affine(ng::rotate_y(-slice_angles[1])) * // 3. align tilt
                ng::linear2affine(ng::rotate_z(-slice_angles[0])) * // 2. align tilt-axis
                ng::translate(-slice_center_3d)                     // 1. slice rotation center
            ).pop_back();
            {
                const auto patch_center_3d = patch_center.push_front(0);
                const Vec<f64, 3> patch_center_transformed = image_to_microscope * patch_center_3d.push_back(1);
                fmt::print("z={:.3f}", patch_center_transformed[0]);
            }

            image_to_microscope = (
                ng::linear2affine(ng::rotate_x(-slice_angles[2])) * // 4. align pitch
                ng::linear2affine(ng::rotate_y(-slice_angles[1])) * // 3. align tilt
                ng::linear2affine(ng::rotate_z(-slice_angles[0])) * // 2. align tilt-axis
                ng::translate(-slice_center_3d)                     // 1. slice rotation center
            ).pop_back();

            {
                const auto patch_center_3d = patch_center.push_front(0);
                const Vec<f64, 3> patch_center_transformed = image_to_microscope * patch_center_3d.push_back(1);
                fmt::println(", z={:.3f}", patch_center_transformed[0]);
            }
        };
        fmt::print("\n");
        patch_z_offset({-90, -60, 0}, {0, 0});
        patch_z_offset({-90, -60, 0}, {512, 512});
        patch_z_offset({-90, -60, 0}, {1024, 1024});
        fmt::print("\n");
        patch_z_offset({-90, 60, 0}, {0, 0});
        patch_z_offset({-90, 60, 0}, {512, 512});
        patch_z_offset({-90, 60, 0}, {1024, 1024});
        fmt::print("\n");
        patch_z_offset({90, 60, 0}, {0, 0});
        patch_z_offset({90, 60, 0}, {512, 512});
        patch_z_offset({90, 60, 0}, {1024, 1024});
    }

    void test_common_line() {
        Logger::initialize();
        Logger::set_level("trace");
        const Path test_dir = "/home/thomas/Projects/quinoa/tests/test_pairwise_shift";
        const Path test_data = "/home/thomas/Projects/datasets/EMPIAR-10304/tilt2/tilt2_preali.mrc";

        auto slice = noa::read_data<f32>(test_data, {.n_threads = 4});
        auto slice_rfft = noa::fft::r2c(slice);

        {
            auto lhs_lines = Array<c32>({1, 1, slice_rfft.shape()[2], slice_rfft.shape()[3]});
            auto rhs_lines = Array<c32>({1, 1, slice_rfft.shape()[2], slice_rfft.shape()[3]});
            ng::spectrum2polar<"h2fc">(slice_rfft.subregion(19), slice.shape().set<0>(1), lhs_lines);
            ng::spectrum2polar<"h2fc">(slice_rfft.subregion(30), slice.shape().set<0>(1), rhs_lines);

            lhs_lines = lhs_lines.permute({2, 0, 1, 3});
            rhs_lines = rhs_lines.permute({2, 0, 1, 3});
            noa::normalize_per_batch(lhs_lines, lhs_lines, {.mode = noa::Norm::L2});
            noa::normalize_per_batch(rhs_lines, rhs_lines, {.mode = noa::Norm::L2});

            Array<c64> scores(lhs_lines.shape()[0]);
            ns::cross_correlation_score(lhs_lines, rhs_lines, scores);

            auto r_scores = noa::like<f64>(scores);
            noa::ewise(scores, r_scores, noa::Real{});
            save_vector_to_text(r_scores.view(), test_dir / "scores1.txt");

            noa::ewise(scores, r_scores, noa::Imag{});
            save_vector_to_text(r_scores.view(), test_dir / "scores2.txt");

            noa::ewise(scores, r_scores, noa::Abs{});
            save_vector_to_text(r_scores.view(), test_dir / "scores3.txt");
            return;
        }
        {
            auto lhs_lines = Array<f32>({1, 1, slice_rfft.shape()[2], slice_rfft.shape()[3]});
            auto rhs_lines = Array<f32>({1, 1, slice_rfft.shape()[2], slice_rfft.shape()[3]});
            ng::spectrum2polar<"h2fc">(slice_rfft.subregion(19), slice.shape().set<0>(1), lhs_lines);
            ng::spectrum2polar<"h2fc">(slice_rfft.subregion(40), slice.shape().set<0>(1), rhs_lines);

            lhs_lines = lhs_lines.permute({2, 0, 1, 3});
            rhs_lines = rhs_lines.permute({2, 0, 1, 3});
            noa::normalize_per_batch(lhs_lines, lhs_lines, {.mode = noa::Norm::L2});
            noa::normalize_per_batch(rhs_lines, rhs_lines, {.mode = noa::Norm::L2});

            Array<f64> scores(lhs_lines.shape()[0]);
            ns::cross_correlation_score(lhs_lines, rhs_lines, scores);
            save_vector_to_text(scores.view(), test_dir / "scores2.txt");
            return;
        }

        {
            auto lhs_lines = Array<c32>({1, 1, slice_rfft.shape()[2], slice_rfft.shape()[3]});
            auto rhs_lines = Array<c32>({1, 1, slice_rfft.shape()[2], slice_rfft.shape()[3]});
            ng::spectrum2polar<"h2fc">(slice_rfft.subregion(19), slice.shape().set<0>(1), lhs_lines);
            ng::spectrum2polar<"h2fc">(slice_rfft.subregion(40), slice.shape().set<0>(1), rhs_lines);

            lhs_lines = lhs_lines.permute({2, 0, 1, 3});
            rhs_lines = rhs_lines.permute({2, 0, 1, 3});

            auto l_lines = noa::like<f32>(lhs_lines);
            auto r_lines = noa::like<f32>(lhs_lines);
            noa::ewise(lhs_lines, l_lines, noa::Abs{});
            noa::ewise(rhs_lines, r_lines, noa::Abs{});

            noa::normalize_per_batch(l_lines, l_lines, {.mode = noa::Norm::L2});
            noa::normalize_per_batch(r_lines, r_lines, {.mode = noa::Norm::L2});

            Array<f64> scores(l_lines.shape()[0]);
            ns::cross_correlation_score(l_lines, r_lines, scores);
            fmt::println("{::.4f}", scores.span_1d());

        }
    }
}
