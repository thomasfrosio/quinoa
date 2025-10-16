#include <noa/Array.hpp>
#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>

#include "quinoa/CommonArea.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/RotationOffset.hpp"

namespace qn {
    struct ReduceHeight {
        using span_t = SpanContiguous<const f32, 3>;
        using interp_t = noa::Interpolator<2, noa::Interp::LINEAR, noa::Border::ZERO, span_t>;

        interp_t images{};
        SpanContiguous<const Mat<f32, 2, 3>> image_transforms{};
        SpanContiguous<const ParallelogramMask> fov_masks{};

        NOA_HD void init(i64 i, i64 h, i64 w, f32& sum) const {
            const auto indices = Vec{h, w};
            const auto image_coordinates = image_transforms[i] * indices.as<f32>().push_back(1);
            const auto mask = fov_masks[i](image_coordinates);

            f32 value{};
            if (mask > 1e-6f)
                value = images.interpolate_at(image_coordinates, i);

            sum += value * mask;
        }

        NOA_HD static void join(f32 isum, f32& sum) { sum += isum; }
    };
}

namespace qn {
    void find_rotation_offset(
        const View<const f32>& stack,
        MetadataStack& metadata,
        const RotationOffsetParameters& options
    ) {
        auto timer = Logger::info_scope_time("Rotation offset alignment using common-lines");

        const f64 initial_rotation_offset = options.reset_rotation ? 0 : metadata[0].angles[0];
        f64 max_shift{};
        for (auto& slice: metadata) {
            if (options.reset_rotation) {
                // For the initial search, we usually want to start from 0 and search +-90 deg.
                slice.angles[0] = 0.;
            } else if (not noa::allclose(initial_rotation_offset, slice.angles[0])) {
                slice.angles[0] = initial_rotation_offset;
                Logger::warn_once(
                    "The rotation search algorithm is assuming a fixed tilt-axis, "
                    "but the provided stack has images with different rotation offsets. "
                    "To continue, the existing values will be overwritten with the "
                    "rotation offset of the lowest tilt."
                );
            }
            max_shift = std::max(max_shift, noa::max(slice.shifts));
        }

        // To project the stack along any axis, we need to zero-pad to make sure the image doesn't go out-of-view.
        const auto image_shape = stack.shape().filter(2, 3);
        const auto n_images = stack.shape()[0];
        auto line_size = static_cast<i64>(std::sqrt(2) * static_cast<f64>(noa::max(image_shape)) + max_shift);
        const auto do_bandpass = options.bandpass.highpass_cutoff > 0 or options.bandpass.lowpass_cutoff < 0.5;
        if (do_bandpass)
            line_size = nf::next_fast_size(line_size);

        const auto image_padded_shape = Shape{line_size, line_size};
        const auto image_padded_center = (image_padded_shape.vec / 2).as<f64>();
        const auto image_center = (image_shape.vec / 2).as<f64>();

        const auto device = stack.device();
        Logger::trace(
            "RotationOffset()::search():\n"
            "  reset_rotation={}\n"
            "  device={}\n"
            "  line_size={} (image_shape={}, max_shift={:.2f})\n"
            "  angle_range={:.2f}deg (rotation_offset={:.2f})\n"
            "  angle_step={:.3f}deg",
            options.reset_rotation, device,
            line_size, image_shape, max_shift,
            options.angle_range, initial_rotation_offset,
            options.angle_step
        );

        // Allocate dereferenceable buffers.
        const auto options_managed = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
        const auto image_transforms = Array<Mat<f32, 2, 3>>(n_images, options_managed);
        const auto fov_masks = Array<ParallelogramMask>(n_images, options_managed);
        const auto lines = Array<f32>({n_images, 1, 1, line_size}, options_managed);
        const auto lines_rfft = do_bandpass ? Array<c32>(lines.shape().rfft(), options_managed) : Array<c32>{};

        // Sort metadata in the same order as the stack.
        auto metadata_sorted = metadata;
        metadata_sorted.sort("index");
        const auto index_reference = metadata_sorted.find_lowest_tilt_index();

        const auto angle_range = Vec{-options.angle_range, +options.angle_range};
        const auto rotation_range = initial_rotation_offset + angle_range;

        std::vector<f64> nccs{};
        f64 best_ncc{};
        f64 best_rotation_offset{};
        auto grid_search = GridSearch<f64>({
            .start = rotation_range[0],
            .end = rotation_range[1],
            .step = options.angle_step
        });
        grid_search.for_each([&](f64 rotation_offset) mutable {
            // Set to the current rotation.
            for (auto& slice: metadata_sorted)
                slice.angles[0] = rotation_offset;

            // Set the image transforms.
            for (auto&& [slice, image_transform]: noa::zip(metadata_sorted, image_transforms.span_1d())) {
                const auto rotation = noa::deg2rad(slice.angles[0]);
                image_transform = (
                    ng::translate(image_padded_center) *
                    ng::rotate<true>(noa::deg2rad(-90.)) * // align tilt-axis on x-axis
                    ng::rotate<true>(-rotation) *
                    ng::translate(-image_center - slice.shifts)
                ).inverse().pop_back().as<f32>();
            }

            // Set the FOV mask.
            auto fov = CommonFOV{};
            fov.set_geometry(image_shape, metadata_sorted);
            fov.set_fovs(metadata_sorted, fov_masks.span_1d(), {
                .smooth_edge_percent = 0.1,
                .add_shifts = true, // the coordinates passed correspond to the unaligned image
            });

            // Compute the lines.
            noa::reduce_axes_iwise(
                image_padded_shape.push_front(n_images), device, f32{0}, lines.permute({1, 0, 2, 3}),
                ReduceHeight{
                    .images = ReduceHeight::interp_t(stack.span().filter(0, 2, 3).as_contiguous(), image_shape),
                    .image_transforms = image_transforms.span_1d(),
                    .fov_masks = fov_masks.span_1d(),
                });

            // Filter the lines, if necessary.
            if (do_bandpass) {
                nf::r2c(lines, lines_rfft);
                ns::bandpass<"h2h">(lines_rfft, lines_rfft, lines.shape(), options.bandpass);
                nf::c2r(lines_rfft, lines);
            }
            lines.eval();

            auto zero_normalized_cross_correlation = [](auto lhs, auto rhs) {
                f64 sum_lhs{};
                f64 sum_rhs{};
                f64 sum_lhs_lhs{};
                f64 sum_rhs_rhs{};
                f64 sum_lhs_rhs{};
                for (i64 i{}; i < lhs.ssize(); ++i) {
                    const auto lhs_ = static_cast<f64>(lhs[i]);
                    const auto rhs_ = static_cast<f64>(rhs[i]);
                    sum_lhs += lhs_;
                    sum_rhs += rhs_;
                    sum_lhs_lhs += lhs_ * lhs_;
                    sum_rhs_rhs += rhs_ * rhs_;
                    sum_lhs_rhs += lhs_ * rhs_;
                }
                const f64 count = static_cast<f64>(lhs.ssize());
                const f64 denominator_lhs = sum_lhs_lhs - sum_lhs * sum_lhs / count;
                const f64 denominator_rhs = sum_rhs_rhs - sum_rhs * sum_rhs / count;
                f64 denominator = denominator_lhs * denominator_rhs;
                if (denominator <= 0.0)
                    return 0.0;
                const f64 numerator = sum_lhs_rhs - sum_lhs * sum_rhs / count;
                return numerator / std::sqrt(denominator);
            };

            // Cross-correlation.
            f64 ncc{};
            const auto reference = lines.span().subregion(index_reference).as_1d();
            const auto targets = lines.span().filter(0, 3).as_contiguous();
            for (i64 i{}; i < targets.shape()[0]; ++i)
                if (i != index_reference)
                    ncc += zero_normalized_cross_correlation(reference, targets[i]);
            ncc /= static_cast<f64>(targets.shape()[0] - 1);

            // Maximize the ncc.
            if (ncc > best_ncc) {
                best_ncc = ncc;
                best_rotation_offset = rotation_offset;
            }
            nccs.push_back(ncc);
        });

        const auto increment = best_rotation_offset - initial_rotation_offset;
        Logger::info(
            "rotation_offset={:.3f}deg (increment={:+.3f}, ncc={:.4f}, n_iter={}), or equivalently {:.3f}deg",
            best_rotation_offset, increment, best_ncc, grid_search.size(),
            MetadataSlice::to_angle_range(best_rotation_offset + 180)
        );
        save_plot_xy(noa::Arange{rotation_range[0], options.angle_step}, nccs,
            options.output_directory / "rotation_offset.txt", {
                .title = "Rotation offset search",
                .x_name = "Rotation offset (degrees)",
                .y_name = "NCC",
            });

        // Update metadata with the new rotation.
        for (auto& slice: metadata)
            slice.angles[0] = MetadataSlice::to_angle_range(slice.angles[0] + increment);
    }
}
