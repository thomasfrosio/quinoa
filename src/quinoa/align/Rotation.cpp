#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>

#include "quinoa/Optimizer.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Plot.hpp"

#include "quinoa/align/CommonFOV.hpp"
#include "quinoa/align/Rotation.hpp"

namespace {
    using namespace qn;

    struct ReduceHeight {
        using span_t = SpanContiguous<const f32, 3>;
        using interp_t = nx::Interpolator<2, nx::Interp::LINEAR, noa::Border::ZERO, span_t>;

        interp_t images{};
        SpanContiguous<const Mat<f32, 2, 3>> matrices{};
        SpanContiguous<const ParallelogramMask> fov_masks{};

        NOA_HD void operator()(isize i, isize h, isize w, f32& sum) const {
            const auto image_coordinates = matrices[i] * Vec<f32, 3>::from_values(h, w, 1);
            const auto mask = fov_masks[i](image_coordinates);

            f32 value{};
            if (mask > 1e-6f)
                value = images.interpolate_at(image_coordinates, i);

            sum += value * mask;
        }

        NOA_HD static void join(f32 isum, f32& sum) { sum += isum; }
    };

    constexpr auto zero_normalized_cross_correlation(auto lhs, auto rhs) {
        f64 sum_lhs{};
        f64 sum_rhs{};
        f64 sum_lhs_lhs{};
        f64 sum_rhs_rhs{};
        f64 sum_lhs_rhs{};
        for (isize i{}; i < lhs.ssize(); ++i) {
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
    }
}

namespace qn {
    void find_rotation_offset(
        const View<const f32>& stack,
        Metadata::Stack& metadata,
        Vec<f64, 3>& angle_offsets,
        const RotationOffsetParameters& options
    ) {
        auto timer = Logger::info_scope_time("Rotation offset");

        const bool full_rotation = options.angle_range < 0;
        const f64 initial_rotation_offset = full_rotation ? 0 : metadata[0].angles[0];
        f64 max_shift{};
        for (auto& slice: metadata) {
            if (full_rotation) {
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

        // The projection kernel reduces the height dimension. To project the stack along any axis,
        // we rotate and center it so that the projection axis is perpendicular to the height.
        // To make sure the image doesn't go out-of-bound when rotating, we need to zero-pad appropriately.
        const auto image_shape = stack.shape().filter(2, 3);
        const auto n_images = stack.shape()[0];
        const auto line_size = static_cast<isize>(std::sqrt(2) * static_cast<f64>(noa::max(image_shape)) + max_shift);

        const auto image_padded_shape = Shape{line_size, line_size};
        const auto image_padded_center = (image_padded_shape.vec / 2).as<f64>();
        const auto image_center = (image_shape.vec / 2).as<f64>();

        // Allocate small dereferenceable buffers.
        const auto device = stack.device();
        const auto options_managed = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
        const auto matrices = Array<Mat<f32, 2, 3>>(n_images, options_managed);
        const auto fov_masks = Array<ParallelogramMask>(n_images, options_managed);
        const auto lines = Array<f32>({n_images, 1, 1, line_size}, options_managed);

        // Sort metadata in the same order as the stack.
        auto meta = metadata;
        meta.sort("index");
        const auto pivot = meta.find_lowest_tilt_index();

        // The function to maximize.
        auto znccs = std::vector<Vec<f64, 2>>{}; // diagnostics
        auto eval = [&](u32 n, const f64* rotation_offset, f64* g) {
            check(n == 1 and g == nullptr);

            // Set to the current rotation.
            for (auto& slice: meta)
                slice.angles[0] = *rotation_offset;

            // Set the image matrices.
            for (auto&& [image, matrix]: noa::zip(meta, matrices.span_1d())) {
                const auto rotation = noa::deg2rad(image.angles[0]);
                matrix = (
                    nx::translate(image_padded_center) *
                    nx::rotate<true>(noa::deg2rad(-90.)) * // align tilt-axis on x-axis
                    nx::rotate<true>(-rotation) *
                    nx::translate(-image_center - image.shifts)
                ).inverse().pop_back().as<f32>();
            }

            // Set the FOV mask.
            auto fov = CommonFOV{};
            fov.set_geometry(image_shape, meta);
            fov.set_fovs(meta, fov_masks.span_1d(), {
                .smooth_edge_percent = 0.1,
                .add_shifts = true, // unaligned image
            });

            // Compute the lines.
            using interp_t = ReduceHeight::interp_t;
            noa::reduce_axes_iwise(
                image_padded_shape.push_front(n_images), device, f32{0}, lines.permute({1, 0, 2, 3}),
                ReduceHeight{
                    .images = interp_t(stack.span_contiguous<const f32, 3>(), image_shape),
                    .matrices = matrices.span_1d(),
                    .fov_masks = fov_masks.span_1d(),
                });

            // Cross-correlation.
            f64 zncc{};
            const auto reference = lines.eval().span().subregion(pivot).as_1d();
            const auto targets = lines.span().filter(0, 3).as_contiguous();
            for (isize i{}; i < n_images; ++i)
                if (i != pivot)
                    zncc += zero_normalized_cross_correlation(reference, targets[i]);
            zncc /= static_cast<f64>(n_images - 1);

            znccs.push_back({*rotation_offset, zncc}); // diagnostics
            return zncc;
        };

        f64 best_ncc{-1};
        f64 best_rotation_offset{initial_rotation_offset};
        i32 n_evaluations{};

        if (full_rotation) {
            Logger::trace(
                "rotation_offset:\n"
                "  device={}\n"
                "  mode=grid-search\n"
                "  line_size={} (image_shape={}, max_shift={:.2f})\n"
                "  angle_range=90.00deg (rotation_offset=0.00)",
                device, line_size, image_shape, max_shift
            );

            auto grid_search = GridSearch<f64>({.start = -90., .end = 90., .step = 1.});
            grid_search.for_each([&](f64 rotation_offset) {
                auto ncc = eval(1, &rotation_offset, nullptr);
                if (ncc > best_ncc) {
                    best_ncc = ncc;
                    best_rotation_offset = rotation_offset;
                }
            });
            n_evaluations = static_cast<i32>(grid_search.size());
        } else {
            Logger::trace(
                "rotation_offset:\n"
                "  device={}\n"
                "  mode=local-optimizer\n"
                "  line_size={} (image_shape={}, max_shift={:.2f})\n"
                "  angle_range={:.2f}deg (rotation_offset={:.2f})",
                device, line_size, image_shape, max_shift,
                options.angle_range, initial_rotation_offset
            );
            auto optimizer = Optimizer(NLOPT_LN_SBPLX, 1);
            optimizer.set_x_tolerance_abs(0.005);
            optimizer.set_bounds(
                initial_rotation_offset - options.angle_range,
                initial_rotation_offset + options.angle_range
            );
            optimizer.set_max_objective(eval);
            best_ncc = optimizer.optimize(&best_rotation_offset);
            n_evaluations += optimizer.n_evaluations();
        }

        angle_offsets[0] += best_rotation_offset - initial_rotation_offset;
        Logger::info(
            "rotation_offset={:.3f}deg (increment={:+.3f}, zncc={:.4f}, n_iter={}), or equivalently {:.3f}deg",
            best_rotation_offset, angle_offsets[0], best_ncc, n_evaluations,
            Metadata::Image::to_angle_range(best_rotation_offset + 180)
        );
        save_plot_xy(
            znccs | stdv::transform([](auto& e) { return e[0]; }),
            znccs | stdv::transform([](auto& e) { return e[1]; }),
            *options.output_directory / "rotation_offset.txt", {
                .title = "Rotation offset search",
                .x_name = "Rotation offset (degrees)",
                .y_name = "ZNCC",
            });

        // Update metadata with the new rotation.
        for (auto& slice: metadata)
            slice.angles[0] = Metadata::Image::to_angle_range(best_rotation_offset);
    }
}
