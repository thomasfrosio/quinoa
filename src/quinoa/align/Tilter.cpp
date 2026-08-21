#include <noa/Runtime.hpp>
#include <noa/Signal.hpp>
#include <noa/FFT.hpp>

#include "quinoa/Logger.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/GridSearch.hpp"

#include "quinoa/align/Tilter.hpp"
#include "quinoa/align/CommonFOV.hpp"

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

    constexpr auto get_indices(i32 idx_target, i32 index_lowest_tilt) {
        if (idx_target >= index_lowest_tilt)
            idx_target += 1;

        // The tilts are sorted in ascending order (e.g. -60,..,60), so if the ith target has:
        //  - a negative tilt angle, then the reference is at i + 1.
        //  - a positive tilt angle, then the reference is at i - 1.
        const bool is_negative = idx_target < index_lowest_tilt;
        const i32 idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;
        return Pair{idx_target, idx_reference};
    }

    struct CreateStacks {
        using span_type = SpanContiguous<const f32, 3, i32>;
        using interpolator_type = nx::Interpolator<2, nx::Interp::LINEAR, noa::Border::ZERO, span_type>;
        using reduce_type = Vec<f32, 5, 8>;

        interpolator_type input{}; // (n,h,w)
        SpanContiguous<f32, 3, i32> references{}; // (n-1,h,w)
        SpanContiguous<f32, 3, i32> stretched_targets{}; // (n-1,h,w)

        SpanContiguous<const ParallelogramMask, 1, i32> fov_masks{}; // (n-1)
        SpanContiguous<const Vec<f32, 4>, 1, i32> reference_plane_coefficients{}; // (n-1)
        SpanContiguous<const Mat<f32, 2, 4>, 1, i32> reference2target{}; // (n-1)

        i32 pivot{};

        NOA_HD auto create_stacks(i32 i, i32 y, i32 x) const {
            auto [index_target, index_reference] = get_indices(i, pivot);
            auto fov_mask = fov_masks[i](y, x);

            auto reference = input(index_reference, y, x) * fov_mask;
            references(i, y, x) = reference;

            // Transform the target onto the reference.
            const auto& [a, b, c, d] = reference_plane_coefficients[i];
            const auto z = -(a * static_cast<f32>(x) + b * static_cast<f32>(y) + d) / c;
            const auto reference_coordinates = Vec<f32, 4>::from_values(z, y, x, 1);
            const auto target_coordinates = reference2target[i] * reference_coordinates;

            // Save the masked and stretched target.
            const auto stretched_target = input.interpolate_at(target_coordinates, index_target) * fov_mask;
            stretched_targets(i, y, x) = stretched_target;

            return noa::make_tuple(reference, stretched_target, fov_mask);
        }

        NOA_HD void operator()(i32 i, i32 y, i32 x) const {
            create_stacks(i, y, x);
        }

        NOA_HD void operator()(i32 i, i32 y, i32 x, reduce_type& reduce) const {
            auto [reference, stretched_target, fov_mask] = create_stacks(i, y, x);
            reduce[0] += reference;
            reduce[1] += stretched_target;
            reduce[2] += reference * reference;
            reduce[3] += stretched_target * stretched_target;
            reduce[4] += fov_mask;
        }

        static NOA_HD void join(const reduce_type& tmp, reduce_type& reduce) {
            reduce += tmp;
        }
    };

    struct CrossCorrelate {
        SpanContiguous<const c32, 3, i32> references{}; // (n-1,h,w/2+1)
        SpanContiguous<c32, 3, i32> stretched_targets{}; // (n-1,h,w/2+1)

        NOA_HD void operator()(i32 i, i32 y, i32 x) const {
            const auto frequency = nf::index2frequency<false, true>(Vec{y, x}, references.shape().filter(1));
            const auto phase_shift = static_cast<f32>(product(1 - 2 * abs(frequency % 2))); // shift by +shape/2
            stretched_targets(i, y, x) *= conj(references(i, y, x)) * phase_shift;
        }
    };

    auto relative2global_shifts_(
        const std::vector<Vec<f64, 2>>& relative_shifts,
        std::vector<Vec<f64, 2>>& global_shifts,
        const Metadata::Stack& metadata,
        isize pivot
    ) {
        // Relative shifts (target->reference) to global shifts (target->volume).
        const auto n = std::size(relative_shifts);
        const auto r_pivot = std::ssize(relative_shifts) - 1 - pivot;
        std::inclusive_scan(relative_shifts.begin() + pivot, relative_shifts.end(), global_shifts.begin() + pivot);
        std::inclusive_scan(relative_shifts.rbegin() + r_pivot, relative_shifts.rend(), global_shifts.rbegin() + r_pivot);

        // Center the shifts.
        auto mean = Vec<f64, 2>{};
        for (const auto& shift: global_shifts)
            mean += shift;
        mean /= static_cast<f64>(n);
        for (auto& shift: global_shifts)
            shift -= mean;

        // Transform the shifts back to image space.
        for (usize i{}; i < n; ++i) {
            const auto angles = noa::deg2rad(metadata[i].angles);
            const auto volume2image = (
                nx::rotate_z(angles[0]) *
                nx::rotate_y(angles[1]) *
                nx::rotate_x(angles[2])
            ).filter_rows(1, 2);
            global_shifts[i] = volume2image * global_shifts[i].push_front(0);
        }
        return mean;
    }

    struct ShiftAlignAndMask {
        using input_type = SpanContiguous<const f32, 3, i32>;
        nx::Interpolator<2, nx::Interp::LINEAR, noa::Border::ZERO, input_type> input{};
        SpanContiguous<f32, 3, i32> output{};
        SpanContiguous<const Mat<f32, 2, 3>, 1, i32> xforms{};
        nx::DrawEllipse<2, f32, true> ellipse{};

        NOA_HD void operator()(i32 i, i32 y, i32 x) const {
            const auto output_coordinates = Vec<f32, 2>::from_values(y, x);
            const auto input_coordinates = xforms[i] * output_coordinates.push_back(1);
            const auto mask = ellipse.draw_at(output_coordinates);
            output(i, y, x) = input.interpolate_at(input_coordinates, i) * mask;
        }
    };

    struct PowerSum {
        NOA_HD void operator()(c32 i, c32& r) const { r += i; }
        NOA_HD void join(c32 r, c32& j) const { j += r; }
        NOA_HD void post(c32 reduced, f32& power_sum) const {
            auto sum = noa::abs(reduced);
            power_sum = sum * sum;
        }
    };
}

namespace qn {
    void Tilter::find_accurate_image_rotation(
        const View<const f32>& stack,
        Metadata::Stack& metadata,
        Vec<f64, 3>& angle_offsets,
        const FindAccurateImageRotationOptions& options
    ) {
        auto timer = Logger::info_scope_time("Finding image rotation");

        const bool full_rotation = options.angle_range < 0 or options.angle_range >= 90.;
        const f64 initial_rotation_offset = full_rotation ? 0 : metadata[0].angles[0];
        f64 max_shift{};
        bool warn{};
        for (auto& slice: metadata) {
            if (full_rotation) {
                slice.angles[0] = 0.;
            } else if (not noa::allclose(initial_rotation_offset, slice.angles[0])) {
                slice.angles[0] = initial_rotation_offset;
                warn = true;
            }
            max_shift = std::max(max_shift, noa::max(slice.shifts));
        }
        if (warn) {
            Logger::warn(
                "The rotation search algorithm is assuming a fixed tilt-axis, but the provided stack has images with different rotation offsets. To continue, the existing values will be overwritten with the rotation offset of the lowest tilt."
            );
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

            // Set the FOV masks.
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
                "  mode=accurate-fov (grid-search)\n"
                "  line_size={} (image_shape={}, max_shift={:.2f})\n"
                "  angle_range=90.00deg (initial_rotation_offset=0.00)",
                device, line_size, image_shape, max_shift
            );

            auto grid_search = GridSearch<f64>({.start = -90., .end = 90., .step = -options.angle_range});
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
                "  mode=accurate-fov (local-optimizer)\n"
                "  line_size={} (image_shape={}, max_shift={:.2f})\n"
                "  angle_range={:.2f}deg (initial_rotation_offset={:.2f})",
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

    Tilter::Tilter(
        const Shape4& shape,
        Device device
    ) {
        const auto allocated_start = Allocator::bytes_currently_allocated(device);

        // Allocate 4 times the stack.
        // We only need 3 times the stack; however, the forward FFT needed for the cross-correlation is the slowest
        // step and is significantly faster when the references and stretched targets are batched into the same array.
        // This alignment is meant for low-resolution images (<=2Kx2K images), so a bigger workspace should be fine.
        const auto n_total_images = shape[0];
        const auto n_target_images = shape[0] - 1;
        const auto buffer_shape = shape.set<0>(n_total_images * 4); // TODO shouldn't it be n_target_images * 4?

        // Use device-only memory (which seems faster than managed memory) for the big buffer, if possible.
        const auto n_bytes_to_allocate = static_cast<usize>(buffer_shape.rfft().n_elements()) * sizeof(c32);
        const bool has_enough_space = n_bytes_to_allocate < device.memory_capacity().free;
        m_buffer_rfft = Array<c32>(buffer_shape.rfft(), {
            .device = device,
            .allocator = has_enough_space ? Allocator::ASYNC : Allocator::MANAGED,
        });
        m_buffer = nf::alias_to_real(m_buffer_rfft.view(), buffer_shape);

        // Use managed memory for the small buffers that need CPU access.
        m_fov_masks = Array<ParallelogramMask>(n_target_images, {.device = device, .allocator = Allocator::MANAGED});
        m_plane_coefficients = noa::like<Vec<f32, 4>>(m_fov_masks);
        m_projection_matrices = noa::like<Mat<f32, 2, 4>>(m_fov_masks);
        m_shift_matrices = Array<Mat<f32, 2, 3>>(n_total_images, m_fov_masks.options());

        m_xmap_centered = Array<f32>({n_target_images, 1, 64, 64}, m_fov_masks.options());
        m_peak_shifts = noa::like<Vec<f32, 2>>(m_fov_masks);
        m_peak_values = noa::like<f32>(m_fov_masks);
        m_peak_stats = noa::like<Vec<f32, 5>>(m_fov_masks);

        // Prepare the shifts.
        m_relative_shifts.resize(static_cast<usize>(n_total_images));
        m_global_shifts.resize(static_cast<usize>(n_total_images));

        // Prepare FFT plans and set the workspace.
        if (device.is_gpu()) {
            // eval_
            nf::r2c(buffer(0, 2), buffer_rfft(0, 2), {.record_and_share_workspace = true});
            nf::c2r(buffer_rfft(1), buffer(0), {.record_and_share_workspace = true});

            // find_image_rotation
            nf::r2c(
                m_buffer.subregion(Slice{0, n_total_images}),
                m_buffer_rfft.view().subregion(Slice{0, n_total_images}),
                {.record_and_share_workspace = true}
            );

            const auto workspace = m_buffer_rfft.subregion(Offset{2 * n_target_images});
            const auto n_plans_set = nf::set_workspace(device, workspace);
            if (auto left = nf::workspace_left_to_allocate(device); n_plans_set == 0 or left > 0) {
                Logger::warn(
                    "Failed to set the FFT workspace. A new workspace will have to be allocated, likely increasing the memory requirements substantially. Please report this. shape={}, workspace_left_to_allocate={}bytes, n_plans_set={}",
                    shape, left, n_plans_set);
            }
        }

        const auto allocated = Allocator::bytes_currently_allocated(device) - allocated_start;
        Logger::trace("Tilter() allocated {:.2f}GB on {} ({})",
                      static_cast<f64>(allocated) * 1e-9, m_buffer.device(), m_buffer.allocator());
    }

    void Tilter::find_image_rotation(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        Vec<f64, 3>& angle_offsets,
        const FindImageRotationOptions& options
    ) {
        if (options.accurate_fov) {
            return find_accurate_image_rotation(stack, metadata, angle_offsets, {
                .angle_range = options.angle_range,
                .output_directory = options.output_directory,
            });
        }

        auto t = Logger::info_scope_time("Finding image rotation");

        // If the full range is searched, center on 0 degree.
        // In any case, make sure images share a common rotation.
        const bool full_rotation = options.angle_range < 0. or options.angle_range >= 90.;
        const f64 initial_rotation_offset = full_rotation ? 0. : metadata[0].angles[0];
        bool warn{};
        for (auto& slice: metadata) {
            if (full_rotation) {
                slice.angles[0] = 0.;
            } else if (not noa::allclose(initial_rotation_offset, slice.angles[0])) {
                slice.angles[0] = initial_rotation_offset;
                warn = true;
            }
        }
        if (warn) {
            Logger::warn(
                "The rotation search algorithm is assuming a fixed tilt-axis, but the provided stack has images with different rotation offsets. To continue, the existing values will be overwritten with the rotation offset of the lowest tilt."
            );
        }

        // Define the angular range around the current tilt-axis
        const auto angle_pivot = initial_rotation_offset + 90.; // polar range is centered on the corresponding line
        const auto angle_start = angle_pivot - options.angle_range;
        const auto angle_end = angle_pivot + options.angle_range;
        const auto n_lines = static_cast<isize>(std::round(
            (angle_end - angle_start + options.angle_step) / options.angle_step));
        Logger::trace(
            "rotation_offset:\n"
            "  device={}\n"
            "  mode=power-sum\n"
            "  line_size={}\n"
            "  n_lines={} (range={:.2f}deg, step={:.2f}deg)\n"
            "  initial_rotation_offset={:.2f}",
            stack.device(), stack.shape()[3] / 2 + 1,
            n_lines, options.angle_range, options.angle_step,
            initial_rotation_offset
        );

        // Credit to Marten Chaillet (and EMAN2): https://github.com/teamtomo/teamtomo/pull/105.
        // This is essentially the same approach, except the spectrum2polar call that is more performant and accurate
        // to what they are doing in torch. The only issue with this method is that the FOV changes due to the different
        // tilt-axes is not explicitly handed (hence the find_accurate_image_rotation). The initial elliptical mask to
        // focus on the center of the FOV should help. Compared to find_accurate_image_rotation, which handles the FOV,
        // the results are very similar but this is much faster (x10-100 depending on the angle range).

        const auto input = stack.span_contiguous().as<const f32, 3, i32>();
        const auto n_images = metadata.ssize();
        const auto aligned_stack = m_buffer.subregion(Slice{0, n_images});
        const auto aligned_stack_rfft = m_buffer_rfft.view().subregion(Slice{0, n_images});

        // Shift-align the stack, and add a mask to focus on the center and reduce the changes in the FOV.
        // Since we mask towards zero, images should be mean normalized already.
        const auto shape_2d = input.shape().pop_front();
        for (auto&& [image, inverse_matrix]: noa::zip(metadata, m_shift_matrices.span_1d()))
            inverse_matrix = nx::translate(image.shifts).pop_back().as<f32>();

        const auto center = (shape_2d.vec / 2).as<f64>();
        using input_span_t = SpanContiguous<const f32, 3, i32>;
        using interpolator_t = nx::Interpolator<2, nx::Interp::LINEAR, noa::Border::ZERO, input_span_t>;
        noa::iwise(input.shape(), aligned_stack.device(), ShiftAlignAndMask{
            .input = interpolator_t(input, shape_2d),
            .output = aligned_stack.span_contiguous().as<f32, 3, i32>(),
            .xforms = m_shift_matrices.span_1d().as_index<i32>(),
            .ellipse = nx::Ellipse{.center = center, .radius = center / 2, .smoothness = min(center / 2)}.draw<f32>(),
        });

        // Compute the sum of the spectra and save the power spectrum of that sum.
        nf::r2c(aligned_stack, aligned_stack_rfft); // inplace
        const auto image_shape = aligned_stack.shape().set<0>(1);
        const auto image_shape_rfft = image_shape.rfft();
        const auto spectral_sum_power = m_buffer_rfft.view().reinterpret_as<f32>() // get contiguous buffer
            .subregion(Offset{n_images}) // offset to unused region
            .flat(0).subregion(Slice{0, image_shape_rfft.n_elements()}) // select slice
            .reshape(image_shape_rfft);
        noa::reduce_axes_ewise(aligned_stack_rfft, c32{}, spectral_sum_power, PowerSum{}); // (n,1,h,w/2+1)->(1,1,h,w/2+1)

        // Compute the lines.
        const auto line_size = image_shape_rfft[3];
        const auto line_pitch = noa::next_multiple_of(line_size, 256);
        const auto lines =  m_buffer_rfft.view().reinterpret_as<f32>() // get contiguous buffer
            .flat(0).subregion(Slice{0, n_lines * line_pitch}) // keep lines well aligned
            .reshape({1, 1, n_lines, line_pitch})
            .subregion(Ellipsis{}, Slice{0, line_size});
        nx::spectrum2polar<"h2fc">(spectral_sum_power, image_shape, lines, {
            .phi_range = noa::Linspace{
                .start = noa::deg2rad(angle_start),
                .stop = noa::deg2rad(angle_end),
                .endpoint = true,
            },
            .interp = nx::Interp::LINEAR,
        });

        // Compute the sum of each line.
        auto sums = Array<f32>(n_lines, {.device = lines.device(), .allocator = Allocator::MANAGED});
        noa::reduce_axes_ewise(lines, f32{}, sums.flat(2), noa::ReduceSum{}); // (1,1,n,w/2+1)->(1,1,n,1)
        sums = sums.reinterpret_as_cpu();

        // The common line should be the line with the highest sum.
        const auto offset = noa::argmax(sums).second;
        const auto common_line_angle = angle_start + static_cast<f64>(offset) * options.angle_step;
        const auto new_rotation_offset = common_line_angle - 90.;

        angle_offsets[0] += new_rotation_offset - initial_rotation_offset;
        Logger::info(
            "rotation_offset={:.3f}deg (increment={:+.3f}), or equivalently {:.3f}deg",
            new_rotation_offset, angle_offsets[0],
            Metadata::Image::to_angle_range(new_rotation_offset + 180)
        );
        save_plot_xy(
            noa::Linspace{.start = angle_start - 90, .stop = angle_end - 90, .endpoint = true}, sums.span_1d(),
            *options.output_directory / "rotation_offset_fast.txt", {
                .title = "Rotation offset search",
                .x_name = "Rotation offset (degrees)",
                .y_name = "Power Sum",
            });

        // Update metadata with the new rotation.
        for (auto& slice: metadata)
            slice.angles[0] = Metadata::Image::to_angle_range(new_rotation_offset);
    }

    void Tilter::find_image_shifts(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const FindImageShiftsOptions& options
    ) {
        auto timer = Logger::info_scope_time("Finding image shifts");
        Logger::trace(
            "  device={}\n"
            "  stretching={}\n"
            "  fov_mask={}\n"
            "  smooth_edge={:.0f}%",
            m_buffer.device(),
            options.cosine_stretch,
            options.fov_mask,
            options.smooth_edge_percent * 100
        );

        metadata.sort("tilt"); // just to make sure the images are sorted

        // Iterating a few times may be required to get a stable shift.
        auto max_shifts = Vec<f64, 2>{};
        auto first_average_shift = Vec<f64, 2>{};
        auto last_average_shift = Vec<f64, 2>{};
        const bool converge = options.update_count < 0;
        const i32 count = converge ? 125 : options.update_count;

        i32 i{};
        while (i < count) {
            const auto average_shift = eval_(
                stack, metadata,
                options.fov_mask,
                options.smooth_edge_percent,
                options.max_shift_percent,
                options.cosine_stretch,
                false
            ).first;

            // Logging.
            if (i == 0)
                first_average_shift = average_shift;
            last_average_shift = average_shift;

            // Add the shifts to the metadata.
            max_shifts = 0;
            for (auto&& [slice, global_shift]: noa::zip(metadata, m_global_shifts)) {
                slice.shifts += global_shift;
                max_shifts = max(max_shifts, abs(global_shift));
            }

            // Loop logic.
            ++i;
            if (converge and noa::sqrt(noa::dot(average_shift, average_shift)) <= 0.001)
                break;
        }

        if (count == 1) {
            Logger::info(
                "average_shift={::.3f}, max_shift={::.3f}, n_iter={}",
                first_average_shift, max_shifts, i
            );
        } else {
            Logger::info(
                "first_average_shift={::.3f}, last_average_shift={::.3f}, max_shift={::.3f}, n_iter={}",
                first_average_shift, last_average_shift, max_shifts, i
            );
        }
    }

    void Tilter::find_specimen_level(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        Vec<f64, 3>& angle_offsets,
        const FindSpecimenLevelOptions& options
    ) {
        auto timer = Logger::info_scope_time("Finding specimen level");
        Logger::trace(
            "  tilt_search_range=+-{:.2f}deg\n"
            "  pitch_search_range=+-{:.2f}deg\n"
            "  fov_mask={}\n"
            "  smooth_edge={:.1f}%",
            options.tilt_search_range,
            options.pitch_search_range,
            options.fov_mask,
            options.smooth_edge_percent * 100
        );

        metadata.sort("tilt"); // just to make sure images are sorted

        auto eval = [&](u32 n, const f64* p, f64* g) -> f64 {
            check(n == 2 and g == nullptr);
            auto meta = metadata;
            meta.add_image_angles({0, p[0], p[1]});
            auto zncc = eval_(
                stack, meta,
                options.fov_mask,
                options.smooth_edge_percent,
                options.max_shift_percent,
                true,
                true
            ).second;
            // Logger::trace("stage=[tilt={:.5f}, pitch={:.5f}deg] (zncc={:.10f})", p[0], p[1], zncc);
            return zncc;
        };

        auto tilt_range = Vec{-options.tilt_search_range, options.tilt_search_range};
        auto pitch_range = Vec{-options.pitch_search_range, options.pitch_search_range};
        auto parameters = Vec{0., 0.};
        auto n_evaluations = i32{0};

        auto optimizer = Optimizer{};
        if (options.n_global_search_evaluations > 0) {
            optimizer = Optimizer(NLOPT_GN_DIRECT_L, 2);
            optimizer.set_max_number_of_evaluations(options.n_global_search_evaluations);
            optimizer.set_bounds(tilt_range, pitch_range);
            optimizer.set_max_objective(eval);
            optimizer.optimize(parameters.data());
            n_evaluations += optimizer.n_evaluations();
            tilt_range = parameters + Vec{-2., 2.};
            pitch_range = parameters + Vec{-2., 2.};
        }
        optimizer = Optimizer(NLOPT_LN_SBPLX, 2);
        optimizer.set_x_tolerance_abs(0.05);
        optimizer.set_bounds(tilt_range, pitch_range);
        optimizer.set_max_objective(eval);
        const f64 zncc = optimizer.optimize(parameters.data());
        n_evaluations += optimizer.n_evaluations();

        // Save the offset.
        metadata.add_image_angles(parameters.push_front(0));
        angle_offsets[1] += parameters[0];
        angle_offsets[2] += parameters[1];

        Logger::info(
            "stage=[tilt={:.2f}deg ({:+.2f}), pitch={:.2f}deg ({:+.2f})] (zncc={:.4f}, n_evaluations={})",
            angle_offsets[1], parameters[0], angle_offsets[2], parameters[1],
            zncc, n_evaluations
        );
    }

    auto Tilter::eval_(
        const View<f32>& stack,
        const Metadata::Stack& metadata,
        bool fov_mask,
        f64 smooth_edge_percent,
        f64 max_shift_percent,
        bool cosine_stretch,
        bool need_score
    ) -> Pair<Vec<f64, 2>, f64> {
        check(metadata.ssize() == stack.shape()[0]);

        const auto device = stack.device();
        const auto pivot = static_cast<i32>(metadata.find_lowest_tilt_index());
        const auto n_total_images = metadata.ssize();
        const auto n_targets = n_total_images - 1;
        const auto image_shape = stack.shape().filter(2, 3);
        const auto image_center = (image_shape.vec / 2).as<f64>();

        // Whether to enforce the common FOV between all the images.
        // This removes the regions from the higher tilts that are not in the lower tilts,
        // which can be quite a lot of information that could be used for the alignment. As such,
        // when the shifts are not known (and large shifts are expected), turning off the common FOV
        // is best. In this case, only the FOV due to the shift difference between the target and
        // its reference is accounted for.
        const auto common_fov = fov_mask ?
            CommonFOV(image_shape, metadata) :
            CommonFOV(image_shape);

        // Prepare for the CreateStacks operator.
        for (i32 i{}; auto&& [reference_plane_coefficients, projection_matrix, mask]: noa::zip(
            m_plane_coefficients.span_1d(),
            m_projection_matrices.span_1d(),
            m_fov_masks.span_1d()
        )) {
            const auto [index_target, index_reference] = get_indices(i++, pivot);
            auto target = metadata[index_target];
            auto reference = metadata[index_reference];
            if (not cosine_stretch)
                reference.angles = target.angles;

            // Compute the plane coefficients of the reference.
            const auto reference_angles = noa::deg2rad(reference.angles);
            const auto reference_plane_rotation = (
                nx::rotate_z(reference_angles[0]) *
                nx::rotate_y(reference_angles[1]) *
                nx::rotate_x(reference_angles[2])
            );
            const auto [c, b, a] = reference_plane_rotation * Vec{1., 0., 0.}; // plane normal
            const auto reference_center = image_center + reference.shifts;
            const auto d = b * -reference_center[0] + a * -reference_center[1]; // precompute coordinate - shifts
            reference_plane_coefficients = Vec{a, b, c, d}.as<f32>();

            // Compute the reference->target transformation.
            const auto target_angles = noa::deg2rad(target.angles);
            projection_matrix = (
                nx::translate(image_center.push_front(0) + target.shifts.push_front(0)) *
                nx::rotate_z<true>(+target_angles[0]) *
                nx::rotate_y<true>(+target_angles[1]) *
                nx::rotate_x<true>(+target_angles[2]) *
                nx::rotate_x<true>(-reference_angles[2]) *
                nx::rotate_y<true>(-reference_angles[1]) *
                nx::rotate_z<true>(-reference_angles[0]) *
                nx::translate(-image_center.push_front(0) - reference.shifts.push_front(0))
            ).filter_rows(1, 2).as<f32>(); // project along z-axis

            mask = common_fov.set_fov(reference, {
                .smooth_edge_percent = smooth_edge_percent,
                .add_shifts = true, // the mask is applied to unaligned images
                .add_tilt_and_pitch = fov_mask,
            });
        }

        // Compute the reference and (stretched-)target stacks.
        using interp_t = CreateStacks::interpolator_type;
        auto iwise_shape = image_shape.push_front(n_targets).as<i32>();
        auto create_stacks = CreateStacks{
            .input = interp_t(stack.span_contiguous<const f32, 3, i32>(), image_shape.as<i32>()),
            .references = buffer(0).span_contiguous<f32, 3, i32>(),
            .stretched_targets = buffer(1).span_contiguous<f32, 3, i32>(),
            .fov_masks = m_fov_masks.span_contiguous<ParallelogramMask, 1, i32>(),
            .reference_plane_coefficients = m_plane_coefficients.span_contiguous<Vec<f32, 4>, 1, i32>(),
            .reference2target = m_projection_matrices.span_contiguous<Mat<f32, 2, 4>, 1, i32>(),
            .pivot = pivot,
        };

        if (need_score) {
            // On top of computing the stacks, compute the stats for the ZNCC score.
            noa::reduce_axes_iwise( // (n,h,w)->(n,1,1)
                iwise_shape, device, CreateStacks::reduce_type{},
                m_peak_stats.view().flat(1), create_stacks
            );
        } else {
            noa::iwise(iwise_shape, device, create_stacks);
        }

        // if (not Logger::s_debug_path.empty()) {
        //     auto filename = Logger::s_debug_path / "stretched_targets.mrc";
        //     noa::write_image(buffer(1), filename, {.dtype = "f32"});
        //     filename = Logger::s_debug_path / "references.mrc";
        //     noa::write_image(buffer(0), filename, {.dtype = "f32"});
        //     Logger::debug("{} saved", filename);
        // }

        // (Conventional) cross-correlation, returning a centered map.
        // Note that to compute a ZNCC between -1 and 1, the FFT normalization should be done on the xmap (BACKWARD).
        nf::r2c(buffer(0, 2), buffer_rfft(0, 2), {.norm = nf::Norm::BACKWARD});
        noa::iwise(buffer_rfft(0).shape().filter(0, 2, 3).as<i32>(), device, CrossCorrelate{
            .references = buffer_rfft(0).span_contiguous<const c32, 3, i32>(),
            .stretched_targets = buffer_rfft(1).span_contiguous<c32, 3, i32>(), // output
        });
        nf::c2r(buffer_rfft(1), buffer(0), {.norm = nf::Norm::BACKWARD}); // centered xmap

        // if (not Logger::s_debug_path.empty()) {
        //     auto filename = Logger::s_debug_path / "xmap.mrc";
        //     noa::write_image(buffer(0), filename, {.dtype = "f32"});
        //     Logger::debug("{} saved", filename);
        // }

        // Compute the shift, i.e., by how much the (stretched-)target is away from the reference.
        // To align the (stretched-)target onto the reference, we would need to subtract this shift from it.
        find_peaks<"fc">(buffer(0), m_xmap_centered.view(), m_peak_shifts.view(), m_peak_values.view(), {
            .distortion_angle_deg = metadata[0].angles[0],
            .max_shift_percent = max_shift_percent,
        });

        // We should now transform the shifts back to each target's reference-frame. However, we'll need to compute the
        // global shifts (and center them) later on. These operations require accumulating the shifts of the lower views
        // up to the global reference. As such, the simplest is to scale all these image-to-image shifts directly to
        // the same reference-frame, process everything there, and then go back to each image's reference-frame at
        // the end. For simplicity, we chose this common reference-frame to be the volume reference-frame, which has
        // no rotation, no tilt, no pitch.
        f64 average_zncc{};
        m_relative_shifts[static_cast<usize>(pivot)] = {};
        for (i32 i{}; auto&& [peak_shift, peak_value, peak_stats]: noa::zip(
            m_peak_shifts.span_1d(),
            m_peak_values.span_1d(),
            m_peak_stats.span_1d()
        )) {
            const auto [index_target, index_reference] = get_indices(i++, pivot);
            auto target = metadata[index_target];
            auto reference = metadata[index_reference];
            if (not cosine_stretch)
                reference.angles = target.angles;

            // Compute the reference-plane coefficients.
            const auto reference_angles = noa::deg2rad(reference.angles);
            const auto reference_plane_rotation = (
                nx::rotate_z(reference_angles[0]) *
                nx::rotate_y(reference_angles[1]) *
                nx::rotate_x(reference_angles[2])
            );

            // Compute the z-coordinate at the image shift.
            const auto projected_shift = peak_shift.as<f64>();
            const auto [c, b, a] = reference_plane_rotation * Vec{1., 0., 0.}; // plane normal
            const auto z = -(a * projected_shift[1] + b * projected_shift[0]) / c;

            // Transform the shifts to volume-space.
            const auto reference2volume = (
                nx::rotate_x<true>(-reference_angles[2]) *
                nx::rotate_y<true>(-reference_angles[1]) *
                nx::rotate_z<true>(-reference_angles[0])
            ).filter_rows(1, 2);
            auto shift = reference2volume * Vec{z, projected_shift[0], projected_shift[1], 1.};

            m_relative_shifts[static_cast<usize>(index_target)] = shift;

            if (need_score) {
                // Center and L2-normalize the peak.
                const auto [lhs_sum, rhs_sum, lhs_sum_sqd, rhs_sum_sqd, mask_sum] = peak_stats.as<f64>();
                const auto lhs_mean = lhs_sum / mask_sum;
                const auto rhs_mean = rhs_sum / mask_sum;
                const auto lhs_variance = lhs_sum_sqd - mask_sum * lhs_mean * lhs_mean;
                const auto rhs_variance = rhs_sum_sqd - mask_sum * rhs_mean * rhs_mean;
                const auto energy = std::sqrt(lhs_variance * rhs_variance);
                if (energy >= 1e-6) {
                    // The ZNCC can be very close to zero, which doesn't inspire confidence.
                    // However, when testing with identical images, it correctly gives one.
                    // On simulated data, it is also fairly accurate.
                    auto zncc = (static_cast<f64>(peak_value) - mask_sum * lhs_mean * rhs_mean) / energy;
                    average_zncc += zncc;
                }
            }
        }
        average_zncc /= static_cast<f64>(n_targets);

        return {relative2global_shifts_(m_relative_shifts, m_global_shifts, metadata, pivot), average_zncc};
    }
}
