#include <noa/Signal.hpp>
#include <noa/FFT.hpp>

#include "quinoa/Logger.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/Utilities.hpp"

#include "quinoa/align/Coarse.hpp"

namespace {
    using namespace qn;

    constexpr auto get_indices(i32 idx_target, i32 index_lowest_tilt) {
        if (idx_target >= index_lowest_tilt)
            idx_target += 1;

        // The tilts are sorted in ascending order, so if the ith target has:
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
}

namespace qn {
    AlignmentCoarse::AlignmentCoarse(
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
        const auto buffer_shape = shape.set<0>(n_total_images * 4);

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

        m_xmap_centered = Array<f32>({n_target_images, 1, 64, 64}, m_fov_masks.options());
        m_peak_shifts = noa::like<Vec<f32, 2>>(m_fov_masks);
        m_peak_values = noa::like<f32>(m_fov_masks);
        m_peak_stats = noa::like<Vec<f32, 5>>(m_fov_masks);

        // Prepare the shifts.
        m_relative_shifts.resize(static_cast<usize>(n_total_images));
        m_global_shifts.resize(static_cast<usize>(n_total_images));

        // Prepare FFT plans and set the workspace.
        if (device.is_gpu()) {
            nf::clear_cache(device);
            nf::set_cache_limit(10, device);
            nf::r2c(buffer(0, 2), buffer_rfft(0, 2), {.record_and_share_workspace = true});
            nf::c2r(buffer_rfft(1), buffer(0), {.record_and_share_workspace = true});
            const auto workspace = m_buffer_rfft.subregion(Offset{2 * n_target_images});
            const auto n_plans_set = nf::set_workspace(device, workspace);
            if (n_plans_set != 2) {
                Logger::warn(
                    "Failed to set the FFT workspace. An new workspace will have to be allocated, "
                    "possibly increasing the memory requirements significantly. Please report this. "
                    "shape={}, workspace_left_to_allocate={}bytes, n_plans_set={}",
                    shape, nf::workspace_left_to_allocate(device), n_plans_set);
            }
        }

        const auto allocated = Allocator::bytes_currently_allocated(device) - allocated_start;
        Logger::trace("AlignmentCoarse() allocated {:.2f}GB on {} ({})",
                      static_cast<f64>(allocated) * 1e-9, m_buffer.device(), m_buffer.allocator());
    }

    void AlignmentCoarse::align_shifts(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const AlignShiftsOptions& options
    ) {
        auto timer = Logger::info_scope_time("Coarse shift alignment");
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

        Logger::info(
            "first_average_shift={::.3f}, last_average_shift={::.3f}, max_shift={::.3f}, n_iter={}",
            first_average_shift, last_average_shift, max_shifts, i
        );
    }

    void AlignmentCoarse::level_stage(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        Vec<f64, 3>& angle_offsets,
        const LevelStageOptions& options
    ) {
        auto timer = Logger::info_scope_time("Leveling the stage");
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
            return zncc;
        };

        auto tilt_range = Vec{-options.tilt_search_range, options.tilt_search_range};
        auto pitch_range = Vec{-options.pitch_search_range, options.pitch_search_range};
        auto parameters = Vec{0., 0.};
        auto n_evaluations = i32{0};

        auto optimizer = Optimizer{};
        if (options.tilt_search_range > 10.) {
            optimizer = Optimizer(NLOPT_GN_DIRECT_L, 2);
            optimizer.set_max_number_of_evaluations(100); // this should be more than enough
            optimizer.set_bounds(tilt_range, pitch_range);
            optimizer.set_max_objective(eval);
            optimizer.optimize(parameters.data());
            n_evaluations += optimizer.n_evaluations();
        }
        optimizer = Optimizer(NLOPT_LN_SBPLX, 2);
        optimizer.set_x_tolerance_abs(0.01);
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

    auto AlignmentCoarse::eval_(
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

        // if (s_debug_path) {
        //     auto filename = *s_debug_path / "stretched_targets.mrc";
        //     noa::write_image(buffer(1), filename, {.dtype = "f16"});
        //     filename = *s_debug_path / "references.mrc";
        //     noa::write_image(buffer(0), filename, {.dtype = "f16"});
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

        // if (s_debug_path) {
        //     auto filename = *s_debug_path / "xmap.mrc";
        //     noa::write_image(buffer(0), filename, {.dtype = "f16"});
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
                    auto zncc = (static_cast<f64>(peak_value) - mask_sum * lhs_mean * rhs_mean) / energy;
                    average_zncc += zncc;
                }
            }
        }
        average_zncc /= static_cast<f64>(n_targets);

        return {relative2global_shifts_(m_relative_shifts, m_global_shifts, metadata, pivot), average_zncc};
    }
}
