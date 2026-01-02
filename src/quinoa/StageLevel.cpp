#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Signal.hpp>

// #include "quinoa/GridSearch.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/StageLevel.hpp"
#include "quinoa/Types.hpp"

namespace {
    using namespace qn;

    Path debug_path;

    constexpr auto get_indices_(i32 idx_target, i32 index_lowest_tilt) {
        if (idx_target >= index_lowest_tilt)
            idx_target += 1;

        // The tilts are sorted in ascending order, so if the ith target has:
        //  - a negative tilt angle, then the reference is at i + 1.
        //  - a positive tilt angle, then the reference is at i - 1.
        const bool is_negative = idx_target < index_lowest_tilt;
        const i32 idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;
        return Pair{idx_target, idx_reference};
    }

    struct CrossCorrelate {
        using span_type = SpanContiguous<const f32, 3, i32>;
        using interpolator_type = noa::Interpolator<2, noa::Interp::LINEAR, noa::Border::ZERO, span_type>;

        interpolator_type stack{};
        SpanContiguous<const Vec<f32, 4>> reference_plane_coefficients{};
        SpanContiguous<const Mat<f32, 2, 4>> reference2target{};
        i32 index_lowest_tilt{};

        SpanContiguous<f32, 3> debug_target_stretched{};
        SpanContiguous<f32, 3> debug_reference{};

        NOA_HD void init(i32 batch, i32 y, i32 x, f32& cc, f32& lhs_cc, f32& rhs_cc) const {
            const auto [index_target, index_reference] = get_indices_(batch, index_lowest_tilt);

            // Get the z at this image coordinate using the plane equation.
            const auto& [a, b, c, d] = reference_plane_coefficients[batch];
            const auto z = -(a * static_cast<f32>(x) + b * static_cast<f32>(y) + d) / c;
            const auto reference_coordinates = Vec<f32, 3>::from_values(z, y, x);

            // Transform from target to reference.
            const auto target_coordinates = reference2target[batch] * reference_coordinates.push_back(1);
            const auto target_stretched = stack.interpolate_at(target_coordinates, index_target);
            const auto& reference = stack(index_reference, y, x);

            if (not debug_target_stretched.is_empty()) {
                debug_target_stretched(batch, y, x) = target_stretched;
                debug_reference(batch, y, x) = reference;
            }

            cc += reference * target_stretched;
            lhs_cc += reference * reference;
            rhs_cc += target_stretched * target_stretched;
        }

        static NOA_HD void join(
            f32 icc, f32 icc_lhs, f32 icc_rhs,
            f32& cc, f32& cc_lhs, f32& cc_rhs
        ) {
            cc += icc;
            cc_lhs += icc_lhs;
            cc_rhs += icc_rhs;
        }

        using remove_defaulted_final = bool;
        static NOA_HD void final(f32 cc, f32 cc_lhs, f32 cc_rhs, f32& ncc) {
            const auto energy = noa::sqrt(cc_lhs) * noa::sqrt(cc_rhs);
            if (noa::abs(energy) > 1e-6f)
                ncc = cc / energy;
        }
    };

    struct CrossCorrelate2 {
        using span_type = SpanContiguous<const f32, 3, i32>;
        using interpolator_type = noa::Interpolator<2, noa::Interp::LINEAR, noa::Border::ZERO, span_type>;

        interpolator_type stack{};
        SpanContiguous<const Vec<f32, 4>> reference_plane_coefficients{};
        SpanContiguous<const Mat<f32, 2, 4>> reference2target{};
        i32 index_lowest_tilt{};

        SpanContiguous<f32, 3, i32> target_stretched{};

        NOA_HD void operator()(i32 batch, i32 y, i32 x) const {
            const auto [index_target, index_reference] = get_indices_(batch, index_lowest_tilt);

            // Get the z at this image coordinate using the plane equation.
            const auto& [a, b, c, d] = reference_plane_coefficients[batch];
            const auto z = -(a * static_cast<f32>(x) + b * static_cast<f32>(y) + d) / c;
            const auto reference_coordinates = Vec<f32, 3>::from_values(z, y, x);

            // Transform from target to reference.
            const auto target_coordinates = reference2target[batch] * reference_coordinates.push_back(1);
            target_stretched(batch, y, x) = stack.interpolate_at(target_coordinates, index_target);
        }
    };

    auto ncc_(
        const View<f32>& stack,
        const Metadata::Stack& metadata,
        const Array<Vec<f32, 4>>& plane_coefficients,
        const Array<Mat<f32, 2, 4>>& projection_matrices,
        const Array<f32>& nccs
    ) {
        check(metadata.ssize() == stack.shape()[0]);

        const auto device = stack.device();
        const auto index_lowest_tilt = static_cast<i32>(metadata.find_lowest_tilt_index());
        const auto n_slices = metadata.ssize() - 1; // remove the lowest tilt
        const auto slice_shape = stack.shape().filter(2, 3);
        const auto slice_center = (slice_shape.vec / 2).as<f64>();

        for (i32 i{}; auto&& [coeffs, matrix]: noa::zip(plane_coefficients.span_1d(), projection_matrices.span_1d())) {
            const auto [index_target, index_reference] = get_indices_(i++, index_lowest_tilt);
            const auto& target = metadata[index_target];
            const auto& reference = metadata[index_reference];

            // Compute the reference-plane coefficients.
            const auto target_angles = noa::deg2rad(target.angles);
            const auto reference_angles = noa::deg2rad(reference.angles);
            const auto reference_plane_rotation = (
                ng::rotate_z(reference_angles[0]) *
                ng::rotate_y(reference_angles[1]) *
                ng::rotate_x(reference_angles[2])
            );
            const auto [c, b, a] = reference_plane_rotation * Vec{1., 0., 0.}; // plane normal
            const auto reference_center = slice_center + reference.shifts;
            const auto d = b * -reference_center[0] + a * -reference_center[1]; // precompute coordinate - shifts
            coeffs = Vec{a, b, c, d}.as<f32>();

            // Compute the reference->target transformation.
            matrix = (
                ng::translate(slice_center.push_front(0) + target.shifts.push_front(0)) *
                ng::rotate_z<true>(target_angles[0]) *
                ng::rotate_y<true>(target_angles[1]) *
                ng::rotate_x<true>(target_angles[2]) *
                ng::rotate_x<true>(-reference_angles[2]) *
                ng::rotate_y<true>(-reference_angles[1]) *
                ng::rotate_z<true>(-reference_angles[0]) *
                ng::translate(-slice_center.push_front(0) - reference.shifts.push_front(0))
            ).filter_rows(1, 2).as<f32>();
        }

        // Normalize cross-correlation between the references and their cosine-stretched targets.
        using interp_t = CrossCorrelate::interpolator_type;
        noa::reduce_axes_iwise( // DHW->D11
            slice_shape.push_front(n_slices), device, noa::wrap(0.f, 0.f, 0.f), nccs.view().flat(1),
            CrossCorrelate{
                .stack = interp_t(stack.span_contiguous<f32, 3, i32>(), slice_shape.as<i32>()),
                .reference_plane_coefficients = plane_coefficients.span_1d(),
                .reference2target = projection_matrices.span_1d(),
                .index_lowest_tilt = index_lowest_tilt,
            });

        // Optimize for the entire stack.
        f64 average{};
        for (auto ncc: nccs.eval().span_1d())
            average += static_cast<f64>(ncc);
        return average / static_cast<f64>(n_slices);
    }

    auto ncc2_(
        const View<f32>& stack,
        const View<f32>& target_stretched,
        const Metadata::Stack& metadata,
        const Array<Vec<f32, 4>>& plane_coefficients,
        const Array<Mat<f32, 2, 4>>& projection_matrices,
        const Array<f32>& nccs
    ) {
        check(metadata.ssize() == stack.shape()[0]);

        const auto device = stack.device();
        const auto index_lowest_tilt = static_cast<i32>(metadata.find_lowest_tilt_index());
        const auto n_slices = metadata.ssize() - 1; // remove the lowest tilt
        const auto slice_shape = stack.shape().filter(2, 3);
        const auto slice_center = (slice_shape.vec / 2).as<f64>();

        for (i32 i{}; auto&& [coeffs, matrix]: noa::zip(plane_coefficients.span_1d(), projection_matrices.span_1d())) {
            const auto [index_target, index_reference] = get_indices_(i++, index_lowest_tilt);
            const auto& target = metadata[index_target];
            const auto& reference = metadata[index_reference];

            // Compute the reference-plane coefficients.
            const auto target_angles = noa::deg2rad(target.angles);
            const auto reference_angles = noa::deg2rad(reference.angles);
            const auto reference_plane_rotation = (
                ng::rotate_z(reference_angles[0]) *
                ng::rotate_y(reference_angles[1]) *
                ng::rotate_x(reference_angles[2])
            );
            const auto [c, b, a] = reference_plane_rotation * Vec{1., 0., 0.}; // plane normal
            const auto reference_center = slice_center + reference.shifts;
            const auto d = b * -reference_center[0] + a * -reference_center[1]; // precompute coordinate - shifts
            coeffs = Vec{a, b, c, d}.as<f32>();

            // Compute the reference->target transformation.
            matrix = (
                ng::translate(slice_center.push_front(0) + target.shifts.push_front(0)) *
                ng::rotate_z<true>(target_angles[0]) *
                ng::rotate_y<true>(target_angles[1]) *
                ng::rotate_x<true>(target_angles[2]) *
                ng::rotate_x<true>(-reference_angles[2]) *
                ng::rotate_y<true>(-reference_angles[1]) *
                ng::rotate_z<true>(-reference_angles[0]) *
                ng::translate(-slice_center.push_front(0) - reference.shifts.push_front(0))
            ).filter_rows(1, 2).as<f32>();
        }

        // Normalize cross-correlation between the references and their cosine-stretched targets.
        using interp_t = CrossCorrelate::interpolator_type;
        noa::iwise(
            slice_shape.push_front(n_slices), device,
            CrossCorrelate2{
                .stack = interp_t(stack.span_contiguous<const f32, 3, i32>(), slice_shape.as<i32>()),
                .reference_plane_coefficients = plane_coefficients.span_1d(),
                .reference2target = projection_matrices.span_1d(),
                .index_lowest_tilt = index_lowest_tilt,
                .target_stretched = target_stretched.span_contiguous<f32, 3, i32>()
            });

        // nf::r2c(target_stretched, projected_target_and_reference_rfft);
        // ns::cross_correlation_map<"h2fc">(
        //     projected_target_and_reference_rfft.subregion(0),
        //     projected_target_and_reference_rfft.subregion(1),
        //     xmap, {.mode = ns::Correlation::CONVENTIONAL}
        // );

        // Optimize for the entire stack.
        f64 average{};
        for (auto ncc: nccs.eval().span_1d())
            average += static_cast<f64>(ncc);
        return average / static_cast<f64>(n_slices);
    }
}

namespace qn {
    void coarse_stage_leveling(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        Vec<f64, 2>& tilt_pitch_offset,
        const StageLevelingParameters& options
    ) {
        auto timer = Logger::info_scope_time("Stage leveling");
        Logger::trace(
            "device={}\n"
            "bounds=[tilt={:.2f}, pitch={:.2f}]",
            stack.device(), options.tilt_search_range, options.pitch_search_range
        );

        // debug_path = "/dls/ebic/data/staff-scratch/thomas2/tmp/kyprianos/03";

        // The algorithm assumes that the images in the stack are sorted by their tilt angles.
        // We know this to be true during the coarse alignment, but just in case we change something:
        auto metadata_sorted = metadata;
        metadata_sorted.sort("tilt");
        bool stack_is_sorted{true};
        for (i32 expected_index{0}; const auto& image: metadata_sorted) {
            if (expected_index++ != image.index) {
                stack_is_sorted = false;
                break;
            }
        }
        check(stack_is_sorted, "The tilts in the stack should be sorted in ascending order");

        // Unified reusable buffers.
        const auto options_managed = ArrayOption{.device = stack.device(), .allocator = Allocator::MANAGED};
        const auto projection_matrices = Array<Mat<f32, 2, 4>>(metadata.ssize() - 1, options_managed);
        const auto plane_coefficients = noa::like<Vec<f32, 4>>(projection_matrices);
        const auto nccs = noa::like<f32>(projection_matrices);

        auto eval = [&](u32 n, const f64* p, f64* g) {
            check(n == 2 and g == nullptr);
            auto i_metadata = metadata_sorted;
            i_metadata.add_image_angles({0, p[0], p[1]});
            return ncc_(stack, i_metadata, plane_coefficients, projection_matrices, nccs);
        };

        auto tilt_range = Vec{-options.tilt_search_range, options.tilt_search_range};
        auto pitch_range = Vec{-options.pitch_search_range, options.pitch_search_range};
        auto parameters = Vec{0., 0.};
        auto n_evaluations = 0;

        // Logger::s_is_debug = true;
        // eval(2, parameters.data(), nullptr);
        // Logger::s_is_debug = false;

        auto optimizer = Optimizer{};
        if (options.tilt_search_range > 10.) {
            optimizer = Optimizer(NLOPT_GN_DIRECT_L, 2);
            optimizer.set_max_number_of_evaluations(100);
            optimizer.set_bounds(tilt_range, pitch_range);
            optimizer.set_max_objective(eval);
            optimizer.optimize(parameters.data());
            n_evaluations += optimizer.n_evaluations();
        }
        optimizer = Optimizer(NLOPT_LN_SBPLX, 2);
        optimizer.set_x_tolerance_abs(0.005);
        optimizer.set_bounds(tilt_range, pitch_range);
        optimizer.set_max_objective(eval);
        const f64 ncc = optimizer.optimize(parameters.data());
        n_evaluations += optimizer.n_evaluations();

        // Logger::s_is_debug = true;
        // eval(2, parameters.data(), nullptr);
        // Logger::s_is_debug = false;

        // Save the offset.
        metadata.add_image_angles(parameters.push_front(0));
        tilt_pitch_offset += parameters;

        Logger::info(
            "stage=[tilt={:.2f}deg ({:+.2f}), pitch={:.2f}deg ({:+.2f})] (ncc={:.4f}, n_evaluations={})",
            tilt_pitch_offset[0], parameters[0], tilt_pitch_offset[1], parameters[1],
            ncc, n_evaluations
        );
    }
}
