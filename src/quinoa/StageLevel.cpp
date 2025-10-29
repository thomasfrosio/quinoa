#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>

#include "quinoa/GridSearch.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/StageLevel.hpp"
#include "quinoa/Types.hpp"

namespace {
    using namespace qn;

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

        NOA_HD void init(i32 batch, i32 y, i32 x, f32& cc, f32& lhs_cc, f32& rhs_cc) const {
            const auto [index_target, index_reference] = get_indices_(batch, index_lowest_tilt);

            // Get the z at this image coordinate by solving the plane equation (d - ax - by) / c.
            const auto& [a, b, c, d] = reference_plane_coefficients[batch];
            const auto z = -(a * static_cast<f32>(x) + b * static_cast<f32>(y) + d) / c;
            const auto reference_coordinates = Vec<f32, 3>::from_values(z, y, x);

            // Transform from target to reference.
            const auto target_coordinates = reference2target[batch] * reference_coordinates.push_back(1);
            const auto target_stretched = stack.interpolate_at(target_coordinates, index_target);
            const auto& reference = stack(index_reference, y, x);

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

    auto ncc_(
        const View<f32>& stack,
        const MetadataStack& metadata,
        const View<Vec<f32, 4>>& plane_coefficients,
        const View<Mat<f32, 2, 4>>& projection_matrices,
        const View<f32>& nccs
    ) {
        check(metadata.ssize() == stack.shape()[0]);

        const auto device = stack.device();
        const auto index_lowest_tilt = static_cast<i32>(metadata.find_lowest_tilt_index());
        const auto n_slices = metadata.ssize() - 1; // remove the lowest tilt
        const auto slice_shape = stack.shape().filter(2, 3);
        const auto slice_center = (slice_shape.vec / 2).as<f64>();

        // Compute the target->reference transformations.
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

            // Compute the reference -> target transformation.
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
            slice_shape.push_front(n_slices), device, noa::wrap(0.f, 0.f, 0.f), nccs.flat(1),
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
}

namespace qn {
    void coarse_stage_leveling(
        const View<f32>& stack,
        MetadataStack& metadata,
        Vec<f64, 2>& tilt_pitch_offset,
        const StageLevelingParameters& options
    ) {
        auto timer = Logger::info_scope_time("Stage leveling");
        Logger::trace(
            "device={}\n"
            "tilt_search=[range={:.2f}, step={:.2f}]deg\n"
            "pitch_search=[range={:.2f}, step={:.2f}]deg",
            stack.device(),
            options.tilt_search_range, options.tilt_search_step,
            options.pitch_search_range, options.pitch_search_step
        );

        // The algorithm assumes that the slices in the stack are sorted by their tilt angles.
        // We know this to be true during the coarse alignment, but just in case we change something:
        auto metadata_sorted = metadata;
        metadata_sorted.sort("tilt");
        bool stack_is_sorted{true};
        for (i32 expected_index{0}; const auto& slice: metadata_sorted) {
            if (expected_index++ != slice.index) {
                stack_is_sorted = false;
                break;
            }
        }
        check(stack_is_sorted, "The tilts in the stack should be sorted in ascending order");

        // Unified reusable buffers.
        const auto projection_matrices = Array<Mat<f32, 2, 4>>(metadata.ssize() - 1, {
            .device = stack.device(),
            .allocator = Allocator::MANAGED,
        });
        const auto plane_coefficients = noa::like<Vec<f32, 4>>(projection_matrices);
        const auto nccs = noa::like<f32>(projection_matrices);

        f64 best_ncc{};
        Vec<f64, 2> best_offsets{};

        GridSearch<f64, f64>(
            {.start = -options.tilt_search_range, .end = options.tilt_search_range, .step = options.tilt_search_step},
            {.start = -options.pitch_search_range, .end = options.pitch_search_range, .step = options.pitch_search_step}
        ).for_each([&](f64 tilt_offset, f64 pitch_offset) {
            auto i_metadata = metadata_sorted;
            i_metadata.add_image_angles({0, tilt_offset, pitch_offset});
            const auto ncc = ncc_(stack.view(), i_metadata, plane_coefficients.view(), projection_matrices.view(), nccs.view());
            if (ncc > best_ncc) {
                best_ncc = ncc;
                best_offsets = {tilt_offset, pitch_offset};
            }
        });

        // Save the offset.
        tilt_pitch_offset += best_offsets;
        metadata.add_image_angles(best_offsets.push_front(0));
        Logger::info(
            "stage=[tilt={:.2f}deg ({:+.2f}), pitch={:.2f}deg ({:+.2f})] (ncc={:.4f})",
            tilt_pitch_offset[0], best_offsets[0], tilt_pitch_offset[1], best_offsets[1], best_ncc
        );
    }
}
