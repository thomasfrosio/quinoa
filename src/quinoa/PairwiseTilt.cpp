#include "quinoa/GridSearch.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/PairwiseTilt.hpp"
#include "quinoa/Types.hpp"
#include "quinoa/Plot.hpp"

#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>

namespace {
    using namespace qn;

    constexpr auto get_indices_(i32 idx_target, i32 index_lowest_tilt) {
        if (idx_target >= index_lowest_tilt)
            idx_target += 1;

        // If ith target has:
        //  - negative tilt angle, then reference is at i + 1.
        //  - positive tilt angle, then reference is at i - 1.
        const bool is_negative = idx_target < index_lowest_tilt;
        const i32 idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;
        return Pair{idx_target, idx_reference};
    }

    template<typename Input, typename Matrices>
    struct CrossCorrelate {
        Input stack;
        Matrices inverse_matrices;
        i32 index_lowest_tilt;

        constexpr void init(i32 batch, i32 y, i32 x, f32& cc, f32& lhs_cc, f32& rhs_cc) const {
            const auto [index_target, index_reference] = get_indices_(batch, index_lowest_tilt);

            const auto coordinates = inverse_matrices[batch] * Vec<f32, 3>::from_values(y, x, 1);
            const auto target_stretched = stack.interpolate_at(coordinates, index_target);
            const auto& reference = stack(index_reference, y, x);

            cc += reference * target_stretched;
            lhs_cc += reference * reference;
            rhs_cc += target_stretched * target_stretched;
        }

        static constexpr void join(
            f32 icc, f32 icc_lhs, f32 icc_rhs,
            f32& cc, f32& cc_lhs, f32& cc_rhs
        ) {
            cc += icc;
            cc_lhs += icc_lhs;
            cc_rhs += icc_rhs;
        }

        using remove_defaulted_final = bool;
        static constexpr void final(f32 cc, f32 cc_lhs, f32 cc_rhs, f32& ncc) {
            // Normalize using autocorrelation.
            const auto energy = noa::sqrt(cc_lhs) * noa::sqrt(cc_rhs);
            ncc = cc / energy;
        }
    };

    auto ncc_(
        const View<f32>& stack,
        const MetadataStack& metadata
    ) {
        check(metadata.ssize() == stack.shape()[0]);

        const auto device = stack.device();
        const auto index_lowest_tilt = static_cast<i32>(metadata.find_lowest_tilt_index());
        const auto n_slices = metadata.ssize() - 1; // remove the lowest tilt
        const auto slice_shape = stack.shape().filter(2, 3);
        const auto slice_center = (slice_shape.vec / 2).as<f64>();

        // Compute the target->reference transformations.
        auto matrices = Array<Mat<f32, 2, 3>>(n_slices);
        for (i32 i{}; auto& matrix: matrices.span_1d_contiguous()) {
            const auto [index_target, index_reference] = get_indices_(i++, index_lowest_tilt);

            // Compute the affine matrix to transform the target "onto" the reference.
            const Vec<f64, 3> target_angles = noa::deg2rad(metadata[index_target].angles);
            const Vec<f64, 3> reference_angles = noa::deg2rad(metadata[index_reference].angles);

            // First, compute the cosine stretching to estimate the tilt (and technically pitch) difference.
            // These angles are flipped, since the cos-scaling is perpendicular to the axis of rotation.
            // So since the tilt axis is along Y, its stretching is along X.
            Vec<f64, 2> cos_factor = noa::cos(reference_angles.filter(2, 1)) / noa::cos(target_angles.filter(2, 1));

            // Cancel the difference (if any) in rotation and shift as well.
            matrix = (
                ng::translate(slice_center + metadata[index_reference].shifts) *
                ng::linear2affine(ng::rotate(reference_angles[0])) *
                ng::linear2affine(ng::scale(cos_factor)) *
                ng::linear2affine(ng::rotate(-target_angles[0])) *
                ng::translate(-slice_center - metadata[index_target].shifts)
            ).inverse().pop_back().as<f32>();
        }
        matrices = std::move(matrices).to({.device = device, .allocator = Allocator::ASYNC});

        // Normalize cross-correlation between the references and their cosine-stretched targets.
        auto nccs = noa::like<f32>(matrices);
        auto stack_span = stack.span<f32, 3, i32>().as_contiguous();
        using interpolator_t = noa::Interpolator<2, noa::Interp::LINEAR, noa::Border::ZERO, decltype(stack_span)>;

        noa::reduce_axes_iwise( // DHW->D11
            slice_shape.push_front(n_slices), device, noa::wrap(0.f, 0.f, 0.f), nccs.flat(1),
            CrossCorrelate{
                .stack = interpolator_t(stack_span, slice_shape.as<i32>()),
                .inverse_matrices = matrices.span_1d_contiguous(),
                .index_lowest_tilt = index_lowest_tilt,
            });
        nccs = std::move(nccs).to_cpu();

        // The optimization is for the entire stack, so return the average ncc.
        f64 average{};
        for (auto ncc: nccs.span_1d_contiguous())
            average += static_cast<f64>(ncc);
        return average / static_cast<f64>(n_slices);
    }
}

namespace qn {
    void coarse_fit_tilt(
        const View<f32>& stack,
        MetadataStack& metadata,
        f64& tilt_offset,
        const PairwiseTiltOptions& options
    ) {
        auto timer = Logger::info_scope_time("Tilt offset alignment using pairwise cosine-stretching");
        Logger::trace(
            "device={}\n"
            "angle_range={:.2f}deg\n"
            "angle_step={:.2f}deg\n",
            stack.device(), options.grid_search_range, options.grid_search_step
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

        f64 best_ncc{};
        f64 best_tilt_offset{};
        std::vector<f64> nccs;
        GridSearch<f64>({
            .start = -options.grid_search_range,
            .end = options.grid_search_range,
            .step = options.grid_search_step,
        }).for_each([&](f64 offset) {
            auto i_metadata = metadata_sorted;
            i_metadata.add_image_angles({0, offset, 0});
            const auto ncc = ncc_(stack.view(), i_metadata);
            if (ncc > best_ncc) {
                best_ncc = ncc;
                best_tilt_offset = offset;
            }
            nccs.emplace_back(ncc);
        });

        // Save the offset.
        tilt_offset += best_tilt_offset;
        metadata.add_image_angles({0, best_tilt_offset, 0});
        Logger::info("tilt_offset={:.3f} ({:+.3f}) degrees (ncc={:.4f})", tilt_offset, best_tilt_offset, best_ncc);

        save_plot_xy(
            noa::Arange{tilt_offset - options.grid_search_range, options.grid_search_step},
            nccs, options.output_directory / "tilt_offsets.txt", {
                .title = "Tilt offset alignment using pairwise cosine-stretching",
                .x_name = "Tilt offsets (degrees)",
                .y_name = "NCC",
            });
    }
}
