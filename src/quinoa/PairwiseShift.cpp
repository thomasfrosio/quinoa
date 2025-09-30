#include <noa/IO.hpp>
#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>
#include <noa/Utils.hpp>

#include "quinoa/PairwiseShift.hpp"
#include "quinoa/Plot.hpp"

namespace {
    using namespace qn;

    auto relative2global_shifts_(
        const std::vector<Vec<f64, 2>>& relative_shifts,
        const MetadataStack& metadata,
        i64 index_of_global_reference
    ) -> Pair<std::vector<Vec<f64, 2>>, Vec<f64, 2>> {
        // Relative shifts (target->reference) to global shifts (target->volume).
        const auto count = std::size(relative_shifts);
        const auto pivot = index_of_global_reference;
        const auto r_pivot = std::ssize(relative_shifts) - 1 - pivot;
        auto global_shifts = std::vector<Vec<f64, 2>>(relative_shifts.size());
        std::inclusive_scan(relative_shifts.begin() + pivot, relative_shifts.end(), global_shifts.begin() + pivot);
        std::inclusive_scan(relative_shifts.rbegin() + r_pivot, relative_shifts.rend(), global_shifts.rbegin() + r_pivot);

        // Center the shifts.
        auto mean = Vec<f64, 2>{};
        for (auto& shift: global_shifts)
            mean += shift;
        mean /= static_cast<f64>(count);
        for (auto& shift: global_shifts)
            shift -= mean;

        // Transform the shifts from volume-space back to image-space.
        // This effectively shrinks the shifts to account for the tilt/pitch of
        // the images and applies the rotation of the images.
        for (size_t i{}; i < count; ++i) {
            const auto angles = noa::deg2rad(metadata[i].angles);
            const auto volume2image =
                ng::rotate(angles[0]) *
                ng::scale(noa::cos(angles.filter(2, 1)));
            global_shifts[i] = volume2image * global_shifts[i];
        }
        return {global_shifts, mean};
    }
}

namespace qn {
    PairwiseShift::PairwiseShift(
        const Shape4<i64>& shape,
        Device compute_device,
        Allocator allocator
    ) {
        // The "target" is the higher (absolute) tilt image that we want to align onto the "reference".
        // Performance-wise, using out-of-place FFTs could be slightly better, but here, prefer the safer
        // option to use less memory and in-place FFTs, since we are not constrained by performance at this step.
        const auto options = ArrayOption{compute_device, allocator};
        m_buffer_rfft = Array<c32>({3, 1, shape[2], shape[3] / 2 + 1}, options);
        m_xmap = Array<f32>({1, 1, shape[2], shape[3]}, options);
        m_xmap_centered = Array<f32>({1, 1, 64, 64}, {.device = compute_device, .allocator = Allocator::MANAGED});

        const auto n_bytes = m_xmap.size() * sizeof(f32) + m_buffer_rfft.size() * sizeof(c32);
        Logger::trace("PairwiseShift(): allocated {:.2f}MB on {} ({})",
                      static_cast<f64>(n_bytes) * 1e-6, options.device, options.allocator);
    }

    void PairwiseShift::update(
        const View<f32>& stack,
        MetadataStack& metadata,
        const PairwiseShiftParameters& parameters
    ) {
        if (m_buffer_rfft.is_empty())
            return;

        auto timer = Logger::info_scope_time("Pairwise shift alignment");
        Logger::trace(
            "device={}\n"
            "cosine_stretching={}\n"
            "area_match={}\n"
            "smooth_edge={}%",
            m_xmap.device(),
            parameters.cosine_stretch,
            parameters.area_match,
            parameters.smooth_edge_percent * 100
        );

        // We'll need the slices sorted by tilt angles, with the lowest absolute tilt being the pivot point.
        metadata.sort("tilt");
        const i64 index_lowest_tilt = metadata.find_lowest_tilt_index();
        const i64 slice_count = metadata.ssize();

        //
        Vec<f64, 2> max_shifts{};
        Vec<f64, 2> first_average_shift{};
        Vec<f64, 2> last_average_shift{};
        const bool converge = parameters.update_count < 0;
        const i32 count = converge ? 125 : parameters.update_count;
        i32 i{};
        while (i < count) {
            // The metadata won't be updated in the loop, we can compute the common area once here.
            if (parameters.area_match)
                m_common_area.set_geometry(stack.shape().filter(2, 3), metadata);

            // The main processing loop. From the lowest to the highest tilt, find the relative shifts.
            // These shifts are the slice-to-slice shifts, i.e. the shift to apply to the target to align
            // it onto its neighbor reference.
            std::vector<Vec<f64, 2>> slice_to_slice_shifts;
            slice_to_slice_shifts.reserve(static_cast<size_t>(slice_count));
            for (i64 idx_target{}; idx_target < slice_count; ++idx_target) {
                if (index_lowest_tilt == idx_target) {
                    // Everything is relative to this reference view, so of course its shifts are 0.
                    // We call it the global reference view.
                    slice_to_slice_shifts.push_back({});
                    continue;
                }

                // If ith target has:
                //  - negative tilt angle, then reference is at i + 1.
                //  - positive tilt angle, then reference is at i - 1.
                const bool is_negative = idx_target < index_lowest_tilt;
                const i64 idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;

                // Compute the shifts.
                const Vec<f64, 2> slice_to_slice_shift = find_relative_shifts_(
                    stack, metadata[idx_reference], metadata[idx_target], parameters);
                slice_to_slice_shifts.push_back(slice_to_slice_shift);
            }

            // This is the main drawback of this method. We need to compute the global shifts from the relative
            // shifts, so the high tilt slices end up accumulating the errors of the lower tilts.
            const auto [global_shifts, i_average_shift] = relative2global_shifts_(
                slice_to_slice_shifts, metadata, index_lowest_tilt);

            // Logging.
            if (i == 0)
                first_average_shift = i_average_shift;
            last_average_shift = i_average_shift;

            // Update the metadata.
            max_shifts = 0;
            for (auto&& [slice, global_shift]: noa::zip(metadata, global_shifts)) {
                slice.shifts += global_shift;
                max_shifts = max(max_shifts, abs(global_shift));
            }

            // Loop logic.
            ++i;
            if (converge and noa::sqrt(noa::dot(i_average_shift, i_average_shift)) <= 0.001)
                break;
        }

        Logger::info(
            "first_average_shift={::.3f}, last_average_shift={::.3f}, max_shift={::.3f}, n_iter={}",
            first_average_shift, last_average_shift, max_shifts, i
        );
        save_plot_shifts(metadata, parameters.output_directory / "coarse_shifts.txt", {.title = "Coarse Shifts"});
    }

    auto PairwiseShift::find_relative_shifts_(
        const View<f32>& stack,
        const MetadataSlice& reference_slice,
        const MetadataSlice& target_slice,
        const PairwiseShiftParameters& parameters
    ) const -> Vec2<f64> {
        // Compute the affine matrix to transform the target "onto" the reference.
        const Vec3<f64> target_angles = noa::deg2rad(target_slice.angles);
        const Vec3<f64> reference_angles = noa::deg2rad(reference_slice.angles);
        const Vec2<f64> slice_center = (m_xmap.shape().vec.filter(2, 3) / 2).as<f64>();

        // First, compute the cosine stretching to estimate the tilt (and technically pitch) difference.
        // These angles are flipped, since the cos-scaling is perpendicular to its axis of rotation.
        // For the tilt, since the tilt-axis is along Y, the corresponding stretching is along X.
        Vec2<f64> cos_factor{1, 1};
        if (parameters.cosine_stretch)
            cos_factor = noa::cos(reference_angles.filter(2, 1)) / noa::cos(target_angles.filter(2, 1));

        // Cancel the difference (if any) in rotation and shift as well.
        const Mat33<f64> target2reference =
            ng::translate(slice_center + reference_slice.shifts) *
            ng::rotate<true>(reference_angles[0]) *
            ng::scale<true>(cos_factor) *
            ng::rotate<true>(-target_angles[0]) *
            ng::translate(-slice_center - target_slice.shifts);
        const auto fwd_target2reference = target2reference.as<f32>();
        const auto inv_target2reference = target2reference.inverse().as<f32>();

        // Get the views from the buffer.
        const auto buffer_shape = m_xmap.shape().set<0>(3);
        const auto buffer_rfft = m_buffer_rfft.view();
        const auto buffer = noa::fft::alias_to_real(buffer_rfft, buffer_shape);

        const auto transformed_target = buffer.subregion(0);
        const auto reference = buffer.subregion(1);
        const auto target = buffer.subregion(2);

        // Real-space masks to guide the alignment and not compare things that cannot or shouldn't be compared.
        // This is relevant for large shifts between images and high-tilt angles, but also to restrict the
        // alignment to a specific region, i.e., the center of the image.
        const auto indices = std::array{reference_slice.index, target_slice.index};
        if (parameters.area_match) {
            // Copy if the stack isn't on the compute-device.
            View<f32> input_reference = stack.subregion(indices[0]);
            View<f32> input_target = stack.subregion(indices[1]);
            if (stack.device() != buffer.device()) {
                input_reference = input_reference.to(reference);
                input_target = input_target.to(target);
            }

            // Enforce a common area across the tilt series.
            // This is more restrictive and removes regions from the higher tilts that aren't in the 0deg view.
            // This is quite good to remove the regions in the higher tilts that varies a lot from one tilt to the
            // next, and where cosine stretching isn't a good approximation of what is happening in 3d space.
            m_common_area.mask(input_reference, reference, reference_slice, false, parameters.smooth_edge_percent);
            m_common_area.mask(input_target, target, target_slice, false, parameters.smooth_edge_percent);
        } else {
            // The area match can be very restrictive in the high tilts. When the shifts are not known and
            // large shifts are present, it is best to turn off the area match and enforce a common FOV only between
            // the two images that are being compared.
            const auto reference_and_target = buffer.subregion(ni::Slice{1, 3});
            const auto hw = reference_and_target.shape().pop_front<2>();
            const auto smooth_edge_size = static_cast<f64>(noa::max(hw)) * parameters.smooth_edge_percent;

            noa::copy_batches(stack, reference_and_target, View(indices.data(), 2));

            const auto mask = ng::Rectangle{
                .center = slice_center,
                .radius = slice_center - smooth_edge_size,
                .smoothness = smooth_edge_size,
            }.draw<f32>();
            ng::draw(reference_and_target, reference_and_target, mask, fwd_target2reference);
            ng::draw(reference_and_target, reference_and_target, mask, inv_target2reference);
        }

        // After this point, the target is stretched and should "overlap" with the reference.
        ng::transform_2d(target, transformed_target, inv_target2reference, {.interp = parameters.interp});

        // Get the views from the buffers.
        const auto transformed_target_and_reference_rfft = buffer_rfft.subregion(ni::Slice{0, 2});
        const auto transformed_target_and_reference = buffer.subregion(ni::Slice{0, 2});
        const auto xmap = m_xmap.view();

        if (Logger::is_debug()) {
            const auto target_reference_filename =
                parameters.output_directory / fmt::format("transformed_target_and_reference_{:0>2}.mrc", indices[1]);
            noa::write(transformed_target_and_reference, target_reference_filename, {.dtype = noa::io::Encoding::F16});
            Logger::debug("{} saved", target_reference_filename);
        }

        // (Conventional) cross-correlation.
        // Technically, we should zero-pad here to cancel the circular property of the DFT (only the zero-lag is
        // unaffected by it). However, while we do expect significant lags, since we don't care about the actual peak
        // value, as long as we can find the highest peak, it is fine.
        noa::fft::r2c(transformed_target_and_reference, transformed_target_and_reference_rfft);
        ns::bandpass<"h2h">(
            transformed_target_and_reference_rfft,
            transformed_target_and_reference_rfft,
            transformed_target_and_reference.shape(),
            parameters.bandpass
        );
        ns::cross_correlation_map<"h2fc">(
            transformed_target_and_reference_rfft.subregion(0),
            transformed_target_and_reference_rfft.subregion(1),
            xmap, {.mode = ns::Correlation::CONVENTIONAL}
        );

        if (Logger::is_debug()) {
            auto xmap_filename = parameters.output_directory / fmt::format("xmap_{:0>2}.mrc", indices[1]);
            noa::write(xmap, xmap_filename, {.dtype = noa::io::Encoding::F16});
            Logger::debug("{} saved", xmap_filename);

            xmap_filename = parameters.output_directory / fmt::format("xmap_centered_{:0>2}.mrc", indices[1]);
            noa::write(m_xmap_centered.view(), xmap_filename, {.dtype = noa::io::Encoding::F16});
            Logger::debug("{} saved", xmap_filename);
        }

        // Find the best peak and compute the shift of the transformed target.
        const auto shift_transformed_target = find_shift<"fc2fc">(xmap, m_xmap_centered.view(), {
            .distortion_angle_deg = reference_slice.angles[0],
            .max_shift_percent = parameters.max_shift_percent,
        });

        // Enforce a maximum-lag by masking the cross-correlation map and argmax.
        // Due to the difference in tilt, the cross-correlation map can be distorted orthogonal to the tilt-axis.
        // To improve the accuracy of the subpixel registration, correct the tilt-axis to have the distortion along x.
        // Since the actual peak is close to argmax, focus on (and only render) a small subregion around argmax.

        // Get the peak and rotate back to the original xmap reference-frame.
        // The resulting shift is by how much the transformed target is away from the reference,
        // so to align it onto the reference, we would beed to subtract this shift from it.



        // We could get the shift of the actual target (rather than the transformed one, which is stretched).
        // However, we'll need to compute the global shifts later on, and we'll need to center the shifts.
        // These operations require accumulating the shifts of the lower views up to the global reference.
        // To do so, the simplest is to scale all these slice-to-slice shifts that we just computed to the same
        // reference frame, process everything there, and then go back to each image's reference-frame at the end.
        // Here, this global reference frame is the volume space, with no rotation, no tilt, no pitch.

        // Note that if the cosine-stretching was turned off, we need to stretch from the target angles to 0,
        // not from the reference. However, the rotation is always from the reference to 0.
        const auto current_angles = parameters.cosine_stretch ? reference_angles : target_angles;
        const auto reference2volume =
            ng::scale(/*cos(0)=*/ 1 / noa::cos(current_angles.filter(2, 1))) *
            ng::rotate(-reference_angles[0]);
        return reference2volume * shift_transformed_target;
    }
}
