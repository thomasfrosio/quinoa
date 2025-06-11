#include <noa/IO.hpp>
#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Stack.hpp"
#include "quinoa/PairwiseShift.hpp"

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
                slice_to_slice_shifts, metadata, index_lowest_tilt, parameters.cosine_stretch);

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
        // These angles are flipped, since the cos-scaling is perpendicular to the axis of rotation.
        // So since the tilt axis is along Y, its stretching is along X.
        Vec2<f64> cos_factor{1, 1};
        if (parameters.cosine_stretch)
            cos_factor = noa::cos(reference_angles.filter(2, 1)) / noa::cos(target_angles.filter(2, 1));

        // Cancel the difference (if any) in rotation and shift as well.
        const Mat33<f64> fwd_stretch_target_to_reference_d =
            ng::translate(slice_center + reference_slice.shifts) *
            ng::linear2affine(ng::rotate(reference_angles[0])) *
            ng::linear2affine(ng::scale(cos_factor)) *
            ng::linear2affine(ng::rotate(-target_angles[0])) *
            ng::translate(-slice_center - target_slice.shifts);
        const auto fwd_stretch_target_to_reference = fwd_stretch_target_to_reference_d.as<f32>();
        const auto inv_stretch_target_to_reference = fwd_stretch_target_to_reference_d.inverse().as<f32>();

        // Get the views from the buffer.
        const auto buffer_shape = m_xmap.shape().set<0>(3);
        const auto buffer_rfft = m_buffer_rfft.view();
        const auto buffer = noa::fft::alias_to_real(buffer_rfft, buffer_shape);

        const auto target_stretched = buffer.subregion(0);
        const auto reference = buffer.subregion(1);
        const auto target = buffer.subregion(2);

        // Real-space masks to guide the alignment and not compare things that cannot or shouldn't be compared.
        // This is relevant for large shifts between images and high-tilt angles, but also to restrict the
        // alignment to a specific region, i.e., the center of the image.
        const auto indices = std::array{reference_slice.index, target_slice.index};
        if (parameters.area_match) {
            // Copy if stack isn't on the compute-device.
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
            };
            ng::draw_shape(reference_and_target, reference_and_target, mask, fwd_stretch_target_to_reference);
            ng::draw_shape(reference_and_target, reference_and_target, mask, inv_stretch_target_to_reference);
        }

        // After this point, the target is stretched and should "overlap" with the reference.
        ng::transform_2d(
            target, target_stretched,
            inv_stretch_target_to_reference,
            {.interp = parameters.interp}
        );

        // Get the views from the buffers.
        const auto target_stretched_and_reference_rfft = buffer_rfft.subregion(ni::Slice{0, 2});
        const auto target_stretched_and_reference = buffer.subregion(ni::Slice{0, 2});
        const auto xmap = m_xmap.view();

        if (Logger::is_debug()) {
            const auto target_reference_filename =
                parameters.output_directory / fmt::format("target_stretched_and_reference_{:0>2}.mrc", indices[0]);
            noa::write(target_stretched_and_reference, target_reference_filename);
            Logger::debug("{} saved", target_reference_filename);
        }

        // (Conventional) cross-correlation. There's no need to normalize here, we just need the shift.
        // TODO Technically we should zero-pad here to cancel the circular property of the DFT (only the zero lag
        //      is unaffected by it). However, while we do expect significant lags, we run this multiple times and
        //      do not care about the peak value, so zero padding shouldn't affect the final result.
        noa::fft::r2c(target_stretched_and_reference, target_stretched_and_reference_rfft);
        ns::bandpass<"h2h">(
            target_stretched_and_reference_rfft,
            target_stretched_and_reference_rfft,
            target_stretched_and_reference.shape(),
            parameters.bandpass
        ); // TODO Use ns::filter_spectrum to apply sampling functions (exposure, mtf) as well?
        ns::cross_correlation_map<"h2f">(
            target_stretched_and_reference_rfft.subregion(0),
            target_stretched_and_reference_rfft.subregion(1),
            xmap, {.mode = ns::Correlation::CONVENTIONAL}
        );

        if (not parameters.debug_directory.empty()) {
            const auto xmap_filename = parameters.debug_directory / fmt::format("xmap_{:0>2}.mrc", indices[0]);
            noa::fft::remap(noa::Remap::F2FC, xmap, target, xmap.shape());
            noa::write(target, xmap_filename);
            Logger::debug("{} saved", xmap_filename);
        }

        // Computes the YX shift of the target. To shift-align the stretched target onto the reference,
        // we would then need to subtract this shift to the stretched target.
        const auto [peak_coordinate, peak_value] = ns::cross_correlation_peak_2d<"f2f">(xmap, {
            .maximum_lag =
                (parameters.max_shift_percent >= 0.99 ? -1 : parameters.max_shift_percent) *
                xmap.shape().vec.filter(2, 3).as<f64>()
        });

        const auto shift_reference = (peak_coordinate - slice_center).as<f64>();
        if (not parameters.cosine_stretch)
            return shift_reference;

        // We could destretch the shift to bring it back to the original target. However, we'll need
        // to compute the global shifts later on, as well as centering the shifts. This implies
        // to accumulate the shifts of the lower views while accounting for their own scaling.
        // Instead, it is simpler to scale all the slice-to-slice shifts to the same reference frame,
        // process everything there, and then go back to whatever higher tilt reference frame at
        // the end. Here, this global reference frame is the planar specimen (no tilt, no pitch).
        const auto fwd_stretch_reference_to_0deg = Mat22<f64>{
            ng::rotate(reference_angles[0]) *
            ng::scale(1 / noa::cos(reference_angles.filter(2, 1))) * // 1 = cos(0deg)
            ng::rotate(-reference_angles[0])
        };
        const auto shift_0deg = fwd_stretch_reference_to_0deg * shift_reference;
        return shift_0deg;
    }

    auto PairwiseShift::relative2global_shifts_(
        const std::vector<Vec<f64, 2>>& relative_shifts,
        const MetadataStack& metadata,
        i64 index_lowest_tilt,
        bool cosine_stretch
    ) -> Pair<std::vector<Vec<f64, 2>>, Vec<f64, 2>> {
        // Compute the global shifts and the mean.
        const size_t count = relative_shifts.size();
        Vec<f64, 2> mean{};
        auto scan_op = [mean_scale = 1 / static_cast<f64>(count), &mean](auto& sum, auto& current) {
            const auto out = sum + current;
            mean += out * mean_scale;
            return out;
        };

        // Inclusive scan.
        // FIXME
        auto global_shifts = std::vector<Vec<f64, 2>>(count);
        auto sum = Vec<f64, 2>{};
        for (i64 i = index_lowest_tilt + 1; i < static_cast<i64>(count); ++i)
            global_shifts.data()[i] = sum = scan_op(sum, relative_shifts.data()[i]);
        sum = 0;
        for (i64 i = index_lowest_tilt - 1; i >= 0; --i)
            global_shifts.data()[i] = sum = scan_op(sum, relative_shifts.data()[i]);

        // Center the shifts.
        for (auto& shift: global_shifts)
            shift -= mean;

        if (not cosine_stretch)
            return {global_shifts, mean};

        // Center the global shifts and scale them back to the original reference frame of their respective slice,
        // i.e. shrink the shifts to account for the slice's tilt and pitch.
        for (size_t i{}; i < count; ++i) {
            const auto angles = noa::deg2rad(metadata[i].angles);
            const auto fwd_shrink_matrix = Mat22<f64>{
                ng::rotate(angles[0]) *
                ng::scale(noa::cos(angles.filter(2, 1))) *
                ng::rotate(-angles[0])
            };
            global_shifts[i] = fwd_shrink_matrix * global_shifts[i];
        }
        return {global_shifts, mean};
    }
}
