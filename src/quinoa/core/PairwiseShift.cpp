#include <noa/Memory.hpp>
#include <noa/IO.hpp>
#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/core/PairwiseShift.hpp"

namespace qn {
    PairwiseShift::PairwiseShift(
            const Shape4<i64>& shape,
            noa::Device compute_device,
            noa::Allocator allocator
    ) {
        // The "target" is the higher (absolute) tilt image that we want to align onto the "reference".
        // Performance-wise, using out-of-place FFTs could be slightly better, but here, prefer the safer
        // option to use less memory and in-place FFTs, since we are not constrained by performance at this step.
        const auto options = noa::ArrayOption(compute_device, allocator);
        m_buffer_rfft = noa::memory::empty<c32>({3, 1, shape[2], shape[3] / 2 + 1}, options);
        m_xmap = noa::memory::empty<f32>({1, 1, shape[2], shape[3]}, options);

        const auto bytes = m_xmap.size() * sizeof(f32) + m_buffer_rfft.size() * sizeof(c32);
        qn::Logger::trace("PairwiseShift(): allocated for {} MB on {} (allocator={})",
                          static_cast<f64>(bytes) * 10e-6, options.device(), options.allocator());
    }

    void PairwiseShift::update(
            const Array<f32>& stack,
            MetadataStack& metadata,
            const PairwiseShiftParameters& parameters,
            bool cosine_stretch,
            bool area_match,
            f64 smooth_edge_percent,
            f64 max_area_loss_percent,
            f64 max_shift_percent
    ) {
        if (m_buffer_rfft.is_empty())
            return;

        qn::Logger::info("Pairwise shift alignment...");
        qn::Logger::trace("Compute device: {}\n"
                          "Cosine stretching: {}\n"
                          "Area match: {}{}\n"
                          "Smooth edge: {}%",
                          m_xmap.device(), cosine_stretch, area_match,
                          area_match ? noa::string::format(" (max area loss: {}%)", max_area_loss_percent) : "",
                          smooth_edge_percent * 100);
        noa::Timer timer;
        timer.start();

        // We'll need the slices sorted by tilt angles, with the lowest absolute tilt being the pivot point.
        metadata.sort("tilt");
        const i64 index_lowest_tilt = metadata.find_lowest_tilt_index();
        const i64 slice_count = static_cast<i64>(metadata.size());

        // The metadata won't be updated in the loop, we can compute the common area once here.
        if (area_match)
            m_common_area.set_geometry(stack.shape().filter(2, 3), metadata, max_area_loss_percent);

        // The main processing loop. From the lowest to the highest tilt, find the relative shifts.
        // These shifts are the slice-to-slice shifts, i.e. the shift to apply to the target to align
        // it onto its neighbor reference.
        std::vector<Vec2<f64>> slice_to_slice_shifts;
        slice_to_slice_shifts.reserve(static_cast<size_t>(slice_count));
        for (i64 idx_target = 0; idx_target < slice_count; ++idx_target) {
            if (index_lowest_tilt == idx_target) {
                // Everything is relative to this reference view, so of course its shifts are 0.
                // We call it the global reference view.
                slice_to_slice_shifts.emplace_back(0);
                continue;
            }

            // If ith target has:
            //  - negative tilt angle, then reference is at i + 1.
            //  - positive tilt angle, then reference is at i - 1.
            const bool is_negative = idx_target < index_lowest_tilt;
            const i64 idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;

            // Compute the shifts.
            const Vec2<f64> slice_to_slice_shift = find_relative_shifts_(
                    stack, metadata[idx_reference], metadata[idx_target],
                    parameters, cosine_stretch, area_match,
                    smooth_edge_percent, max_shift_percent);
            slice_to_slice_shifts.emplace_back(slice_to_slice_shift);
        }

        // This is the main drawback of this method. We need to compute the global shifts from the relative
        // shifts, so the high tilt slices end up accumulating the errors of the lower tilts.
        const std::vector<Vec2<f64>> global_shifts =
                relative2global_shifts_(slice_to_slice_shifts, metadata, index_lowest_tilt);

        // Update the metadata.
        for (i64 i = 0; i < slice_count; ++i)
            metadata[i].shifts += global_shifts[static_cast<size_t>(i)];

        qn::Logger::info("Pairwise shift alignment... done. Took {:.2f}ms\n", timer.elapsed());
    }

    auto PairwiseShift::find_relative_shifts_(
            const Array<f32>& stack,
            const MetadataSlice& reference_slice,
            const MetadataSlice& target_slice,
            const PairwiseShiftParameters& parameters,
            bool cosine_stretch,
            bool area_match,
            f64 smooth_edge_percent,
            f64 max_shift_percent
    ) -> Vec2<f64> {
        // Compute the affine matrix to transform the target "onto" the reference.
        const Vec3<f64> target_angles = noa::math::deg2rad(target_slice.angles);
        const Vec3<f64> reference_angles = noa::math::deg2rad(reference_slice.angles);
        const Vec2<f32> slice_center = MetadataSlice::center(m_xmap.shape());

        // First, compute the cosine stretching to estimate the tilt (and technically elevation) difference.
        // These angles are flipped, since the cos-scaling is perpendicular to the axis of rotation.
        // So since the tilt axis is along Y, its stretching is along X.
        Vec2<f64> cos_factor{1};
        if (cosine_stretch)
            cos_factor = noa::math::cos(reference_angles.filter(2, 1)) / noa::math::cos(target_angles.filter(2, 1));

        // Cancel the difference (if any) in rotation and shift as well.
        const Double33 fwd_stretch_target_to_reference_d =
                noa::geometry::translate(slice_center.as<f64>() + reference_slice.shifts) *
                noa::geometry::linear2affine(noa::geometry::rotate(reference_angles[0])) *
                noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                noa::geometry::linear2affine(noa::geometry::rotate(-target_angles[0])) *
                noa::geometry::translate(-slice_center.as<f64>() - target_slice.shifts);
        const auto fwd_stretch_target_to_reference = fwd_stretch_target_to_reference_d.as<f32>();
        const auto inv_stretch_target_to_reference = fwd_stretch_target_to_reference_d.inverse().as<f32>();

        // Get the views from the buffer.
        const auto buffer_shape = m_xmap.shape().pop_front().push_front(3);
        const auto buffer_rfft = m_buffer_rfft.view();
        const auto buffer = noa::fft::alias_to_real(buffer_rfft, buffer_shape);

        const auto target_stretched = buffer.subregion(0);
        const auto reference = buffer.subregion(1);
        const auto target = buffer.subregion(2);

        // Real-space masks to guide the alignment and not compare things that cannot or shouldn't be compared.
        // This is relevant for large shifts between images and high-tilt angles, but also to restrict the
        // alignment to a specific region, i.e., the center of the image.
        if (area_match) {
            // Copy if stack isn't on the compute-device.
            View<f32> input_reference = stack.view().subregion(reference_slice.index);
            View<f32> input_target = stack.view().subregion(target_slice.index);
            if (stack.device() != buffer.device()) {
                input_reference.to(reference);
                input_target.to(target);
                input_reference = reference;
                input_target = target;
            }

            // Enforce a common area across the tilt series.
            // This is more restrictive and removes regions from the higher tilts that aren't in the 0deg view.
            // This is quite good to remove the regions in the higher tilts that varies a lot from one tilt to the
            // next, and where cosine stretching isn't a good approximation of what is happening in 3d space.
            m_common_area.mask_view(input_reference, reference, reference_slice, smooth_edge_percent);
            m_common_area.mask_view(input_target, target, target_slice, smooth_edge_percent);

        } else {
            // The area match can be very restrictive in the high tilts. When the shifts are not known and
            // large shifts are present, it is best to turn off the area match and enforce a common FOV only between
            // the two images that are being compared.
            const auto reference_and_target = buffer.subregion(noa::indexing::Slice{1, 3});
            const auto hw = reference_and_target.shape().pop_front<2>();
            const auto smooth_edge_size = static_cast<f32>(
                    static_cast<f64>(noa::math::max(hw)) *
                    smooth_edge_percent);

            std::array indexes{reference_slice.index, target_slice.index};
            noa::memory::copy_batches(stack.view(), reference_and_target, View<i32>(indexes.data(), 2));

            noa::geometry::rectangle(
                    reference_and_target, reference_and_target, slice_center,
                    slice_center - smooth_edge_size, smooth_edge_size,
                    fwd_stretch_target_to_reference);
            noa::geometry::rectangle(
                    reference_and_target, reference_and_target, slice_center,
                    slice_center - smooth_edge_size, smooth_edge_size,
                    inv_stretch_target_to_reference);
        }

        // After this point, the target is stretched and should "overlap" with the reference.
        noa::geometry::transform_2d(
                target, target_stretched,
                inv_stretch_target_to_reference,
                parameters.interpolation_mode);

        // Get the views from the buffers.
        const auto target_stretched_and_reference_rfft = buffer_rfft.subregion(noa::indexing::Slice{0, 2});
        const auto target_stretched_and_reference = buffer.subregion(noa::indexing::Slice{0, 2});
        const auto xmap = m_xmap.view();

        if (!parameters.debug_directory.empty()) {
            const auto output_index = reference_slice.index;
            const auto target_reference_filename = noa::string::format(
                    "target_stretched_and_reference_{:>02}.mrc", output_index);
            noa::io::save(target_stretched_and_reference,
                          parameters.debug_directory / target_reference_filename);
        }

        // (Conventional) cross-correlation. There's no need to normalize here, we just need the shift.
        noa::fft::r2c(target_stretched_and_reference, target_stretched_and_reference_rfft);
        noa::signal::fft::bandpass<noa::fft::H2H>(
                target_stretched_and_reference_rfft,
                target_stretched_and_reference_rfft,
                target_stretched_and_reference.shape(),
                parameters.highpass_filter[0], parameters.lowpass_filter[0],
                parameters.highpass_filter[1], parameters.lowpass_filter[1]);
        // TODO Apply sampling functions (exposure, mtf)?
        noa::signal::fft::xmap<noa::fft::H2F>(
                target_stretched_and_reference_rfft.subregion(0),
                target_stretched_and_reference_rfft.subregion(1),
                xmap, noa::signal::CorrelationMode::CONVENTIONAL);

        if (!parameters.debug_directory.empty()) {
            const auto output_index = reference_slice.index;
            const auto xmap_filename = noa::string::format("xmap_{:>02}.mrc", output_index);
            noa::fft::remap(noa::fft::F2FC, xmap, target, xmap.shape());
            noa::io::save(target, parameters.debug_directory / xmap_filename);
        }

        // Possibly restrict the shifts by masking the xmap.
        max_shift_percent = max_shift_percent >= 0.99 ? -1 : max_shift_percent; // if negative, it will be ignored
        const Vec2<f64> max_shift = max_shift_percent * xmap.shape().vec().filter(2, 3).as<f64>();

        // Computes the YX shift of the target. To shift-align the stretched target onto the reference,
        // we would then need to subtract this shift to the stretched target.
        const auto [peak_coordinate, peak_value] = noa::signal::fft::xpeak_2d<noa::fft::F2F>(
                xmap, max_shift.as<f32>(), noa::signal::PeakMode::PARABOLA_1D, Vec2<i64>{1});
        const auto shift_reference = (peak_coordinate - slice_center).as<f64>();
        qn::Logger::trace("{:>2} peak: pos={::> 8.3f}, value={:.6g}",
                          reference_slice.index, shift_reference, peak_value);

        // We could destretch the shift to bring it back to the original target. However, we'll need
        // to compute the global shifts later on, as well as centering the shifts. This implies
        // to accumulate the shifts of the lower views while accounting for their own scaling.
        // Instead, it is simpler to scale all the slice-to-slice shifts to the same reference
        // frame, process everything there, and then go back to whatever higher tilt reference at
        // the end. Here, this global reference frame is the planar specimen (no tilt, no elevation).
        const Vec2<f64> reference_to_no_elevation_no_tilt{reference_angles[2], reference_angles[1]};
        const Double22 fwd_stretch_reference_to_0deg{
                noa::geometry::rotate(reference_angles[0]) *
                noa::geometry::scale(1 / math::cos(reference_to_no_elevation_no_tilt)) * // 1 = cos(0deg)
                noa::geometry::rotate(-reference_angles[0])
        };
        const Vec2<f64> shift_0deg = fwd_stretch_reference_to_0deg * shift_reference;
        return shift_0deg;
    }

    auto PairwiseShift::relative2global_shifts_(
            const std::vector<Vec2<f64>>& relative_shifts,
            const MetadataStack& metadata,
            i64 index_lowest_tilt
    ) -> std::vector<Vec2<f64>> {
        // Compute the global shifts and the mean.
        const size_t count = relative_shifts.size();
        std::vector<Vec2<f64>> global_shifts(count);
        const auto index_pivot = index_lowest_tilt;
        const auto index_rpivot = static_cast<i64>(count) - 1 - index_pivot;
        const auto mean_scale = 1 / static_cast<f64>(count);
        Vec2<f64> mean{0};
        auto scan_op = [mean_scale, &mean](auto& current, auto& next) {
            const auto out = current + next;
            mean += out * mean_scale;
            return out;
        };
        std::inclusive_scan(relative_shifts.begin() + index_pivot,
                            relative_shifts.end(),
                            global_shifts.begin() + index_pivot,
                            scan_op);
        std::inclusive_scan(relative_shifts.rbegin() + index_rpivot,
                            relative_shifts.rend(),
                            global_shifts.rbegin() + index_rpivot,
                            scan_op);

        qn::Logger::info("Average shift: {::.3f}", mean);

        // Center the global shifts (optional) and scale them back to the original reference frame
        // of their respective slice, i.e. shrink the shifts to account for the slice's tilt and pitch.
        for (size_t i = 0; i < count; ++i) {
            const Vec3<f64> angles(noa::math::deg2rad(metadata[i].angles));
            const Double22 fwd_shrink_matrix{
                    noa::geometry::rotate(angles[0]) *
                    noa::geometry::scale(noa::math::cos(Vec2<f64>(angles[2], angles[1]))) *
                    noa::geometry::rotate(-angles[0])
            };
            const auto corrected_shift = fwd_shrink_matrix * (global_shifts[i] - mean);
            qn::Logger::trace("view={:>02}, global shift={::> 8.3f}, global corrected shift={::> 8.3f}",
                              i, global_shifts[i], corrected_shift);

            global_shifts[i] = corrected_shift;
        }

        return global_shifts;
    }
}
