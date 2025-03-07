#include <noa/Array.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>
#include <noa/FFT.hpp>

#include "quinoa/Thickness.hpp"
#include "quinoa/Optimizer.hpp"

namespace {
    using namespace qn;

    struct SIRTWeight {
        using value_type = f32;
        f32 fake_iter;

        // With a large enough level (>1000), this is equivalent to a radial weighting.
        constexpr explicit SIRTWeight(f32 level) {
            fake_iter = level <= 15.f ? level :
                        level <= 30.f ? 15.f + 0.4f * (level - 15.f) :
                                        27.f * 0.6f * (level - 30.f);
        }

        NOA_HD auto operator()(const Vec<f32, 2>& fftfreq_2d, i32) const {
            return noa::sqrt(dot(fftfreq_2d, fftfreq_2d)) * 2;

            // const f32 fftfreq = noa::sqrt(dot(fftfreq_2d, fftfreq_2d));
            // if (fftfreq < 1e-6f or fftfreq > 0.5f)
            //     return 0.f;
            // const f32 max = 0.5f * (1 - noa::pow(1 - 0.00195f / 0.5f, fake_iter));
            // const f32 current = fftfreq * (1 - noa::pow(1 - 0.00195f / fftfreq, fake_iter));
            // return current / max;
        }
    };

    struct ReduceAbsSum {
        using enable_vectorization = bool;

        static constexpr void init(const f32& value, f32& sum) {
            sum += noa::abs(value);
        }
        static constexpr void join(const f32& ireduced, f32& reduced) {
            reduced += ireduced;
        }
    };

    auto compute_profile_(
        StackLoader& stack_loader,
        MetadataStack& metadata,
        const Path& debug_directory
    ) -> Array<f32> {
        // Zero pad.
        const auto slice_shape = stack_loader.slice_shape();
        const auto slice_center = (slice_shape / 2).vec.as<f64>();
        const auto padded_size = noa::fft::next_fast_size(noa::max(slice_shape) * 2);
        const auto slice_padded_shape = Shape4<i64>{1, 1, padded_size, padded_size};
        const auto stack_padded_shape = slice_padded_shape.set<0>(metadata.ssize());
        const auto border_right = (Vec2<i64>{padded_size, padded_size} - slice_shape.vec).push_front<2>(0);
        const auto options = ArrayOption(stack_loader.compute_device(), stack_loader.allocator());

        // Compute one central-slice at a time to save memory.
        auto slices = noa::Array<f32>({2, 1, slice_shape[0], slice_shape[1]}, options);
        auto slice_0 = slices.view().subregion(0);
        auto slice_1 = slices.view().subregion(1);

        auto slice_padded = noa::empty<f32>(slice_padded_shape, options);
        auto stack_padded_rfft = noa::empty<c32>(stack_padded_shape.rfft(), options);

        for (i64 i{}; const auto& slice: metadata) {
            stack_loader.read_slice(slice_0, slice.index_file);
            noa::resize(slice_0, slice_padded.view(), {}, border_right);
            noa::fft::r2c(slice_padded.view(), stack_padded_rfft.subregion(i++));
        }

        // Load the geometry.
        auto shifts = noa::Array<Vec<f32, 2>>(metadata.ssize());
        auto rotations = noa::Array<ng::Quaternion<f32>>(metadata.ssize());
        for (auto&& [shift, rotation, slice]: noa::zip(shifts.span_1d(), rotations.span_1d(), metadata)) {
            shift = -(slice_center + slice.shifts).as<f32>();
            rotation = ng::matrix2quaternion((
                ng::rotate_x(noa::deg2rad(slice.angles[2])) *
                ng::rotate_y(noa::deg2rad(slice.angles[1])) *
                ng::rotate_z(noa::deg2rad(-slice.angles[0]))
            ).inverse()).as<f32>();
        }
        if (not stack_loader.compute_device().is_cpu()) {
            shifts = std::move(shifts).to(options);
            rotations = std::move(rotations).to(options);
        }

        // Prepare the central-slices for the Fourier insertion.
        ns::phase_shift_2d<"h2h">(stack_padded_rfft, stack_padded_rfft, stack_padded_shape, shifts);

        // Sample the Fourier slice.
        auto slice_padded_rfft = noa::Array<c32>(slice_padded_shape.rfft(), options);
        ng::fourier_insert_interpolate_and_extract_3d<"h2h">(
            std::move(stack_padded_rfft), {}, stack_padded_shape,
            slice_padded_rfft, {}, slice_padded_shape,
            {}, std::move(rotations),
            {}, ng::rotate_x(noa::deg2rad(90.)).as<f32>(),
            {
                .interp = noa::Interp::LINEAR,
                .input_windowed_sinc = {-1, 0.003},
                .add_to_output = false,
                .correct_weights = false, // TODO test with and without correction
            }
        );

        // Reconstruct the projection.
        ns::phase_shift_2d<"h2h">(slice_padded_rfft, slice_padded_rfft, slice_padded_shape, slice_center);
        // ns::filter_spectrum_2d<"h2h">(slice_padded_rfft, slice_padded_rfft, slice_padded_shape, SIRTWeight{35});
        noa::fft::c2r(std::move(slice_padded_rfft), slice_padded);
        noa::resize(std::move(slice_padded), slice_0, {}, -border_right);

        if (not debug_directory.empty()) {
            auto variances = noa::variance(slice_0, ReduceAxes{.width = true}).to_cpu().flat();
            noa::normalize(variances, variances, {.mode = noa::Norm::L2});

            noa::write(slice_0, debug_directory / "projected_slice_rotx_90.mrc");
            save_vector_to_text(variances.view(), debug_directory / "variances.txt");
        }

        // Median filter and gaussian blur.
        // This helps a lot removing background noise that is picked up when computing the variance.
        ns::median_filter_2d(slice_0, slice_1, {.window_size = 11});
        const auto gaussian = ns::window_gaussian<f32>(7, 3, {.normalize = true}).to(options);
        ns::convolve_separable(slice_1, slice_0, {}, gaussian, {});
        ns::convolve_separable(slice_0, slice_1, {}, {}, gaussian);

        // The convolution is not great with the edges, and it messes up the gradient.
        // As such, remove a few pixels at the edges before computing the variances.
        // Note: the center should be preserved so we can later shift the center.
        slice_1 = slice_1.subregion(ni::Ellipsis{}, ni::Slice{10, -10}, ni::Slice{10, -10});

        // Variance of every row.
        auto variances = noa::variance(slice_1, ReduceAxes{.width = true}).to_cpu().flat();
        noa::normalize(variances, variances, {.mode = noa::Norm::L2});

        if (not debug_directory.empty()) {
            noa::write(slice_1, debug_directory / "projected_slice_rotx_90.mrc");
            save_vector_to_text(variances.view(), debug_directory / "variances.txt");
        }

        noa::Session::clear_fft_cache(options.device);
        return variances;
    }

    auto compute_abs_gradient(
        const Span<f32>& array,
        f64 mask_diameter,
        const Path& debug_directory
    ) -> Array<f32> {
        auto gradient = noa::Array<f32>(array.ssize());
        const auto span = gradient.span_1d_contiguous();
        const i64 last = span.ssize() - 1;

        span[0] = std::abs(array[1] - array[0]); // forward difference
        for (i64 i = 1; i < last; ++i)
            span[i] = std::abs(array[i - 1] - array[i + 1]) / 2; // central difference
        span[last] = std::abs(array[last] - array[last - 1]); // backward difference

        if (not debug_directory.empty())
            save_vector_to_text(gradient.view(), debug_directory / "gradient.txt");

        noa::normalize(gradient, gradient); // FIXME min to 0?
        const auto min = noa::min(gradient);
        for (auto& e: span)
            e -= min;
        if (not debug_directory.empty())
            save_vector_to_text(gradient.view(), debug_directory / "gradient_normalized.txt");

        // Apply the Tuckey-like mask to remove everything outside the mask.
        // const auto center = MetadataSlice::center<f64>(span.ssize());
        // const auto radius = std::min(mask_diameter / 2, static_cast<f64>(span.size()) / 2);
        // const auto edge = radius * 0.05;
        // auto line = ng::guts::DrawLine<f32, false, f64>(center, radius, edge); // TODO Add window_tuckey()
        // for (i64 i = 0; i < span.ssize(); ++i)
        //     span[i] *= line(static_cast<f64>(i));

//        noa::math::normalize(gradient, gradient);
        if (not debug_directory.empty())
            save_vector_to_text(gradient.view(), debug_directory / "gradient_masked.txt");

        return gradient;
    }

//     auto adjust_to_center_of_mass(const Span<f32>& array, f64 spacing) -> std::pair<Span<f32>, f64> {
//         const auto size = array.ssize();
//         const auto size_f = static_cast<f64>(size);
//
//         // Compute the center-of-mass (supporting negative values).
//         const auto min = static_cast<f64>(*stdr::min_element(array));
//         f64 com{}, sum{};
//         for (i64 i = 0; i < size; ++i) {
//             const auto value = static_cast<f64>(array[i]) - min;
//             com += value * static_cast<f64>(i);
//             sum += value;
//         }
//         com /= sum;
//
//         // The shift to apply. This will ultimately be applied to the metadata.
//         const auto current_center = MetadataSlice::center<f64>(size);
//         auto shift = com - current_center;
//         Logger::trace("center={}, com={}, shift={}", current_center, com, shift);
//         if (std::abs(shift) > size_f * 0.25) {
//             panic("The estimated center-of-mass ({:.2f}nm) is too far away from the current center ({:.2f}nm). "
//                   "This cannot be a good sign and may break the next steps, possibly silently, so stop now",
//                   com * spacing, current_center * spacing);
//         }
//         if (std::abs(shift) > size_f * 0.125) {
//             Logger::warn(
//                 "The estimated center-of-mass ({:.2f}nm) is far from the current center ({:.2f}nm). "
//                 "The center will still be updated (shift={:.3f}nm) and the alignment will continue, "
//                 "but this may not a good sign!",
//                 com * spacing, current_center * spacing, shift * spacing);
//         }
//
//         // Truncate the window to have the c-o-m fairly close to the window center.
//         // This will not affect the metadata, it's just here to (hopefully) improve the thickness estimate.
//         i64 offset = static_cast<i64>(std::round(shift));
//         const auto left_offset = offset > 0 ? offset * 2 : 0;
//         const auto right_offset = offset < 0 ? offset * 2 : 0;
//         auto new_span = Span<f32>(array.data() + left_offset, array.ssize() - left_offset - right_offset);
//
//         return {new_span, shift};
//     }
//
//     auto find_window(const Span<f32>& array, f64 spacing_nm, const Path& debug_directory) -> std::pair<f64, f64> {
//         std::vector<i32> peaks_x;
//         std::vector<f32> peaks_y;
//         peaks_x.reserve(array.size() / 4);
//         peaks_y.reserve(peaks_x.capacity());
//
//         // 1. Find all peaks (x, y).
//         for (i64 i = 1; i < array.ssize() - 1; ++i) {
//             const f32 n0 = array[i - 1];
//             const f32 n1 = array[i];
//             const f32 n2 = array[i + 1];
//             const f32 slope_0 = n1 - n0;
//             const f32 slope_1 = n2 - n1;
//
//             if (slope_0 >= 0 && slope_1 < 0) { // peak: positive/zero slope to negative slope.
//                 peaks_x.emplace_back(static_cast<i32>(i));
//                 peaks_y.emplace_back(n1);
//             }
//         }
//
//         const auto span_x = Span<const i32>(peaks_x.data(), peaks_x.size());
//         const auto span_y = Span<const f32>(peaks_y.data(), peaks_y.size());
//         const i64 n_peaks = span_y.ssize();
//         qn::Logger::trace("n_peaks={}", n_peaks);
//         QN_CHECK(n_peaks > 3, "Something went wrong... A lot more peaks are expected...");
//
//         // 2. Select the first and last peaks.
//         // TODO We could limit this search to +- 100nm around the center and let the dilation do the rest.
//         const auto initial_peaks = [&]() {
//             constexpr f32 HIGH_SIGMA = 1.25;
//             std::pair<i64, i64> output{0, 0};
//             i64 total{0};
//             for (i64 i = 0; i < n_peaks; ++i) {
//                 if (span_y[i] < HIGH_SIGMA)
//                     continue;
//                 if (output.first == 0)
//                     output.first = i;
//                 output.second = i;
//                 total += 1;
//             }
//             qn::Logger::trace("n_selected_peaks={}, threshold={}, thickness={:.2f}nm",
//                               total, HIGH_SIGMA, (span_x[output.second] - span_x[output.first]) * spacing_nm);
//             return output;
//         }();
//
//         // 3. Connectivity-based extension, ie dilation and select above a lower threshold.
//         //    Try to expand the window to neighboring peaks.
//         const auto [first, last] = [&]() {
//             constexpr f32 MEDIUM_SIGMA = 0.5;
//             constexpr i64 EXTEND_UP_TO = 10;
//             // FIXME Make sure these are not too far away (check consecutive peaks below threshold?).
//             //       Here this is intended to stay very close to the big peaks but simply extend at the edges
//             //       for significant (but weaker) signal in direct contact with the peaks.
//             std::pair<i64, i64> output = initial_peaks;
//
//             i64 left_extended{0};
//             i64 right_extended{0};
//             for (i64 i = 1; i < EXTEND_UP_TO + 1; ++i) {
//                 const auto previous = initial_peaks.first - i;
//                 if (previous >= 0 and span_y[previous] >= MEDIUM_SIGMA) {
//                     output.first = previous;
//                     left_extended += 1;
//                 }
//                 const auto next = initial_peaks.second + i;
//                 if (next < n_peaks and span_y[next] >= MEDIUM_SIGMA) {
//                     output.second = next;
//                     right_extended += 1;
//                 }
//             }
//             qn::Logger::trace("extended_left={}, extended_right={}, threshold={}, thickness={:.2f}nm",
//                               left_extended, right_extended, MEDIUM_SIGMA,
//                               (span_x[output.second] - span_x[output.first]) * spacing_nm);
//             return output;
//         }();
//
//         // 4. Go to the zeros of these peaks.
//         //    The cutoff can be useful in case the peak ends up going further than it should
//         //    because it is fused with other peaks in the background. So for simplicity, cut
//         //    everything below a given threshold.
//         auto find_zero = [&array](i64 start, i64 end, i64 increment) {
//             constexpr f32 CUTOFF = 0.2f;
//             for (i64 i = start; i != (end - increment); i += increment) {
//                 const f32 n0 = array[i - increment];
//                 const f32 n1 = array[i];
//                 const f32 n2 = array[i + increment];
//                 const f32 slope_0 = n1 - n0;
//                 const f32 slope_1 = n2 - n1;
//
//                 if (n1 < CUTOFF or (slope_0 < 0 and slope_1 >= 0))
//                     return i;
//             }
//             return start; // something is likely wrong
//         };
//         const i64 start = find_zero(span_x[first], span_x[std::max(i64{0}, first - 1)], -1);
//         const i64 end = find_zero(span_x[last], span_x[std::min(last + 1, n_peaks - 1)], +1);
//         qn::Logger::trace("thickness={:.2f}nm", static_cast<f64>(end - start) * spacing_nm);
//
//         // Make sure it's not too far off the center.
//         auto f_start = static_cast<f64>(start);
//         auto f_end = static_cast<f64>(end);
//         auto window_radius = (f_end - f_start) / 2;
//         auto window_center = f_start + window_radius;
//         auto expected_center = static_cast<f64>(array.size()) / 2;
//         if (window_center < expected_center * 0.75 or expected_center * 1.25 < window_center) {
//             qn::Logger::warn("The sample z-window search may have failed. "
//                              "The window center is off by more than 25% of the expected center...");
//         }
//
// //        // We assume the expected center is the true center. So enforce it by possibly enlarging the window.
// //        auto radius = std::max(expected_center - f_start, f_end - expected_center); // the window is now symmetric
//         auto thickness = window_radius * 2;
//         auto thickness_nm = thickness * spacing_nm;
//
//
// //        qn::Logger::trace("window_center={:.3f}, expected_center={:.3f}, thickness={:.2f}nm after symmetry",
// //                          window_center, expected_center, thickness_nm);
//         return {thickness_nm, window_center};
//     }
}

namespace qn {
    auto estimate_sample_thickness(
        const Path& stack_filename,
        MetadataStack& metadata,
        const EstimateSampleThicknessParameters& parameters
    ) -> f64 {
        auto timer = Logger::info_scope_time("Thickness estimation");

        auto stack_loader = StackLoader(stack_filename, {
            .compute_device = parameters.compute_device,
            .allocator = Allocator::MANAGED,
            .precise_cutoff = true,
            .rescale_target_resolution = parameters.resolution,
            .rescale_min_size = 512,
            .exposure_filter = false,
            .bandpass{
                .highpass_cutoff = 0.02,
                .highpass_width = 0.02,
                .lowpass_cutoff = 0.50,
                .lowpass_width = 0.01,
            },
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        auto rescaled_metadata = metadata;
        rescaled_metadata.rescale_shifts(stack_loader.file_spacing(), stack_loader.stack_spacing());

        const f64 average_spacing_nm = 1e-1 * noa::mean(stack_loader.stack_spacing());
        const auto max_thickness_pixels = static_cast<i64>(std::ceil(parameters.maximum_thickness_nm / average_spacing_nm));

        // 1. Compute the profile of the tomogram (forward projection at pitch=90deg).
        //    Compute the variance of the profile along the rows.
        //    Compute the gradient of this. This is what we use for the window fitting.
        //    The main advantages of the gradient are that it removes the background
        //    and still preserves the peak information.
        const Array variances = compute_profile_(stack_loader, rescaled_metadata, parameters.debug_directory);
        const Array gradient = compute_abs_gradient(variances.span_1d_contiguous(), max_thickness_pixels, parameters.debug_directory);

        // // 2. Adjust the z-center of the tomogram to the estimated center-of-mass.
        // //    This should be fairly robust, and in most cases, it's only a small shift.
        // auto adjusted_gradient = gradient.span_1d_contiguous();
        // if (parameters.adjust_com) {
        //     const auto [span, z_shift] = adjust_to_center_of_mass(adjusted_gradient, average_spacing_nm);
        //     metadata.add_global_shift({z_shift, 0, 0});
        //     adjusted_gradient = span;
        // }
        //
        // // 3. Window finder.
        // const auto [thickness_nm, new_center] = find_window(
        //     adjusted_gradient, average_spacing_nm, parameters.debug_directory);
        // if (parameters.adjust_com) {
        //     const auto current_center = MetadataSlice::center<f64>(adjusted_gradient.ssize());
        //     auto z_shift = new_center - current_center;
        //     metadata.add_global_shift({z_shift, 0, 0});
        //     Logger::trace("current_center={}, new_center={}, z_shift={}", current_center, new_center, z_shift);
        // }
        // if (parameters.initial_thickness_nm != std::numeric_limits<f64>::max()) {
        //     if (thickness_nm < parameters.initial_thickness_nm * 0.5 or
        //         thickness_nm > parameters.initial_thickness_nm * 1.5) {
        //         panic("Thickness estimate ({:.2f}nm) is too far off from the user-provided value ({:.2f}nm)",
        //               thickness_nm, parameters.initial_thickness_nm);
        //     } else if (thickness_nm < parameters.initial_thickness_nm * 0.75 or
        //                thickness_nm > parameters.initial_thickness_nm * 1.25) {
        //         Logger::warn("Thickness estimate ({:.2f}nm) is far from the user-provided value ({:.2f}nm)",
        //                      thickness_nm, parameters.initial_thickness_nm);
        //     }
        // }

        return 10.;
    }
}
