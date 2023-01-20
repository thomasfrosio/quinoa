#pragma once

#include <numeric>

#include <noa/Session.h>
#include <noa/Array.h>
#include <noa/Memory.h>
#include <noa/Math.h>
#include <noa/Geometry.h>
#include <noa/FFT.h>
#include <noa/Signal.h>
#include <noa/Utils.h>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/io/Logging.h"

namespace qn {
    struct PairwiseCosineParameters {
        float2_t max_shift = {};
        float smooth_edge_percent = 0.5f;
        float2_t highpass_filter{0.03, 0.03};
        float2_t lowpass_filter{0.25, 0.05};

        bool center_tilt_axis = true;
        noa::InterpMode interpolation_mode = noa::InterpMode::INTERP_LINEAR_FAST;
        path_t debug_directory;
    };

    /// Shift alignment using cosine stretching on the higher tilt images.
    class PairwiseCosine {
    public:
        /// Allocates for the internal buffers.
        /// \param shape                (BD)HW shape of the slices. The BD dimensions are ignored.
        /// \param compute_device       Device where to perform the alignment.
        /// \param smooth_edge_percent  Size, in percent of the maximum-sized dimension, of the zero-edge taper.
        /// \param interpolation_mode   Interpolation mode used for the cosine-stretching.
        /// \param allocator            Allocator to use for \p compute_device.
        PairwiseCosine(dim4_t shape, noa::Device compute_device,
                       noa::InterpMode interpolation_mode,
                       noa::Allocator allocator = noa::Allocator::DEFAULT_ASYNC) {
            const auto options = noa::ArrayOption(compute_device, allocator);

            // The "target" is the higher (absolute) tilt image that we want to align onto the "reference".
            // Performance-wise, using out-of-place FFTs could be slightly better, but here, prefer the safer
            // option to use less memory and in-place FFTs.
            const auto buffer_shape = dim4_t{2, 1, shape[2], shape[3]};
            std::tie(m_target_reference, m_target_reference_fft) = noa::fft::empty<float>(buffer_shape, options);

            // For ease of use, separate into distinct objects. This way we can refer to the target
            // and reference individually, or use both at the same time for batched operations.
            m_target = m_target_reference.subregion(0);
            m_reference = m_target_reference.subregion(1);
            m_target_fft = m_target_reference_fft.subregion(0);
            m_reference_fft = m_target_reference_fft.subregion(1);

            // Since we need to transform the target, use a GPU texture since we'll need to copy
            // the target to a new array anyway. For CPU devices, this doesn't allocate anything,
            // and will directly use the input stack for the transformation.
            const auto slice_shape = dim4_t{1, 1, shape[2], shape[3]};
            m_target_texture = noa::Texture<float>(
                    slice_shape, compute_device, interpolation_mode, noa::BorderMode::BORDER_ZERO);

            // The cross-correlation map.
            m_xmap = noa::memory::empty<float>(slice_shape, options);
        }

        /// 2D in-place shifts alignment using cosine-stretching.
        /// \details Starts with the view at the lowest tilt angle. To align a neighbouring view
        ///          (which has a higher tilt angle), stretch that view by a factor of cos(x)
        ///          perpendicular to the tilt-axis, where x is the tilt angle of the neighbouring
        ///          view, in radians. Then use conventional correlation to find the XY shift between images.
        /// \param[in] stack        Input stack.
        /// \param[in,out] metadata Metadata of \p stack. This is updated with the new shifts.
        /// \param max_shift        Maximum YX shift a slice is allowed to have. If <= 0, it is ignored.
        /// \param center           Whether the average shift (in the microscope reference frame) should be centered.
        void updateShifts(const Array<float>& stack,
                          MetadataStack& metadata,
                          const PairwiseCosineParameters& parameters) {
            qn::Logger::info("Pairwise cosine-stretch shift alignment...");
            qn::Logger::trace("Compute device: {}", m_xmap.device());
            Timer timer;
            timer.start();

            const auto slice_center = MetadataSlice::center(m_xmap.shape());
            const auto smooth_edge_size =
                    static_cast<float>(std::max(m_xmap.shape()[2], m_xmap.shape()[3])) *
                    parameters.smooth_edge_percent;

            // We'll need the slices sorted by tilt angles, with the lowest absolute tilt being the pivot point.
            metadata.sort("tilt");
            const size_t index_lowest_tilt = metadata.lowestTilt();
            const size_t slice_count = metadata.size();

            // The main processing loop. From the lowest to the highest tilt, find the relative shifts.
            // These shifts are the slice-to-slice shifts, i.e. the shift to apply to the target to align
            // it onto its neighbour reference.
            std::vector<double2_t> slice_to_slice_shifts;
            slice_to_slice_shifts.reserve(slice_count);
            for (size_t idx_target = 0; idx_target < slice_count; ++idx_target) {
                if (index_lowest_tilt == idx_target) {
                    slice_to_slice_shifts.emplace_back(0);
                    continue;
                }

                // If ith target has:
                //  - negative tilt angle, then reference is at i + 1.
                //  - positive tilt angle, then reference is at i - 1.
                const bool is_negative = idx_target < index_lowest_tilt;
                const size_t idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;

                // Compute the shifts.
                slice_to_slice_shifts.emplace_back(
                        findRelativeShifts_(stack, metadata[idx_reference], metadata[idx_target],
                                            slice_center, smooth_edge_size, parameters));
            }

            // This is the main drawback of this method. We need to compute the global shifts from the relative
            // shifts, so the high tilt slices end up accumulating the errors of the lower tilts.
            const std::vector<double2_t> global_shifts =
                    relative2globalShifts_(slice_to_slice_shifts, metadata, index_lowest_tilt,
                                           parameters.center_tilt_axis);

            // Update the metadata.
            for (size_t i = 0; i < slice_count; ++i)
                metadata[i].shifts += static_cast<float2_t>(global_shifts[i]);

            qn::Logger::info("Pairwise cosine-stretch shift alignment... done. Took {:.2f}ms\n", timer.elapsed());
        }

    private:
        // Compute the conventional-correlation between the reference and target, accounting for the known
        // difference in rotation and shift between the two. The yaw and shift difference can be directly
        // corrected, but the tilt and pitch cannot (these are 3D transformations for 2D slices). To mitigate
        // this difference, a scaling is applied onto the target, usually resulting in a stretch perpendicular
        // to the tilt and pitch axes.
        // Importantly, the common field-of-view (FOV) is aligned, and edges are smooth out. This FOV is computed
        // using the common geometry, which can be far off. The better the current geometry is, the better the
        // estimation of the FOV.
        // As explained below, the output shift is the shift of the target relative to the reference in the
        // "microscope" reference frame (i.e. 0-degree tilt and pitch). This simplifies computation later
        // when the global shifts and centering needs to be computed.
        double2_t findRelativeShifts_(const Array<float>& stack,
                                      const MetadataSlice& reference_slice,
                                      const MetadataSlice& target_slice,
                                      float2_t slice_center,
                                      float smooth_edge_size,
                                      const PairwiseCosineParameters& parameters) {
            // Get the reference and target in their respective buffers.
            noa::memory::copy(stack.subregion(reference_slice.index), m_reference);
            m_target_texture.update(stack.subregion(target_slice.index));

            const float3_t target_angles = noa::math::deg2rad(target_slice.angles);
            const float3_t reference_angles = noa::math::deg2rad(reference_slice.angles);

            // Compute the affine matrix to transform the target "onto" the reference.
            // These angles are flipped, since the cos-scaling is perpendicular to the axis of rotation.
            const float2_t cos_factor{noa::math::cos(reference_angles[2]) / noa::math::cos(target_angles[2]),
                                      noa::math::cos(reference_angles[1]) / noa::math::cos(target_angles[1])};

            // Apply the scaling for the tilt and pitch difference,
            // and cancel the difference (if any) in yaw and shift as well.
            // After this point, the target should "overlap" with the reference.
            const float33_t fwd_stretch_target_to_reference =
                    noa::geometry::translate(slice_center + reference_slice.shifts) *
                    float33_t{noa::geometry::rotate(reference_angles[0])} *
                    float33_t{noa::geometry::scale(cos_factor)} *
                    float33_t{noa::geometry::rotate(-target_angles[0])} *
                    noa::geometry::translate(-slice_center - target_slice.shifts);
            const float33_t inv_stretch_target_to_reference = noa::math::inverse(fwd_stretch_target_to_reference);
            noa::geometry::transform2D(m_target_texture, m_target, inv_stretch_target_to_reference);

            // Enforce common field-of-view, hopefully to guide the alignment and not compare things that
            // cannot be compared. This is relevant for large shifts between images and high tilt angles,
            // but also to restrict the alignment to a region of the images.
            noa::signal::rectangle(
                    m_target_reference, m_target_reference, slice_center,
                    slice_center - smooth_edge_size, smooth_edge_size,
                    fwd_stretch_target_to_reference);
            noa::signal::rectangle(
                    m_target_reference, m_target_reference, slice_center,
                    slice_center - smooth_edge_size, smooth_edge_size,
                    inv_stretch_target_to_reference);

            if (!parameters.debug_directory.empty()) {
                const auto output_index = reference_slice.index;
                const auto target_reference_filename = noa::string::format("target_reference_{:>02}.mrc", output_index);
                noa::io::save(m_target_reference, parameters.debug_directory / target_reference_filename);
            }

            // (Conventional) cross-correlation:
            noa::fft::r2c(m_target_reference, m_target_reference_fft);
            noa::signal::fft::bandpass<noa::fft::H2H>(
                    m_target_fft, m_target_fft, m_xmap.shape(),
                    parameters.highpass_filter[0], parameters.lowpass_filter[0],
                    parameters.highpass_filter[1], parameters.lowpass_filter[1]);
            noa::signal::fft::xmap<noa::fft::H2F>(m_target_fft, m_reference_fft, m_xmap,
                                                  noa::signal::CONVENTIONAL_CORRELATION);

            if (!parameters.debug_directory.empty()) {
                const auto output_index = reference_slice.index;
                const auto xmap_filename = noa::string::format("xmap_{:>02}.mrc", output_index);
                noa::fft::remap(noa::fft::F2FC, m_xmap, m_target, m_xmap.shape());
                noa::io::save(m_target, parameters.debug_directory / xmap_filename);
            }

            // Computes the YX shift of the target. To shift-align the stretched target onto the reference,
            // we would then need to subtract this shift to the stretched target.
            const float2_t peak_reference = noa::signal::fft::xpeak2D<noa::fft::F2F>(
                    m_xmap, parameters.max_shift, noa::signal::PEAK_PARABOLA_1D, long2_t{1});
            const auto shift_reference = double2_t(peak_reference - slice_center);

            // We could destretch the shift to bring it back to the original target. However, we'll need
            // to compute the global shifts later on, as well as "centering the tilt-axis". This implies
            // to accumulate the shifts of the lower views while accounting for their own scaling.
            // Instead, it is simpler to scale all the slice-to-slice shifts to the same reference
            // frame, process everything there, and then go back to whatever higher tilt reference at
            // the end. Here, this common reference frame is the untilted and unpitched specimen.
            const float2_t reference_pivot_tilt{reference_angles[2], reference_angles[1]};
            const double22_t fwd_stretch_reference_to_0deg{
                    noa::geometry::rotate(reference_angles[0]) *
                    noa::geometry::scale(1 / math::cos(reference_pivot_tilt)) * // 1 = cos(0deg)
                    noa::geometry::rotate(-reference_angles[0])
            };
            const double2_t shift_0deg = fwd_stretch_reference_to_0deg * shift_reference;
            return shift_0deg;
        };

        // Compute the global shifts, i.e. the shifts to apply to a slice so that it becomes aligned with the
        // reference slice. At this point, we have the relative (i.e. slice-to-slice) shifts in the 0deg reference
        // frame, so we need to accumulate the shifts of the lower degree slices.
        // At the same time, we can center the global shifts to minimize the overall movement of the slices.
        static std::vector<double2_t> relative2globalShifts_(
                const std::vector<double2_t>& relative_shifts,
                const MetadataStack& metadata,
                size_t index_lowest_tilt,
                bool center_tilt_axis) {

            // Compute the global shifts and the mean.
            const size_t count = relative_shifts.size();
            std::vector<double2_t> global_shifts(count);
            const auto index_pivot = static_cast<int64_t>(index_lowest_tilt);
            const auto index_rpivot = static_cast<int64_t>(count) - 1 - index_pivot;
            const auto mean_scale = 1 / static_cast<double>(count);
            double2_t mean{0};
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

            qn::Logger::trace("Average shift: {:.3f}", mean);
            if (!center_tilt_axis)
                mean = 0;

            // Center the global shifts (optional) and scale them back to the original reference frame
            // of their respective slice, i.e. shrink the shifts to account for the slice's tilt and pitch.
            for (size_t i = 0; i < count; ++i) {
                const double3_t angles(noa::math::deg2rad(metadata[i].angles));
                const double22_t fwd_shrink_matrix{
                        noa::geometry::rotate(angles[0]) *
                        noa::geometry::scale(noa::math::cos(double2_t(angles[2], angles[1]))) *
                        noa::geometry::rotate(-angles[0])
                };
                global_shifts[i] = fwd_shrink_matrix * (global_shifts[i] - mean);
            }

            return global_shifts;
        }

    private:
        noa::Array<float> m_target_reference;
        noa::Array<cfloat_t> m_target_reference_fft;

        noa::Array<float> m_target;
        noa::Array<float> m_reference;
        noa::Array<cfloat_t> m_target_fft;
        noa::Array<cfloat_t> m_reference_fft;

        noa::Texture<float> m_target_texture;
        noa::Array<float> m_xmap;
    };
}
