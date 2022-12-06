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
#include "quinoa/core/Geometry.h"
#include "quinoa/io/Logging.h"

namespace qn::alignment {
    /// Shift alignment using cosine stretching on the higher tilt images.
    class PairwiseCosine {
    public:
        /// Allocates for the internal buffers.
        /// \param shape                (BD)HW shape of the slices. The BD dimensions are ignored.
        /// \param shift_scaling_factor           HW pixel size of the slices.
        /// \param compute_device       Device where to perform the alignment.
        ///                             If it is a GPU, the input stack for the alignment can be on any device,
        ///                             including the CPU. Otherwise, the stack will be expected to be on the CPU.
        /// \param smooth_edge_percent  Size, in percent of the maximum-sized dimension, of the zero-edge taper.
        /// \param interpolation_mode   Interpolation mode used for the cosine stretching.
        /// \param allocator            Allocator to use for \p compute_device.
        PairwiseCosine(dim4_t shape, Device compute_device,
                       float smooth_edge_percent = 0.1f,
                       InterpMode interpolation_mode = INTERP_LINEAR_FAST,
                       Allocator allocator = Allocator::DEFAULT_ASYNC)
                : m_center(MetadataSlice::center(shape)),
                  m_smooth_edge_size(static_cast<float>(std::max(shape[2], shape[3])) * smooth_edge_percent) {
            const dim4_t buffer_shape{2, 1, shape[2], shape[3]};
            const dim4_t slice_shape{1, 1, shape[2], shape[3]};
            const ArrayOption options(compute_device, allocator);

            // The "target" is the higher (absolute) tilt image that we want to align onto the "reference".
            // Performance-wise, using out-of-place FFTs could be slightly better, but here prefer the safer
            // option to use less memory and in-place FFTs.
            std::tie(m_target_reference, m_target_reference_fft) = fft::empty<float>(buffer_shape, options);
            m_target = m_target_reference.subregion(0);
            m_reference = m_target_reference.subregion(1);
            m_target_fft = m_target_reference_fft.subregion(0);
            m_reference_fft = m_target_reference_fft.subregion(1);

            // Since we need to transform the target, use a GPU texture since we'll need to copy
            // the target to a new array anyway. For CPU devices, this doesn't allocate anything,
            // and we'll directly use the input stack as input for the transformation.
            m_target_texture = Texture<float>(slice_shape, compute_device, interpolation_mode, BORDER_ZERO);

            // The (phase) cross-correlation map.
            m_xmap = memory::empty<float>(slice_shape, options);
        }

        /// 2D XY shifts alignment using cosine stretching on the reference images.
        /// \details Starts with the view with the lowest tiltY angle. To align a neighbouring view
        ///          (which has a higher tiltY angle), stretch that view by a factor of cosine(tiltY)
        ///          perpendicular to the tilt-axis, where tiltY is the tilt angle of the neighbouring
        ///          view, in radians. Then use phase correlation to find the XY shift between images.
        ///          The returned shifts are unstretched and the average shift is centered.
        /// \param[in] stack        Input stack.
        /// \param[in,out] metadata Metadata of \p stack. The slice indexes should correspond to the batch indexes
        ///                         in the \p stack. Excluded slices are ignored. When the function returns, this
        ///                         metadata is updated with the new shifts. The order of the slices is unchanged
        ///                         and excluded slices are preserved.
        /// \param max_shift        Maximum YX shift a slice is allowed to have. If <= 0, it is ignored.
        /// \param center           Whether the average shift (in the microscope reference frame) should be centered.
        void updateShifts(const Array<float>& stack,
                          MetadataStack& metadata,
                          float2_t max_shift = {},
                          bool center = true) {
            QN_CHECK(!m_xmap.empty(), "Empty object detected");
            QN_CHECK(stack.device().cpu() || m_xmap.device().gpu(),
                     "The input device {} is not supported for the compute device {}",
                     stack.device(), m_xmap.device());
            qn::Logger::trace("Pairwise cosine-stretch shift alignment...");
            Timer timer;
            timer.start();

            // We'll need the slices sorted by tilt angles, with the lowest absolute tilt being the pivot point.
            // The metadata is allowed to contain excluded views, so for simplicity here, squeeze them out.
            MetadataStack metadata_ = metadata;
            metadata_.squeeze().sort("tilt");
            const size_t index_lowest_tilt = metadata_.lowestTilt();
            const size_t slice_count = metadata_.size();

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
                        findRelativeShifts_(stack, metadata_[idx_reference], metadata_[idx_target], max_shift));
            }

            // This is the main drawback of this method. We need to compute the global shifts from the relative
            // shifts, so the high tilt slices end up accumulating the errors of the lower tilts.
            const std::vector<double2_t> global_shifts =
                    relative2globalShifts_(slice_to_slice_shifts, metadata_, index_lowest_tilt, center);

            // Update the metadata.
            for (size_t i = 0; i < slice_count; ++i) {
                for (auto& original_slice: metadata.slices()) {
                    if (original_slice.index == metadata_[i].index)
                        original_slice.shifts += static_cast<float2_t>(global_shifts[i]);
                }
            }

            qn::Logger::trace("Pairwise cosine-stretch shift alignment... took {}ms", timer.elapsed());
        }

    private:
        // Compute the phase-correlation between the reference and target, accounting for the known difference
        // in rotation and shift between the two. The yaw and shift difference can be directly corrected, but
        // the tilt and pitch cannot (these are 3D transformations for 2D slices). To mitigate this difference,
        // a scaling is applied onto the target, usually resulting in a stretch perpendicular to the tilt
        // and pitch axes.
        // As explained below, the output shift is the shift of the target relative to the reference in the
        // "microscope" reference frame (i.e. 0-degree tilt and pitch). This simplifies computation later
        // when the global shifts and centering needs to be computed.
        double2_t findRelativeShifts_(const Array<float>& stack,
                                      const MetadataSlice& reference_slice,
                                      const MetadataSlice& target_slice,
                                      float2_t max_shift) {
            // Get the reference and target in their respective buffers.
            memory::copy(stack.subregion(reference_slice.index), m_reference);
            m_target_texture.update(stack.subregion(target_slice.index));

            const float3_t target_angles = math::deg2rad(target_slice.angles);
            const float3_t reference_angles = math::deg2rad(reference_slice.angles);

            // Compute the affine matrix to transform the target "onto" the reference.
            // These angles are flipped, since the cos-scaling is perpendicular to the axis of rotation.
            const float2_t cos_factor{math::cos(reference_angles[2]) / math::cos(target_angles[2]),
                                      math::cos(reference_angles[1]) / math::cos(target_angles[1])};

            // Apply the scaling for the tilt and pitch difference,
            // and cancel the difference (if any) in yaw and shift as well.
            // After this point, the target should "overlap" with the reference.
            const float33_t fwd_stretch_target_to_reference =
                    noa::geometry::translate(m_center + reference_slice.shifts) *
                    float33_t{noa::geometry::rotate(reference_angles[0])} *
                    float33_t{noa::geometry::scale(cos_factor)} *
                    float33_t{noa::geometry::rotate(-target_angles[0])} *
                    noa::geometry::translate(-m_center - target_slice.shifts);
            const float33_t inv_stretch_target_to_reference = math::inverse(fwd_stretch_target_to_reference);
            noa::geometry::transform2D(m_target_texture, m_target, inv_stretch_target_to_reference);

            // Enforce common field-of-view, hopefully to guide the alignment and not compare things that
            // cannot be compared. This is mostly relevant for large shifts between images and high tilt angles.
            noa::signal::rectangle(
                    m_target_reference, m_target_reference, m_center,
                    m_center - m_smooth_edge_size, m_smooth_edge_size,
                    fwd_stretch_target_to_reference);
            noa::signal::rectangle(
                    m_target_reference, m_target_reference, m_center,
                    m_center - m_smooth_edge_size, m_smooth_edge_size,
                    inv_stretch_target_to_reference);

            // (Conventional) cross-correlation:
            fft::r2c(m_target_reference, m_target_reference_fft);
            signal::fft::xmap<fft::H2F>(m_target_fft, m_reference_fft, m_xmap);

            // Computes the YX shift of the target. To shift-align the stretched target onto the reference,
            // we would then need to subtract this shift to the stretched target.
            const float2_t peak_reference = signal::fft::xpeak2D<fft::F2F>(m_xmap, max_shift);
            const double2_t shift_reference(peak_reference - m_center);

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
                bool center) {

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
            if (!center)
                mean = 0;

            // Center the global shifts (optional) and scale them back to the original reference frame
            // of their respective slice, i.e. shrink the shifts to account for the slice's tilt and pitch.
            for (size_t i = 0; i < count; ++i) {
                const double3_t angles(math::deg2rad(metadata[i].angles));
                const double22_t fwd_shrink_matrix{
                        noa::geometry::rotate(angles[0]) *
                        noa::geometry::scale(math::cos(double2_t(angles[2], angles[1]))) *
                        noa::geometry::rotate(-angles[0])
                };
                global_shifts[i] = fwd_shrink_matrix * (global_shifts[i] - mean);
            }

            return global_shifts;
        }

    private:
        Array<float> m_target_reference;
        Array<cfloat_t> m_target_reference_fft;

        Array<float> m_target;
        Array<float> m_reference;
        Array<cfloat_t> m_target_fft;
        Array<cfloat_t> m_reference_fft;

        Texture<float> m_target_texture;
        Array<float> m_xmap;
        float2_t m_center{};
        float m_smooth_edge_size{};
    };
}
