#include <noa/Array.hpp>
#include <noa/Signal.hpp>
#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/CommonArea.hpp"
#include "quinoa/Types.hpp"

namespace {
    using namespace qn;

    struct SIRTWeight {
        using value_type = f32;
        f32 fake_iter;

        // With a large enough level (>1000), this is equivalent to a radial weighting.
        NOA_HD constexpr explicit SIRTWeight(f32 level) {
            fake_iter = level <= 15.f ? level :
                        level <= 30.f ? 15.f + 0.4f * (level - 15.f) :
                                        27.f * 0.6f * (level - 30.f);
        }

        NOA_HD constexpr auto operator()(const Vec<f32, 3>& fftfreq_3d, i32) const {
            const f32 fftfreq = noa::sqrt(dot(fftfreq_3d, fftfreq_3d));
            if (fftfreq < 1e-6f or fftfreq > 0.5f)
                return 0.f;
            const f32 max = 0.5f * (1 - noa::pow(1 - 0.00195f / 0.5f, fake_iter));
            const f32 current = fftfreq * (1 - noa::pow(1 - 0.00195f / fftfreq, fake_iter));
            return current / max;
        }
    };

    struct ReduceMeanWithinMask {
        static constexpr void init(f32 v, f32& s, i32& m) {
            if (v != 0.f) {
                s += v;
                m += 1;
            }
        }
        static constexpr void join(f32 i_s, i32 i_m, f32& s, i32& m) {
            s += i_s;
            m += i_m;
        }
    };

    struct SubtractMeanWithinMask {
        constexpr void operator()(f32 s, i32 m, f32& v) const {
            if (v != 0.f) {
                v -= s / static_cast<f32>(m);
            }
        }
    };

    class Projector {
        Array<f32> m_input_images;
        Array<c32> m_references_rfft;
        Array<f32> m_target_and_references;
        Array<c32> m_target_and_references_padded_rfft;
        Array<f32> m_xmap;

        CommonArea m_common_area;

        [[nodiscard]] auto options_() const -> ArrayOption {
            return {.device = m_input_images.device(), .allocator = Allocator::ASYNC};
        }

        [[nodiscard]] auto slice_shape_() const -> Shape<i64, 4> {
            return m_input_images.shape().set<0>(1);
        }

        [[nodiscard]] auto references_rfft_(i64 n) const -> View<c32> {
            return m_references_rfft.view().subregion(ni::Slice{0, n});
        }

        [[nodiscard]] auto references_(i64 n) const -> View<f32> {
            return noa::fft::alias_to_real(references_rfft_(n), slice_shape_().set<0>(n));
        }

        [[nodiscard]] auto target_and_references_(i64 n) const -> View<f32> {
            return m_target_and_references.view().subregion(ni::Slice{0, n + 1});
        }

        [[nodiscard]] auto target_and_references_padded_rfft_(i64 n) const -> View<c32> {
            return m_target_and_references_padded_rfft.view().subregion(ni::Slice{0, n + 1});
        }

        [[nodiscard]] auto target_and_references_padded_(i64 n) const -> View<f32> {
            return noa::fft::alias_to_real(target_and_references_padded_rfft_(n), slice_shape_().set<0>(n + 1));
        }

        void compute_projections(
            const View<const f32>& tilt_series,
            const MetadataStack& tilt_series_metadata,
            const MetadataStack& output_metadata,
            i64 volume_thickness
        ) {
            //
            const auto options = options_();
            const auto slice_shape = tilt_series.shape().filter(2, 3);
            const auto volume_shape = slice_shape.push_front(volume_thickness);
            const auto volume_center = (volume_shape / 2).vec.as<f64>();

            // Retrieve the input(s) and output(s).
            const auto n_inputs = tilt_series_metadata.ssize();
            auto input_images = m_input_images.view().subregion(ni::Slice{0, n_inputs});

            // Apply the common FOV to the images about to be backward projected.
            m_common_area = CommonArea(slice_shape, tilt_series_metadata);
            m_common_area.mask(tilt_series, input_images, tilt_series_metadata, false, 0.1);

            // Backward projection matrices.
            auto input_matrices = Array<Mat<f32, 2, 4>>(tilt_series_metadata.ssize());
            for (auto&& [input_matrix, slice]: noa::zip(input_matrices.span_1d_contiguous(), tilt_series_metadata)) {
                input_matrix = (
                    ng::translate((volume_center.pop_front() + slice.shifts).push_front(0)) *
                    ng::linear2affine(ng::rotate_z(noa::deg2rad(slice.angles[0]))) *
                    ng::linear2affine(ng::rotate_y(noa::deg2rad(slice.angles[1]))) *
                    ng::linear2affine(ng::rotate_x(noa::deg2rad(slice.angles[2]))) *
                    ng::translate(-volume_center)
                ).inverse().filter_rows(1, 2).as<f32>();
            }
            if (options.device.is_gpu())
                input_matrices = std::move(input_matrices).to(options);

            // Forward projection matrices.
            i64 projection_window{};
            auto output_matrices = Array<Mat<f32, 3, 4>>(output_metadata.ssize());
            for (auto&& [output_matrix, slice]: noa::zip(output_matrices.span_1d_contiguous(), output_metadata)) {
                output_matrix = (
                    ng::translate((volume_center.pop_front() + slice.shifts).push_front(0)) *
                    ng::linear2affine(ng::rotate_z(noa::deg2rad(slice.angles[0]))) *
                    ng::linear2affine(ng::rotate_y(noa::deg2rad(slice.angles[1]))) *
                    ng::linear2affine(ng::rotate_x(noa::deg2rad(slice.angles[2]))) *
                    ng::translate(-volume_center)
                ).pop_back().as<f32>();

                projection_window = std::max(
                    projection_window, ng::forward_projection_window_size(volume_shape, output_matrix));
            }
            if (options.device.is_gpu())
                output_matrices = std::move(output_matrices).to(options);

            // Retrieve the outputs.
            const auto n_outputs = output_metadata.ssize();
            auto references_rfft = references_rfft_(n_outputs);
            auto references = references_(n_outputs);

            // Backward and forward project.
            ng::backward_and_forward_project_3d(
                input_images, references, volume_shape,
                std::move(input_matrices), std::move(output_matrices), projection_window,
                {.interp = noa::Interp::LINEAR, .add_to_output = false}
            );

            // Radial weighting.
            // TODO Add bandpass?
            noa::fft::r2c(references, references_rfft);
            ns::filter_spectrum<"h2h">(references_rfft, references_rfft, references.shape(), SIRTWeight{300});
            noa::fft::c2r(references_rfft, references);
        }

        struct NCCResult {

        };

        /// Computes the normalized cross-correlation (NCC), between the target and each reference, at the best lag.
        /// At such, this function returns, for each reference:
        ///     1. the relative shift.
        ///     2. the NCC at this shift.
        void cross_correlate_projections(
            const View<f32>& target,
            const MetadataSlice& target_metadata,
            const MetadataStack& references_metadata
        ) {
            // Retrieve the buffers.
            const auto n_outputs = references_metadata.ssize();
            auto references = references_(n_outputs);
            auto target_and_references = target_and_references_(n_outputs);
            auto target_and_references_padded_rfft = target_and_references_padded_rfft_(n_outputs);
            auto target_and_references_padded = target_and_references_padded_(n_outputs);

            // Common FOV.
            m_common_area.mask(target, target_and_references.subregion(0), target_metadata, false, 0.1);
            m_common_area.mask(references, target_and_references.subregion(ni::Slice{1, n_outputs}), references_metadata, false, 0.1);

            // Normalize the mean within the FOV.
            const auto device = options_().device;
            const auto options = options_().set_allocator(Allocator::MANAGED);
            auto sums = noa::Array<f32>(n_outputs + 1, options).flat(0);
            auto count = noa::Array<i32>(n_outputs + 1, options).flat(0);
            noa::reduce_axes_ewise(
                target_and_references,
                noa::wrap(f32{}, i32{}),
                noa::wrap(sums, count),
                ReduceMeanWithinMask{}
            );
            noa::ewise(
                noa::wrap(sums, count),
                target_and_references,
                SubtractMeanWithinMask{}
            );

            // Compute the L2-norm.
            // This will be used to normalize the cross-correlation peak.
            // This is equivalent (but faster) to normalizing the peak with the auto-correlation.
            const auto norms = sums.view(); // shape={n,1,1,1}
            noa::l2_norm(target_and_references, norms);

            // Zero-pad and compute the cross-correlation.
            // This is the correct way to do it, but the padding is a bit overkill for most cases.
            const auto padding = (target_and_references_padded.shape() - target_and_references.shape()).vec;
            noa::resize(target_and_references, target_and_references_padded, {}, padding);
            noa::fft::r2c(target_and_references_padded, target_and_references_padded_rfft, {
                .norm = noa::fft::Norm::NONE
            });

            // To save memory, reuse the weights to store the cross-correlation map.
            const auto xmap = m_xmap.view().subregion(ni::Slice{0, n_outputs});

            // We rotate the xmap before the picking, so compute the centered xmap.
            ns::cross_correlation_map<"h2fc">(
                target_and_references_padded_rfft.subregion(0),
                target_and_references_padded_rfft.subregion(ni::Slice{1, n_outputs}),
                xmap, {.ifft_norm = noa::fft::Norm::NONE}
            );

            // Work on a small subregion located at the center of the xmap, because the peak should be very close
            // to the center and because it's more efficient. Moreover, the peak can be distorted perpendicular to
            // the tilt-axis due to the tilt (although the z-mask should mostly fix that). To help the picking,
            // rotate so that the distortion is along the X-axis. At the same time, setup transform_2d to only
            // render the small subregion at the center of the xmap.
            const View<f32> peak_window = m_peak_window.view();

            const auto xmap_center = (xmap.shape().vec.filter(2, 3) / 2).as<f64>();
            const auto peak_window_center = (peak_window.shape().vec.filter(2, 3) / 2).as<f64>();
            const auto xmap_matrices = Array<Mat<f32, 2, 3>>(n_outputs, {.allocator = Allocator::MANAGED});
            for (auto&& [matrix, slice]: noa::zip(xmap_matrices.span_1d_contiguous(), references_metadata)) {
                matrix = (
                    ng::translate(peak_window_center) *
                    ng::linear2affine(ng::rotate(noa::deg2rad(-slice.angles[0]))) *
                    ng::translate(-xmap_center)
                ).inverse().pop_back().as<f32>();
            }
            ng::transform_2d(xmap, peak_window, xmap_matrices.view().reinterpret_as(device.type()));

            // Find the peak, with subpixel-registration.
            // TODO Test without peak_window and transform_2d.
            auto peak_coordinates = Array<Vec<f32, 2>>(n_outputs, options);
            auto peak_values = Array<f32>(n_outputs, options);
            ns::cross_correlation_peak_2d<"fc2fc">(peak_window, peak_coordinates, peak_values);

            Vec<Vec<f64, 2>, 3> shifts;
            Vec<f64, 3> scores;
            auto norm_target = norms.first();

            for (auto&& [peak_coordinate, peak_value, norm_reference, slice]:
                 noa::zip(peak_coordinates.span_1d_contiguous(),
                          peak_values.span_1d_contiguous(),
                          norms.span_1d_contiguous().offset_inplace(1),
                          references_metadata)
            ) {
                // Deduce the shift from the peak position.
                // Don't forget to rotate back; the shift is not rotated!
                const Mat<f64, 2, 2> rotation_offset = ng::rotate(noa::deg2rad(slice.angles[0]));
                const Vec<f64, 2> shift_rotated = peak_coordinate.as<f64>() - peak_window_center;
                const Vec<f64, 2> shift = rotation_offset * shift_rotated.as<f64>();

                // Peak normalization.
                const auto denominator =
                        static_cast<f64>(norm_target) * // target norm
                        static_cast<f64>(norm_reference) * // projected norm
                        static_cast<f64>(xmap.shape().filter(2, 3).n_elements()); // fft scale
                const auto score = static_cast<f64>(peak_value) / denominator;
            }
            return {score, shift};
        }
    };


}
