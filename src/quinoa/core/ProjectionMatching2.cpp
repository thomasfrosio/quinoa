#include <random>

#include <noa/FFT.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/core/Ewise.hpp"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/ProjectionMatching.hpp"
#include "quinoa/core/CubicGrid.hpp"
#include "quinoa/io/Logging.h"

namespace {
    using namespace qn;

    // Compute the projected-references and shift-aligns it to the corresponding target.
    // The reference used to compute the first projected-reference is added to the lists
    // of references for the next projected-reference.
    class Projector {
    private:
        // Layout: [reference 0, ..., reference n, target, projected_reference_dst, projected_reference_src]
        i64 m_n_references{0};
        View<c32> m_slices_padded_rfft;
        View<f32> m_weights_padded_rfft;
        View<f32> m_two_slices;

        CTFAnisotropic64 m_average_ctf;
        Array<f32> m_two_values;
        Array<CTFAnisotropic64> m_ctfs;
        Array<Quaternion<f32>> m_insertion_inv_rotations;
        Array<f32> m_peak_window;

    public:
        Projector(
                const View<c32>& slices_padded_rfft,
                const View<f32>& weights_padded_rfft,
                const View<f32>& two_slices,
                const CTFAnisotropic64& average_ctf
        ) :
                m_slices_padded_rfft(slices_padded_rfft),
                m_weights_padded_rfft(weights_padded_rfft),
                m_two_slices(two_slices),
                m_average_ctf(average_ctf)
        {
            // Small utility arrays...
            const auto device = slices_padded_rfft.device();
            const auto options = ArrayOption(device, Allocator::MANAGED);
            const auto options_cpu = ArrayOption(Device{}, Allocator::MANAGED);

            m_two_values = noa::memory::empty<f32>({2, 1, 1, 1}, options);
            m_ctfs = noa::memory::empty<CTFAnisotropic64>(2, options_cpu);
            m_insertion_inv_rotations = noa::memory::empty<Quaternion<f32>>({slices_padded_rfft.shape()[0], 1, 1, 1}, options_cpu);
            m_peak_window = noa::memory::empty<f32>({1, 1, 128, 128}, {device, Allocator::DEFAULT_ASYNC});
        }

    public:
        void reset() { m_n_references = 0; }

        [[nodiscard]] auto project_and_correlate_next(
                const View<const f32>& stack,
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) -> std::pair<f64, Vec2<f64>> {
            // Add the reference to the list of previous reference and project.
            increment_reference_count_();
            preprocess_(stack, reference_metadata, target_metadata, common_area, parameters);
            fourier_insert_and_extract_(reference_metadata, target_metadata, parameters);

            // Peak the best shift.
            Vec2<f64> final_shifts{};
            f64 final_score;
            auto updated_target_metadata = target_metadata;

            for (auto i: noa::irange(200)) {
                postprocessing_(stack, updated_target_metadata, common_area, parameters);
                auto [i_score, i_shift] = cross_correlate_(updated_target_metadata, common_area, parameters);

                updated_target_metadata.shifts += i_shift;
                final_score = i_score;
                final_shifts += i_shift;

                if (noa::all(noa::math::abs(i_shift) < parameters.shift_tolerance)) {
//                    qn::Logger::trace("count={}", i);
                    break;
                }
            }
            return {final_score, final_shifts};
        }

    private:
        constexpr void increment_reference_count_() noexcept { m_n_references += 1; }
        [[nodiscard]] constexpr auto n_references_() const noexcept -> i64 { return m_n_references; }
        [[nodiscard]] constexpr auto reference_index_() const noexcept -> i64 { return m_n_references - 1; }
        [[nodiscard]] constexpr auto target_index_() const noexcept -> i64 { return m_n_references; }
        [[nodiscard]] constexpr auto projected_dst_index_() const noexcept -> i64 { return m_n_references + 1; }
        [[nodiscard]] constexpr auto projected_src_index_() const noexcept -> i64 { return m_n_references + 2; }

        [[nodiscard]] auto size_padded_() const noexcept -> i64 {
            return m_slices_padded_rfft.shape()[2];
        }
        [[nodiscard]] auto shape_padded_() const noexcept -> Shape4<i64> {
            return {1, 1, size_padded_(), size_padded_()};
        }
        [[nodiscard]] auto center_padded_() const noexcept -> Vec2<f64> {
            return MetadataSlice::center<f64>(shape_padded_());
        }
        [[nodiscard]] auto shape_original_() const noexcept -> Shape4<i64> {
            return {1, 1, m_two_slices.shape()[2], m_two_slices.shape()[3]};
        }
        [[nodiscard]] auto center_original_() const noexcept -> Vec2<f64> {
            return MetadataSlice::center<f64>(shape_original_());
        }
        [[nodiscard]] auto border_right_() const noexcept -> Vec4<i64> {
            return (shape_padded_() - shape_original_()).vec();
        }

        [[nodiscard]] auto compute_device() const noexcept -> Device {
            return m_two_slices.device();
        }

        [[nodiscard]] auto set_reference_and_target_ctfs_(
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata
        ) const {
            // This is the sampling function, and it contains:
            //  - The CTF set to the defocus of the slice. The defocus gradient should be negligible
            //    thanks to the common-area mask.
            //  - Approximated exposure filter using the CTF B-factor.
            //  - Approximated relative-SNR from electron mean-free-path, using cos(tilt) as scaling factor.
            const auto [_, defocus_astig, defocus_angle] = m_average_ctf.defocus();
            Span ctfs = m_ctfs.span();
            ctfs[0] = m_average_ctf;
            ctfs[0].set_defocus({reference_metadata.defocus, defocus_astig, defocus_angle});
            ctfs[0].set_bfactor(-reference_metadata.exposure[1] * 2);
            ctfs[0].set_scale(noa::math::cos(noa::math::deg2rad(reference_metadata.angles[1])));

            ctfs[1] = m_average_ctf;
            ctfs[1].set_defocus({target_metadata.defocus, defocus_astig, defocus_angle});
            ctfs[1].set_bfactor(-target_metadata.exposure[1] * 2);
            ctfs[1].set_scale(noa::math::cos(noa::math::deg2rad(target_metadata.angles[1])));
        }

        void preprocess_(
                const View<const f32>& stack,
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) {
            // Copy from stack and apply the common-area mask.
            const View<f32> reference = m_two_slices.subregion(0);
//            common_area.mask_view(
//                    stack.subregion(reference_metadata.index),
//                    reference.subregion(0),
//                    reference_metadata,
//                    parameters.smooth_edge_percent);
            stack.subregion(reference_metadata.index).to(reference.subregion(0));

            // Zero-(right-)pad and in-place rfft.
            const View<c32> reference_padded_rfft = m_slices_padded_rfft.subregion(reference_index_());
            const View<f32> reference_padded = noa::fft::alias_to_real(reference_padded_rfft, shape_padded_());
            noa::memory::resize(reference, reference_padded, {}, border_right_());
            noa::fft::r2c(reference_padded, reference_padded_rfft);

            // Pre-processing for Fourier insertion.
            // 1. Remap/fftshift, since the Fourier insertion requires a centered input.
            // 2. Phase-shift the rotation-center to the origin. Note that the right-padding doesn't change the center.
            noa::fft::remap(noa::fft::H2HC, reference_padded_rfft, reference_padded_rfft, shape_padded_());
            noa::signal::fft::phase_shift_2d<noa::fft::HC2HC>(
                    reference_padded_rfft, reference_padded_rfft, shape_padded_(),
                    (-center_original_() - reference_metadata.shifts).as<f32>());

            // Compute and apply the sampling functions of both the reference and target.
            // The current sampling function simply contains the exposure filter and the average CTF of the slice.
            // The CTF at the tilt-axis (this is still per-slice) should be a first good approximation for the
            // sampling function. The field-of-view is restrained to the common-area, so it should be good even
            // for tilted slices.
            if (parameters.use_ctfs) {
                set_reference_and_target_ctfs_(reference_metadata, target_metadata);
                const auto ctfs = m_ctfs.view().as(compute_device().type(), /*prefetch=*/ true);

                const auto slice = noa::indexing::Slice{reference_index_(), target_index_() + 1};
                const auto reference_and_target_padded_rfft = m_slices_padded_rfft.subregion(slice);
                const auto reference_and_target_weights_padded_rfft = m_weights_padded_rfft.subregion(slice);

                // Note that the CTF is multiplied once onto the data, but since the microscope already multiplies
                // the specimen with the CTF once, the sampling function should instead be CTF^2.
                noa::signal::fft::ctf_anisotropic<noa::fft::H2H>( // data
                        reference_and_target_padded_rfft, reference_and_target_padded_rfft,
                        shape_padded_().set<0>(2), ctfs
                );
                noa::signal::fft::ctf_anisotropic<noa::fft::H2H>( // weights
                        reference_and_target_weights_padded_rfft,
                        shape_padded_().set<0>(2), ctfs,
                        /*ctf_abs=*/ false, /*ctf_square=*/ true
                );

                // These are also used for the projection, so center them.
                const View<f32> reference_weights_padded_rfft = reference_and_target_weights_padded_rfft.subregion(0);
                noa::fft::remap(noa::fft::H2HC, reference_weights_padded_rfft,
                                reference_weights_padded_rfft, shape_padded_());

                noa::io::save(reference_and_target_weights_padded_rfft, parameters.debug_directory / "weights.mrc");
            } else {
                noa::memory::fill(m_weights_padded_rfft.subregion(reference_index_()), 1.f);
            }
        }

        void fourier_insert_and_extract_(
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata,
                const ProjectionMatchingParameters& parameters
        ) {
            // The rotation is the CCW angle of the tilt-axis in the slices. For the projection, we want to align
            // the tilt-axis along the Y axis, so subtract this angle and then apply the tilt and elevation.
            // For the Fourier insertion, noa needs the inverse rotation matrix, hence the transposition.
            const Vec3<f64> insertion_angles = noa::math::deg2rad(reference_metadata.angles);
            const Vec3<f64> extraction_angles = noa::math::deg2rad(target_metadata.angles);
            m_insertion_inv_rotations.span()[reference_index_()] =
                    noa::geometry::matrix2quaternion(
                            noa::geometry::euler2matrix(
                                    Vec3<f64>{-insertion_angles[0], insertion_angles[1], insertion_angles[2]},
                                    "zyx", /*intrinsic=*/ false).transpose().as<f32>()
                    );

            const Float33 extraction_fwd_rotation = noa::geometry::euler2matrix(
                    Vec3<f64>{-extraction_angles[0], extraction_angles[1], extraction_angles[2]},
                    "zyx", /*intrinsic=*/ false).as<f32>();

            // Windowed sinc.
            using WindowedSinc = noa::geometry::fft::WindowedSinc;
            const auto i_windowed_sinc = WindowedSinc{
                    static_cast<f32>(parameters.fftfreq_sinc),
                    static_cast<f32>(parameters.fftfreq_blackman)};
            const auto o_windowed_sinc = WindowedSinc{
                    static_cast<f32>(parameters.fftfreq_z_sinc),
                    static_cast<f32>(parameters.fftfreq_z_blackman)};

            const auto references_slice = noa::indexing::Slice{0, n_references_()};
            const auto slices_padded_shape = shape_padded_().set<0>(n_references_());
            auto insertion_inv_rotations = m_insertion_inv_rotations
                    .view().subregion(references_slice)
                    .as(compute_device().type(), /*prefetch=*/ true);

            // Note: This is by far the most expensive step!
            noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                    m_slices_padded_rfft.subregion(references_slice),
                    m_weights_padded_rfft.subregion(references_slice), slices_padded_shape,
                    m_slices_padded_rfft.subregion(projected_src_index_()),
                    m_weights_padded_rfft.subregion(projected_src_index_()), shape_padded_(),
                    Float22{}, insertion_inv_rotations,
                    Float22{}, extraction_fwd_rotation,
                    i_windowed_sinc, o_windowed_sinc,
                    /*add_to_output=*/ false,
                    /*correct_multiplicity=*/ true);
        }

        void postprocessing_(
                const View<const f32>& stack,
                const MetadataSlice& target_metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) {
            // We have first the target, then the buffer for the post-processed projected-reference (projected_dst),
            // and then the projected-reference that should be kept intact (projected_src).
            const auto target_and_projected_slice = noa::indexing::Slice{target_index_(), target_index_() + 2};
            const View<c32> target_and_projected_padded_rfft = m_slices_padded_rfft
                    .subregion(target_and_projected_slice);

            // Applying the weights of the projected-reference onto the target and vice versa.
            const auto target_padded_rfft = target_and_projected_padded_rfft.subregion(0);
            const auto projected_padded_rfft = target_and_projected_padded_rfft.subregion(1);

            // First, prepare the target.
            // 1. Copy from stack and apply the common-area mask.
            // 2. Zero-pad and in-place rfft.
            const View<f32> target = m_two_slices.subregion(0);
//            common_area.mask_view(
//                    stack.subregion(target_metadata.index),
//                    target, target_metadata,
//                    parameters.smooth_edge_percent);
            stack.subregion(target_metadata.index).to(target);
            const View<f32> target_padded = noa::fft::alias_to_real(target_padded_rfft, shape_padded_());
            noa::memory::resize(target, target_padded, {}, border_right_());
            noa::fft::r2c(target_padded, target_padded_rfft);

            // Then prepare the projected-reference.
            // 1. Copy it to a temporary buffer, so we can reuse it later with a difference target shift.
            // 2. Center projection back and shift to the target reference-frame.
            noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                    m_slices_padded_rfft.subregion(projected_src_index_()), projected_padded_rfft, shape_padded_(),
                    (center_original_() + target_metadata.shifts).as<f32>());

            // Apply the weights.
            noa::ewise_binary( // target * projected_weights
                    target_padded_rfft,
                    m_weights_padded_rfft.subregion(projected_src_index_()),
                    target_padded_rfft, noa::multiply_t{});
            if (parameters.use_ctfs) {
                noa::ewise_binary( // projected * target_weights
                        projected_padded_rfft,
                        m_weights_padded_rfft.subregion(target_index_()),
                        projected_padded_rfft, noa::multiply_t{});
            }

//            noa::signal::fft::bandpass<noa::fft::H2H>(
//                    target_and_projected_padded_rfft, target_and_projected_padded_rfft,
//                    shape_padded_().set<0>(2),
//                    parameters.highpass_filter[0], parameters.lowpass_filter[0],
//                    parameters.highpass_filter[1], parameters.lowpass_filter[1]);

            // Go back to real-space and crop.
            const View<f32> target_and_projected_padded = noa::fft::alias_to_real(
                    target_and_projected_padded_rfft, shape_padded_().set<0>(2));
            noa::fft::c2r(target_and_projected_padded_rfft, target_and_projected_padded);
            noa::memory::resize(target_and_projected_padded, m_two_slices, {}, -border_right_());
        }

        auto cross_correlate_(
                const MetadataSlice& target_metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) -> std::pair<f64, Vec2<f64>> {
            const auto target_and_projected_slice = noa::indexing::Slice{target_index_(), target_index_() + 2};

            // 1. Compute the mask.
            // To save memory, reuse the weights to store the mask.
            const auto mask = m_slices_padded_rfft
                    .subregion(target_and_projected_slice).flat(/*axis=*/ 0).as<f32>()
                    .subregion(noa::indexing::Slice{0, shape_original_().elements()})
                    .reshape(shape_original_());
//            common_area.compute(mask, target_metadata, parameters.smooth_edge_percent);
            noa::memory::fill(mask, 1.f);

            // 2. Apply the mask to both the target and projected reference.
            const auto target_and_projected = m_two_slices;
            noa::ewise_binary(target_and_projected, mask, target_and_projected, noa::multiply_t{});

            // 3. Compute the sum of the two views and use the mask area to get the means under the mask.
            const auto sums = m_two_values.view(); // shape={2,1,1,1}
            noa::math::sum(target_and_projected, sums);
            const auto mask_area = noa::math::sum(mask); // synced
            const auto target_mean_within_mask = sums.span()[0] / mask_area;
            const auto projected_mean_within_mask = sums.span()[1] / mask_area;

            // 4. Subtract the mean under the mask.
            noa::ewise_trinary(
                    target_and_projected.subregion(0), target_mean_within_mask, mask,
                    target_and_projected.subregion(0), subtract_within_mask_t{});
            noa::ewise_trinary(
                    target_and_projected.subregion(1), projected_mean_within_mask, mask,
                    target_and_projected.subregion(1), subtract_within_mask_t{});

            if (!parameters.debug_directory.empty()) {
                noa::io::save(target_and_projected,
                              parameters.debug_directory /
                              noa::string::format("target_and_projected_{:>02}.mrc", target_metadata.index));
            }

            // 5. Compute the L2-norm.
            // This will be used to normalize the cross-correlation peak.
            // This is equivalent (but faster) to normalizing the peak with the auto-correlation.
            const auto norms = m_two_values.view(); // shape={2,1,1,1}
            noa::math::norm(target_and_projected, norms);

            // 6. Zero-pad (again) and compute the cross-correlation.
            // This is the correct way to do it, but the padding is a bit overkill for most cases.
            const View<c32> target_and_projected_padded_rfft = m_slices_padded_rfft
                    .subregion(target_and_projected_slice);
            const View<f32> target_and_projected_padded = noa::fft::alias_to_real(
                    target_and_projected_padded_rfft, shape_padded_().set<0>(2));
            noa::memory::resize(target_and_projected, target_and_projected_padded, {}, border_right_());
            noa::fft::r2c(target_and_projected_padded, target_and_projected_padded_rfft, noa::fft::Norm::NONE);

            // To save memory, reuse the weights to store the cross-correlation map.
            const auto xmap = m_slices_padded_rfft
                    .subregion(target_and_projected_slice).flat(/*axis=*/ 0).as<f32>()
                    .subregion(noa::indexing::Slice{0, shape_padded_().elements()})
                    .reshape(shape_padded_());

            // We rotate the xmap before the picking, so compute the centered xmap.
            noa::signal::fft::xmap<noa::fft::H2FC>(
                    target_and_projected_padded_rfft.subregion(0),
                    target_and_projected_padded_rfft.subregion(1),
                    xmap, noa::signal::CorrelationMode::CONVENTIONAL,
                    noa::fft::Norm::NONE);

            // Work on a small subregion located at the center of the xmap, because the peak should be very close
            // to the center and because it's more efficient. Moreover, the peak can be distorted perpendicular to
            // the tilt-axis due to the tilt (although the z-mask should mostly fix that). To help the picking,
            // rotate so that the distortion is along the X-axis. At the same time, setup transform_2d to only
            // render the small subregion at the center of the xmap.
            const View<f32> peak_window = m_peak_window.view();

            // Note that the center of the cross-correlation is fixed at N//2 (integral division), so don't use
            // MetadataSlice::center in case we switch of convention one day and use N/2 (floating-point division).
            const auto xmap_center = (xmap.shape().vec().filter(2, 3) / 2).as<f64>();
            const auto peak_window_center = (peak_window.shape().vec().filter(2, 3) / 2).as<f64>();

            const auto rotation_angle_rad = noa::math::deg2rad(target_metadata.angles[0]);
            const Double22 rotation = noa::geometry::rotate(rotation_angle_rad);
            const Double33 xmap_fwd_transform =
                    noa::geometry::translate(peak_window_center) *
                    noa::geometry::linear2affine(rotation.transpose()) *
                    noa::geometry::translate(-xmap_center);
            noa::geometry::transform_2d(xmap, peak_window, xmap_fwd_transform.inverse().as<f32>());

            if (!parameters.debug_directory.empty()) {
                noa::io::save(xmap,
                              parameters.debug_directory /
                              noa::string::format("xmap_{:>02}.mrc", target_metadata.index));
                noa::io::save(peak_window,
                              parameters.debug_directory /
                              noa::string::format("xmap_rotated_{:>02}.mrc", target_metadata.index));
            }

            // Find the peak, with subpixel-registration.
            auto [peak_coordinate, peak_value] = noa::signal::fft::xpeak_2d<noa::fft::FC2FC>(peak_window);

            // Deduce the shift from the peak position.
            // Don't forget to rotate back; the shift is not rotated!
            const Vec2<f64> shift_rotated = peak_coordinate.as<f64>() - peak_window_center;
            const Vec2<f64> shift = rotation * shift_rotated.as<f64>();

            // Peak normalization.
            const auto denominator =
                    static_cast<f64>(norms.span()[0]) *             // target norm
                    static_cast<f64>(norms.span()[1]) *             // projected norm
                    static_cast<f64>(shape_padded_().elements());   // fft scale
            const auto score = static_cast<f64>(peak_value) / denominator;
            return {score, shift};
        }
    };

    // Utility class for the projection-matching optimization.
    class Fitter {
    private:
        View<const f32> m_stack;
        Projector* m_projector;
        const MetadataStack* m_metadata;
        const CommonArea* m_common_area;
        const ProjectionMatchingParameters* m_parameters;

        Memoizer m_memoizer;
        Vec2<f64> m_tilt_minmax;
        std::vector<f64> m_scores;
        MetadataStack m_updated_metadata;
        i64 m_n_evaluations{0}; // just for logging

    public:
        Fitter(
                const View<const f32>& stack,
                Projector& projector,
                const MetadataStack& metadata, // already sorted in the desired projection order
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) :
                m_stack(stack),
                m_projector(&projector),
                m_metadata(&metadata),
                m_common_area(&common_area),
                m_parameters(&parameters)
        {
            m_memoizer = Memoizer(/*n_parameters=*/ parameters.rotation_resolution, /*resolution=*/ 25);
            const auto minmax = metadata.minmax_tilts();
            m_tilt_minmax = {minmax.first, minmax.second};
        }

    private:
        auto memoizer_() -> Memoizer& { return m_memoizer; }

        [[nodiscard]] auto to_normalized_tilt_(f64 tilt) const noexcept -> f64 {
            const auto& [min, max] = m_tilt_minmax;
            return (tilt - min) / (max - min);
        }

    public:
        [[nodiscard]] auto aligned_metadata() noexcept -> MetadataStack& { return m_updated_metadata; }
        [[nodiscard]] auto n_evaluations() const noexcept -> i64 { return m_n_evaluations; }

        auto align(const Span<f64>& rotation_offsets) -> f64 {
            m_scores.clear();
            m_projector->reset();
            m_updated_metadata = *m_metadata; // the original metadata should stay unchanged

            // Update the metadata with the new rotation angles.
            const auto spline = CubicSplineGrid<f64, 1>(rotation_offsets.ssize(), 1, rotation_offsets.data());
            for (auto& slice: m_updated_metadata) {
                const f64 normalized_tilt = to_normalized_tilt_(slice.angles[1]);
                const f64 rotation_offset = spline.interpolate(normalized_tilt);
                slice.angles[0] = MetadataSlice::to_angle_range(slice.angles[0] + rotation_offset);
            }

            // Projection-matching of the entire stack.
            for (i64 target_index = 1; target_index < m_updated_metadata.ssize(); ++target_index) {
                const MetadataSlice& new_reference_slice = m_updated_metadata[target_index - 1];
                MetadataSlice& target_slice = m_updated_metadata[target_index];

                const auto [score, shift_offset] = m_projector->project_and_correlate_next(
                        m_stack, new_reference_slice, target_slice,
                        *m_common_area, *m_parameters
                );

                // Align the target slice. This will be the new reference at the next iteration.
                target_slice.shifts += shift_offset;

                // Collect the score of the slice-alignment.
                m_scores.emplace_back(score);

//                qn::Logger::trace("{:>02}: tilt={} score={:.8f}, shift_offset={::.8f}",
//                                  target_index, target_slice.angles[1], score, shift_offset);
            }

            ++m_n_evaluations;
            return std::accumulate(m_scores.begin(), m_scores.end(), f64{0}) / static_cast<f64>(m_scores.size());
        }

        static auto maximization_function(
                u32 n_parameters, const f64* parameters, f64* gradients, void* instance
        ) -> f64 {
            noa::Timer timer;
            timer.start();

            QN_CHECK(n_parameters <= 3 && parameters && instance, "Invalid parameters");
            auto& self = *static_cast<Fitter*>(instance);
            auto& memoizer = self.memoizer_();
            const auto original_parameters = Span<const f64>(parameters, n_parameters);

            // Memoization.
            std::optional<f64> memoized_cost = memoizer.find(parameters, gradients, 1e-6);
            if (memoized_cost.has_value()) {
                f64 cost = memoized_cost.value();
                qn::Logger::trace("rotation_offsets={:.6f}, cost={:.4f}, elapsed={:.2f}ms, memoized=true",
                                  fmt::join(original_parameters, ","), cost, timer.elapsed());
                return cost;
            }

            // Save a copy of the parameters.
            std::array<f64, 3> buffer{};
            const auto copied_parameters = Span<f64>(buffer.data(), n_parameters);
            for (u32 i = 0; i < n_parameters; ++i)
                copied_parameters[i] = original_parameters[i];

            const f64 cost = self.align(copied_parameters);

            if (gradients != nullptr) {
                constexpr f64 DELTA = 0.2; // in degrees // FIXME try 0.1 or scale gradient
                for (i64 i: noa::irange(n_parameters)) {
                    f64 original_value = copied_parameters[i];

                    copied_parameters[i] += DELTA;
                    const f64 cost_plus_delta = self.align(copied_parameters);

                    copied_parameters[i] = original_value - DELTA;
                    const f64 cost_minus_delta = self.align(copied_parameters);

                    copied_parameters[i] = original_value;
                    const auto gradient = CentralFiniteDifference::get(cost_minus_delta, cost_plus_delta, DELTA);
                    gradients[i] = gradient;
                    qn::Logger::trace("gradient[{}]={} (minus={}, plus={})",
                                      i, gradient, cost_minus_delta, cost_plus_delta);
                }
            }

            qn::Logger::trace("rotation_offsets={:.6f}, cost={:.4f}, elapsed={:.2f}ms",
                              fmt::join(original_parameters, ","), cost, timer.elapsed());
            memoizer.record(parameters, cost, gradients);
            return cost;
        }
    };
}

namespace qn {
    ProjectionMatching::ProjectionMatching(
            i64 n_slices,
            const Shape2<i64>& shape,
            Device compute_device,
            Allocator allocator
    ) {
        const auto options = ArrayOption(compute_device, allocator);
        m_two_slices = noa::memory::empty<f32>(shape.push_front<2>({2, 1}), options);

        n_slices += 3; // +target, +projected-reference x2
        const i64 size_padded = noa::fft::next_fast_size(noa::math::max(shape) * 2);
        const auto slice_padded_shape = Shape4<i64>{n_slices, 1, size_padded, size_padded};
        m_slices_padded_rfft = noa::memory::empty<c32>(slice_padded_shape.rfft(), options);
        m_weights_padded_rfft = noa::memory::like<f32>(m_slices_padded_rfft);

        const auto total_bytes = static_cast<f64>(
                m_two_slices.size() * sizeof(f32) +
                m_slices_padded_rfft.size() * sizeof(c32) +
                m_weights_padded_rfft.size() * sizeof(f32));
        qn::Logger::info("Projection-matching allocated {:.2f} GB on device={}", total_bytes * 1e-9, compute_device);
    }

    void ProjectionMatching::update(
            // inputs
            const View<f32>& stack,
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters,
            const CTFAnisotropic64& average_ctf,

            // input/output
            MetadataStack& metadata
    ) {
        qn::Logger::info("Projection-matching alignment...");
        noa::Timer timer;
        timer.start();

        // Order for projection matching.
        auto projection_metadata = metadata;
        projection_metadata.sort("exposure");

        // Helpers for the alignment.
        auto projector = Projector(m_slices_padded_rfft.view(), m_weights_padded_rfft.view(), m_two_slices.view(), average_ctf);
        auto fitter = Fitter(stack, projector, projection_metadata, common_area, parameters);

        QN_CHECK(parameters.rotation_resolution <= 3, "Invalid rotation resolution");
        std::array<f64, 3> buffer{};
        const auto rotation_offsets = Span<f64>(buffer.data(), parameters.rotation_resolution);

        // Optimize the rotation by maximizing the score from the
        // projection-matching of every slice in the stack.
        if (parameters.rotation_range > 1e-2) {
            auto optimizer = Optimizer(
                    NLOPT_LN_SBPLX, //parameters.rotation_resolution > 1 ? NLOPT_LD_LBFGS : NLOPT_LN_SBPLX,
                    parameters.rotation_resolution);
            optimizer.set_max_objective(Fitter::maximization_function, &fitter);
            optimizer.set_bounds(-parameters.rotation_range, parameters.rotation_range);
            optimizer.set_initial_step(parameters.rotation_range * 0.2);
            optimizer.set_x_tolerance_abs(0.05);
            optimizer.set_fx_tolerance_abs(5e-5);
            optimizer.optimize(rotation_offsets.data());
        }

        // The optimizer can do some post-processing on the parameters, so the last projection matching
        // might not have been run on these final parameters. As such, do a final pass with the actual
        // best rotation offset returned by the optimizer.
        fitter.align(rotation_offsets);

        // Update the original metadata with the aligned one.
        metadata.update_from(
                fitter.aligned_metadata(),
                /*update_angles=*/ true,
                /*update_shifts=*/ true,
                /*update_defocus=*/false);

        qn::Logger::info("rotation_offset={}", metadata[0].angles[0]);
        qn::Logger::info("Projection-matching alignment... done. Took {:.3f}s (n_evaluations={})",
                          timer.elapsed() * 1e-3, fitter.n_evaluations());
    }
}
