#include <noa/FFT.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/core/Ewise.hpp"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/ProjectionMatching.hpp"
#include "quinoa/io/Logging.h"

namespace {
    using namespace qn;

    // Compute the projected-references and shift-aligns it to the corresponding target.
    // The reference used to compute the first projected-reference is added to the lists
    // of references for the next projected-reference.
    class Projector {
    private:
        // Layout: [reference 0, ..., reference n, target, projected_reference]
        i64 m_n_references{0};
        View<c32> m_slices_padded_rfft;
        View<f32> m_weights_padded_rfft;
        View<f32> m_two_slices;

        CTFAnisotropic64 m_average_ctf;
        Array<f32> m_two_values;
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
            m_two_values = noa::memory::empty<f32>({2, 1, 1, 1}, options);
            m_insertion_inv_rotations = noa::memory::empty<Quaternion<f32>>({slices_padded_rfft.shape()[0], 1, 1, 1}, options);
            m_peak_window = noa::memory::empty<f32>({1,1,128,128}, {device, Allocator::DEFAULT_ASYNC});
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
            increment_reference_count_();
            preprocess_(stack, reference_metadata, target_metadata, common_area, parameters);
            fourier_insert_and_extract_(reference_metadata, target_metadata, parameters);
            postprocessing_(target_metadata, parameters);
            return cross_correlate_(target_metadata, common_area, parameters);
        }

    private:
        constexpr void increment_reference_count_() noexcept { m_n_references += 1; }
        [[nodiscard]] constexpr auto n_references_() const noexcept -> i64 { return m_n_references; }
        [[nodiscard]] constexpr auto reference_index_() const noexcept -> i64 { return m_n_references - 1; }
        [[nodiscard]] constexpr auto target_index_() const noexcept -> i64 { return m_n_references; }
        [[nodiscard]] constexpr auto projected_index_() const noexcept -> i64 { return m_n_references + 1; }

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

        [[nodiscard]] auto reference_and_target_ctfs_(
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata
        ) const -> std::array<CTFAnisotropic64, 2> {
            // This is the sampling function, and it contains:
            //  - The CTF set to the defocus of the slice. The defocus gradient should be negligible
            //    thanks to the common-area mask.
            //  - Approximated exposure filter using the CTF B-factor.
            //  - Approximated relative-SNR from electron mean-free-path, using cos(tilt) as scaling factor.
            const auto [_, defocus_astig, defocus_angle] = m_average_ctf.defocus();
            std::array<CTFAnisotropic64, 2> ctfs;

            ctfs[0] = m_average_ctf;
            ctfs[0].set_defocus({reference_metadata.defocus, defocus_astig, defocus_angle});
            ctfs[0].set_bfactor(-reference_metadata.exposure[1] * 2);
            ctfs[0].set_scale(noa::math::cos(noa::math::deg2rad(reference_metadata.angles[1])));

            ctfs[1] = m_average_ctf;
            ctfs[1].set_defocus({target_metadata.defocus, defocus_astig, defocus_angle});
            ctfs[1].set_bfactor(-target_metadata.exposure[1] * 2);
            ctfs[1].set_scale(noa::math::cos(noa::math::deg2rad(target_metadata.angles[1])));

            return ctfs;
        }

        void preprocess_(
                const View<const f32>& stack,
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) {
            noa::Timer timer;
            timer.start();
            const auto reference_and_target_slice = noa::indexing::Slice{reference_index_(), target_index_() + 1};

            // Copy from stack and apply the common-area mask.
            const View<f32> reference_and_target = m_two_slices;
            common_area.mask_view(
                    stack.subregion(reference_metadata.index),
                    reference_and_target.subregion(0),
                    reference_metadata,
                    parameters.smooth_edge_percent);
            common_area.mask_view( // TODO save the mask for the cross-correlation?
                    stack.subregion(target_metadata.index),
                    reference_and_target.subregion(1),
                    target_metadata,
                    parameters.smooth_edge_percent);

            // Extract necessary views.
            const View<c32> reference_and_target_padded_rfft = m_slices_padded_rfft
                    .subregion(reference_and_target_slice);
            const View<f32> reference_and_target_padded = noa::fft::alias_to_real(
                    reference_and_target_padded_rfft, shape_padded_().set<0>(2));

            // Zero-(right-)pad and in-place rfft.
            noa::memory::resize(reference_and_target, reference_and_target_padded, {}, border_right_());
            if (!parameters.debug_directory.empty()) {
                noa::io::save(reference_and_target_padded,
                              parameters.debug_directory /
                              fmt::format("reference_and_target_padded_{:>02}.mrc", target_metadata.index));
            }
            noa::fft::r2c(reference_and_target_padded, reference_and_target_padded_rfft);

            // Compute and apply the sampling functions of both the reference and target.
            // The current sampling function simply contains the exposure filter and the average CTF of the slice.
            // The CTF at the tilt-axis (this is still per-slice) should be a first good approximation for the
            // sampling function. The field-of-view is restrained to the common-area, so it should be good even
            // for tilted slices. Also note that the CTF is multiplied once with the slice, but since the microscope
            // already multiplies the specimen by the CTF once, the sampling function ends up being CTF^2.
            const auto ctfs = reference_and_target_ctfs_(reference_metadata, target_metadata);
            noa::memory::fill(m_weights_padded_rfft.subregion(reference_and_target_slice), 1.f);
//            noa::signal::fft::ctf_anisotropic<noa::fft::H2H>( // weights
//                    m_weights_padded_rfft.subregion(reference_and_target_slice),
//                    shape_padded_().set<0>(2), View(ctfs.data(), 2),
//                    /*ctf_abs=*/ false, /*ctf_square=*/ true
//            );
//            noa::signal::fft::ctf_anisotropic<noa::fft::H2H>( // data
//                    reference_and_target_padded_rfft, reference_and_target_padded_rfft,
//                    shape_padded_().set<0>(2), View(ctfs.data(), 2)
//            );

            // Pre-processing for Fourier insertion.
            // 1. Remap/fftshift, since the Fourier insertion requires a centered input.
            // 2. Phase-shift the rotation-center to the origin. Note that the right-padding doesn't change the center.
            const View<c32> reference_padded_rfft = reference_and_target_padded_rfft.subregion(0);
            noa::fft::remap(noa::fft::H2HC, reference_padded_rfft, reference_padded_rfft, shape_padded_());
            noa::signal::fft::phase_shift_2d<noa::fft::HC2HC>(
                    reference_padded_rfft, reference_padded_rfft, shape_padded_(),
                    (-center_original_() - reference_metadata.shifts).as<f32>());

            reference_padded_rfft.eval();
            fmt::print("preprocessing took {}ms\n", timer.elapsed());
        }

        void fourier_insert_and_extract_(
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata,
                const ProjectionMatchingParameters& parameters
        ) {
            noa::Timer timer;
            timer.start();
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
            noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                    m_slices_padded_rfft.subregion(references_slice),
                    m_weights_padded_rfft.subregion(references_slice), slices_padded_shape,
                    m_slices_padded_rfft.subregion(projected_index_()),
                    m_weights_padded_rfft.subregion(projected_index_()), shape_padded_(),
                    Float22{}, m_insertion_inv_rotations.subregion(references_slice),
                    Float22{}, extraction_fwd_rotation,
                    i_windowed_sinc, o_windowed_sinc,
                    /*add_to_output=*/ false,
                    /*correct_multiplicity=*/ true);

            m_weights_padded_rfft.eval();
            fmt::print("projection took {}ms\n", timer.elapsed());
        }

        void postprocessing_(
                const MetadataSlice& target_metadata,
                const ProjectionMatchingParameters& parameters
        ) {
            noa::Timer timer;
            timer.start();

            // We have first the target and then the projected-reference.
            const auto target_and_projected_slice = noa::indexing::Slice{target_index_(), projected_index_() + 1};
            const View<c32> target_and_projected_padded_rfft = m_slices_padded_rfft
                    .subregion(target_and_projected_slice);
            const View<f32> target_and_projected_weights_padded_rfft = m_weights_padded_rfft
                    .subregion(target_and_projected_slice);

            // Applying the weights of the projected-reference onto the target and vice versa.
            const auto target_padded_rfft = target_and_projected_padded_rfft.subregion(0);
            const auto projected_padded_rfft = target_and_projected_padded_rfft.subregion(1);

            noa::ewise_binary( // target * projected_weights
                    target_padded_rfft,
                    target_and_projected_weights_padded_rfft.subregion(1),
                    target_padded_rfft, noa::multiply_t{});
            noa::ewise_binary( // projected * target_weights
                    projected_padded_rfft,
                    target_and_projected_weights_padded_rfft.subregion(0),
                    projected_padded_rfft, noa::multiply_t{});

            // Center projection back and shift to the target reference-frame.
            noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                    projected_padded_rfft, projected_padded_rfft, shape_padded_(),
                    (center_original_() + target_metadata.shifts).as<f32>());

            noa::signal::fft::bandpass<noa::fft::H2H>(
                    target_and_projected_padded_rfft, target_and_projected_padded_rfft,
                    shape_padded_().set<0>(2),
                    parameters.highpass_filter[0], parameters.lowpass_filter[0],
                    parameters.highpass_filter[1], parameters.lowpass_filter[1]);

            // Go back to real-space and crop.
            const View<f32> target_and_projected_padded = noa::fft::alias_to_real(
                    target_and_projected_padded_rfft, shape_padded_().set<0>(2));
            noa::fft::c2r(target_and_projected_padded_rfft, target_and_projected_padded);
            noa::memory::resize(target_and_projected_padded, m_two_slices, {}, -border_right_());

            m_two_slices.eval();
            fmt::print("postprocessing took {}ms\n", timer.elapsed());
        }

        auto cross_correlate_(
                const MetadataSlice& target_metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) -> std::pair<f64, Vec2<f64>> {
            noa::Timer timer;
            timer.start();
            const auto target_and_projected_slice = noa::indexing::Slice{target_index_(), projected_index_() + 1};

            // 1. Compute the mask.
            // To save memory, reuse the weights to store the mask.
            const auto mask = m_weights_padded_rfft
                    .subregion(target_and_projected_slice).flat(/*axis=*/ 0)
                    .subregion(noa::indexing::Slice{0, shape_original_().elements()})
                    .reshape(shape_original_());
            common_area.compute(mask, target_metadata, parameters.smooth_edge_percent);

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
            const auto xmap = m_weights_padded_rfft
                    .subregion(target_and_projected_slice).flat(/*axis=*/ 0)
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

            fmt::print("cross-correlation took {}ms\n", timer.elapsed());

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

        std::vector<f64> m_scores;
        Memoizer m_memoizer;

    public:
        Fitter(
                const View<const f32>& stack,
                Projector& projector,
                const MetadataStack& metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) :
                m_stack(stack),
                m_projector(&projector),
                m_metadata(&metadata),
                m_common_area(&common_area),
                m_parameters(&parameters)
        {
            m_scores.reserve(metadata.size());
            m_memoizer = Memoizer(/*n_parameters=*/ 1, /*resolution=*/ 6);
        }

        auto memoizer() -> Memoizer& { return m_memoizer; }

        auto align(const f64 rotation_offset) -> std::pair<f64, MetadataStack> {
            m_scores.clear();
            m_projector->reset();

            // Take a copy of the metadata, because the projection-matching updates it and
            // 1) we actually don't care about the shift updates from the projection-matching at this point,
            // 2) we want to use the same metadata for every evaluation.
            auto updated_metadata = *m_metadata;
            updated_metadata.sort("exposure");
            updated_metadata.add_global_angles({rotation_offset, 0, 0});

            qn::Logger::trace("Computing cost with rotation_offset={:.8f}", rotation_offset);
            for (i64 target_index = 1; target_index < static_cast<i64>(updated_metadata.size()); ++target_index) {

                MetadataSlice& new_reference_slice = updated_metadata[target_index - 1];
                MetadataSlice& target_slice = updated_metadata[target_index];

                const auto [score, shift_offset] = m_projector->project_and_correlate_next(
                        m_stack, new_reference_slice, target_slice,
                        *m_common_area, *m_parameters
                );

                // Align the target slice. This will be the new reference at the next iteration.
                target_slice.shifts += shift_offset;

                // Collect the score of the slice-alignment.
                m_scores.push_back(score);

                qn::Logger::trace("{:>02}: score={:.8f}, shift_offset={::.8f}",
                                  target_index, score, shift_offset);
            }

            const auto final_score =
                    std::accumulate(m_scores.begin(), m_scores.end(), f64{0}) /
                    static_cast<f64>(m_scores.size());
            qn::Logger::trace("final_score={:.8f}", final_score);
            return {final_score, updated_metadata};
        }

        auto cost(const f64 rotation_offset) -> f64 {
            auto [score, _] = align(rotation_offset);
            return score;
        }

        static auto maximization_function(
                u32 n_parameters, [[maybe_unused]] const f64* parameters,
                f64* gradients, void* instance
        ) -> f64 {
            noa::Timer timer;
            timer.start();

            QN_CHECK(n_parameters == 1 && parameters && instance, "Invalid parameters");
            auto& self = *static_cast<Fitter*>(instance);
            const f64 rotation_offset = parameters[0];

            // Memoization.
            std::optional<f64> memoized_cost = self.memoizer().find(parameters, gradients, 1e-8);
            if (memoized_cost.has_value()) {
                f64 cost = memoized_cost.value();
                qn::Logger::trace("cost={:.4f}, elapsed={:.2f}ms, memoized=true", cost, timer.elapsed());
                return cost;
            }

            const f64 cost = self.cost(rotation_offset);
            f64 gradient{};
            if (gradients != nullptr) {
                constexpr f64 DELTA = 0.05; // in degrees
                const f64 cost_minus_delta = self.cost(rotation_offset - DELTA);
                const f64 cost_plus_delta = self.cost(rotation_offset + DELTA);
                gradient = CentralFiniteDifference::get(cost_minus_delta, cost_plus_delta, DELTA);
                *gradients = gradient;
            }

            qn::Logger::trace("cost={:.4f}, gradient={}, elapsed={:.2f}ms", cost, gradient, timer.elapsed());
            self.memoizer().record(parameters, cost, gradients);
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

        n_slices += 2; // +target, +projected-reference
        const i64 size_padded = noa::fft::next_fast_size(noa::math::max(shape) * 2);
        const auto slice_padded_shape = Shape4<i64>{n_slices, 1, size_padded, size_padded};
        m_slices_padded_rfft = noa::memory::empty<c32>(slice_padded_shape.rfft(), options);
        m_weights_padded_rfft = noa::memory::like<f32>(m_slices_padded_rfft);

        const auto total_bytes = static_cast<f64>(
                m_two_slices.size() * sizeof(f32) +
                m_slices_padded_rfft.size() * sizeof(c32) +
                m_weights_padded_rfft.size() * sizeof(f32));
        qn::Logger::info("Projection-matching allocated {:.2f} GB", total_bytes * 1e-9);
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

        auto projector = Projector(
                m_slices_padded_rfft.view(),
                m_weights_padded_rfft.view(),
                m_two_slices.view(),
                average_ctf);
        auto fitter = Fitter(stack, projector, metadata, common_area, parameters);

        // Optimizer.
        const Optimizer optimizer(parameters.use_estimated_gradients ? NLOPT_LD_LBFGS : NLOPT_LN_SBPLX, 1);
        optimizer.set_max_objective(Fitter::maximization_function, &fitter);
        optimizer.set_bounds(-5, 5); // at this point, this should be way more than enough
        optimizer.set_initial_step(0.5);

        // Optimize the rotation model by maximizing the score from the projection-matching of every slice in the stack.
        f64 rotation_offset{0};
//        optimizer.optimize(&rotation_offset);

        // The optimizer can do some post-processing on the parameters, so the last projection matching
        // might not have been run on these final parameters. As such, do a final pass with the actual
        // best parameters returned by the optimizer.
        const auto [final_score, new_metadata] = fitter.align(rotation_offset);
        metadata.update_from(new_metadata, true, true, true); // contains the new rotation and the new shifts

        qn::Logger::trace("Projection-matching alignment... done. Took {:.3f}ms ({} evaluations)",
                          timer.elapsed(), optimizer.number_of_evaluations() + 1);
    }
}
