#include <noa/FFT.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/core/Ewise.hpp"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/ProjectionMatching.hpp"
#include "quinoa/io/Logging.h"

namespace {
    using namespace qn;

    // Utility class for the projection-matching optimization.
    class Fitter {
    private:
        View<const f32> m_stack;
        const ProjectionMatching* m_projector;
        const MetadataStack* m_metadata;
        const CommonArea* m_common_area;
        const ProjectionMatchingParameters* m_parameters;
        CTFAnisotropic64 m_average_ctf;

        std::vector<f64> m_scores;
        Memoizer m_memoizer;

    public:
        Fitter(
                const View<const f32>& stack,
                const ProjectionMatching& projector,
                const MetadataStack& metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters,
                const CTFAnisotropic64& average_ctf
        ) :
                m_stack(stack),
                m_projector(&projector),
                m_metadata(&metadata),
                m_common_area(&common_area),
                m_parameters(&parameters),
                m_average_ctf(average_ctf) {
            m_scores.reserve(metadata.size());
            m_memoizer = Memoizer(/*n_parameters=*/1, /*resolution=*/6);
        }

        auto memoizer() -> Memoizer& { return m_memoizer; }

        auto align(const f64 rotation_offset) -> std::pair<f64, MetadataStack> {
            m_scores.clear();
            m_projector->reset_();

            // Take a copy of the metadata, because the projection-matching updates it and
            // 1) we actually don't care about the shift updates from the projection-matching at this point,
            // 2) we want to use the same metadata for every evaluation.
            auto updated_metadata = *m_metadata;
            updated_metadata.add_global_angles({rotation_offset, 0, 0});

            qn::Logger::trace("Computing cost with rotation_offset={:.8f}");
            for (i64 target_index = 1; target_index < static_cast<i64>(updated_metadata.size()); ++target_index) {

                MetadataSlice& reference_slice = updated_metadata[target_index - 1];
                MetadataSlice& target_slice = updated_metadata[target_index];

                const auto [shift_offset, score] = m_projector->align_next_slice_(
                        m_stack, reference_slice, target_slice,
                        *m_common_area, *m_parameters
                );

                // Align the target slice. At the next iteration, it is added to the list of references.
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
                constexpr f64 DELTA = 0.0125; // in degrees
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
            const Shape2<i64>& shape,
            Device compute_device,
            Allocator allocator
    ) {
        i64 size_padded = noa::fft::next_fast_size(noa::math::max(shape) * 2);
        const auto slice_shape = shape.push_front<2>({1, 1});
        const auto slice_padded_shape = Shape4<i64>{1, 1, size_padded, size_padded};
        const auto options = ArrayOption(compute_device, allocator);

        m_reference_and_target = noa::memory::empty<f32>(slice_shape.set<0>(2), options);
        m_reference_and_target_rfft = noa::memory::empty<c32>(slice_shape.set<0>(2).rfft(), options);
        m_reference_and_target_padded_rfft = noa::memory::empty<c32>(slice_padded_shape.set<0>(2).rfft(), options);
        m_reference_and_target_weights_padded_rfft = noa::memory::empty<f32>(slice_shape.set<0>(2).rfft(), options);
        m_projected_reference_padded_rfft = noa::memory::empty<c32>(slice_padded_shape.rfft(), options);
        m_projected_weights_padded_rfft = noa::memory::empty<f32>(slice_padded_shape.rfft(), options);
        m_projected_multiplicity_padded_rfft = noa::memory::empty<f32>(slice_padded_shape.rfft(), options);

        const auto total_bytes = static_cast<f64>(
                slice_padded_shape.set<0>(8).rfft().elements() *
                slice_shape.set<0>(6).rfft().elements() *
                static_cast<i64>(sizeof(f32)));
        qn::Logger::info("Projection-matching allocated ~{} GB", total_bytes * 1e-9);
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

        auto fitter = Fitter(stack, *this, metadata, common_area, parameters, average_ctf);

        // Optimizer.
        const Optimizer optimizer(parameters.use_estimated_gradients ? NLOPT_LD_LBFGS : NLOPT_LN_SBPLX, 1);
        optimizer.set_max_objective(Fitter::maximization_function, &fitter);
        optimizer.set_bounds(-5, 5); // at this point, this should be way more than enough
        optimizer.set_initial_step(0.5);

        // Optimize the rotation model by maximizing the score from the projection-matching of every slice in the stack.
        f64 rotation_offset{0};
        optimizer.optimize(&rotation_offset);

        // The optimizer can do some post-processing on the parameters, so the last projection matching
        // might not have been run on these final parameters. As such, do a final pass with the actual
        // best parameters returned by the optimizer.
        const auto [final_score, new_metadata] = fitter.align(rotation_offset);
        metadata = new_metadata; // contains the new rotation and the new shifts

        qn::Logger::trace("Projection-matching alignment... done. Took {}ms ({} evaluations)",
                          timer.elapsed(), optimizer.number_of_evaluations() + 1);
    }

    // Resets the projected data. Required to start a new projection-matching run.
    void ProjectionMatching::reset_() const {
        noa::memory::fill(m_projected_reference_padded_rfft, c32{0});
        noa::memory::fill(m_projected_weights_padded_rfft, f32{0});
        noa::memory::fill(m_projected_multiplicity_padded_rfft, f32{0});
    }

    // Computes the projected-reference for the current target. Then cross-correlates the projected-reference
    // with the target and returns the peak score and shift. It assumes that the number and order of the slices
    // in the metadata is left unchanged until reset() is called.
    auto ProjectionMatching::align_next_slice_(
            const View<const f32>& stack,
            const MetadataSlice& reference_metadata,
            const MetadataSlice& target_metadata,
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters
    ) const -> std::pair<Vec2<f64>, f64> {
        compute_next_projection_(
                stack, reference_metadata, target_metadata,
                common_area, parameters);
        return cross_correlate_(target_metadata, parameters);
    }

    void ProjectionMatching::compute_next_projection_(
            const View<const f32>& stack,
            const MetadataSlice& reference_metadata,
            const MetadataSlice& target_metadata,
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters
    ) const {
        // Copy from stack and apply the common-area mask.
        const View<f32> reference_and_target = m_reference_and_target.view();
        common_area.mask_view(
                stack.subregion(reference_metadata.index),
                reference_and_target.subregion(0),
                reference_metadata,
                parameters.smooth_edge_percent);
        common_area.mask_view(
                stack.subregion(target_metadata.index),
                reference_and_target.subregion(1),
                target_metadata,
                parameters.smooth_edge_percent);

        // Get the view where the new reference and target should go.
        const View<c32> reference_and_target_padded_rfft = m_reference_and_target_padded_rfft.view();
        const View<c32> reference_padded_rfft = reference_and_target_padded_rfft.subregion(0);
        const View<c32> target_padded_rfft = reference_and_target_padded_rfft.subregion(1);
        const View<f32> reference_and_target_padded = noa::fft::alias_to_real(
                reference_and_target_padded_rfft, shape_padded_().push_front<2>({2, 1}));

        // Zero-pad and rfft.
        noa::memory::resize(reference_and_target, reference_and_target_padded);
        if (!parameters.debug_directory.empty()) {
            noa::io::save(reference_and_target_padded,
                          parameters.debug_directory /
                          fmt::format("reference_and_target_padded_{:>02}.mrc", reference_metadata.index));
        }
        noa::fft::r2c(reference_and_target_padded, reference_and_target_padded_rfft);

        // -- Sampling functions --
        const auto reference_and_target_weights_padded_rfft = m_reference_and_target_weights_padded_rfft.view();
        const auto reference_weights_padded_rfft = reference_and_target_weights_padded_rfft.subregion(0);
        const auto target_weights_padded_rfft = reference_and_target_weights_padded_rfft.subregion(1);
        compute_and_apply_weights_(
                reference_and_target_padded_rfft,
                reference_and_target_weights_padded_rfft,
                reference_metadata, target_metadata,
                reference_and_target_padded.shape()
        );

        // Remap the reference for the Fourier insertion.
        const auto slice_padded_shape = shape_padded_().push_front<2>({1, 1});
        const auto slice_padded_center = MetadataSlice::center<f64>(slice_padded_shape);
        noa::fft::remap(noa::fft::H2HC, reference_padded_rfft,
                        reference_padded_rfft, slice_padded_shape);

        // Phase-shift the rotation centre of the reference for Fourier insertion.
        noa::signal::fft::phase_shift_2d<noa::fft::HC2HC>(
                reference_padded_rfft, reference_padded_rfft, slice_padded_shape,
                -(slice_padded_center + reference_metadata.shifts).as<f32>());

        // -- Rotations --
        // The rotation is the CCW angle of the tilt-axis in the slices. For the projection, we want to align
        // the tilt-axis along the Y axis, so subtract this angle and then apply the tilt and elevation.
        // For the Fourier insertion, noa needs the inverse rotation matrix, hence the transposition.
        const Vec3<f64> insertion_angles = noa::math::deg2rad(reference_metadata.angles);
        const Vec3<f64> target_angles = noa::math::deg2rad(target_metadata.angles);
        const Float33 insertion_inv_rotation = noa::geometry::euler2matrix(
                Vec3<f64>{-insertion_angles[0], insertion_angles[1], insertion_angles[2]},
                "zyx", /*intrinsic=*/ false).transpose().as<f32>();
        const Float33 extraction_fwd_rotation = noa::geometry::euler2matrix(
                Vec3<f64>{-target_angles[0], target_angles[1], target_angles[2]},
                "zyx", /*intrinsic=*/ false).as<f32>();

        // -- Fourier-insertion --
        using WindowedSinc = noa::geometry::fft::WindowedSinc;
        const auto i_windowed_sinc = WindowedSinc{
            static_cast<f32>(parameters.fftfreq_sinc),
            static_cast<f32>(parameters.fftfreq_blackman)};
        const auto o_windowed_sinc = WindowedSinc{
            static_cast<f32>(parameters.fftfreq_z_sinc),
            static_cast<f32>(parameters.fftfreq_z_blackman)};
        const View<c32> projected_reference_padded_rfft = m_projected_reference_padded_rfft.view();
        const View<f32> projected_weights_padded_rfft = m_projected_weights_padded_rfft.view();
        const View<f32> projected_multiplicity_padded_rfft = m_projected_multiplicity_padded_rfft.view();

        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                reference_padded_rfft, slice_padded_shape,
                projected_reference_padded_rfft, slice_padded_shape,
                Float22{}, insertion_inv_rotation,
                Float22{}, extraction_fwd_rotation,
                i_windowed_sinc, o_windowed_sinc,
                /*add_to_output=*/ true);
        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                reference_weights_padded_rfft, slice_padded_shape,
                projected_weights_padded_rfft, slice_padded_shape,
                Float22{}, insertion_inv_rotation,
                Float22{}, extraction_fwd_rotation,
                i_windowed_sinc, o_windowed_sinc,
                /*add_to_output=*/ true);
        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                1.f, slice_padded_shape,
                projected_multiplicity_padded_rfft, slice_padded_shape,
                Float22{}, insertion_inv_rotation,
                Float22{}, extraction_fwd_rotation,
                i_windowed_sinc, o_windowed_sinc,
                /*add_to_output=*/ true);

        // Correct for the multiplicity and prepare for the CC by applying the weights
        // of the projected-reference to the target and vice versa.
        noa::ewise_trinary(
                projected_reference_padded_rfft, projected_multiplicity_padded_rfft, target_weights_padded_rfft,
                reference_padded_rfft, correct_multiplicity_and_multiply_t{});
        noa::ewise_trinary(
                projected_weights_padded_rfft, projected_multiplicity_padded_rfft, target_padded_rfft,
                target_padded_rfft, correct_multiplicity_and_multiply_t{});

        // Center projection back and shift onto the target.
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                reference_padded_rfft, reference_padded_rfft, slice_padded_shape,
                (slice_padded_center + target_metadata.shifts).as<f32>());

        // -- Go back to real-space --
        noa::fft::c2r(reference_and_target_padded_rfft, reference_and_target_padded);
        noa::memory::resize(reference_and_target_padded, reference_and_target);

        // Apply the common-area mask again just to remove small artefacts from the Fourier-space
        // transformations. Note that here both the projected-reference and target are on the "target
        // reference frame".
        common_area.mask_view(
                reference_and_target.subregion(0), reference_and_target.subregion(0),
                target_metadata, parameters.smooth_edge_percent);
        common_area.mask_view(
                reference_and_target.subregion(1), reference_and_target.subregion(1),
                target_metadata, parameters.smooth_edge_percent);
    }

    // Compute the sampling function of both the new reference view and the target.
    // The current sampling function simply contains the exposure filter and the average CTF
    // of the slice. The average CTF is not ideal (it's not local), but it should be a first
    // good approximation, especially given that the field-of-view is restrained to the common-
    // area. Also note that the CTF is multiplied once with the slice, but since the microscope
    // already multiplies by the CTF, the sampling function ends up with CTF^2.
    void ProjectionMatching::compute_and_apply_weights_(
            const View<c32>& reference_and_target_rfft,
            const View<f32>& reference_and_target_weights_rfft,
            const MetadataSlice& reference_metadata,
            const MetadataSlice& target_metadata,
            const Shape4<i64>& slice_shape
    ) const {
        // This is the sampling function, and it contains:
        //  - The CTF set to the defocus of the slice. The defocus gradient should be negligible
        //    thanks to the common-area mask.
        //  - Approximation exposure filter using the CTF B-factor.
        //  - Approximation of the SNR from electron mean-free-path, using cos(tilt) as scaling factor.

        // TODO Is it worth having an isotropic version?
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

        // The data is already CTF multiplied by the microscope, so the weights should be ctf^2.
        noa::signal::fft::ctf_anisotropic<noa::fft::H2H>(
                reference_and_target_weights_rfft, reference_and_target_weights_rfft,
                slice_shape, View(ctfs.data(), 2), /*ctf_abs=*/ false, /*ctf_square=*/ true
        );
        noa::signal::fft::ctf_anisotropic<noa::fft::H2H>(
                reference_and_target_rfft, reference_and_target_rfft,
                slice_shape, View(ctfs.data(), 2)
        );
    }

    auto ProjectionMatching::cross_correlate_(
            const MetadataSlice& target_metadata,
            const ProjectionMatchingParameters& parameters
    ) const -> std::pair<Vec2<f64>, f64> {
        const auto reference_and_target = m_reference_and_target.view();
        const auto reference = reference_and_target.subregion(0);
        const auto target = reference_and_target.subregion(1);

        // Normalize so we can compare the peaks between iterations.
        // The normalization is not explicitly within a mask, but it is effectively within the mask
        // since everything outside the mask is 0, and we only compute the norm and min/max.
        noa::math::normalize_per_batch(reference_and_target, reference_and_target);
        const f32 energy_target = noa::math::sqrt(noa::math::sum(target, noa::abs_squared_t{})); // TODO norm?
        const f32 energy_reference = noa::math::sqrt(noa::math::sum(reference, noa::abs_squared_t{}));

        if (!parameters.debug_directory.empty()) {
            noa::io::save(reference_and_target,
                          parameters.debug_directory /
                          noa::string::format("reference_and_target_{:>02}.mrc", target_metadata.index));
        }

        const auto reference_and_target_rfft = m_reference_and_target_rfft.view();
        noa::fft::r2c(reference_and_target, reference_and_target_rfft); // TODO Norm::NONE?
        noa::signal::fft::bandpass<noa::fft::H2H>(
                reference_and_target_rfft, reference_and_target_rfft, reference_and_target.shape(),
                parameters.highpass_filter[0], parameters.lowpass_filter[0],
                parameters.highpass_filter[1], parameters.lowpass_filter[1]);

        // We rotate the xmap before the picking, so compute the centered xmap.
        const auto xmap = reference; // reference data is not used anymore, so recycle it
        noa::signal::fft::xmap<noa::fft::H2FC>(
                reference_and_target_rfft.subregion(1),
                reference_and_target_rfft.subregion(0),
                xmap);

        if (!parameters.debug_directory.empty()) {
            noa::io::save(xmap,
                          parameters.debug_directory /
                          fmt::format("xmap_{:>02}.mrc", target_metadata.index));
        }

        // Note that the center of the cross-correlation is fixed at N//2 (integral division), so don't use
        // MetadataSlice::center in case we switch of convention one day and use N/2 (floating-point division).
        const auto xmap_center = (xmap.shape().vec().filter(2, 3) / 2).as<f64>();

        // TODO Better fitting of the peak. 2D parabola?
        auto [peak_coordinate, peak_value] = noa::signal::fft::xpeak_2d<noa::fft::FC2FC>(xmap);
        const Vec2<f64> shift = peak_coordinate.as<f64>() - xmap_center;

        // Normalize the peak.
        // TODO Check that this is equivalent to dividing by the auto-correlation peak.
        const auto score = static_cast<f64>(peak_value) /
                           noa::math::sqrt(static_cast<f64>(energy_target) *
                                           static_cast<f64>(energy_reference));

        return {shift, score};
    }
}
