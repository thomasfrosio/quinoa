#include "quinoa/core/YawFinder.h"
#include "quinoa/core/Optimizer.hpp"

namespace qn {
    GlobalYawSolver::GlobalYawSolver(
            const Array<f32>& stack,
            const MetadataStack& metadata,
            Device compute_device,
            const GlobalYawOffsetParameters& parameters
    ) {
        auto metadata_sorted = metadata;
        metadata_sorted.sort("absolute_tilt");

        // Allocate reference buffer. This will be compared with the stretched targets.
        const auto options = noa::ArrayOption(compute_device, noa::Allocator::DEFAULT_ASYNC);
        m_reference_index = metadata_sorted[0].index; // lowest abs tilt
        std::tie(m_reference, m_reference_fft) = noa::fft::empty<f32>(
                Shape4<i64>{1, 1, stack.shape()[2], stack.shape()[3]},
                options);
        stack.subregion(m_reference_index).to(m_reference);
        noa::fft::r2c(m_reference, m_reference_fft);

        // Get the target indexes.
        const auto reference_tilt_angle = metadata_sorted[0].angles[1];
        for (size_t i = 1; i < metadata_sorted.size(); ++i) { // skip the reference
            const auto& slice = metadata_sorted[i];
            const auto absolute_tilt_difference = std::abs(reference_tilt_angle - slice.angles[1]);
            if (absolute_tilt_difference <= parameters.absolute_max_tilt_difference)
                m_target_indexes.emplace_back(slice.index);
        }

        // Allocate buffers for the optimization function.
        // The matrices need to be accessed from the CPU, so use a GPU pinned array.
        const i64 targets_count = static_cast<i64>(m_target_indexes.size());
        m_inv_stretching_matrices = noa::memory::empty<Float33>(
                Shape4<i64>{targets_count, 1, 1, 1},
                ArrayOption(compute_device, Allocator::PINNED));
        m_xcorr_coefficients = noa::memory::empty<f32>(
                Shape4<i64>{targets_count, 1, 1, 1},
                ArrayOption(compute_device, Allocator::PINNED));

        // Allocate stretched-targets buffer.
        const auto targets_shape = Shape4<i64>{targets_count, 1, stack.shape()[2], stack.shape()[3]};
        std::tie(m_targets_stretched, m_targets_stretched_fft) =
                noa::fft::empty<f32>(targets_shape, options);

        // On the CPU, we need to allocate another buffer for the input targets.
        // On the GPU, we can temporarily use the output buffer.
        auto targets_buffer = compute_device.is_cpu() ?
                              noa::memory::like(m_targets_stretched) : m_targets_stretched;
        noa::memory::copy_batches(stack, targets_buffer,
                                  View<i32>(m_target_indexes.data(), targets_count));

        m_targets = noa::Texture<f32>(
                targets_shape, compute_device,
                parameters.interpolation_mode, noa::BorderMode::ZERO, 0.f, /*layered=*/ true);
        m_targets.update(targets_buffer);
    }

    // This initializes the yaw by looking at the 360deg range.
    // We do expect two symmetric roots (separated by 180 deg). The info we have doesn't allow us to select the
    // correct root (the defocus gradient could be a solution to that). Therefore, try to find the two peaks and
    // select the one with the highest score (which doesn't mean it's the correct one since the difference is likely
    // to be non-significant).
    void GlobalYawSolver::initialize_yaw(MetadataStack& metadata, const GlobalYawOffsetParameters& parameters) {
        // Set the yaws to 0 since the search is relative to the yaw saved in the metadata.
        for (auto& slice : metadata.slices())
            slice.angles[0] = 0.f;

        // Save some info directly in the class. This is used to pass data to the optimizer.
        // Another approach could be to capture this info into the lambda, but the lambda will
        // then end up in a std::function once passed to nlopt, which is slightly less efficient.
        m_metadata = &metadata;
        m_highpass_filter = parameters.highpass_filter;
        m_lowpass_filter = parameters.lowpass_filter;

        // Objective function for nlopt.
        auto func = [](u32, const f64* x, f64* gx, void* instance) -> f64 {
            auto* global_yaw_finder = static_cast<qn::GlobalYawSolver*>(instance);
            const f32 yaw_offset = static_cast<f32>(*x);

            const auto out = global_yaw_finder->max_objective_fx_(yaw_offset);
            if (gx != nullptr)
                *gx = static_cast<f64>(global_yaw_finder->max_objective_gx_(yaw_offset));

            qn::Logger::debug("x={}, fx={}, gx={}", yaw_offset, out, gx ? fmt::format("{}", *gx) : "null");
            return static_cast<f64>(out);
        };

        // While we could run a global optimization, since we know how our function looks like,
        // running 2 symmetric local search should be ok too.
        const auto algorithm =
                parameters.solve_using_estimated_gradient ?
                NLOPT_LD_LBFGS : NLOPT_LN_NELDERMEAD;
        const Optimizer optimizer(algorithm, 1);
        optimizer.set_max_objective(func, this);
        optimizer.set_x_tolerance_abs(0.001);
        optimizer.set_fx_tolerance_abs(0.0001);

        std::array<f64, 2> x{0, 180};
        std::array<f64, 2> fx{};
        for (size_t i = 0; i < 2; ++i) {
            optimizer.set_bounds(x[i] - 180, x[i] + 180);
            optimizer.optimize(x.data() + i, fx.data() + i);
        }

        const size_t best_index = fx[0] >= fx[1] ? 0 : 1;
        const auto fx_best = static_cast<f32>(fx[best_index]);

        // Final yaw, in range [-180, 180] deg.
        auto x_best = static_cast<f32>(x[best_index]);
        if (x_best > 180)
            x_best -= 360;

        qn::Logger::trace("Found initial global yaw of {:.3f} degrees (score={:.3f})", x_best, fx_best);

        for (auto& slice : metadata.slices())
            slice.angles[0] = x_best;
    }

    void GlobalYawSolver::update_yaw(MetadataStack& metadata,
                                     const GlobalYawOffsetParameters& parameters,
                                     f32 low_bound, f32 high_bound) {
        m_metadata = &metadata;
        m_highpass_filter = parameters.highpass_filter;
        m_lowpass_filter = parameters.lowpass_filter;

        // Objective function for nlopt.
        auto func = [](u32, const f64* x, f64* gx, void* instance) -> f64 {
            auto* global_yaw_finder = static_cast<qn::GlobalYawSolver*>(instance);
            const f32 yaw_offset = static_cast<f32>(*x);

            const auto out = global_yaw_finder->max_objective_fx_(yaw_offset);
            if (gx != nullptr)
                *gx = static_cast<f64>(global_yaw_finder->max_objective_gx_(yaw_offset));

            qn::Logger::debug("x={}, fx={}, gx={}", yaw_offset, out, gx ? fmt::format("{}", *gx) : "null");
            return static_cast<f64>(out);
        };

        const auto algorithm =
                parameters.solve_using_estimated_gradient ?
                NLOPT_LD_LBFGS : NLOPT_LN_NELDERMEAD;
        Optimizer optimizer(algorithm, 1);

        optimizer.set_max_objective(func, this);
        optimizer.set_bounds(static_cast<f64>(low_bound), static_cast<f64>(high_bound));
        optimizer.set_initial_step(static_cast<f64>(high_bound) * 0.1);
        optimizer.set_x_tolerance_abs(0.001);
        optimizer.set_fx_tolerance_abs(0.001);

        // Initial offset guess is 0. This is NOT the actual yaw.
        // The actual yaw is whatever is already saved in the metadata.
        f64 x{0.};
        f64 fx{};
        optimizer.optimize(&x, &fx);
        qn::Logger::trace("Found global yaw offset of {:.3f} degrees (score={:.3f})", x, fx);

        // Update the metadata.
        const auto yaw_offset = static_cast<f32>(x);
        for (auto& slice : metadata.slices())
            slice.angles[0] += yaw_offset;
    }

    f32 GlobalYawSolver::max_objective_fx_(f32 yaw_offset_deg) {
        // Cosine stretch the targets views using this yaw.
        update_stretching_matrices_(*m_metadata, yaw_offset_deg);
        noa::geometry::transform_2d(m_targets, m_targets_stretched, m_inv_stretching_matrices);

        // Cross-correlation between the target and the stretched references.
        noa::fft::r2c(m_targets_stretched, m_targets_stretched_fft);
        noa::signal::fft::bandpass<noa::fft::H2H>(
                m_targets_stretched_fft, m_targets_stretched_fft, m_targets_stretched.shape(),
                m_highpass_filter[0], m_lowpass_filter[0],
                m_highpass_filter[1], m_lowpass_filter[1]);
        noa::signal::fft::xcorr<noa::fft::H2H>(
                m_reference_fft, m_targets_stretched_fft,
                m_targets_stretched.shape(), m_xcorr_coefficients);
        return noa::math::sum(m_xcorr_coefficients);
    }

    // Numerically estimate the derivative using the central finite difference.
    f32 GlobalYawSolver::max_objective_gx_(f32 yaw_offset) {
        auto make_xph_representable = [](f32 x, f32 h) {
            // From https://github.com/boostorg/math/blob/develop/include/boost/math/differentiation/finite_difference.hpp
            // Redefine h so that x + h is representable. Not using this trick leads to
            // large error. The compiler flag -ffast-math evaporates these operations...
            f32 temp = x + h;
            h = temp - x;
            // Handle the case x + h == x:
            if (h == 0)
                h = std::nextafter(x, (std::numeric_limits<f32>::max)()) - x;
            return h;
        };

        f32 eps = std::numeric_limits<f32>::epsilon();
        f32 h = std::pow(3 * eps, static_cast<f32>(1) / static_cast<f32>(3));
        const f32 delta = make_xph_representable(yaw_offset, h);

        const f32 yh = max_objective_fx_(yaw_offset + delta);
        const f32 ymh = max_objective_fx_(yaw_offset - delta);
        const f32 diff = yh - ymh;
        return diff / (2 * delta);
    }

    void GlobalYawSolver::update_stretching_matrices_(const MetadataStack& metadata, f32 yaw_offset) {
        const auto yaw_offset_rad = static_cast<f64>(noa::math::deg2rad(yaw_offset));
        const auto slice_center = MetadataSlice::center(m_reference.shape()).as<f64>();
        const MetadataSlice& reference_slice = metadata[m_reference_index];
        const Vec3<f64> reference_angles = noa::math::deg2rad(reference_slice.angles.as<f64>());

        for (size_t i = 0; i < m_target_indexes.size(); ++i) {
            const MetadataSlice& target_slice = metadata[m_target_indexes[i]];
            const Vec3<f64> target_angles = noa::math::deg2rad(target_slice.angles.as<f64>());

            // Compute the affine matrix to transform the target "onto" the reference.
            // These angles are flipped, since the cos-scaling is perpendicular to the axis of rotation.
            const Vec2<f64> cos_factor{noa::math::cos(reference_angles[2]) / noa::math::cos(target_angles[2]),
                                       noa::math::cos(reference_angles[1]) / noa::math::cos(target_angles[1])};

            // Apply the scaling for the tilt and pitch difference,
            // and cancel the difference (if any) in yaw and shift as well.
            // After this point, the target should "overlap" with the reference.
            const Double33 fwd_stretch_target_to_reference =
                    noa::geometry::translate(slice_center + reference_slice.shifts.as<f64>()) *
                    noa::geometry::linear2affine(noa::geometry::rotate(yaw_offset_rad + reference_angles[0])) *
                    noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                    noa::geometry::linear2affine(noa::geometry::rotate(-yaw_offset_rad -target_angles[0])) *
                    noa::geometry::translate(-slice_center - target_slice.shifts.as<f64>());
            m_inv_stretching_matrices(i, 0, 0, 0) = fwd_stretch_target_to_reference.inverse().as<f32>();
        }
    }
}
