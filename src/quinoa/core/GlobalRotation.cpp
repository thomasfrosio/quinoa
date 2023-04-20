#include "quinoa/core/GlobalRotation.hpp"
#include "quinoa/core/Optimizer.hpp"

namespace {
    struct OptimizerData {
        qn::GlobalRotation* global_rotation{};
        const qn::MetadataStack* metadata{};
        const qn::GlobalRotationParameters* parameters{};
    };
}

namespace qn {
    GlobalRotation::GlobalRotation(
            const Array<f32>& stack,
            const MetadataStack& metadata,
            Device compute_device,
            const GlobalRotationParameters& parameters
    ) {
        auto metadata_sorted = metadata;
        metadata_sorted.sort("absolute_tilt");

        const auto slice_shape = Shape4<i64>{1, 1, stack.shape()[2], stack.shape()[3]};
        const auto array_options = noa::ArrayOption(compute_device, noa::Allocator::DEFAULT_ASYNC);

        // Allocate and prepare reference buffer. This will be compared with the stretched targets.
        m_reference_index = metadata_sorted[0].index; // lowest abs tilt
        m_reference_rfft = noa::memory::empty<c32>(slice_shape.fft(), array_options);
        const auto reference_fft = m_reference_rfft.view();
        const auto reference = noa::fft::alias_to_real(reference_fft, slice_shape);
        stack.subregion(m_reference_index).to(reference);
        noa::fft::r2c(reference, reference_fft);

        // Get the target indexes.
        const auto reference_tilt_angle = metadata_sorted[0].angles[1];
        for (size_t i = 1; i < metadata_sorted.size(); ++i) { // skip the reference at index 0
            const auto& slice = metadata_sorted[i];
            const auto absolute_tilt_difference = std::abs(reference_tilt_angle - slice.angles[1]);
            if (absolute_tilt_difference <= parameters.absolute_max_tilt_difference)
                m_target_indexes.emplace_back(slice.index);
        }

        // Allocate buffers for the optimization function.
        // The matrices need to be accessed from the CPU and GPU, so use managed memory.
        const i64 targets_count = static_cast<i64>(m_target_indexes.size());
        m_inv_stretching_matrices = noa::memory::empty<Float33>(
                Shape4<i64>{targets_count, 1, 1, 1},
                ArrayOption(compute_device, Allocator::MANAGED));
        m_xcorr_coefficients = noa::memory::empty<f32>(
                Shape4<i64>{targets_count, 1, 1, 1},
                ArrayOption(compute_device, Allocator::MANAGED));

        // Allocate stretched-targets buffer.
        const auto targets_shape = Shape4<i64>{targets_count, 1, stack.shape()[2], stack.shape()[3]};
        m_targets_stretched_rfft = noa::memory::empty<c32>(targets_shape.fft(), array_options);
        const auto targets_stretched_fft = m_targets_stretched_rfft.view();
        const auto targets_stretched = noa::fft::alias_to_real(targets_stretched_fft, targets_shape);

        m_targets = noa::Texture<f32>(
                targets_shape, compute_device,
                parameters.interpolation_mode, noa::BorderMode::ZERO, 0.f, /*layered=*/ true);

        const auto target_indexes = View<i32>(m_target_indexes.data(), targets_count);
        if (compute_device.is_cpu()) {
            // On the CPU, we need to allocate another buffer for the input targets.
            const auto targets_buffer = noa::memory::like(targets_stretched);
            noa::memory::copy_batches(stack, targets_buffer, target_indexes);
            m_targets.update(targets_buffer);
        } else {
            // On the GPU, we can temporarily use the output buffer.
            noa::memory::copy_batches(stack, targets_stretched, target_indexes);
            m_targets.update(targets_stretched);
        }
    }

    void GlobalRotation::initialize(MetadataStack& metadata, const GlobalRotationParameters& parameters) {
        if (m_reference_rfft.is_empty())
            return;

        qn::Logger::info("Global rotation offset alignment...");
        qn::Logger::trace("Compute device: {}\n"
                          "Max absolute angle difference: {:.2f}",
                          m_reference_rfft.device(),
                          parameters.absolute_max_tilt_difference);
        noa::Timer timer;
        timer.start();

        // Set the rotations to 0 since the search is relative to the rotation saved in the metadata.
        for (auto& slice : metadata.slices())
            slice.angles[0] = 0.f;

        // Save some info directly in the class. This is used to pass data to the optimizer.
        OptimizerData optimizer_data;
        optimizer_data.global_rotation = this;
        optimizer_data.metadata = &metadata;
        optimizer_data.parameters = &parameters;

        // Objective function to maximize.
        auto func = [](u32, const f64* x, f64* gx, void* instance) -> f64 {
            const auto& data = *static_cast<OptimizerData*>(instance);
            const auto& self = *data.global_rotation;
            const auto rotation_offset = static_cast<f32>(*x);

            const auto out = self.max_objective_fx_(
                    rotation_offset, *data.metadata, *data.parameters);
            if (gx != nullptr) {
                *gx = static_cast<f64>(self.max_objective_gx_(
                        rotation_offset, *data.metadata, *data.parameters));
            }
            qn::Logger::debug("x={:> 8.3f}, fx={:.6f}, gx={}",
                              rotation_offset, out,
                              gx ? fmt::format("{:.3f}", *gx) : "null");
            return static_cast<f64>(out);
        };

        // While we could run a global optimization, since we know what our function looks like,
        // running 2 symmetric local search should be ok too.
        const auto algorithm =
                parameters.solve_using_estimated_gradient ?
                NLOPT_LD_LBFGS : NLOPT_LN_SBPLX;
        const Optimizer optimizer(algorithm, 1);
        optimizer.set_max_objective(func, &optimizer_data);
        optimizer.set_x_tolerance_abs(0.008);

        std::array<f64, 2> x{0, 180};
        std::array<f64, 2> fx{};
        for (size_t i = 0; i < 2; ++i) {
            optimizer.set_bounds(x[i] - 180, x[i] + 180);
            optimizer.optimize(x.data() + i, fx.data() + i);
        }

        const size_t best_index = fx[0] >= fx[1] ? 0 : 1;
        const auto fx_best = static_cast<f32>(fx[best_index]);

        // Final rotation, in range [-180, 180] deg.
        auto x_best = static_cast<f32>(x[best_index]);
        if (x_best > 180)
            x_best -= 360;

        for (auto& slice : metadata.slices())
            slice.angles[0] = x_best;

        qn::Logger::trace("Found initial global rotation of {:.3f} degrees (score={:.3f})", x_best, fx_best);
        qn::Logger::info("Global rotation offset alignment... done. Took {}\n", timer.elapsed());
    }

    void GlobalRotation::update(MetadataStack& metadata,
                                const GlobalRotationParameters& parameters,
                                f32 bound) {
        if (m_reference_rfft.is_empty())
            return;

        qn::Logger::info("Global rotation offset alignment...");
        qn::Logger::trace("Compute device: {}\n"
                          "Max absolute angle difference: {:.2f}\n"
                          "Max rotation offset: {:.2f}",
                          m_reference_rfft.device(),
                          parameters.absolute_max_tilt_difference,
                          bound);
        noa::Timer timer;
        timer.start();

        OptimizerData optimizer_data;
        optimizer_data.global_rotation = this;
        optimizer_data.metadata = &metadata;
        optimizer_data.parameters = &parameters;

        // Objective function for nlopt.
        auto func = [](u32, const f64* x, f64* gx, void* instance) -> f64 {
            const auto& data = *static_cast<OptimizerData*>(instance);
            const auto& self = *data.global_rotation;
            const auto rotation_offset = static_cast<f32>(*x);

            const auto out = self.max_objective_fx_(
                    rotation_offset, *data.metadata, *data.parameters);
            if (gx != nullptr) {
                *gx = static_cast<f64>(self.max_objective_gx_(
                        rotation_offset, *data.metadata, *data.parameters));
            }
            qn::Logger::debug("x={:> 8.3f}, fx={:.6f}, gx={}",
                              rotation_offset, out,
                              gx ? fmt::format("{:.6f}", *gx) : "null");
            return static_cast<f64>(out);
        };

        const auto algorithm =
                parameters.solve_using_estimated_gradient ?
                NLOPT_LD_LBFGS : NLOPT_LN_SBPLX;
        Optimizer optimizer(algorithm, 1);

        optimizer.set_max_objective(func, &optimizer_data);
        optimizer.set_bounds(-static_cast<f64>(bound), static_cast<f64>(bound));
        optimizer.set_initial_step(static_cast<f64>(bound) * 0.1);
        optimizer.set_x_tolerance_abs(0.005);

        f64 x{0.}; // initial rotation offset relative to whatever is in the metadata
        f64 fx{};
        optimizer.optimize(&x, &fx); // returns the best rotation in x

        // Update the metadata.
        const auto rotation_offset = static_cast<f32>(x);
        for (auto& slice : metadata.slices())
            slice.angles[0] += rotation_offset;

        qn::Logger::trace("Found global rotation offset of {:.3f} degrees (score={:.3f})", x, fx);
        qn::Logger::info("Global rotation offset alignment... done. Took {}\n", timer.elapsed());
    }

    auto GlobalRotation::max_objective_fx_(
            f32 rotation_offset,
            const MetadataStack& metadata,
            const GlobalRotationParameters& parameters
    ) const -> f32 {
        const auto targets_stretched_rfft = m_targets_stretched_rfft.view();
        const auto targets_stretched = noa::fft::alias_to_real(targets_stretched_rfft, m_targets.shape());

        // Cosine stretch the target views using this rotation.
        update_stretching_matrices_(metadata, rotation_offset);
        noa::geometry::transform_2d(m_targets, targets_stretched, m_inv_stretching_matrices.view());

        // TODO Normalize?

        // Cross-correlation between the target and the stretched references.
        noa::fft::r2c(targets_stretched, targets_stretched_rfft);
        noa::signal::fft::bandpass<noa::fft::H2H>(
                targets_stretched_rfft, targets_stretched_rfft, targets_stretched.shape(),
                parameters.highpass_filter[0], parameters.lowpass_filter[0],
                parameters.highpass_filter[1], parameters.lowpass_filter[1]);
        noa::signal::fft::xcorr<noa::fft::H2H>(
                m_reference_rfft.view(), targets_stretched_rfft,
                targets_stretched.shape(), m_xcorr_coefficients);
        return noa::math::sum(m_xcorr_coefficients);
    }

    // Numerically estimate the derivative using the central finite difference.
    auto GlobalRotation::max_objective_gx_(
            f32 rotation_offset,
            const MetadataStack& metadata,
            const GlobalRotationParameters& parameters
    ) const -> f32 {
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
        const f32 delta = make_xph_representable(rotation_offset, h);

        const f32 yh = max_objective_fx_(rotation_offset + delta, metadata, parameters);
        const f32 ymh = max_objective_fx_(rotation_offset - delta, metadata, parameters);
        const f32 diff = yh - ymh;
        return diff / (2 * delta);
    }

    void GlobalRotation::update_stretching_matrices_(const MetadataStack& metadata, f32 rotation_offset) const {
        const auto rotation_offset_rad = static_cast<f64>(noa::math::deg2rad(rotation_offset));
        const auto slice_center = MetadataSlice::center(m_targets.shape()).as<f64>();
        const MetadataSlice& reference_slice = metadata[m_reference_index];
        const Vec3<f64> reference_angles = noa::math::deg2rad(reference_slice.angles.as<f64>());

        for (size_t i = 0; i < m_target_indexes.size(); ++i) {
            const MetadataSlice& target_slice = metadata[m_target_indexes[i]];
            const Vec3<f64> target_angles = noa::math::deg2rad(target_slice.angles.as<f64>());

            // Compute the affine matrix to transform the target "onto" the reference.
            // These angles are flipped, since the cos-scaling is perpendicular to the axis of rotation.
            // While we account for the elevation increment, it is sort of expected to be 0...
            const Vec2<f64> cos_factor{noa::math::cos(reference_angles[2]) / noa::math::cos(target_angles[2]),
                                       noa::math::cos(reference_angles[1]) / noa::math::cos(target_angles[1])};

            // Apply the scaling for the tilt and elevation difference,
            // and cancel the difference (if any) in rotation and shift as well.
            // After this point, the target should "overlap" with the reference.
            const Double33 fwd_stretch_target_to_reference =
                    noa::geometry::translate(slice_center + reference_slice.shifts.as<f64>()) *
                    noa::geometry::linear2affine(noa::geometry::rotate(rotation_offset_rad + reference_angles[0])) *
                    noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                    noa::geometry::linear2affine(noa::geometry::rotate(-rotation_offset_rad -target_angles[0])) *
                    noa::geometry::translate(-slice_center - target_slice.shifts.as<f64>());
            m_inv_stretching_matrices(i, 0, 0, 0) = fwd_stretch_target_to_reference.inverse().as<f32>();
        }
    }
}
