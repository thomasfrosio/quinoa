#include <noa/core/utils/Timer.hpp>
#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Math.hpp>
#include <noa/Memory.hpp>
#include <noa/Signal.hpp>

#include "quinoa/core/GlobalRotation.hpp"
#include "quinoa/core/Optimizer.hpp"

namespace {
    using namespace ::qn;

    void update_stretching_matrices_(
            const MetadataStack& metadata,
            const std::vector<i32>& target_indexes,
            i64 reference_index,
            f64 rotation_offset_rad,
            Vec2<f64> slice_center,
            Span<Float33> inv_stretching_matrices
    ) {
        const MetadataSlice& reference_slice = metadata[reference_index];
        const Vec3<f64> reference_angles = noa::math::deg2rad(reference_slice.angles);

        for (size_t i = 0; i < target_indexes.size(); ++i) {
            const MetadataSlice& target_slice = metadata[target_indexes[i]];
            const Vec3<f64> target_angles = noa::math::deg2rad(target_slice.angles);

            // Compute the affine matrix to transform the target "onto" the reference.
            // These angles are flipped, since the cos-scaling is perpendicular to the axis of rotation.
            // While we account for the elevation increment, it is sort of expected to be 0...
            const Vec2<f64> cos_factor{noa::math::cos(reference_angles[2]) / noa::math::cos(target_angles[2]),
                                       noa::math::cos(reference_angles[1]) / noa::math::cos(target_angles[1])};

            // Apply the scaling for the tilt and elevation difference,
            // and cancel the difference (if any) in rotation and shift as well.
            // After this point, the target should "overlap" with the reference.
            const Double33 fwd_stretch_target_to_reference =
                    noa::geometry::translate(slice_center + reference_slice.shifts) *
                    noa::geometry::linear2affine(noa::geometry::rotate(rotation_offset_rad + reference_angles[0])) *
                    noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                    noa::geometry::linear2affine(noa::geometry::rotate(-rotation_offset_rad - target_angles[0])) *
                    noa::geometry::translate(-slice_center - target_slice.shifts);
            inv_stretching_matrices[i] = fwd_stretch_target_to_reference.inverse().as<f32>();
        }
    }

    class RotationOffsetFitter {
    public:
        const MetadataStack* m_metadata;
        const GlobalRotationParameters* m_parameters;
        Memoizer m_memoizer;

        Texture<f32> m_targets;
        View<c32> m_targets_stretched_rfft;
        View<c32> m_reference_rfft;

        View<Float33> m_inv_stretching_matrices;
        View<f32> m_xcorr_coefficients;
        const std::vector<i32>* m_target_indexes;
        i32 m_reference_index{};

    public:
        RotationOffsetFitter(
                const MetadataStack& metadata,
                const GlobalRotationParameters& parameters,
                const Texture<f32>& targets,
                const View<c32>& targets_stretched_rfft,
                const View<c32>& reference_rfft,
                View<Float33> inv_stretching_matrices,
                View<f32> xcorr_coefficients,
                const std::vector<i32>& target_indexes,
                i32 reference_index
        ) :
                m_metadata(&metadata),
                m_parameters(&parameters),
                m_memoizer(/*n_parameters=*/ 1, /*resolution=*/ 4),
                m_targets(targets),
                m_targets_stretched_rfft(targets_stretched_rfft),
                m_reference_rfft(reference_rfft),
                m_inv_stretching_matrices(inv_stretching_matrices),
                m_xcorr_coefficients(xcorr_coefficients),
                m_target_indexes(&target_indexes),
                m_reference_index(reference_index) {}

    public:
        [[nodiscard]] auto cost(f64 rotation_offset_rad) const -> f64 {
            const auto targets_stretched = noa::fft::alias_to_real(m_targets_stretched_rfft, m_targets.shape());

            // Cosine-stretch the target views using this rotation offset.
            const auto slice_center = MetadataSlice::center(m_targets.shape()).as<f64>();
            update_stretching_matrices_(
                    *m_metadata, *m_target_indexes, m_reference_index,
                    rotation_offset_rad, slice_center,
                    m_inv_stretching_matrices.span());
            noa::geometry::transform_2d(m_targets, targets_stretched, m_inv_stretching_matrices);

            if (!m_parameters->debug_directory.empty()) {
                const auto filename = m_parameters->debug_directory / "targets_stretched.mrc";
                noa::io::save(targets_stretched, filename);
                qn::Logger::debug("{} saved", filename);
            }

            // We could/should normalize here, but the targets are the same (and normalized) and the stretching doesn't
            // affect the image stats that much. At the end, we get the same results with and without normalization.

            // Cross-correlation coefficient between the reference and the stretched targets.
            noa::fft::r2c(targets_stretched, m_targets_stretched_rfft);
            noa::signal::fft::bandpass<noa::fft::H2H>(
                    m_targets_stretched_rfft, m_targets_stretched_rfft, targets_stretched.shape(),
                    m_parameters->highpass_filter[0], m_parameters->lowpass_filter[0],
                    m_parameters->highpass_filter[1], m_parameters->lowpass_filter[1]);
            noa::signal::fft::xcorr<noa::fft::H2H>(
                    m_reference_rfft, m_targets_stretched_rfft,
                    targets_stretched.shape(), m_xcorr_coefficients);
            return static_cast<f64>(noa::math::mean(m_xcorr_coefficients));
        }

        static auto function_to_maximise(
                u32 n_parameters, const f64* parameters, f64* gradients, void* instance
        ) -> f64 {
            NOA_ASSERT(n_parameters == 1 && parameters != nullptr);
            auto& self = *static_cast<RotationOffsetFitter*>(instance);
            auto& memoizer = self.m_memoizer;
            const f64 rotation_offset = parameters[0];

            std::optional<f64> memoized_cost = memoizer.find(parameters, gradients, /*epsilon=*/ 1e-8);
            if (memoized_cost.has_value()) {
                auto cost = memoized_cost.value();
                log(cost, rotation_offset, gradients, true);
                return cost;
            }

            const f64 cost = self.cost(rotation_offset);
            if (gradients) {
                const f64 delta = 5e-6;
                const f64 f_plus = self.cost(rotation_offset + delta);
                const f64 f_minus = self.cost(rotation_offset - delta);
                gradients[0] = CentralFiniteDifference::get(f_minus, f_plus, delta);
            }

            memoizer.record(parameters, cost, gradients);
            log(cost, rotation_offset, gradients);
            return cost;
        }

        static void log(f64 cost, f64 rotation_offset, const f64* gradient, bool memoized = false) {
            qn::Logger::trace(
                    "v={:.8f}, f={:.8f}{}{}",
                    rotation_offset, cost,
                    gradient ? noa::string::format(", g={:.8f}", *gradient) : "",
                    memoized ? ", memoized=true" : "");
        }
    };
}

namespace qn {
    GlobalRotation::GlobalRotation(
            const Array<f32>& stack,
            const MetadataStack& metadata,
            const GlobalRotationParameters& parameters,
            Device compute_device,
            Allocator allocator
    ) {
        auto metadata_sorted = metadata;
        metadata_sorted.sort("absolute_tilt");

        const auto slice_shape = Shape4<i64>{1, 1, stack.shape()[2], stack.shape()[3]};
        const auto options = noa::ArrayOption(compute_device, allocator);

        // Allocate and prepare the reference buffer. This will be compared with the stretched targets.
        // Note: we do expect the stack slices to be MEAN_STD normalized, so don't do it here again.
        m_reference_index = metadata_sorted[0].index; // lowest abs tilt
        m_reference_rfft = noa::memory::empty<c32>(slice_shape.rfft(), options);
        const auto reference_rfft = m_reference_rfft.view();
        const auto reference = noa::fft::alias_to_real(reference_rfft, slice_shape);
        stack.subregion(m_reference_index).to(reference);
        noa::fft::r2c(reference, reference_rfft);
        noa::signal::fft::bandpass<noa::fft::H2H>(
                reference_rfft, reference_rfft, slice_shape,
                parameters.highpass_filter[0], parameters.lowpass_filter[0],
                parameters.highpass_filter[1], parameters.lowpass_filter[1]);

        if (!parameters.debug_directory.empty()) {
            const auto filename = parameters.debug_directory / "reference_rfft.mrc";
            noa::io::save(noa::ewise_unary(reference_rfft, noa::abs_one_log_t{}), filename);
            qn::Logger::debug("{} saved", filename);
        }

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
        m_xcorr_coefficients = noa::memory::like<f32>(m_inv_stretching_matrices);

        // Allocate stretched-targets rfft buffer.
        const auto targets_shape = Shape4<i64>{targets_count, 1, stack.shape()[2], stack.shape()[3]};
        m_targets_stretched_rfft = noa::memory::empty<c32>(targets_shape.rfft(), options);
        const auto targets_stretched = noa::fft::alias_to_real(m_targets_stretched_rfft.view(), targets_shape);

        // Allocate and initialize texture for the targets.
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

        qn::Logger::info("Rotation offset alignment using cosine-stretching...");
        qn::Logger::trace("Compute device: {}\n"
                          "Max absolute angle difference: {:.2f}",
                          m_reference_rfft.device(),
                          parameters.absolute_max_tilt_difference);
        noa::Timer timer;
        timer.start();

        // Set the rotations to 0 since the search is relative to the rotations saved in the metadata.
        for (auto& slice : metadata.slices())
            slice.angles[0] = 0.;

        auto fitter = RotationOffsetFitter(
                metadata,
                parameters,
                m_targets,
                m_targets_stretched_rfft.view(),
                m_reference_rfft.view(),
                m_inv_stretching_matrices.view(),
                m_xcorr_coefficients.view(),
                m_target_indexes,
                m_reference_index
        );

        const auto algorithm =
                parameters.solve_using_estimated_gradient ?
                NLOPT_LD_LBFGS : NLOPT_LN_SBPLX;
        const Optimizer optimizer(algorithm, 1);
        optimizer.set_max_objective(RotationOffsetFitter::function_to_maximise, &fitter);
        optimizer.set_x_tolerance_abs(noa::math::deg2rad(0.01));
        optimizer.set_bounds(noa::math::deg2rad(-1.), noa::math::deg2rad(181.));

        f64 rotation_offset{noa::math::deg2rad(90.)};
        optimizer.optimize(&rotation_offset);
        rotation_offset = noa::math::rad2deg(rotation_offset);

        // Update metadata.
        metadata.add_global_angles({rotation_offset, 0, 0});

        qn::Logger::info("rotation_offset={:.3f} degrees "
                         "(note: we cannot distinguish with the {:.3f}+180={:.3f} offset at this point)",
                         rotation_offset, rotation_offset, rotation_offset + 180);
        qn::Logger::info("Rotation offset alignment using cosine-stretching... done. Took {}\n", timer.elapsed());
    }

    void GlobalRotation::update(
            MetadataStack& metadata,
            const GlobalRotationParameters& parameters,
            f64 range_degrees
    ) {
        if (m_reference_rfft.is_empty())
            return;

        qn::Logger::info("Global rotation offset alignment...");
        qn::Logger::trace("Compute device: {}\n"
                          "Max absolute angle difference: {:.2f}\n"
                          "Max rotation offset: {:.2f}",
                          m_reference_rfft.device(),
                          parameters.absolute_max_tilt_difference,
                          range_degrees);
        noa::Timer timer;
        timer.start();

        auto fitter = RotationOffsetFitter(
                metadata,
                parameters,
                m_targets,
                m_targets_stretched_rfft.view(),
                m_reference_rfft.view(),
                m_inv_stretching_matrices.view(),
                m_xcorr_coefficients.view(),
                m_target_indexes,
                m_reference_index
        );

        const auto algorithm =
                parameters.solve_using_estimated_gradient ?
                NLOPT_LD_LBFGS : NLOPT_LN_SBPLX;
        Optimizer optimizer(algorithm, 1);
        optimizer.set_max_objective(RotationOffsetFitter::function_to_maximise, &fitter);
        optimizer.set_x_tolerance_abs(noa::math::deg2rad(0.005));
        optimizer.set_bounds(-noa::math::deg2rad(range_degrees), noa::math::deg2rad(range_degrees));

        f64 rotation_offset{0.}; // initial rotation offset relative to whatever is in the metadata
        optimizer.optimize(&rotation_offset); // returns the best rotation in x
        rotation_offset = noa::math::rad2deg(rotation_offset);

        // Update the metadata.
        f64 average_rotation_offset{0};
        for (auto& slice : metadata.slices()) {
            slice.angles[0] += rotation_offset;
            average_rotation_offset += slice.angles[0];
        }
        average_rotation_offset /= static_cast<f64>(metadata.ssize());

        qn::Logger::info("rotation_offset={:.3f} ({:+.3f}) degrees", average_rotation_offset, rotation_offset);
        qn::Logger::info("Global rotation offset alignment... done. Took {}\n", timer.elapsed());
    }
}
