#include <noa/IO.hpp>
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/Utilities.h"

namespace {
    using namespace qn;

    // Data passed through the optimizer.
    struct OptimizerData {
        ProjectionMatching* projector{};
        const ProjectionMatchingParameters* parameters{};

        MetadataStack metadata;

        // This is set at each iteration so that the projector knows what target
        // (and therefore projected-reference) to compute.
        i64 target_index{};
        std::vector<i64> reference_indexes;
        Vec2<f64> final_shift;
    };
}

namespace qn {
    ProjectionMatching::ProjectionMatching(
            const noa::Shape4<i64>& shape,
            noa::Device compute_device,
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters,
            noa::Allocator allocator) {
        // Zero padding:
        const i64 size_padded = std::max(shape[2], shape[3]) * 2;

        // Find the maximum number of reference slices we'll need to hold at a given time.
        const i64 max_reference_count = max_references_count_(metadata, parameters);
        const auto target_reference_shape = Shape4<i64>{2, 1, shape[2], shape[3]};
        const auto target_reference_padded_shape = Shape4<i64>{2, 1, size_padded, size_padded};

        // Add an extra slice here to store the target at the end.
        const auto slices_shape = Shape4<i64>{max_reference_count + 1, 1, shape[2], shape[3]};
        const auto slices_padded_shape = Shape4<i64>{max_reference_count + 1, 1, size_padded, size_padded};

        // Device-only buffers.
        const auto device_options = ArrayOption(compute_device, allocator);
        m_slices = noa::memory::empty<f32>(slices_shape, device_options);
        m_slices_padded_fft = noa::memory::empty<c32>(slices_padded_shape.fft(), device_options);
        m_target_reference_fft = noa::memory::empty<c32>(target_reference_shape.fft(), device_options);
        m_target_reference_padded_fft = noa::memory::empty<c32>(target_reference_padded_shape.fft(), device_options);
        m_multiplicity_padded_fft = noa::memory::empty<f32>({1, 1, size_padded, size_padded / 2 + 1}, device_options);
        m_peak_window = noa::memory::empty<f32>({1, 1, 1, 256 * 256}, device_options);

        // Managed buffers. This allocates ~n*25*4 bytes.
        const auto managed_options = ArrayOption(compute_device, Allocator::MANAGED);
        const auto max_reference_shape = Shape4<i64>{max_reference_count, 1, 1, 1};
        m_reference_weights = noa::memory::empty<f32>(max_reference_shape, managed_options);
        m_reference_batch_indexes = noa::memory::like<Vec4<i32>>(m_reference_weights);
        m_insert_inv_references_rotation = noa::memory::like<Float33>(m_reference_weights);
        m_reference_shifts_center2origin = noa::memory::like<Vec2<f32>>(m_reference_weights);
    }

    void ProjectionMatching::update(
            const Array<f32>& stack,
            MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters,
            bool shift_only) {
        qn::Logger::trace("Projection matching alignment...");
        noa::Timer timer;
        timer.start();

        auto max_objective_function = [](u32, const f64* x, f64*, void* instance) -> f64 {
            auto* data = static_cast<OptimizerData*>(instance);

            const auto [shift, score] = data->projector->project_and_correlate_(
                    data->metadata,
                    data->target_index,
                    data->reference_indexes,
                    *data->parameters,
                    noa::signal::CorrelationMode::CONVENTIONAL,
                    *x
            );
            qn::Logger::debug("rotation offset={:> 7.4f}, score={:> 7.4f}", *x, score);

            data->final_shift = shift;
            return static_cast<f64>(score);
        };

        // Set up the optimizer.
        OptimizerData optimizer_data;
        optimizer_data.projector = this;
        optimizer_data.parameters = &parameters;
        optimizer_data.metadata = metadata;
        optimizer_data.metadata.sort("absolute_tilt");

        const Optimizer optimizer(NLOPT_LN_SBPLX, 1);
        optimizer.set_max_objective(max_objective_function, &optimizer_data);
        optimizer.set_x_tolerance_abs(0.005);
        const f64 bound = !shift_only ? 1.5 : 0;
        optimizer.set_bounds(-bound, bound);

        // Convergence loop.
        bool last_iteration = parameters.max_iterations == 1;
        for (i64 iter = 0; iter < parameters.max_iterations; ++iter) {

            // Enforce 3rd degree polynomial on the rotation angles.
            // For the first iteration, this should just fit a line because the rotation is likely
            // to be constant. In this case, it will simply update the shift, which is what we want anyway.
            const ThirdDegreePolynomial polynomial = poly_fit_rotation(optimizer_data.metadata);

            // Compute the rotation of the global reference using the polynomial. So whilst we cannot align
            // the global reference, we can still move the average rotation (including the global reference's)
            // progressively using projection matching.
            optimizer_data.metadata[0].angles[0] = polynomial(optimizer_data.metadata[0].angles[1]);

            // For every slice (excluding the lowest tilt which defines the reference-frame), find the best
            // geometry parameters using the previously aligned slice(s) as reference for alignment. The geometry
            // parameters are, for each slice, the rotation and the (y,x) shifts.
            const auto slice_count = static_cast<i64>(optimizer_data.metadata.size());
            for (i64 target_index = 1; target_index < slice_count; ++target_index) {

                // The target.
                optimizer_data.target_index = target_index;
                auto& slice = optimizer_data.metadata[target_index];

                // Get the indexes of the reference views for this target view.
                set_reference_indexes_(
                        target_index, optimizer_data.metadata,
                        parameters, optimizer_data.reference_indexes);

                // Prepare the target and references for Fourier insertion.
                prepare_for_insertion_(
                        stack.view(), optimizer_data.metadata, target_index,
                        optimizer_data.reference_indexes, parameters);

                // Enforce polynomial curve.
                const f64 rotation_offset_to_polynomial_curve = polynomial(slice.angles[1]) - slice.angles[0];
                const auto [shift, _] = project_and_correlate_(
                        optimizer_data.metadata,
                        optimizer_data.target_index,
                        optimizer_data.reference_indexes,
                        *optimizer_data.parameters,
                        noa::signal::CorrelationMode::DOUBLE_PHASE,
                        rotation_offset_to_polynomial_curve
                );
                slice.angles[0] += rotation_offset_to_polynomial_curve;
                slice.shifts += shift.as<f64>();

                // Find the best rotation offset (and its corresponding shift) by maximising the
                // cross-correlation between the target and the projected-reference. This is turned
                // off for the last iteration, since at this point we want to keep the rotations on
                // the polynomial curve.
                if (!shift_only && !last_iteration) {
                    f64 x{0};
                    f64 fx;
                    optimizer.optimize(&x, &fx);
                    slice.angles[0] += x;
                    slice.shifts += optimizer_data.final_shift;

                    qn::Logger::debug("{:>02}: rotation offset={:> 6.3f}, score={:.6g}, shift={::> 6.3f}",
                                      target_index, x, fx, optimizer_data.final_shift);
                }
            }

            // TODO Check convergence.
            last_iteration = iter == (parameters.max_iterations - 1);

            if (parameters.center_tilt_axis)
                center_tilt_axis_(optimizer_data.metadata);
        }

        // Update the metadata.
        for (const auto& updated_metadata: optimizer_data.metadata.slices()) {
            for (auto& original_slice: metadata.slices()) {
                if (original_slice.index == updated_metadata.index) {
                    original_slice.angles = updated_metadata.angles;
                    original_slice.shifts = updated_metadata.shifts;
                }
            }
        }
        qn::Logger::trace("Projection matching alignment... took {}ms", timer.elapsed());
    }
}

// Private methods:
namespace qn {
    auto ProjectionMatching::extract_peak_window_(const Vec2<f32>& max_shift) -> View<f32> {
        const auto radius = noa::math::ceil(max_shift).as<i64>();
        const auto elements = noa::math::product(radius * 2 + 1);
        return m_peak_window
                .view()
                .subregion(noa::indexing::ellipsis_t{}, noa::indexing::slice_t{0, elements})
                .reshape(Shape4<i64>{1, 1, radius[0] * 2 + 1, radius[1] * 2 + 1});
    }

    void ProjectionMatching::set_reference_indexes_(
            i64 target_index,
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters,
            std::vector<i64>& output_reference_indexes
    ) {
        const f64 target_tilt_angle = metadata[target_index].angles[1];
        const f64 max_tilt_difference = parameters.backward_tilt_angle_difference;
        const i64 max_index = parameters.backward_use_aligned_only ? target_index : static_cast<i64>(metadata.size());

        output_reference_indexes.clear();
        output_reference_indexes.reserve(static_cast<size_t>(max_index));

        for (i64 reference_index = 0; reference_index < max_index; ++reference_index) {
            const f64 reference_tilt_angle = metadata[reference_index].angles[1];
            const f64 tilt_difference = noa::math::abs(target_tilt_angle - reference_tilt_angle);

            // Of course, do not include the target in the projected-reference.
            if (reference_index != target_index && tilt_difference <= max_tilt_difference)
                output_reference_indexes.emplace_back(reference_index);
        }
    }

    i64 ProjectionMatching::max_references_count_(
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters) {
        // While we could return the number of slices in the metadata (-1 to ignore the target),
        // we try to find the maximum number of reference slices we'll need to hold at any given time.
        // Depending on the parameters for back-projection, this can save a good amount of memory.

        // Ensure at least 3 slices for reusing buffer for various data before or after the projection.
        i64 max_count{3};

        const auto slice_count = static_cast<i64>(metadata.size());

        // This is O(N^2), but it's fine because N is small (<60), and we do it once in the constructor.
        for (i64 target_index = 0; target_index < slice_count; ++target_index) {
            const f64 target_tilt_angle = metadata[target_index].angles[1];
            const i64 max_index = parameters.backward_use_aligned_only ? target_index : slice_count;

            // Count how many references are needed for the current target.
            i64 count = 0;
            for (i64 reference_index = 0; reference_index < max_index; ++reference_index) {
                const f64 reference_tilt_angle = metadata[reference_index].angles[1];
                const f64 tilt_difference = noa::math::abs(target_tilt_angle - reference_tilt_angle);
                if (reference_index != target_index && tilt_difference <= parameters.backward_tilt_angle_difference)
                    ++count;
            }

            max_count = std::max(max_count, count);
        }

        return max_count;
    }

    void ProjectionMatching::apply_area_mask_(
            const View<f32>& input,
            const View<f32>& output,
            const MetadataSlice& metadata,
            const ProjectionMatchingParameters& parameters
    ) {
        const Vec2<f32> center = MetadataSlice::center(input.shape());
        const Vec3<f32> angles = noa::math::deg2rad(metadata.angles).as<f32>();
        const Vec2<f32> shifts = metadata.shifts.as<f32>();

        const auto hw = input.shape().pop_front<2>();
        const auto smooth_edge_size =
                static_cast<f32>(noa::math::max(hw)) *
                parameters.area_match_taper;

        // Find the ellipse radius.
        // We start by a sphere to make sure it fits regardless of the shape and rotation angle.
        // The elevation shrinks the height of the ellipse. The tilt shrinks the width of the ellipse.
        const auto radius = static_cast<f32>(noa::math::min(hw)) / 2;
        const Vec2<f32> ellipse_radius = radius * noa::math::abs(noa::math::cos(angles.filter(2, 1)));

        // Then center the ellipse and rotate to have its height aligned with the tilt-axis.
        const Float33 inv_transform =
                noa::geometry::translate(center) *
                noa::geometry::linear2affine(noa::geometry::rotate(-angles[0])) *
                noa::geometry::translate(-(center + shifts));

        // Multiply the view with the ellipse. Everything outside is set to 0.
        noa::geometry::ellipse(
                input, output,
                /*center=*/ center,
                /*radius=*/ ellipse_radius - smooth_edge_size,
                /*edge_size=*/ smooth_edge_size,
                inv_transform);
    }

    void ProjectionMatching::apply_area_mask_(
            const View<f32>& input,
            const View<f32>& output,
            const MetadataStack& metadata,
            const std::vector<i64>& indexes,
            const ProjectionMatchingParameters& parameters
    ) {
        // TODO Batch this using a spherical mask and stretch it to create the ellipse.
        //      This will also stretch/shrink the taper, which isn't great, but it should be ok.
        for (size_t i = 0; i < indexes.size(); ++i) {
            // Assume the order in input/output/indexes match.
            apply_area_mask_(input.subregion(i), output.subregion(i),
                             metadata[indexes[i]], parameters);
        }
    }

    void ProjectionMatching::prepare_for_insertion_(
            const View<f32>& stack,
            const MetadataStack& metadata,
            i64 target_index,
            const std::vector<i64>& reference_indexes,
            const ProjectionMatchingParameters& parameters
    ) {
        const auto references_count = static_cast<i64>(reference_indexes.size());
        const auto references_range = noa::indexing::slice_t{0, references_count};

        const auto reference_weights = m_reference_weights.view().subregion(references_range);
        const auto reference_batch_indexes = m_reference_batch_indexes.view().subregion(references_range);
        const auto reference_shifts_center2origin = m_reference_shifts_center2origin.view().subregion(references_range);
        const auto insert_inv_references_rotation = m_insert_inv_references_rotation.view().subregion(references_range);

        const f64 target_tilt = noa::math::deg2rad(metadata[target_index].angles[1]);

        for (i64 i = 0; i < references_count; ++i) {
            const i64 reference_index = reference_indexes[static_cast<size_t>(i)];
            const Vec2<f64>& reference_shifts = metadata[reference_index].shifts;
            const Vec3<f64> reference_angles = noa::math::deg2rad(metadata[reference_index].angles);

            // TODO Weighting based on the order of collection? Or use exposure weighting?
            // Multiplicity and weight.
            // How much the slice should contribute to the final projected-reference.
            [[maybe_unused]] constexpr auto PI = noa::math::Constant<f64>::PI;
            [[maybe_unused]] const f64 tilt_difference = std::abs(target_tilt - reference_angles[1]);
            const f32 weight = 1; //noa::math::sinc(tilt_difference * PI / parameters.backward_tilt_angle_difference);
            reference_weights(i, 0, 0, 0) = 1 / weight; // FIXME?

            // Get the references indexes. These are the indexes of the reference slices within the input stack.
            reference_batch_indexes(i, 0, 0, 0) = {metadata[reference_index].index, 0, 0, 0};

            // Shifts to phase-shift the rotation center at the array origin.
            reference_shifts_center2origin(i, 0, 0, 0) = -slice_padded_center() - reference_shifts.as<f32>();

            // For the Fourier insertion, noa needs the inverse rotation matrix, hence the transposition.
            insert_inv_references_rotation(i, 0, 0, 0) = noa::geometry::euler2matrix(
                    Vec3<f64>{-reference_angles[0], reference_angles[1], reference_angles[2]},
                    "ZYX", false).transpose().as<f32>();
        }

        // We are going to store the target and the references next to each other.
        const auto target_references_range = noa::indexing::slice_t{0, references_count + 1};
        const auto only_references_range = noa::indexing::slice_t{1, references_count + 1};
        const View<f32> target_references = m_slices.view().subregion(target_references_range);
        const View<f32> references = target_references.subregion(only_references_range);

        const View<c32> target_references_padded_fft = m_slices_padded_fft.view().subregion(target_references_range);
        const View<f32> target_references_padded = noa::fft::alias_to_real(
                target_references_padded_fft, Shape4<i64>{references_count + 1, 1, size_padded(), size_padded()});

        // Get the target and mask it.
        apply_area_mask_(stack.subregion(metadata[target_index].index),
                         target_references.subregion(0), metadata[target_index], parameters);

        // Get the reference slices and mask them.
        View<f32> input_reference_slices;
        if (is_consecutive_range(reference_batch_indexes)) {
            // If the reference slices are already consecutive, no need to copy to a new array.
            const auto start = reference_batch_indexes(0, 0, 0, 0)[0];
            const auto end = start + references_count;
            input_reference_slices = stack.subregion(noa::indexing::slice_t{start, end});
        } else {
            noa::memory::extract_subregions(stack, references, reference_batch_indexes, noa::BorderMode::NOTHING);
            input_reference_slices = references;
        }
        apply_area_mask_(input_reference_slices, references, metadata, reference_indexes, parameters);

        // Zero-pad and fft both.
        noa::memory::resize(target_references, target_references_padded);
//        {
//            noa::io::save(target_references_padded,
//                          "/home/thomas/Projects/quinoa/tests/tilt2_v2/debug_pm/target_references_padded.mrc");
//        }
        noa::fft::r2c(target_references_padded, target_references_padded_fft);

        // Prepare the references for Fourier insertion.
        // The shift of the reference slices should be removed to have the rotation center at the origin.
        // phase_shift_2d can do the remap, but not in-place, so use remap to center the slice.
        const View<c32> references_padded_fft = target_references_padded_fft.subregion(only_references_range);
        const auto references_padded_shape = Shape4<i64>{references_count, 1, size_padded(), size_padded()};
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                references_padded_fft, references_padded_fft,
                references_padded_shape, reference_shifts_center2origin);
        noa::fft::remap(noa::fft::H2HC, references_padded_fft,
                        references_padded_fft, references_padded_shape);

        // At this point...
        //  - The zero padded rfft target and references are in m_slices_padded_fft
        //  - The references are phase-shifted and centered, ready for Fourier insertion.
    }

    void ProjectionMatching::compute_target_reference_(
            const MetadataStack& metadata,
            i64 target_index,
            f64 target_rotation_offset,
            const std::vector<i64>& reference_indexes,
            const ProjectionMatchingParameters& parameters
    ) {
        // ...At this point
        //  - The zero padded rfft target and references are in m_slices_padded_fft
        //  - The references are phase-shifted and centered, ready for Fourier insertion.
        const auto references_count = static_cast<i64>(reference_indexes.size());
        const View<const c32> input_target_references_padded_fft = m_slices_padded_fft.view()
                .subregion(noa::indexing::slice_t{0, references_count + 1});

        const auto references_range = noa::indexing::slice_t{0, references_count};
        const auto reference_weights = m_reference_weights.view().subregion(references_range);
        const auto insert_inv_references_rotation = m_insert_inv_references_rotation.view().subregion(references_range);

        // We'll save the target and reference next to each other.
        const View<c32> target_reference_padded_fft = m_target_reference_padded_fft.view();
        const View<c32> target_padded_fft = target_reference_padded_fft.subregion(0);
        const View<c32> reference_padded_fft = target_reference_padded_fft.subregion(1);
        const View<f32> multiplicity_padded_fft = m_multiplicity_padded_fft.view();
        const View<f32> target_reference_padded = noa::fft::alias_to_real(
                target_reference_padded_fft, Shape4<i64>{2, 1, size_padded(), size_padded()}); // output

        // Target geometry:
        // The yaw is the CCW angle of the tilt-axis in the slices. For the projection, we want to align
        // the tilt-axis along the Y axis, so subtract this angle and then apply the tilt and pitch.
        const Vec2<f64>& target_shifts = metadata[target_index].shifts;
        const Vec3<f64> target_angles = noa::math::deg2rad(
                metadata[target_index].angles +
                Vec3<f64>{target_rotation_offset, 0, 0});
        const Float33 extract_fwd_target_rotation = noa::geometry::euler2matrix(
                Vec3<f64>{-target_angles[0], target_angles[1], target_angles[2]},
                "zyx", /*intrinsic=*/ false).as<f32>();

        // Projection:
        // - The projected reference (and its corresponding sampling function) is reconstructed
        //   by adding the contribution of the input reference slices to the relevant "projected" frequencies.
        const auto references_padded_shape = Shape4<i64>{references_count, 1, size_padded(), size_padded()};
        const auto slice_padded_shape = Shape4<i64>{1, 1, size_padded(), size_padded()};

        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                input_target_references_padded_fft.subregion(noa::indexing::slice_t{1, references_count + 1}),
                references_padded_shape,
                reference_padded_fft, slice_padded_shape,
                Float22{}, insert_inv_references_rotation,
                Float22{}, extract_fwd_target_rotation,
                parameters.backward_slice_z_radius, false,
                parameters.projection_cutoff);
        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                noa::indexing::broadcast(reference_weights, references_padded_shape.fft()),
                references_padded_shape,
                multiplicity_padded_fft, slice_padded_shape,
                Float22{}, insert_inv_references_rotation,
                Float22{}, extract_fwd_target_rotation,
                parameters.backward_slice_z_radius, false,
                parameters.projection_cutoff);

        // Center projection back and shift onto the target.
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                reference_padded_fft, reference_padded_fft,
                slice_padded_shape, slice_padded_center() + target_shifts.as<f32>());

        // Correct for the multiplicity.
        // If weight is 0 (i.e. less than machine epsilon), the frequency is set to 0.
        // Otherwise, divide by the multiplicity.
        noa::ewise_binary(reference_padded_fft, multiplicity_padded_fft,
                          reference_padded_fft, noa::divide_safe_t{});

        // For normalization, binary mask the target with the multiplicity to only keep
        // the shared frequencies between the target and the projected reference.
        // TODO Create new operator to merge these two calls.
        // FIXME I think we should normalize to get a sampling function, and then multiply with the target.
        noa::ewise_binary(multiplicity_padded_fft, 1e-6f, multiplicity_padded_fft, noa::greater_t{});
        noa::ewise_binary(multiplicity_padded_fft, input_target_references_padded_fft.subregion(0),
                          target_padded_fft, noa::multiply_t{});

        // Go back to real-space.
        noa::fft::c2r(target_reference_padded_fft, target_reference_padded);

        // The target and the projected reference are saved next to each other.
        const auto target_reference = m_slices.view().subregion(noa::indexing::slice_t{0, 2});
        noa::memory::resize(target_reference_padded, target_reference);

        // Apply the area mask again.
        apply_area_mask_(target_reference.subregion(0), target_reference.subregion(0),
                         metadata[target_index], parameters);
        apply_area_mask_(target_reference.subregion(1), target_reference.subregion(1),
                         metadata[target_index], parameters);

        // ...At this point
        // - The real-space target and projected reference are in the first two slice of m_slices.
        // - They are ready for cross-correlation.
    }

    auto ProjectionMatching::project_and_correlate_(
            const MetadataStack& metadata,
            i64 target_index,
            const std::vector<i64>& reference_indexes,
            const ProjectionMatchingParameters& parameters,
            noa::signal::CorrelationMode correlation_mode,
            f64 angle_offset
    ) -> std::pair<Vec2<f64>, f64> {
        // First, compute the target and projected-reference slice.
        compute_target_reference_(metadata, target_index, angle_offset, reference_indexes, parameters);
        const auto target_reference = m_slices.subregion(noa::indexing::slice_t{0, 2});
        const auto target = target_reference.subregion(0);
        const auto reference = target_reference.subregion(1);

        // Normalization is crucial here to get a normalized peak.
        noa::math::normalize(target, target); // TODO normalize within mask?
        noa::math::normalize(reference, reference);
        const f32 energy_target = noa::math::sqrt(noa::math::sum(target, noa::abs_squared_t{}));
        const f32 energy_reference = noa::math::sqrt(noa::math::sum(reference, noa::abs_squared_t{}));

        if (!parameters.debug_directory.empty()) {
            noa::io::save(target_reference,
                          parameters.debug_directory /
                          noa::string::format("target_reference_{:>02}.mrc", target_index));
        }

        const auto target_reference_fft = m_target_reference_fft.view();
        noa::fft::r2c(target_reference, target_reference_fft); // TODO Norm::NONE?
        noa::signal::fft::bandpass<noa::fft::H2H>(
                target_reference_fft, target_reference_fft, target_reference.shape(),
                parameters.highpass_filter[0], parameters.lowpass_filter[0],
                parameters.highpass_filter[1], parameters.lowpass_filter[1]);
        // TODO spectral_whitening?

        // At this point, we can use any of the m_references slices for the xmap.
        // We rotate the xmap before the picking, so compute the centered xmap.
        const auto xmap = m_slices.view().subregion(0);
        noa::signal::fft::xmap<noa::fft::H2FC>(
                target_reference_fft.subregion(0),
                target_reference_fft.subregion(1),
                xmap, correlation_mode);
        if (!parameters.debug_directory.empty()) {
            noa::io::save(xmap, parameters.debug_directory /
                                noa::string::format("xmap_{:>02}.mrc", target_index));
        }

        // transform_2d will only render this small peak_window, which should
        // end up making the transformation and the picking very cheap to compute.
        const bool is_double_phase = correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE;
        const auto max_shift = noa::math::clamp(parameters.max_shift, Vec2<f32>{32}, Vec2<f32>{128});
        const View<f32> peak_window = extract_peak_window_(is_double_phase ? max_shift * 2 : max_shift);
        const Vec2<f32> peak_window_center = MetadataSlice::center(peak_window.shape());

        // The xmap is distorted perpendicular to the tilt-axis due to the tilt geometry.
        // To help the picking, rotate so that the distortion is along the X-axis.
        const auto rotation = noa::math::deg2rad(metadata[target_index].angles[0]);
        const Float33 xmap_inv_transform(
                noa::geometry::translate(slice_center().as<f64>()) *
                noa::geometry::linear2affine(noa::geometry::rotate(rotation)) *
                noa::geometry::translate(-peak_window_center.as<f64>()));
        noa::geometry::transform_2d(xmap, peak_window, xmap_inv_transform);

        // TODO Better fitting of the peak. 2D parabola?
        auto [peak_coordinate, peak_value] = noa::signal::fft::xpeak_2d<noa::fft::FC2FC>(peak_window);
        if (is_double_phase)
            peak_coordinate /= 2;
        const Vec2<f32> shift_rotated = peak_coordinate - peak_window_center;
        const Vec2<f64> shift = noa::geometry::rotate(rotation) * shift_rotated.as<f64>();

        // Normalize the peak. This doesn't work for mutual cross-correlation.
        // TODO Check that this is equivalent to dividing by the auto-correlation peak.
        const auto score = static_cast<f64>(peak_value) /
                           noa::math::sqrt(static_cast<f64>(energy_target) *
                                           static_cast<f64>(energy_reference));

        return {shift, score};
    }

    auto ProjectionMatching::poly_fit_rotation(
            const MetadataStack& metadata
    ) -> ThirdDegreePolynomial {
        // Exclude the first view, assuming it's the global reference.
        const auto rows = static_cast<i64>(metadata.size()) - 1;

        // Find x in Ax=b. Shapes: A(M.N) * x(N.1) = b(M.1)
        const Array<f64> A({1, 1, rows, 4});
        const Array<f64> b({1, 1, rows, 1});
        const auto A_ = A.accessor_contiguous<f64, 2>();
        const auto b_ = b.accessor_contiguous_1d();

        // d + cx + bx^2 + ax^3 = 0
        for (i64 row = 1; row < rows + 1; ++row) { // skip 0
            const MetadataSlice& slice = metadata[row];
            const auto rotation = static_cast<f64>(slice.angles[0]);
            const auto tilt = static_cast<f64>(slice.angles[1]);
            A_(row, 0) = 1;
            A_(row, 1) = tilt;
            A_(row, 2) = tilt * tilt;
            A_(row, 3) = tilt * tilt * tilt;
            b_(row) = rotation;
        }

        // Least-square solution using SVD.
        std::array<f64, 4> svd{};
        std::array<f64, 4> x{};
        noa::math::lstsq(
                A.view(),
                b.view(),
                View<f64>(x.data(), {1, 1, 4, 1}),
                0.,
                View<f64>(svd.data(), 4)
        );

        return ThirdDegreePolynomial{x[3], x[2], x[1], x[0]};
    }

    void ProjectionMatching::center_tilt_axis_(MetadataStack& metadata) {
        Vec2<f64> mean{0};
        auto mean_scale = 1 / static_cast<f64>(metadata.size());
        for (size_t i = 0; i < metadata.size(); ++i) {
            const Vec3<f64> angles = noa::math::deg2rad(metadata[i].angles);
            const Vec2<f64> pitch_tilt = angles.filter(2, 1);
            const Double22 stretch_to_0deg{
                    noa::geometry::rotate(angles[0]) *
                    noa::geometry::scale(1 / noa::math::cos(pitch_tilt)) * // 1 = cos(0deg)
                    noa::geometry::rotate(-angles[0])
            };
            const Vec2<f64> shift_at_0deg = stretch_to_0deg * metadata[i].shifts;
            mean += shift_at_0deg * mean_scale;
        }
        for (size_t i = 0; i < metadata.size(); ++i) {
            const Vec3<f64> angles = noa::math::deg2rad(metadata[i].angles);
            const Vec2<f64> pitch_tilt = angles.filter(2, 1);
            const Double22 shrink_matrix{
                    noa::geometry::rotate(angles[0]) *
                    noa::geometry::scale(noa::math::cos(pitch_tilt)) *
                    noa::geometry::rotate(-angles[0])
            };
            metadata[i].shifts -= shrink_matrix * mean;
        }
    }
}
