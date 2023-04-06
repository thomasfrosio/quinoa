#include <noa/IO.hpp>
#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Optimizer.hpp"

namespace {
    using namespace qn;

    // Data passed through the optimizer.
    struct OptimizerData {
        ProjectionMatching* projector{};
        const ProjectionMatchingParameters* parameters{};

        View<f32> stack;
        MetadataStack metadata;
        View<f32> peak_window_2d;
        Vec2<f32> peak_window_center;

        // This is set at each iteration so that the projector knows what target
        // (and therefore projected-reference) to compute.
        i64 target_index{};
        std::vector<i64> reference_indexes;

        // Saved shifts for each optimizer call. That way, we can always retrieve
        // the shift of the best peak (the best peak is the last one).
        // This is also set at each iteration.
        std::vector<Vec2<f32>> shifts;
    };
}

namespace qn {
    ProjectionMatching::ProjectionMatching(
            const noa::Shape4<i64>& shape,
            noa::Device compute_device,
            noa::Allocator allocator)
            : m_slice_center(MetadataSlice::center(shape)) {
        // Zero padding:
        m_max_size = std::max(shape[2], shape[3]);
        const i64 size_pad = m_max_size * 2;
        m_slice_shape = {1, 1, shape[2], shape[3]};
        m_slice_shape_padded = {1, 1, size_pad, size_pad};
        m_slice_center_padded = static_cast<f32>(size_pad / 2);

        // TODO For the buffers, keep them as separated entities (no alias for in-place FFT).
        //      This could be improved and when/if textures are added this is likely to change.
        const auto options = ArrayOption(compute_device, allocator);
        m_slices = noa::memory::empty<f32>({2, 1, shape[2], shape[3]}, options);
        m_slices_fft = noa::memory::empty<c32>(m_slices.shape().fft(), options);
        m_slices_padded = noa::memory::empty<f32>({2, 1, size_pad, size_pad}, options);
        m_slices_padded_fft = noa::memory::empty<c32>(m_slices_padded.shape().fft(), options);
        m_slice_weight_padded_fft = noa::memory::empty<f32>(m_slice_shape_padded.fft(), options);
        m_cumulative_fov = noa::memory::empty<f32>(m_slice_shape, options);

        //
        m_peak_window = noa::memory::empty<f32>({1, 1, 1, 128 * 128}, options);
    }

    void ProjectionMatching::update_geometry(
            const Array<f32>& stack,
            MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters) {
        qn::Logger::trace("Projection matching alignment...");
        noa::Timer timer;
        timer.start();

        //
        const auto max_shift = noa::math::clamp(parameters.max_shift, Vec2<f32>{16}, Vec2<f32>{128});

        OptimizerData optimizer_data;
        optimizer_data.projector = this;
        optimizer_data.parameters = &parameters;
        optimizer_data.stack = stack.view();
        optimizer_data.metadata = metadata;
        optimizer_data.peak_window_2d = extract_peak_window(max_shift);
        optimizer_data.peak_window_center = MetadataSlice::center(optimizer_data.peak_window_2d.shape());

        auto max_objective_function = [](u32, const f64* x, f64*, void* instance) -> f64 {
            auto* data = static_cast<OptimizerData*>(instance);
            const auto angle_offsets = Vec3<f64>(x).as<f32>();

            const auto [fx, shift] = data->projector->project_and_correlate_(
                    data->stack,
                    data->peak_window_2d,
                    data->metadata,
                    data->target_index,
                    data->reference_indexes,
                    *data->parameters,
                    angle_offsets,
                    data->peak_window_center
            );
            data->shifts.emplace_back(shift);
            qn::Logger::debug("x={}, fx={}", angle_offsets, fx);
            return static_cast<f64>(fx);
        };

        // Set up the optimizer.
        const Optimizer optimizer(NLOPT_LN_NELDERMEAD, 3);
        optimizer.set_max_objective(max_objective_function, &optimizer_data);
        optimizer.set_x_tolerance_abs(0.001);
        optimizer.set_fx_tolerance_abs(0.0001);
        const auto upper_bounds = Vec3<f64>{0.5, 0, 1};
        const auto lower_bounds = -upper_bounds;
        optimizer.set_bounds(lower_bounds.data(), upper_bounds.data());

        // For every slice (excluding the lowest tilt which defines the reference-frame), find the best
        // geometry parameters using the previously aligned slice as reference for alignment. The geometry
        // parameters are, for each slice, the 3D rotation (yaw, tilt, pitch) and the (y,x) shifts.
        optimizer_data.metadata.sort("exposure");
        const auto slice_count = static_cast<i64>(optimizer_data.metadata.size());

        for (i64 target_index = 1; target_index < slice_count; ++target_index) {
            noa::Timer timer_iter;
            timer_iter.start();
            optimizer_data.target_index = target_index;

            // Get the indexes of the reference views for this target view.
            set_reference_indexes_(
                    target_index, optimizer_data.metadata,
                    parameters, optimizer_data.reference_indexes);

            Vec3<f64> x{0};
            f64 fx;
            optimizer.optimize(x.data(), &fx);
            qn::Logger::debug("{}: angles={}, score={}", target_index, x, fx);

            // Update the metadata.
            auto& slice = optimizer_data.metadata[target_index];
            slice.angles += x.as<f32>();
            slice.shifts += optimizer_data.shifts.back();

            qn::Logger::debug("Projection matching shift alignment... iter {} took {}ms",
                              target_index, timer_iter.elapsed());
        }

        if (parameters.center_tilt_axis)
            center_tilt_axis_(optimizer_data.metadata);

        // Update the metadata.
        for (const auto& updated_metadata: optimizer_data.metadata.slices()) {
            for (auto& original_slice: metadata.slices()) {
                if (original_slice.index == updated_metadata.index) {
                    original_slice.angles[0] = updated_metadata.angles[0];
                    original_slice.shifts = updated_metadata.shifts;
                }
            }
        }
        qn::Logger::trace("Projection matching alignment... took {}ms", timer.elapsed());
    }

    auto ProjectionMatching::extract_peak_window(const Vec2<f32>& max_shift) -> View<f32> {
        const auto radius = noa::math::ceil(max_shift).as<i64>();
        const auto elements = noa::math::product(radius * 2);
        return m_peak_window
                .view()
                .subregion(noa::indexing::ellipsis_t{}, noa::indexing::slice_t{0, elements})
                .reshape(Shape4<i64>{1, 1, radius[0] * 2, radius[1] * 2});
    }

    void ProjectionMatching::set_reference_indexes_(
            i64 target_index,
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters,
            std::vector<i64>& output_reference_indexes
    ) {
        const f32 target_tilt_angle = metadata[target_index].angles[1];
        const f32 max_tilt_difference = parameters.backward_tilt_angle_difference;
        const i64 max_index = parameters.backward_use_aligned_only ? target_index : static_cast<i64>(metadata.size());

        output_reference_indexes.clear();
        output_reference_indexes.reserve(static_cast<size_t>(max_index));

        for (i64 reference_index = 0; reference_index < max_index; ++reference_index) {
            const f32 reference_tilt_angle = metadata[reference_index].angles[1];
            const f32 tilt_difference = noa::math::abs(target_tilt_angle - reference_tilt_angle);

            // Of course, do not include the target in the projected-reference.
            if (reference_index != target_index && tilt_difference <= max_tilt_difference)
                output_reference_indexes.emplace_back(reference_index);
        }
    }

    auto ProjectionMatching::project_and_correlate_(
            const View<f32>& stack,
            const View<f32>& peak_window,
            const MetadataStack& metadata,
            i64 target_index,
            const std::vector<i64>& reference_indexes,
            const ProjectionMatchingParameters& parameters,
            const Vec3<f32>& angle_offsets,
            const Vec2<f32>& peak_window_center
    ) -> std::pair<f32, Vec2<f32>> {
        compute_target_and_reference_(stack, metadata, target_index, angle_offsets,
                                      reference_indexes, parameters);

        const auto target_reference = m_slices.view();
        const auto target_reference_fft = m_slices_fft.view();
        if (!parameters.debug_directory.empty())
            noa::io::save(target_reference, parameters.debug_directory / "target_reference.mrc");

        noa::math::normalize(target_reference, target_reference);
        noa::fft::r2c(target_reference, target_reference_fft);
        noa::signal::fft::bandpass<noa::fft::H2H>(
                target_reference_fft, target_reference_fft,
                target_reference.shape(), 0.10f, 0.40f, 0.08f, 0.05f);

        // Overwrite the target with the xmap.
        const auto target_fft = target_reference_fft.subregion(0);
        const auto reference_fft = target_reference_fft.subregion(1);
        const auto xmap = target_reference.subregion(0);
        noa::signal::fft::xmap<noa::fft::H2FC>(target_fft, reference_fft, xmap); // TODO Double phase
        if (!parameters.debug_directory.empty())
            noa::io::save(xmap, parameters.debug_directory / "xmap.mrc");

        const auto [shift, peak_value] = extract_peak_from_xmap_(
                xmap, peak_window,
                m_slice_center, peak_window_center,
                metadata[target_index]);
        return {peak_value, shift};
    }

    void ProjectionMatching::compute_target_and_reference_(
            const View<f32>& stack,
            const MetadataStack& metadata,
            i64 target_index,
            const Vec3<f32>& target_angles_offset,
            const std::vector<i64>& reference_indexes,
            const ProjectionMatchingParameters& parameters
    ) {
        const Vec2<f32>& target_shifts = metadata[target_index].shifts;
        const Vec3<f32> target_angles = noa::math::deg2rad(metadata[target_index].angles + target_angles_offset);

        // The yaw is the CCW angle of the tilt-axis in the slices. For the projection, we want to align
        // the tilt-axis along the Y axis, so subtract this angle and then apply the tilt and pitch.
        const Float33 fwd_target_rotation = noa::geometry::euler2matrix(
                Vec3<f32>{-target_angles[0], target_angles[1], target_angles[2]},
                "zyx", /*intrinsic=*/ false);

        // Use a view to not reference count arrays since nothing is destructed here.
        const auto cumulative_fov = m_cumulative_fov.view();
        const auto slice_weight_padded_fft = m_slice_weight_padded_fft.view();

        // Alias to individual buffers:
        const auto slice0 = m_slices.view().subregion(0);
        const auto slice1 = m_slices.view().subregion(1);
        const auto slice0_padded = m_slices_padded.view().subregion(0);
        const auto slice1_padded = m_slices_padded.view().subregion(1);
        const auto slice0_padded_fft = m_slices_padded_fft.view().subregion(0);
        const auto slice1_padded_fft = m_slices_padded_fft.view().subregion(1);

        // Go through the stack and backward project the reference slices.
        // Reset the buffers for backward projection.
        noa::memory::fill(slice1_padded_fft, c32{0});
        noa::memory::fill(slice_weight_padded_fft, f32{0});
        noa::memory::fill(cumulative_fov, f32{0});

        const auto zero_taper_size = static_cast<f32>(m_max_size) * parameters.smooth_edge_percent;

        // TODO We could batch this and remove the loop. That should be more efficient since we
        //      have a fair amount of reference slices. The issue might be memory though...
        f32 total_weight{0};
        for (i64 reference_index: reference_indexes) {
            const Vec2<f32>& reference_shifts = metadata[reference_index].shifts;
            const Vec3<f32> reference_angles = noa::math::deg2rad(metadata[reference_index].angles);

            // TODO Weighting based on the order of collection? Or is exposure weighting enough?
            // How much the slice should contribute to the final projected-reference.
            [[maybe_unused]] constexpr auto PI = noa::math::Constant<f32>::PI;
            [[maybe_unused]] const f32 tilt_difference = std::abs(target_angles[1] - reference_angles[1]);
            const f32 weight = 1;//noa::math::sinc(tilt_difference * PI / max_tilt_difference);

            // Collect the FOV of this reference slice.
            add_fov_to_cumulative_fov(
                    cumulative_fov, weight,
                    target_angles, target_shifts,
                    reference_angles, reference_shifts,
                    zero_taper_size, m_slice_center);
            total_weight += weight;

            // Get the reference slice ready for back-projection.
            const View reference = stack.subregion(metadata[reference_index].index);
            apply_fov_of_target(
                    reference, slice0,
                    target_angles, target_shifts,
                    reference_angles, reference_shifts,
                    zero_taper_size, m_slice_center);
            noa::memory::resize(slice0, slice0_padded);
            noa::fft::r2c(slice0_padded, slice0_padded_fft);

            // The shift of the reference slice should be removed to have the rotation center at the origin.
            // phase_shift_2d can do the remap, but not in-place, so use remap to center the slice.
            noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                    slice0_padded_fft, slice0_padded_fft,
                    m_slice_shape_padded, -m_slice_center_padded - reference_shifts);
            noa::fft::remap(noa::fft::H2HC, slice0_padded_fft,
                            slice0_padded_fft, m_slice_shape_padded);

            // For the insertion, noa needs the inverse rotation matrix, hence the transpose.
            // For the extraction, it needs the forward matrices, so all good.
            const Float33 inv_reference_rotation = noa::geometry::euler2matrix(
                    Vec3<f32>{-reference_angles[0], reference_angles[1], reference_angles[2]},
                    "ZYX", false).transpose();
            noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                    slice0_padded_fft, m_slice_shape_padded,
                    slice1_padded_fft, m_slice_shape_padded,
                    Float22{}, inv_reference_rotation,
                    Float22{}, fwd_target_rotation,
                    parameters.backward_slice_z_radius,
                    parameters.forward_cutoff);
            noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                    1 / weight, m_slice_shape_padded,
                    slice_weight_padded_fft, m_slice_shape_padded,
                    Float22{}, inv_reference_rotation,
                    Float22{}, fwd_target_rotation,
                    parameters.backward_slice_z_radius,
                    parameters.forward_cutoff);
        }

        // For the target view, simply extract it from the stack and apply the cumulative FOV.
        View target_view = stack.subregion(metadata[target_index].index);
        noa::ewise_trinary(m_cumulative_fov, 1 / total_weight, target_view,
                           slice0, noa::multiply_t{});

        // For the reference view, center the output projected slice onto the target,
        // apply the projected-weight/multiplicity, and apply the cumulative FOV.
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                slice1_padded_fft, slice1_padded_fft,
                m_slice_shape_padded, m_slice_center_padded + target_shifts);
        noa::ewise_trinary(slice1_padded_fft, m_slice_weight_padded_fft, 1e-3f,
                           slice1_padded_fft, noa::divide_epsilon_t{});

        noa::fft::c2r(slice1_padded_fft, slice1_padded);
        noa::memory::resize(slice1_padded, slice1);
        noa::ewise_trinary(m_cumulative_fov, 1 / total_weight, slice1,
                           slice1, noa::multiply_t{});
    }

    void ProjectionMatching::compute_target_and_reference_2(
            const View<f32>& stack,
            const MetadataStack& metadata,
            i64 target_index,
            const Vec3<f32>& target_angles_offset,
            const std::vector<i64>& reference_indexes,
            const ProjectionMatchingParameters& parameters
    ) {
        const Vec2<f32>& target_shifts = metadata[target_index].shifts;
        const Vec3<f32> target_angles = noa::math::deg2rad(metadata[target_index].angles + target_angles_offset);

        // The yaw is the CCW angle of the tilt-axis in the slices. For the projection, we want to align
        // the tilt-axis along the Y axis, so subtract this angle and then apply the tilt and pitch.
        const Float33 fwd_target_rotation = noa::geometry::euler2matrix(
                Vec3<f32>{-target_angles[0], target_angles[1], target_angles[2]},
                "zyx", /*intrinsic=*/ false);

        // Use a view to not reference count arrays since nothing is destructed here.
        const auto cumulative_fov = m_cumulative_fov.view();
        const auto slice_weight_padded_fft = m_slice_weight_padded_fft.view();

        //
        const auto references = noa::memory::empty<f32>(m_slice_shape);
        const auto [references_padded, references_padded_fft] = noa::fft::empty<f32>(m_slice_shape_padded);
        const auto [projected_padded, projected_padded_fft] = noa::fft::empty<f32>(m_slice_shape_padded);

        const auto reference_weights = noa::memory::empty<f32>(reference_indexes.size()); // FIXME batched
        const auto reference_batch_indexes = noa::memory::empty<i32>(reference_indexes.size());
        const auto fov_inv_reference2target = noa::memory::empty<Float23>(reference_indexes.size()); // FIXME
        const auto fov_inv_target2reference = noa::memory::empty<Float23>(reference_indexes.size());

        // Go through the stack and backward project the reference slices.
        // Reset the buffers for backward projection.
        noa::memory::fill(projected_padded_fft, c32{0});
        noa::memory::fill(slice_weight_padded_fft, f32{0});
        noa::memory::fill(cumulative_fov, f32{0});

        const auto zero_taper_size = static_cast<f32>(m_max_size) * parameters.smooth_edge_percent;

        // Utility loop.
        f32 total_weight{0};
        for (i64 i = 0; i < reference_indexes.size(); ++i) {
            const i64 reference_index = reference_indexes[i];
            const Vec2<f32>& reference_shifts = metadata[reference_index].shifts;
            const Vec3<f32> reference_angles = noa::math::deg2rad(metadata[reference_index].angles);

            // TODO Weighting based on the order of collection? Or is exposure weighting enough so leave to 1?
            // How much the slice should contribute to the final projected-reference.
            [[maybe_unused]] constexpr auto PI = noa::math::Constant<f32>::PI;
            [[maybe_unused]] const f32 tilt_difference = std::abs(target_angles[1] - reference_angles[1]);
            const f32 weight =  1;//noa::math::sinc(tilt_difference * PI / max_tilt_difference);
            reference_weights(i, 0, 0, 0) = weight;
            total_weight += weight;

            // Get the references indexes
            reference_batch_indexes(0, 0, 0, i) = metadata[reference_index].index;

            // Compute the matrices for the FOV.
            const auto cos_factor =
                    noa::math::cos(target_angles.filter(2, 1)) /
                    noa::math::cos(reference_angles.filter(2, 1));
            const auto inv_reference2target = noa::math::inverse( // TODO Compute inverse transformation directly
                    noa::geometry::translate(slice_center + target_shifts) *
                    noa::geometry::linear2affine(noa::geometry::rotate(target_angles[0])) *
                    noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                    noa::geometry::linear2affine(noa::geometry::rotate(-reference_angles[0])) *
                    noa::geometry::translate(-slice_center - reference_shifts)
            );

            const auto cos_factor =
                    noa::math::cos(reference_angles.filter(2, 1)) /
                    noa::math::cos(target_angles.filter(2, 1));
            const auto inv_target2reference = noa::math::inverse( // TODO Compute inverse transformation directly?
                    noa::geometry::translate(slice_center + reference_shifts) *
                    noa::geometry::linear2affine(noa::geometry::rotate(reference_angles[0])) *
                    noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                    noa::geometry::linear2affine(noa::geometry::rotate(-target_angles[0])) *
                    noa::geometry::translate(-slice_center - target_shifts)
            );
        }

        // Collect the FOV of this reference slice.
        noa::geometry::rectangle(
                reference_weights, cumulative_fov,
                m_slice_center, m_slice_center - zero_taper_size,
                zero_taper_size, inv_reference2target);

            // Get the reference slice ready for back-projection.
            noa::memory::copy_batches(stack, references, reference_batch_indexes);

            apply_fov_of_target(
                    references, references,
                    target_angles, target_shifts,
                    reference_angles, reference_shifts,
                    zero_taper_size, m_slice_center);
            noa::memory::resize(references, references_padded);
            noa::fft::r2c(references_padded, references_padded_fft);

            // The shift of the reference slice should be removed to have the rotation center at the origin.
            // phase_shift_2d can do the remap, but not in-place, so use remap to center the slice.
            noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                    references_padded_fft, references_padded_fft,
                    m_slice_shape_padded, -m_slice_center_padded - reference_shifts);
            noa::fft::remap(noa::fft::H2HC, references_padded_fft,
                            references_padded_fft, m_slice_shape_padded);

            // For the insertion, noa needs the inverse rotation matrix, hence the transpose.
            // For the extraction, it needs the forward matrices, so all good.
            const Float33 inv_reference_rotation = noa::geometry::euler2matrix(
                    Vec3<f32>{-reference_angles[0], reference_angles[1], reference_angles[2]},
                    "ZYX", false).transpose();
            noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                    references_padded_fft, m_slice_shape_padded,
                    projected_padded_fft, m_slice_shape_padded,
                    Float22{}, inv_reference_rotation,
                    Float22{}, fwd_target_rotation,
                    parameters.backward_slice_z_radius,
                    parameters.forward_cutoff);
            noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                    1 / weight, m_slice_shape_padded,
                    slice_weight_padded_fft, m_slice_shape_padded,
                    Float22{}, inv_reference_rotation,
                    Float22{}, fwd_target_rotation,
                    parameters.backward_slice_z_radius,
                    parameters.forward_cutoff);
        }

        // For the target view, simply extract it from the stack and apply the cumulative FOV.
        View target_view = stack.subregion(metadata[target_index].index);
        noa::ewise_trinary(m_cumulative_fov, 1 / total_weight, target_view,
                           slice0, noa::multiply_t{});

        // For the reference view, center the output projected slice onto the target,
        // apply the projected-weight/multiplicity, and apply the cumulative FOV.
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                slice1_padded_fft, slice1_padded_fft,
                m_slice_shape_padded, m_slice_center_padded + target_shifts);
        noa::ewise_trinary(slice1_padded_fft, m_slice_weight_padded_fft, 1e-3f,
                           slice1_padded_fft, noa::divide_epsilon_t{});

        noa::fft::c2r(slice1_padded_fft, slice1_padded);
        noa::memory::resize(slice1_padded, slice1);
        noa::ewise_trinary(m_cumulative_fov, 1 / total_weight, slice1,
                           slice1, noa::multiply_t{});
    }

    void ProjectionMatching::apply_fov_of_target(
            const View<f32>& input_slice, const View<f32>& output_slice,
            const Vec3<f32>& target_angles, const Vec2<f32>& target_shifts,
            const Vec3<f32>& reference_angles, const Vec2<f32>& reference_shifts,
            f32 zero_taper_size, const Vec2<f32>& slice_center) {
        const auto cos_factor =
                noa::math::cos(reference_angles.filter(2, 1)) /
                noa::math::cos(target_angles.filter(2, 1));
        const auto inv_target2reference = noa::math::inverse( // TODO Compute inverse transformation directly?
                noa::geometry::translate(slice_center + reference_shifts) *
                noa::geometry::linear2affine(noa::geometry::rotate(reference_angles[0])) *
                noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                noa::geometry::linear2affine(noa::geometry::rotate(-target_angles[0])) *
                noa::geometry::translate(-slice_center - target_shifts)
        );
        noa::geometry::rectangle(
                input_slice, output_slice,
                slice_center, slice_center - zero_taper_size,
                zero_taper_size, inv_target2reference);
    }

    void ProjectionMatching::add_fov_to_cumulative_fov(
            const View<f32>& cumulative_fov, f32 weight,
            const Vec3<f32>& target_angles, const Vec2<f32>& target_shifts,
            const Vec3<f32>& reference_angles, const Vec2<f32>& reference_shifts,
            f32 zero_taper_size, const Vec2<f32>& slice_center) {
        const auto cos_factor =
                noa::math::cos(target_angles.filter(2, 1)) /
                noa::math::cos(reference_angles.filter(2, 1));
        const auto inv_reference2target = noa::math::inverse( // TODO Compute inverse transformation directly
                noa::geometry::translate(slice_center + target_shifts) *
                noa::geometry::linear2affine(noa::geometry::rotate(target_angles[0])) *
                noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                noa::geometry::linear2affine(noa::geometry::rotate(-reference_angles[0])) *
                noa::geometry::translate(-slice_center - reference_shifts)
        );
        noa::geometry::rectangle(
                cumulative_fov, cumulative_fov,
                slice_center, slice_center - zero_taper_size,
                zero_taper_size, inv_reference2target, noa::plus_t{}, weight);
    }

void ProjectionMatching::add_fov_to_cumulative_fov2(
        const View<f32>& reference_weights, const View<f32>& cumulative_fov,
        const Vec3<f32>& target_angles, const Vec2<f32>& target_shifts,
        const Vec3<f32>& reference_angles, const Vec2<f32>& reference_shifts,
        f32 zero_taper_size, const Vec2<f32>& slice_center) {
    const auto cos_factor =
            noa::math::cos(target_angles.filter(2, 1)) /
            noa::math::cos(reference_angles.filter(2, 1));
    const auto inv_reference2target = noa::math::inverse( // TODO Compute inverse transformation directly
            noa::geometry::translate(slice_center + target_shifts) *
            noa::geometry::linear2affine(noa::geometry::rotate(target_angles[0])) *
            noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
            noa::geometry::linear2affine(noa::geometry::rotate(-reference_angles[0])) *
            noa::geometry::translate(-slice_center - reference_shifts)
    );
    noa::geometry::rectangle(
            cumulative_fov, cumulative_fov,
            slice_center, slice_center - zero_taper_size,
            zero_taper_size, inv_reference2target, noa::plus_t{}, weight);
}

    auto ProjectionMatching::extract_peak_from_xmap_(
            const View<f32>& xmap,
            const View<f32>& peak_window,
            Vec2<f32> xmap_center,
            Vec2<f32> peak_window_center,
            const MetadataSlice& slice
    ) -> std::pair<Vec2<f32>, f32> {
        // Rotate around the xmap center, but then shift to the peak_window center.
        // transform_2d will only render the peak_window
        const auto yaw = noa::math::deg2rad(slice.angles[0]);
        const Float33 xmap_inv_transform(
                noa::geometry::translate(xmap_center) *
                noa::geometry::linear2affine(noa::geometry::rotate(yaw)) *
                noa::geometry::translate(-peak_window_center));
        noa::geometry::transform_2d(xmap, peak_window, xmap_inv_transform);
//            noa::io::save(peak_window, "/home/thomas/Projects/quinoa/tests/simtilt/debug_pm/xmap_cropped.mrc");

        // TODO Better fitting of the peak. 2D parabola?
        const auto [peak_rotated, peak_value] = noa::signal::fft::xpeak_2d<noa::fft::FC2FC>(peak_window);
        Vec2<f32> shift_rotated = peak_rotated - peak_window_center;
        const Vec2<f32> shift = noa::geometry::rotate(yaw) * shift_rotated;
        return {shift, peak_value};
    }

    void ProjectionMatching::center_tilt_axis_(MetadataStack& metadata) {
        Vec2<f64> mean{0};
        auto mean_scale = 1 / static_cast<f64>(metadata.size());
        for (size_t i = 0; i < metadata.size(); ++i) {
            const Vec3<f64> angles = noa::math::deg2rad(metadata[i].angles).as<f64>();
            const Vec2<f64> pitch_tilt = angles.filter(2, 1);
            const Double22 stretch_to_0deg{
                    noa::geometry::rotate(angles[0]) *
                    noa::geometry::scale(1 / noa::math::cos(pitch_tilt)) * // 1 = cos(0deg)
                    noa::geometry::rotate(-angles[0])
            };
            const Vec2<f64> shift_at_0deg = stretch_to_0deg * metadata[i].shifts.as<f64>();
            mean += shift_at_0deg * mean_scale;
        }
        for (size_t i = 0; i < metadata.size(); ++i) {
            const Vec3<f64> angles = noa::math::deg2rad(metadata[i].angles).as<f64>();
            const Vec2<f64> pitch_tilt = angles.filter(2, 1);
            const Double22 shrink_matrix{
                    noa::geometry::rotate(angles[0]) *
                    noa::geometry::scale(noa::math::cos(pitch_tilt)) *
                    noa::geometry::rotate(-angles[0])
            };
            metadata[i].shifts -= (shrink_matrix * mean).as<f32>();
        }
    }
}
