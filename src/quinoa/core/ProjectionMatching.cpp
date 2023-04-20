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

        View<f32> stack;
        MetadataStack metadata;

        // This is set at each iteration so that the projector knows what target
        // (and therefore projected-reference) to compute.
        i64 target_index{};
        std::vector<i64> reference_indexes;
    };
}

namespace qn {
    ProjectionMatching::ProjectionMatching(
            const noa::Shape4<i64>& shape,
            noa::Device compute_device,
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters,
            noa::Allocator allocator)
            : m_slice_center(MetadataSlice::center(shape)) {
        // Zero padding:
        m_max_size = std::max(shape[2], shape[3]);
        const i64 size_pad = m_max_size * 2;
        m_slice_padded_shape = {1, 1, size_pad, size_pad};
        m_slice_center_padded = static_cast<f32>(size_pad / 2);

        // Find the maximum number of reference slices we'll need to hold at a given time.
        const i64 max_reference_count = max_references_count_(metadata, parameters);
        const auto target_reference_shape = Shape4<i64>{2, 1, shape[2], shape[3]};
        const auto target_reference_padded_shape = Shape4<i64>{2, 1, size_pad, size_pad};
        const auto slices_shape = Shape4<i64>{max_reference_count, 1, shape[2], shape[3]};
        m_slices_padded_shape = Shape4<i64>{max_reference_count, 1, size_pad, size_pad};

        // Device-only buffers.
        const auto device_options = ArrayOption(compute_device, allocator);
        m_target_reference_fft = noa::memory::empty<c32>(target_reference_shape.fft(), device_options);
        m_references = noa::memory::empty<f32>(slices_shape, device_options);
        m_references_padded_fft = noa::memory::empty<c32>(m_slices_padded_shape.fft(), device_options);
        m_target_reference_padded_fft = noa::memory::empty<c32>(target_reference_padded_shape.fft(), device_options);
        m_multiplicity_padded_fft = noa::memory::empty<f32>({1, 1, size_pad, size_pad / 2 + 1}, device_options);
        m_peak_window = noa::memory::empty<f32>({1, 1, 1, 128 * 128}, device_options);

        // Managed buffers. This allocates ~n*25*4 bytes.
        const auto managed_options = ArrayOption(compute_device, Allocator::MANAGED);
        const auto max_reference_shape = Shape4<i64>{max_reference_count, 1, 1, 1};
        m_reference_weights = noa::memory::empty<f32>(max_reference_shape, managed_options);
        m_reference_batch_indexes = noa::memory::like<Vec4<i32>>(m_reference_weights);
        m_fov_inv_reference2target = noa::memory::like<Float23>(m_reference_weights); // FIXME Batch?
        m_fov_inv_target2reference = noa::memory::like<Float23>(m_reference_weights); // FIXME Batch?
        m_insert_inv_references_rotation = noa::memory::like<Float33>(m_reference_weights);
        m_reference_shifts_center2origin = noa::memory::like<Vec2<f32>>(m_reference_weights);
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
        const View<f32> peak_window_2d = extract_peak_window_(max_shift);
        const Vec2<f32> peak_window_center = MetadataSlice::center(peak_window_2d.shape());

        OptimizerData optimizer_data;
        optimizer_data.projector = this;
        optimizer_data.parameters = &parameters;
        optimizer_data.stack = stack.view();
        optimizer_data.metadata = metadata;

        auto max_objective_function = [](u32, const f64* x, f64*, void* instance) -> f64 {
            auto* data = static_cast<OptimizerData*>(instance);
            const auto angle_offsets = Vec3<f64>(x).as<f32>();

            const auto fx = data->projector->project_and_correlate_(
                    data->stack,
                    data->metadata,
                    data->target_index,
                    data->reference_indexes,
                    *data->parameters,
                    angle_offsets
            );
            qn::Logger::debug("x={}, fx={}", angle_offsets, fx);
            return static_cast<f64>(fx);
        };

        // Set up the optimizer.
        const Optimizer optimizer(NLOPT_LN_NELDERMEAD, 3);
        optimizer.set_max_objective(max_objective_function, &optimizer_data);
        optimizer.set_x_tolerance_abs(0.005);
//        optimizer.set_fx_tolerance_abs(0.0001);
        const auto upper_bounds = Vec3<f64>{2, 0, 0};
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
//            optimizer.optimize(x.data(), &fx);
            fx = project_and_correlate_(
                    optimizer_data.stack,
                    optimizer_data.metadata,
                    optimizer_data.target_index,
                    optimizer_data.reference_indexes,
                    *optimizer_data.parameters,
                    {}
            );

            // Now compute the shift for these last/best angles.
            // The target and projected-reference are already computed,
            // so we can just use them instead of re-projecting everything.
            const Vec2<f32> shifts = extract_final_shift_(
                    optimizer_data.metadata, target_index, parameters, peak_window_2d, peak_window_center);
            qn::Logger::debug("{}: angles={}, score={}, shift={}", target_index, x, fx, shifts);

            // Update the metadata.
            auto& slice = optimizer_data.metadata[target_index];
            slice.angles += x.as<f32>();
            slice.shifts += shifts;

            qn::Logger::debug("Projection matching shift alignment... iter {} took {}ms",
                              target_index, timer_iter.elapsed());
        }

        if (parameters.center_tilt_axis)
            center_tilt_axis_(optimizer_data.metadata);

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

    auto ProjectionMatching::extract_peak_window_(const Vec2<f32>& max_shift) -> View<f32> {
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
            const f32 target_tilt_angle = metadata[target_index].angles[1];
            const i64 max_index = parameters.backward_use_aligned_only ? target_index : slice_count;

            // Count how many references are needed for the current target.
            i64 count = 0;
            for (i64 reference_index = 0; reference_index < max_index; ++reference_index) {
                const f32 reference_tilt_angle = metadata[reference_index].angles[1];
                const f32 tilt_difference = noa::math::abs(target_tilt_angle - reference_tilt_angle);
                if (reference_index != target_index && tilt_difference <= parameters.backward_tilt_angle_difference)
                    ++count;
            }

            max_count = std::max(max_count, count);
        }

        return max_count;
    }

    auto ProjectionMatching::project_and_correlate_(
            const View<f32>& stack,
            const MetadataStack& metadata,
            i64 target_index,
            const std::vector<i64>& reference_indexes,
            const ProjectionMatchingParameters& parameters,
            const Vec3<f32>& angle_offsets
    ) -> f32 {
        // First, get the target slice, and compute the projected-reference slice.
        compute_target_and_reference_(stack, metadata, target_index, angle_offsets,
                                      reference_indexes, parameters);
        const auto target_reference = m_references.subregion(noa::indexing::slice_t{0, 2});
        if (!parameters.debug_directory.empty())
            noa::io::save(target_reference, parameters.debug_directory /
            noa::string::format("target_reference_{:>02}.mrc", target_index));

        const auto target = target_reference.subregion(0);
        const auto reference = target_reference.subregion(1);
        noa::math::normalize(target, target);
        noa::math::normalize(reference, reference);

        const auto target_reference_fft = m_target_reference_fft.view();
        noa::fft::r2c(target_reference, target_reference_fft); // TODO Norm::NONE?
        noa::signal::fft::bandpass<noa::fft::H2H>(
                target_reference_fft, target_reference_fft,
                target_reference.shape(), 0.10f, 0.40f, 0.08f, 0.05f);

//        return -noa::math::rmsd(target, reference);



//        // Compute the cross-correlation coefficient.
//        const auto target_fft = target_reference_fft.subregion(0);
//        const auto reference_fft = target_reference_fft.subregion(1);
//        const auto cc_score = noa::signal::fft::xcorr<fft::H2H>(target_fft, reference_fft, target.shape());
//        return cc_score;

        // At this point, m_target_reference_fft is valid and contains the target
        // and projected-reference ready for extract_final_shift_().
    }

    auto ProjectionMatching::extract_final_shift_(
            const MetadataStack& metadata,
            i64 target_index,
            const ProjectionMatchingParameters& parameters,
            const View<f32>& peak_window,
            const Vec2<f32>& peak_window_center
    ) -> Vec2<f32> {
        // Assume this is called after project_and_correlate_().
        const auto target_reference_fft = m_target_reference_fft.view();
        const auto target_fft = target_reference_fft.subregion(0);
        const auto reference_fft = target_reference_fft.subregion(1);

        // At this point, we can use any of the m_references slices for the xmap.
        // We rotate the xmap before the picking, so compute the centered xmap.
        const auto xmap = m_references.view().subregion(0);
        noa::signal::fft::xmap<noa::fft::H2FC>(
                target_fft, reference_fft, xmap); // TODO DOUBLE_PHASE
        if (!parameters.debug_directory.empty())
            noa::io::save(xmap, parameters.debug_directory /
                    noa::string::format("xmap_{:>02}.mrc", target_index));

        const auto [shift, peak_value] = extract_peak_from_xmap_(
                xmap, peak_window,
                m_slice_center, peak_window_center,
                metadata[target_index]); // TODO DOUBLE_PHASE
        return shift;
    }

    void ProjectionMatching::compute_target_and_reference_(
            const View<f32>& stack,
            const MetadataStack& metadata,
            i64 target_index,
            const Vec3<f32>& target_angles_offset,
            const std::vector<i64>& reference_indexes,
            const ProjectionMatchingParameters& parameters
    ) {
        // Use views to not reference count arrays since nothing is destructed here.
        const auto references_count = static_cast<i64>(reference_indexes.size());
        const auto range = noa::indexing::slice_t{0, references_count};
        const auto references_padded_shape = m_slices_padded_shape.pop_front().push_front(references_count);

        const auto references_all = m_references.view();
        const View<f32> references = references_all.subregion(range);
        const View<c32> references_padded_fft = m_references_padded_fft.view().subregion(range);
        const View<f32> references_padded = noa::fft::alias_to_real(references_padded_fft, references_padded_shape);

        const auto reference_weights = m_reference_weights.view().subregion(range);
        const auto reference_batch_indexes = m_reference_batch_indexes.view().subregion(range);
        const auto fov_inv_reference2target = m_fov_inv_reference2target.view().subregion(range);
        const auto fov_inv_target2reference = m_fov_inv_target2reference.view().subregion(range);
        const auto insert_inv_references_rotation = m_insert_inv_references_rotation.view().subregion(range);
        const auto reference_shifts_center2origin = m_reference_shifts_center2origin.view().subregion(range);

        // Target geometry:
        // The yaw is the CCW angle of the tilt-axis in the slices. For the projection, we want to align
        // the tilt-axis along the Y axis, so subtract this angle and then apply the tilt and pitch.
        const Vec2<f32>& target_shifts = metadata[target_index].shifts;
        const Vec3<f32> target_angles = noa::math::deg2rad(metadata[target_index].angles + target_angles_offset);
        const Float33 extract_fwd_target_rotation = noa::geometry::euler2matrix(
                Vec3<f32>{-target_angles[0], target_angles[1], target_angles[2]},
                "zyx", /*intrinsic=*/ false);

        // Utility loop:
        f32 total_weight{0};
        for (i64 i = 0; i < references_count; ++i) {
            const i64 reference_index = reference_indexes[static_cast<size_t>(i)];
            const Vec2<f32>& reference_shifts = metadata[reference_index].shifts;
            const Vec3<f32> reference_angles = noa::math::deg2rad(metadata[reference_index].angles);

            // TODO Weighting based on the order of collection? Or use exposure weighting?
            // Multiplicity and weight.
            // How much the slice should contribute to the final projected-reference.
            [[maybe_unused]] constexpr auto PI = noa::math::Constant<f32>::PI;
            [[maybe_unused]] const f32 tilt_difference = std::abs(target_angles[1] - reference_angles[1]);
            const f32 weight = 1; //noa::math::sinc(tilt_difference * PI / parameters.backward_tilt_angle_difference);
            reference_weights(i, 0, 0, 0) = 1 / weight; // FIXME?
            total_weight += weight;

            // Get the references indexes. These are the indexes of the reference slices within the input stack.
            reference_batch_indexes(i, 0, 0, 0) = {metadata[reference_index].index, 0, 0, 0};

            // Compute the matrices for the FOV.
            const auto cos_factor =
                    noa::math::cos(target_angles.filter(2, 1)) /
                    noa::math::cos(reference_angles.filter(2, 1));
            const auto reference2target =
                    noa::geometry::translate(m_slice_center + target_shifts) *
                    noa::geometry::linear2affine(noa::geometry::rotate(target_angles[0])) *
                    noa::geometry::linear2affine(noa::geometry::scale(cos_factor)) *
                    noa::geometry::linear2affine(noa::geometry::rotate(-reference_angles[0])) *
                    noa::geometry::translate(-m_slice_center - reference_shifts);
            fov_inv_reference2target(i, 0, 0, 0) = noa::geometry::affine2truncated(reference2target.inverse());
            fov_inv_target2reference(i, 0, 0, 0) = noa::geometry::affine2truncated(reference2target);

            // Shifts to phase-shift rotation center at the array origin.
            reference_shifts_center2origin(i, 0, 0, 0) = -m_slice_center_padded - reference_shifts;

            // For the insertion, noa needs the inverse rotation matrix, hence the transpose call.
            insert_inv_references_rotation(i, 0, 0, 0) = noa::geometry::euler2matrix(
                    Vec3<f32>{-reference_angles[0], reference_angles[1], reference_angles[2]},
                    "ZYX", false).transpose();
        }

        // Get the reference slices ready for back-projection.
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

        const auto zero_taper_size = static_cast<f32>(m_max_size) * parameters.smooth_edge_percent;
        noa::geometry::rectangle(
                input_reference_slices, references,
                m_slice_center, m_slice_center - zero_taper_size,
                zero_taper_size, fov_inv_target2reference);
        noa::memory::resize(references, references_padded);
        noa::fft::r2c(references_padded, references_padded_fft);

        // The shift of the reference slice should be removed to have the rotation center at the origin.
        // phase_shift_2d can do the remap, but not in-place, so use remap to center the slice.
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                references_padded_fft, references_padded_fft,
                references_padded_shape, reference_shifts_center2origin);
        noa::fft::remap(noa::fft::H2HC, references_padded_fft,
                        references_padded_fft, references_padded_shape);

        // We'll save the target and reference next to each other.
        const View<c32> target_reference_padded_fft = m_target_reference_padded_fft.view();
        const View<c32> target_padded_fft = target_reference_padded_fft.subregion(0);
        const View<c32> reference_padded_fft = target_reference_padded_fft.subregion(1);
        const View<f32> multiplicity_padded_fft = m_multiplicity_padded_fft.view();
        const View<f32> target_reference_padded = noa::fft::alias_to_real(
                target_reference_padded_fft, m_slice_padded_shape.pop_front().push_front(2)); // output

        // Projection:
        // - The projected reference (and its corresponding sampling function) is reconstructed
        //   by adding the contribution of the input reference slices to the relevant "projected" frequencies.
        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                references_padded_fft, references_padded_shape,
                reference_padded_fft, m_slice_padded_shape,
                Float22{}, insert_inv_references_rotation,
                Float22{}, extract_fwd_target_rotation,
                parameters.backward_slice_z_radius, false,
                parameters.forward_cutoff);
        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                noa::indexing::broadcast(reference_weights, references_padded_shape.fft()), references_padded_shape,
                multiplicity_padded_fft, m_slice_padded_shape,
                Float22{}, insert_inv_references_rotation,
                Float22{}, extract_fwd_target_rotation,
                parameters.backward_slice_z_radius, false,
                parameters.forward_cutoff);

        // Center projection back and shift onto the target.
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                reference_padded_fft, reference_padded_fft,
                m_slice_padded_shape, m_slice_center_padded + target_shifts);

        // We want to apply the same projection-weighting onto the target too,
        // so zero pad and fft the target too.
        const View<f32> target_padded = target_reference_padded.subregion(0);
        View target_view = stack.subregion(metadata[target_index].index);
        noa::memory::resize(target_view, target_padded);
        noa::fft::r2c(target_padded, target_padded_fft);

        // Apply weight/multiplicity to both the target and projected reference.
        // If weight is 0 (i.e. less than machine epsilon), the frequency is set to 0.
        // Otherwise, divide by the multiplicity, effectively applying the sampling function.
        noa::ewise_binary(reference_padded_fft, multiplicity_padded_fft,
                          reference_padded_fft, noa::divide_safe_t{}); // FIXME

//        noa::ewise_binary(multiplicity_padded_fft, 1e-5f, multiplicity_padded_fft, noa::greater_t{});
//        noa::ewise_binary(multiplicity_padded_fft, target_padded_fft, target_padded_fft, noa::multiply_t{});

//        noa::signal::fft::bandpass<noa::fft::H2H>(
//                target_reference_padded_fft, target_reference_padded_fft,
//                target_reference_padded.shape(), 0.10f, 0.40f, 0.08f, 0.05f);

        noa::fft::c2r(target_reference_padded_fft, target_reference_padded);

        // Gather the FOV of the reference slices into a single cumulative FOV.
        // The target and the projected reference are saved next to each other.
        const auto target_reference = references_all.subregion(noa::indexing::slice_t{0, 2});
        const auto cumulative_fov = references_all.subregion(2);
        noa::geometry::rectangle(
                reference_weights, cumulative_fov,
                m_slice_center, m_slice_center - zero_taper_size,
                zero_taper_size, fov_inv_reference2target);
        noa::memory::resize(target_reference_padded, target_reference);
        noa::ewise_trinary(cumulative_fov, 1 / total_weight, target_reference,
                           target_reference, noa::multiply_t{});
    }

    auto ProjectionMatching::extract_peak_from_xmap_(
            const View<f32>& xmap,
            const View<f32>& peak_window,
            Vec2<f32> xmap_center,
            Vec2<f32> peak_window_center,
            const MetadataSlice& slice
    ) -> std::pair<Vec2<f32>, f32> {
        // The xmap is distorted perpendicular to the tilt-axis. To help the picking, rotate so that
        // the distortion is along the X-axis. Rotate around the xmap center, but then shift to the
        // peak_window center. transform_2d will only render the small peak_window, which should
        // make the transformation and the picking very cheap to compute.
        const auto yaw = noa::math::deg2rad(slice.angles[0]);
        const Float33 xmap_inv_transform(
                noa::geometry::translate(xmap_center) *
                noa::geometry::linear2affine(noa::geometry::rotate(yaw)) *
                noa::geometry::translate(-peak_window_center));
        noa::geometry::transform_2d(xmap, peak_window, xmap_inv_transform);

        // TODO Better fitting of the peak. 2D parabola?
        const auto [peak_rotated, peak_value] = noa::signal::fft::xpeak_2d<noa::fft::FC2FC>(peak_window);
        const Vec2<f32> shift_rotated = peak_rotated - peak_window_center;
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
