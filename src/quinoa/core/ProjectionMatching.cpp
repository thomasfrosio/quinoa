#include <noa/IO.hpp>
#include <optional>

#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/Utilities.h"
#include "quinoa/core/CommonArea.hpp"
#include "quinoa/core/Ewise.hpp"

namespace {
    using namespace qn;

    // Data passed through the optimizer.
    struct OptimizerData {
        const View<f32>* stack{};
        const ProjectionMatching* projection_matching{};
        const ProjectionMatchingParameters* parameters{};

        MetadataStack metadata{};
        CommonArea common_area{};

        const CubicSplineGrid<f64, 1>* rotation_model{};
    };
}

namespace qn {
    u64 ProjectionMatching::predict_memory_usage(
            const noa::Shape2<i64>& shape,
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters) noexcept {
        i64 size_padded = std::max(shape[0], shape[1]) * 2;
        if (parameters.zero_pad_to_fast_fft_size)
            size_padded = noa::fft::next_fast_size(size_padded);
        const auto size_padded_u = static_cast<u64>(size_padded);

        const auto max_reference_count = static_cast<u64>(max_references_count_(metadata, parameters));

        const u64 slices_elements = shape.as<u64>().elements() * (max_reference_count + 3);
        const u64 slices_padded_elements = size_padded_u * size_padded_u * (max_reference_count + 3);
        const u64 multiplicity_padded_rfft_elements = size_padded_u * (size_padded_u / 2 + 1);
        const u64 peak_window = 256 * 256;
        const u64 managed_buffers = max_reference_count * 25;

        return slices_elements * slices_padded_elements *
               multiplicity_padded_rfft_elements *
               peak_window * managed_buffers * sizeof(f32);
    }

    ProjectionMatching::ProjectionMatching(
            const noa::Shape2<i64>& shape,
            noa::Device compute_device,
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters,
            noa::Allocator allocator) {
        // Zero padding:
        i64 size_padded = std::max(shape[0], shape[1]) * 2;
        if (parameters.zero_pad_to_fast_fft_size)
            size_padded = noa::fft::next_fast_size(size_padded);

        // Find the maximum number of reference slices we'll need to hold at a given time.
        const i64 max_reference_count = max_references_count_(metadata, parameters);
        const auto target_reference_shape = Shape4<i64>{2, 1, shape[0], shape[1]};
        const auto target_reference_padded_shape = Shape4<i64>{2, 1, size_padded, size_padded};

        // Add an extra slice here to store the target at the end.
        const auto slices_shape = Shape4<i64>{max_reference_count + 1, 1, shape[0], shape[1]};
        const auto slices_padded_shape = Shape4<i64>{max_reference_count + 1, 1, size_padded, size_padded};

        // Device-only buffers.
        const auto device_options = ArrayOption(compute_device, allocator);
        m_slices = noa::memory::empty<f32>(slices_shape, device_options);
        m_slices_padded_fft = noa::memory::empty<c32>(slices_padded_shape.rfft(), device_options);
        m_target_reference_fft = noa::memory::empty<c32>(target_reference_shape.rfft(), device_options);
        m_target_reference_padded_fft = noa::memory::empty<c32>(target_reference_padded_shape.rfft(), device_options);
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
            const View<f32>& stack,
            MetadataStack& metadata,
            const CommonArea& common_area,
            const CubicSplineGrid<f64, 1>& rotation_model,
            const ProjectionMatchingParameters& parameters
    ) {
        qn::Logger::trace("Projection matching alignment...");
        noa::Timer timer;
        timer.start();

        const auto maximization_function = [](
                u32 n_parameters, [[maybe_unused]] const f64* parameters,
                f64* gradients, void* instance
        ) -> f64 {
            auto& data = *static_cast<OptimizerData*>(instance);
            // TODO memoize parameters/gradients/score
            //      Memoizer<std::array<f64, 3>, std::pair<std::array<f64, 3>, std::array<f64, 3>>>;

            // Take a copy of the metadata, because the projection matching updates it and
            // 1) we actually don't care about the shift updates from the projection matching at this point,
            // 2) we want to use the same metadata for every evaluation.
            auto metadata = data.metadata;

            const f64 score = data.projection_matching->run_projection_matching_(
                    *data.stack, metadata, data.common_area, *data.parameters,
                    *data.rotation_model);

            // Numerically estimate the gradients (along each axis) using the central finite difference.
            // This requires 2n evaluations, n being the number of parameters.
            if (gradients != nullptr) {
                // This is the same as the input parameters
                f64* model_parameters = data.rotation_model->data();

                for (u32 i = 0; i < n_parameters; ++i) {
                    const f64 parameter = model_parameters[i];

                    // Use 32-bits precision to make the delta larger (~0.007), making sure it's
                    // not lost by noa, which uses 32-bits precisions for vector and matrices.
                    const f64 delta = static_cast<f64>(CentralFiniteDifference::delta(static_cast<f32>(parameter)));

                    model_parameters[i] = parameter - delta;
                    const f64 score_minus = data.projection_matching->run_projection_matching_(
                            *data.stack, metadata, data.common_area, *data.parameters,
                            *data.rotation_model);

                    model_parameters[i] = parameter + delta;
                    const f64 score_plus = data.projection_matching->run_projection_matching_(
                            *data.stack, metadata, data.common_area, *data.parameters,
                            *data.rotation_model);

                    gradients[i] = (score_plus - score_minus) / (2 * delta); // central finite difference
                    model_parameters[i] = parameter; // reset original value
                }
            }

            return score;
        };

        // Set up the data used by the maximization function.
        OptimizerData optimizer_data;
        optimizer_data.stack = &stack;
        optimizer_data.projection_matching = this;
        optimizer_data.parameters = &parameters;
        optimizer_data.metadata = metadata;
        optimizer_data.metadata.sort("absolute_tilt"); // TODO exposure?
        optimizer_data.common_area = common_area;
        optimizer_data.common_area.reserve(metadata.size(), m_slices.device());
        optimizer_data.rotation_model = &rotation_model;

        // Set up the optimizer.
        const Optimizer optimizer(NLOPT_LN_SBPLX, 1);
        optimizer.set_max_objective(maximization_function, &optimizer_data);
        const f64 bound = rotation_model.data()[0];
        optimizer.set_bounds(bound - parameters.rotation_range, bound + parameters.rotation_range);
        optimizer.set_x_tolerance_abs(parameters.rotation_tolerance_abs);
        optimizer.set_initial_step(parameters.rotation_initial_step);

        // Optimize the rotation model by maximizing the normalized CC of the projection matching.
        const f64 best_score = optimizer.optimize(rotation_model.data());

        // The optimizer can do some post-processing on the parameters, so the last projection matching
        // might not have been run on these final parameters. As such, do a final pass with the actual
        // best parameters returned by the optimizer.
        const f64 final_score = run_projection_matching_(
                stack, optimizer_data.metadata, common_area, parameters, rotation_model);

        // Just for debugging, log this small difference, if any.
        qn::Logger::debug("best_score={:.6g}, final_score:{:.6g}", best_score, final_score);

        // Update the metadata.
        for (const auto& updated_metadata: optimizer_data.metadata.slices()) {
            for (auto& original_slice: metadata.slices()) {
                if (original_slice.index == updated_metadata.index) {
                    original_slice.angles = updated_metadata.angles;
                    original_slice.shifts = updated_metadata.shifts;
                }
            }
        }

        qn::Logger::trace("Projection matching alignment... took {}ms ({} evaluations)",
                          timer.elapsed(), optimizer.number_of_evaluations());
    }
}

// Private methods:
namespace qn {
    auto ProjectionMatching::extract_peak_window_(const Vec2<f64>& max_shift) const -> View<f32> {
        const auto radius = noa::math::ceil(max_shift).as<i64>();
        const auto diameter = radius * 2 + 1;
        const auto shape = noa::fft::next_fast_shape(Shape4<i64>{1, 1, diameter[0], diameter[1]});
        return m_peak_window
                .view()
                .subregion(noa::indexing::ellipsis_t{}, noa::indexing::slice_t{0, shape.elements()})
                .reshape(shape);
    }

    void ProjectionMatching::set_reference_indexes_(
            i64 target_index,
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters,
            std::vector<i64>& output_reference_indexes
    ) {
        // Assume the metadata is sorted so that all the references are before the target!
        const f64 target_tilt_angle = metadata[target_index].angles[1];
        const f64 max_tilt_difference = parameters.projection_max_tilt_angle_difference;
        const i64 max_index = target_index;

        output_reference_indexes.clear();
        output_reference_indexes.reserve(static_cast<size_t>(max_index));

        fmt::print("reproject={} from ", target_tilt_angle);
        for (i64 reference_index = 0; reference_index < max_index; ++reference_index) {
            const f64 reference_tilt = metadata[reference_index].angles[1];
            const f64 tilt_difference = std::abs(target_tilt_angle - reference_tilt);

            // Do not include the target in the projected-reference to remove any auto-correlation.
            if (reference_index != target_index) { // && tilt_difference <= max_tilt_difference
                output_reference_indexes.emplace_back(reference_index);
                fmt::print("{},", reference_tilt);
            }
        }
        fmt::print("\n");
    }

    i64 ProjectionMatching::max_references_count_(
            const MetadataStack& metadata,
            const ProjectionMatchingParameters& parameters) {
        // While we could return the number of slices in the metadata (-1 to ignore the target),
        // we try to find the maximum number of reference slices we'll need to hold at any given time.
        // Depending on the "max tilt angle difference" parameter, this can save a good amount of memory.

        // Ensure at least 3 slices for reusing buffer for various data before or after the projection.
        i64 max_count{3};

        const auto slice_count = static_cast<i64>(metadata.size());
        const f64 max_tilt_difference = parameters.projection_max_tilt_angle_difference;

        // This is O(N^2), but it's fine because N is small (<60), and we do it once in the constructor.
        for (i64 target_index = 0; target_index < slice_count; ++target_index) {
            const f64 target_tilt_angle = metadata[target_index].angles[1];
            const i64 max_index = target_index;

            // Count how many references are needed for the current target.
            i64 count = 0;
            for (i64 reference_index = 0; reference_index < max_index; ++reference_index) {
                const f64 tilt_difference = std::abs(target_tilt_angle - metadata[reference_index].angles[1]);
                if (reference_index != target_index && tilt_difference <= max_tilt_difference)
                    ++count;
            }

            max_count = std::max(max_count, count);
        }

        return max_count;
    }

    i64 ProjectionMatching::find_tilt_neighbour_(
            const MetadataStack& metadata,
            i64 target_index,
            const std::vector<i64>& reference_indexes
    ) {
        const f64 tilt_angle = metadata[target_index].angles[1];
        i64 neighbour_index{0};
        f64 min_tilt_difference = std::numeric_limits<f64>::max();
        for (size_t i = 0; i < reference_indexes.size(); ++i) {
            const f64 tilt_difference = std::abs(tilt_angle - metadata[i].angles[1]);
            if (tilt_difference < min_tilt_difference) {
                neighbour_index = static_cast<i64>(i);
                min_tilt_difference = tilt_difference;
            }
        }
        return neighbour_index;
    }

    void ProjectionMatching::prepare_for_insertion_(
            const View<f32>& stack,
            const MetadataStack& metadata,
            i64 target_index,
            const std::vector<i64>& reference_indexes,
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters
    ) const {
        const auto references_count = static_cast<i64>(reference_indexes.size());
        const auto references_range = noa::indexing::slice_t{0, references_count};

        const auto reference_weights = m_reference_weights.view().subregion(references_range);
        const auto reference_batch_indexes = m_reference_batch_indexes.view().subregion(references_range);
        const auto reference_shifts_center2origin = m_reference_shifts_center2origin.view().subregion(references_range);
        const auto insert_inv_references_rotation = m_insert_inv_references_rotation.view().subregion(references_range);

        for (i64 i = 0; i < references_count; ++i) {
            const i64 reference_index = reference_indexes[static_cast<size_t>(i)];
            const Vec2<f64>& reference_shifts = metadata[reference_index].shifts;
            const Vec3<f64> reference_angles = noa::math::deg2rad(metadata[reference_index].angles);

            // Weight by the angle difference. At 90, the weight is 0.
            const auto tilt_difference = metadata[target_index].angles[1] - metadata[reference_index].angles[1];
            constexpr f64 PI = noa::math::Constant<f64>::PI;
            const f64 weight = noa::math::cos(tilt_difference * PI / 90) / 2 + 0.5;
            reference_weights(i, 0, 0, 0) = static_cast<f32>(weight);

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
        common_area.mask_view(
                stack.subregion(metadata[target_index].index), target_references.subregion(0),
                metadata[target_index], parameters.smooth_edge_percent);

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
        noa::ewise_binary(input_reference_slices, reference_weights, references, noa::multiply_t{});
        common_area.mask_views(
                references, references,
                metadata, reference_indexes, parameters.smooth_edge_percent);

        // Zero-pad.
        noa::memory::resize(target_references, target_references_padded);
        if (!parameters.debug_directory.empty()) {
            noa::io::save(target_references_padded,
                          parameters.debug_directory /
                          noa::string::format("target_references_padded_{:>02}.mrc", target_index));
        }

        // Here we don't cache this transform since the number of references changes at every call.
        // For CUDA in-place transforms, this can save a lot of device memory (GBytes!).
        noa::fft::r2c(target_references_padded, target_references_padded_fft,
                      noa::fft::NORM_DEFAULT, /*cache_plan=*/ false);

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
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters
    ) const {
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
                parameters.projection_slice_z_radius, false,
                parameters.projection_cutoff);
        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                noa::indexing::broadcast(reference_weights, references_padded_shape.rfft()),
                references_padded_shape,
                multiplicity_padded_fft, slice_padded_shape,
                Float22{}, insert_inv_references_rotation,
                Float22{}, extract_fwd_target_rotation,
                parameters.projection_slice_z_radius, false,
                parameters.projection_cutoff);

        // Center projection back and shift onto the target.
        noa::signal::fft::phase_shift_2d<noa::fft::H2H>(
                reference_padded_fft, reference_padded_fft,
                slice_padded_shape, slice_padded_center() + target_shifts.as<f32>());

        // Correct for the multiplicity on the projected_reference.
        // Everything below 1 is left unchanged because we do want to keep these frequencies down-weighted.
        noa::ewise_binary(reference_padded_fft, multiplicity_padded_fft,
                          reference_padded_fft, qn::divide_max_one_t{});

        // Apply the weights of the projected_reference onto the target.
        // Here, everything below 1 is multiplied, effectively down-weighting the frequencies with low confidence
        // or without any signal in the projected reference, but everything above 1 is left unchanged.
        noa::ewise_binary(input_target_references_padded_fft.subregion(0), multiplicity_padded_fft,
                          target_padded_fft, qn::multiply_min_one_t{});

        // Go back to real-space.
        noa::fft::c2r(target_reference_padded_fft, target_reference_padded);

        // The target and the projected reference are saved next to each other.
        const auto target_reference = m_slices.view().subregion(noa::indexing::slice_t{0, 2});
        noa::memory::resize(target_reference_padded, target_reference);

        // Apply the area mask again.
        common_area.mask_view(
                target_reference.subregion(0), target_reference.subregion(0),
                metadata[target_index], parameters.smooth_edge_percent);
        common_area.mask_view(
                target_reference.subregion(1), target_reference.subregion(1),
                metadata[target_index], parameters.smooth_edge_percent);

        // ...At this point
        // - The real-space target and projected reference are in the first two slice of m_slices.
        // - They are ready for cross-correlation.
    }

    auto ProjectionMatching::select_peak_(
            const View<f32>& xmap,
            const MetadataStack& metadata,
            i64 target_index,
            f64 target_angle_offset,
            const std::vector<i64>& reference_indexes,
            const std::vector<Vec2<f64>>& shift_offsets,
            const ProjectionMatchingParameters& parameters
    ) const -> std::pair<Vec2<f64>, f64> {

        // Find the elliptical mask to apply onto the xmap to enforce a maximum shift.
        // The goal is to prevent large jumps in shift from one tilt to the next because we assume that
        // the pairwise alignment did a good job at aligning tilt neighbours. However, we still want to
        // be able to correct 1) for a global drift of the stack, something that pairwise alignment isn't
        // good at correcting or even creates by its systematic errors when accumulating the pairwise shifts,
        // and 2) for small shifts that could not be corrected at the resolution used by the pairwise alignment.
        // As such, find by how much the low tilt neighbour was shifted, and use it as an estimate of how much
        // the target view is allowed to move.

        // First, get the lower tilt neighbour.
        const auto neighbour_index = static_cast<size_t>(
                find_tilt_neighbour_(metadata, target_index, reference_indexes));
        const Vec2<f64>& neighbour_shift_offset = shift_offsets[neighbour_index];
        Vec3<f64> neighbour_angles = noa::math::deg2rad(metadata[neighbour_index].angles);

        Vec3<f64> target_angles = noa::math::deg2rad(metadata[target_index].angles);
        target_angles[0] += noa::math::deg2rad(target_angle_offset);

        // Estimate this shift of the target based on how much the neighbour was shifted.
        // To do this, we need to shrink the shift to account for the elevation and tilt difference.
        // We also take into account the rotation difference.
        const Vec2<f64> shrink_neighbour_to_target =
                noa::math::cos(target_angles.filter(2, 1)) /
                noa::math::cos(neighbour_angles.filter(2, 1));
        const Double22 xform_neighbour_to_target =
                noa::geometry::rotate(target_angles[0]) *
                noa::geometry::scale(shrink_neighbour_to_target) *
                noa::geometry::rotate(-neighbour_angles[0]);
        const auto ellipse_shift = xform_neighbour_to_target * neighbour_shift_offset;

        // Set an ellipse radius to account for small shifts.
        const auto ellipse_radius = noa::math::min(
                Vec2<f64>{4},
                xmap.shape().vec().filter(2, 3).as<f64>() * parameters.allowed_shift_percent);

        // Increase performance by working on a small subregion located at the center of the xmap.
        const Vec2<f64> max_shift = ellipse_shift + ellipse_radius + 10;
        const View<f32> peak_window = extract_peak_window_(max_shift);
        const Vec2<f32> peak_window_center = MetadataSlice::center(peak_window.shape());

        // Note that the center of the cross-correlation is fixed at N//2 (integral division), so don't use
        // MetadataSlice::center in case we switch of convention one day and use N/2 (floating-point division).
        const auto xmap_center = (xmap.shape().vec().filter(2, 3) / 2).as<f64>();

        // The xmap is distorted perpendicular to the tilt-axis due to the tilt.
        // To help the picking, rotate so that the distortion is along the X-axis.
        const Double22 target_rotation = noa::geometry::rotate(target_angles[0]);
        const Float33 xmap_inv_transform(
                noa::geometry::translate(xmap_center) *
                noa::geometry::linear2affine(target_rotation) *
                noa::geometry::translate(-peak_window_center.as<f64>()));
        noa::geometry::transform_2d(xmap, peak_window, xmap_inv_transform);

        // Enforce a max shift.
        noa::geometry::ellipse(
                peak_window, peak_window,
                peak_window_center + ellipse_shift.as<f32>(),
                ellipse_radius.as<f32>(), 0.f);

        if (!parameters.debug_directory.empty()) {
            noa::io::save(xmap,
                          parameters.debug_directory /
                          noa::string::format("xmap_{:>02}.mrc", target_index));
            noa::io::save(peak_window,
                          parameters.debug_directory /
                          noa::string::format("xmap_rotated_{:>02}.mrc", target_index));
        }

        // TODO Better fitting of the peak. 2D parabola?
        auto [peak_coordinate, peak_value] = noa::signal::fft::xpeak_2d<noa::fft::FC2FC>(peak_window);
        const Vec2<f32> shift_rotated = peak_coordinate - peak_window_center;
        Vec2<f64> shift = target_rotation * shift_rotated.as<f64>();

        return {shift, peak_value};
    }

    auto ProjectionMatching::project_and_correlate_(
            const MetadataStack& metadata,
            i64 target_index,
            f64 target_rotation_offset,
            const std::vector<i64>& reference_indexes,
            const std::vector<Vec2<f64>>& shift_offsets,
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters
    ) const -> std::pair<Vec2<f64>, f64> {
        // First, compute the target and projected-reference slice.
        compute_target_reference_(
                metadata, target_index, target_rotation_offset,
                reference_indexes, common_area, parameters);
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
                xmap);

        // Select the best peak. The returned shifts are ready to be added to the metadata.
        const auto [shift, peak_value] = select_peak_(
                xmap, metadata,
                target_index, target_rotation_offset,
                reference_indexes, shift_offsets, parameters);

        // Normalize the peak.
        // TODO Check that this is equivalent to dividing by the auto-correlation peak.
        const auto score = static_cast<f64>(peak_value) /
                           noa::math::sqrt(static_cast<f64>(energy_target) *
                                           static_cast<f64>(energy_reference));

        return {shift, score};
    }

    f64 ProjectionMatching::run_projection_matching_(
            const View<f32>& stack,
            MetadataStack& metadata,
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters,
            const CubicSplineGrid<f64, 1>& rotation_target) const {

        // Store the score of every slice.
        std::vector<f64> scores;
        scores.reserve(metadata.size() - 1);

        // We need to store the shift offsets for each slice. This vector must match the metadata.
        std::vector shift_offsets(metadata.size(), Vec2<f64>{0});

        // Compute the initial rotation of the global reference.
        metadata[0].angles[0] = rotation_target.interpolate(Vec1<f64>{metadata[0].angles[1]}, 0);

        // For every slice (excluding the lowest tilt which defines the reference-frame), move the rotation
        // to the target rotation and update (y,x) shifts using the previously aligned slice(s) as reference
        // for alignment.
        std::vector<i64> reference_indexes;
        for (i64 target_index = 1; target_index < static_cast<i64>(metadata.size()); ++target_index) {

            // Get the indexes of the reference views for this target view.
            set_reference_indexes_(
                    target_index, metadata,
                    parameters, reference_indexes);

            // Prepare the target and references for Fourier insertion.
            prepare_for_insertion_(
                    stack, metadata, target_index,
                    reference_indexes, common_area, parameters);

            // Get the target rotation for this tilt.
            auto& slice = metadata[target_index];
            const f64 desired_rotation = rotation_target.interpolate(Vec1<f64>{slice.angles[1]}, 0);
            const f64 rotation_offset = desired_rotation - slice.angles[0];

            const auto [shift_offset, score] = project_and_correlate_(
                    metadata,
                    target_index,
                    rotation_offset,
                    reference_indexes,
                    shift_offsets,
                    common_area,
                    parameters
            );

            qn::Logger::debug("{:>02}: rotation={:> 6.3f} ({:>+5.3f}), "
                              "cc_score={:.6g}, shift={::> 6.3f} ({::>+5.3f})",
                              target_index,
                              desired_rotation, rotation_offset,
                              score,
                              slice.shifts, shift_offset);

            // Update the metadata to use this new shift and new rotation in the next projected reference.
            slice.angles[0] += rotation_offset; // or slice.angles[0] = desired_rotation
            slice.shifts += shift_offset;

            scores.push_back(score);
            shift_offsets[static_cast<size_t>(target_index)] = shift_offset;
        }

        // Return the sum of the normalized CC scores. This is used to score this rotation target(s).
        const auto score = std::accumulate(scores.begin(), scores.end(), f64{0});
        qn::Logger::debug("total score={:.6g}", score);
        return score;
    }
}
