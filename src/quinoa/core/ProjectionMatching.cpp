#include <noa/IO.hpp>
#include <optional>

#include "quinoa/core/ProjectionMatching.h"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/Utilities.h"
#include "quinoa/core/CommonArea.hpp"

namespace {
    using namespace qn;

    // Data passed through the optimizer.
    struct OptimizerData {
        ProjectionMatching* projector{};
        const ProjectionMatchingParameters* parameters{};

        MetadataStack metadata;
        CommonArea common_area;

        // This is set at each iteration so that the projector knows what target
        // (and therefore projected-reference) to compute.
        i64 target_index{};
        std::vector<i64> reference_indexes;

        // This is used to compute the global score of this optimization.
        // Comparing the sum of these scores tells us how good the rotation
        // of the global reference was.
        std::vector<f64> scores;

        // The shift offsets that were applied are saved to, in the same order as in the metadata.
        // The offsets are used to restrain the shits of the next neighbour view.
        std::vector<Vec2<f64>> shift_offsets;
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

    f64 ProjectionMatching::update(
            const Array<f32>& stack,
            MetadataStack& metadata,
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters,
            bool shift_only,
            f64 rotation_offset_bound,
            std::optional<ThirdDegreePolynomial> initial_rotation_target
    ) {
        qn::Logger::trace("Projection matching alignment...");
        noa::Timer timer;
        timer.start();

        auto max_objective_function = [](u32, const f64* x, f64*, void* instance) -> f64 {
            auto* data = static_cast<OptimizerData*>(instance);

            const auto [shift, score] = data->projector->project_and_correlate_(
                    data->metadata,
                    data->target_index,
                    *x,
                    data->reference_indexes,
                    data->shift_offsets,
                    data->common_area,
                    *data->parameters
            );
            qn::Logger::debug("rotation offset={:> 7.4f}, score={:> 7.4f}", *x, score);

            const auto target_index_u = static_cast<size_t>(data->target_index);
            data->scores[target_index_u] = score;
            data->shift_offsets[target_index_u] = shift;
            return score;
        };

        // Set up the optimizer.
        OptimizerData optimizer_data;
        optimizer_data.projector = this;
        optimizer_data.parameters = &parameters;
        optimizer_data.metadata = metadata;
        optimizer_data.metadata.sort("absolute_tilt"); // TODO exposure?
        optimizer_data.common_area = common_area;
        optimizer_data.common_area.reserve(m_slices.shape()[0], m_slices.device());
        optimizer_data.reference_indexes.reserve(metadata.size());

        // The indexing to access these vector is the same as the metadata.
        optimizer_data.shift_offsets = std::vector<Vec2<f64>>(metadata.size(), Vec2<f64>{0});
        optimizer_data.scores = std::vector<f64>(metadata.size(), 0.);

        const Optimizer optimizer(NLOPT_LN_SBPLX, 1);
        optimizer.set_max_objective(max_objective_function, &optimizer_data);
        optimizer.set_x_tolerance_abs(0.005);
        const f64 bound = !shift_only ? rotation_offset_bound : 0;
        optimizer.set_bounds(-bound, bound);

        // For every slice (excluding the lowest tilt which defines the reference-frame), find the best
        // geometry parameters using the previously aligned slice(s) as reference for alignment. The geometry
        // parameters are, for each slice, the rotation and the (y,x) shifts.
        const auto slice_count = static_cast<i64>(optimizer_data.metadata.size());
        for (i64 target_index = 1; target_index < slice_count; ++target_index) {
            const auto target_index_u = static_cast<size_t>(target_index);

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
                    optimizer_data.reference_indexes, optimizer_data.common_area, parameters);

            // Either:
            //  - impose no rotation offset and simply find the best shift given the current rotation.
            //  - Add a rotation offset to match the curve and find the shift corresponding to this rotation.
            const f64 rotation_offset_to_polynomial_curve =
                    initial_rotation_target.has_value() ?
                    initial_rotation_target.value()(slice.angles[1]) - slice.angles[0] : 0;
            const auto [first_shift, first_score] = project_and_correlate_(
                    optimizer_data.metadata,
                    optimizer_data.target_index,
                    rotation_offset_to_polynomial_curve,
                    optimizer_data.reference_indexes,
                    optimizer_data.shift_offsets,
                    optimizer_data.common_area,
                    *optimizer_data.parameters
            );

            // Add this score and shift offset. These will be updated by the maximization function.
            optimizer_data.scores[target_index_u] = first_score;
            optimizer_data.shift_offsets[target_index_u] = first_shift;
            qn::Logger::debug("{:>02}: rotation offset={:> 6.3f}, score={:.6g}, shift={::> 6.3f}",
                              target_index, rotation_offset_to_polynomial_curve, first_score, first_shift);

            // Update the metadata to use this new shift and, optionally, new rotation.
            slice.angles[0] += rotation_offset_to_polynomial_curve;
            slice.shifts += first_shift.as<f64>();

            // Find the best rotation offset (and its corresponding shift) by maximising the
            // cross-correlation between the target and the projected-reference.
            if (!shift_only) {
                f64 x_rotation_offset{0}, fx_cc_score;
                optimizer.optimize(&x_rotation_offset, &fx_cc_score);
                qn::Logger::debug("{:>02}: rotation offset={:> 6.3f}, score={:.6g}, shift={::> 6.3f}",
                                  target_index, x_rotation_offset, fx_cc_score,
                                  optimizer_data.shift_offsets[target_index_u]);

                // Update the metadata to use this new shift and new rotation.
                slice.angles[0] += x_rotation_offset;
                slice.shifts += optimizer_data.shift_offsets[target_index_u];
            }
        }

        qn::Logger::trace("Projection matching alignment... took {}ms", timer.elapsed());

        // Update the metadata.
        for (const auto& updated_metadata: optimizer_data.metadata.slices()) {
            for (auto& original_slice: metadata.slices()) {
                if (original_slice.index == updated_metadata.index) {
                    qn::Logger::trace("{:>02},{},{}",
                                      original_slice.index,
                                      original_slice.angles - updated_metadata.angles,
                                      original_slice.shifts - updated_metadata.shifts
                    );
                    original_slice.angles = updated_metadata.angles;
                    original_slice.shifts = updated_metadata.shifts;
                }
            }
        }

        // Return the sum of the normalized CC scores. This is used to select the best "scenario"
        // and the initial rotation of the global reference.
        const f64 score = std::accumulate(optimizer_data.scores.begin(), optimizer_data.scores.end(), f64{0});
        return score;
    }
}

// Private methods:
namespace qn {
    auto ProjectionMatching::extract_peak_window_(const Vec2<f64>& max_shift) -> View<f32> {
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

        for (i64 reference_index = 0; reference_index < max_index; ++reference_index) {
            const f64 tilt_difference = std::abs(target_tilt_angle - metadata[reference_index].angles[1]);

            // Do not include the target in the projected-reference to remove any auto-correlation.
            if (reference_index != target_index && tilt_difference <= max_tilt_difference)
                output_reference_indexes.emplace_back(reference_index);
        }
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

        // This is O(N^2), but it's fine because N is small (<60), and we do it once in the constructor.
        for (i64 target_index = 0; target_index < slice_count; ++target_index) {
            const f64 target_tilt_angle = metadata[target_index].angles[1];
            const i64 max_index = target_index;

            // Count how many references are needed for the current target.
            i64 count = 0;
            for (i64 reference_index = 0; reference_index < max_index; ++reference_index) {
                const f64 tilt_difference = std::abs(target_tilt_angle - metadata[reference_index].angles[1]);

                if (reference_index != target_index &&
                    noa::math::abs(metadata[reference_index].angles[1]) < parameters.projection_max_tilt_angle_difference)
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
        for (i64 row = 0; row < rows; ++row) { // skip 0
            const MetadataSlice& slice = metadata[row + 1];
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

    void ProjectionMatching::prepare_for_insertion_(
            const View<f32>& stack,
            const MetadataStack& metadata,
            i64 target_index,
            const std::vector<i64>& reference_indexes,
            const CommonArea& common_area,
            const ProjectionMatchingParameters& parameters
    ) {
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

            // TODO Weighting based on the order of collection? Or use exposure weighting?
            // Multiplicity and weight.
            reference_weights(i, 0, 0, 0) = 1; // FIXME?

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
        common_area.mask_views(
                input_reference_slices, references,
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
                parameters.projection_slice_z_radius, false,
                parameters.projection_cutoff);
        noa::geometry::fft::insert_interpolate_and_extract_3d<noa::fft::HC2H>(
                noa::indexing::broadcast(reference_weights, references_padded_shape.fft()),
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
            const std::vector<Vec2<f64>>& shift_offsets
    ) -> std::pair<Vec2<f64>, f64> {

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
                Vec2<f64>{5},
                xmap.shape().vec().filter(2, 3).as<f64>() * 0.005);

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
    ) -> std::pair<Vec2<f64>, f64> {
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
        if (!parameters.debug_directory.empty()) {
            noa::io::save(xmap, parameters.debug_directory /
                                noa::string::format("xmap_{:>02}.mrc", target_index));
        }

        // Select the best peak. The returned shifts are ready to be added to the metadata.
        const auto [shift, peak_value] = select_peak_(
                xmap, metadata,
                target_index, target_rotation_offset,
                reference_indexes, shift_offsets);

        // Normalize the peak.
        // TODO Check that this is equivalent to dividing by the auto-correlation peak.
        const auto score = static_cast<f64>(peak_value) /
                           noa::math::sqrt(static_cast<f64>(energy_target) *
                                           static_cast<f64>(energy_reference));

        return {shift, score};
    }
}
