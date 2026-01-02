#include <noa/IO.hpp>
#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>
#include <noa/Utils.hpp>

#include "quinoa/PairwiseShift.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/Optimizer.hpp"

namespace {
    using namespace qn;

    constexpr auto get_indices_(i32 idx_target, i32 index_lowest_tilt) {
        if (idx_target >= index_lowest_tilt)
            idx_target += 1;

        // The tilts are sorted in ascending order, so if the ith target has:
        //  - a negative tilt angle, then the reference is at i + 1.
        //  - a positive tilt angle, then the reference is at i - 1.
        const bool is_negative = idx_target < index_lowest_tilt;
        const i32 idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;
        return Pair{idx_target, idx_reference};
    }

    struct StretchTarget {
        using span_type = SpanContiguous<const f32, 3, i32>;
        using interpolator_type = noa::Interpolator<2, noa::Interp::LINEAR, noa::Border::ZERO, span_type>;

        interpolator_type target{};
        SpanContiguous<f32, 2, i32> target_stretched{};
        Vec<f32, 4> reference_plane_coefficients{};
        Mat<f32, 2, 4> reference2target{};

        NOA_HD void operator()(i32 y, i32 x) const {
            // Get the z at this image coordinate using the plane equation.
            const auto& [a, b, c, d] = reference_plane_coefficients;
            const auto z = -(a * static_cast<f32>(x) + b * static_cast<f32>(y) + d) / c;
            const auto reference_coordinates = Vec<f32, 4>::from_values(z, y, x, 1);

            // Transform from target to reference.
            const auto target_coordinates = reference2target * reference_coordinates;
            target_stretched(y, x) = target.interpolate_at(target_coordinates);
        }
    };

    struct CreateStacks {
        using span_type = SpanContiguous<const f32, 3, i32>;
        using interpolator_type = noa::Interpolator<2, noa::Interp::LINEAR, noa::Border::ZERO, span_type>;
        interpolator_type input{}; // (n,h,w)

        SpanContiguous<f32, 3, i32> references{}; // (n-1,h,w)
        SpanContiguous<f32, 3, i32> stretched_targets{}; // (n-1,h,w)

        SpanContiguous<const ParallelogramMask, 1, i32> reference_masks{}; // (n)
        SpanContiguous<const ParallelogramMask, 1, i32> stretched_target_masks{}; // (n-1)

        SpanContiguous<const Vec<f32, 4>, 1, i32> reference_plane_coefficients{}; // (n-1)
        SpanContiguous<const Mat<f32, 2, 4>, 1, i32> reference2target{}; // (n-1)

        i32 pivot{};

        NOA_HD void operator()(i32 i, i32 y, i32 x) const {
            const auto n_images = reference_masks.ssize();

            // Save as reference.
            const auto reference = input(i, y, x) * reference_masks[i](y, x);
            if (i > 0 and i <= pivot)
                references(i - 1, y, x) = reference;
            if (i >= pivot and i < n_images - 1)
                references(i, y, x) = reference;

            // Save as target.
            if (i != pivot) {
                const auto index_target = i > pivot ? i - 1 : i;

                // Transform the target onto the reference.
                const auto& [a, b, c, d] = reference_plane_coefficients[index_target];
                const auto z = -(a * static_cast<f32>(x) + b * static_cast<f32>(y) + d) / c;
                const auto reference_coordinates = Vec<f32, 4>::from_values(z, y, x, 1);
                const auto target_coordinates = reference2target[index_target] * reference_coordinates;

                // Save the masked and stretched target.
                const auto stretched_target = input.interpolate_at(target_coordinates, i);
                stretched_targets(index_target, y, x) = stretched_target * stretched_target_masks[index_target](y, x);
            }
            // TODO reduction operator so it can compute the l2-norm in one pass. this is still optional and can be iwise only
        }
    };

    struct CreateStacks2 {
        using span_type = SpanContiguous<const f32, 3, i32>;
        using interpolator_type = noa::Interpolator<2, noa::Interp::LINEAR, noa::Border::ZERO, span_type>;
        using reduce_type = Vec<f32, 5, 8>;

        interpolator_type input{}; // (n,h,w)
        SpanContiguous<f32, 3, i32> references{}; // (n-1,h,w)
        SpanContiguous<f32, 3, i32> stretched_targets{}; // (n-1,h,w)

        SpanContiguous<const ParallelogramMask, 1, i32> fov_masks{}; // (n - 1)
        SpanContiguous<const Vec<f32, 4>, 1, i32> reference_plane_coefficients{}; // (n-1)
        SpanContiguous<const Mat<f32, 2, 4>, 1, i32> reference2target{}; // (n-1)

        i32 pivot{};

        NOA_HD auto create_stacks(i32 i, i32 y, i32 x) const {
            auto [index_target, index_reference] = get_indices_(i, pivot);
            auto fov_mask = fov_masks[i](y, x);

            auto reference = input(index_reference, y, x) * fov_mask;
            references(i, y, x) = reference;

            // Transform the target onto the reference.
            const auto& [a, b, c, d] = reference_plane_coefficients[i];
            const auto z = -(a * static_cast<f32>(x) + b * static_cast<f32>(y) + d) / c;
            const auto reference_coordinates = Vec<f32, 4>::from_values(z, y, x, 1);
            const auto target_coordinates = reference2target[i] * reference_coordinates;

            // Save the masked and stretched target.
            const auto stretched_target = input.interpolate_at(target_coordinates, index_target) * fov_mask;
            stretched_targets(i, y, x) = stretched_target;

            return noa::make_tuple(reference, stretched_target, fov_mask);
        }

        NOA_HD void operator()(i32 i, i32 y, i32 x) const {
            create_stacks(i, y, x);
        }

        NOA_HD void init(i32 i, i32 y, i32 x, reduce_type& reduce) const {
            auto [reference, stretched_target, fov_mask] = create_stacks(i, y, x);
            reduce[0] += reference;
            reduce[1] += stretched_target;
            reduce[2] += reference * reference;
            reduce[3] += stretched_target * stretched_target;
            reduce[4] += fov_mask;
        }

        static NOA_HD void join(const reduce_type& tmp, reduce_type& reduce) {
            reduce += tmp;
        }
    };

    struct CrossCorrelate {
        SpanContiguous<c32, 3, i32> references{}; // (n-1,h,w/2+1)
        SpanContiguous<c32, 3, i32> stretched_targets{}; // (n-1,h,w/2+1)

        NOA_HD void operator()(i32 i, i32 y, i32 x) const {
            const auto frequency = nf::index2frequency<false, true>(Vec{y, x}, references.shape().pop_front());
            const auto phase_shift = static_cast<f32>(product(1 - 2 * abs(frequency % 2))); // shift by +shape/2
            auto cc = stretched_targets(i, y, x) * conj(references(i, y, x));
            cc *= phase_shift;
            stretched_targets(i, y, x) = cc;
        }
    };

    auto relative2global_shifts_(
        const std::vector<Vec<f64, 2>>& relative_shifts,
        const Metadata::Stack& metadata,
        i64 pivot
    ) {
        // Relative shifts (target->reference) to global shifts (target->volume).
        const auto n = std::size(relative_shifts);
        const auto r_pivot = std::ssize(relative_shifts) - 1 - pivot;
        auto global_shifts = std::vector<Vec<f64, 2>>(relative_shifts.size());
        std::inclusive_scan(relative_shifts.begin() + pivot, relative_shifts.end(), global_shifts.begin() + pivot);
        std::inclusive_scan(relative_shifts.rbegin() + r_pivot, relative_shifts.rend(), global_shifts.rbegin() + r_pivot);

        // Center the shifts.
        auto mean = Vec<f64, 2>{};
        for (auto& shift: global_shifts)
            mean += shift;
        mean /= static_cast<f64>(n);
        for (auto& shift: global_shifts)
            shift -= mean;

        // Transform the shifts back to image space.
        for (size_t i{}; i < n; ++i) {
            const auto angles = noa::deg2rad(metadata[i].angles);
            const auto volume2image = (
                ng::rotate_z(angles[0]) *
                ng::rotate_y(angles[1]) *
                ng::rotate_x(angles[2])
            ).filter_rows(1, 2);
            global_shifts[i] = volume2image * global_shifts[i].push_front(0);
        }
        return Pair{global_shifts, mean};
    }

    auto relative2global_shifts_(
        const std::vector<Vec<f64, 2>>& relative_shifts,
        std::vector<Vec<f64, 2>>& global_shifts,
        const Metadata::Stack& metadata,
        i64 pivot
    ) {
        // Relative shifts (target->reference) to global shifts (target->volume).
        const auto n = std::size(relative_shifts);
        const auto r_pivot = std::ssize(relative_shifts) - 1 - pivot;
        std::inclusive_scan(relative_shifts.begin() + pivot, relative_shifts.end(), global_shifts.begin() + pivot);
        std::inclusive_scan(relative_shifts.rbegin() + r_pivot, relative_shifts.rend(), global_shifts.rbegin() + r_pivot);

        // Center the shifts.
        auto mean = Vec<f64, 2>{};
        for (const auto& shift: global_shifts)
            mean += shift;
        mean /= static_cast<f64>(n);
        for (auto& shift: global_shifts)
            shift -= mean;

        // Transform the shifts back to image space.
        for (size_t i{}; i < n; ++i) {
            const auto angles = noa::deg2rad(metadata[i].angles);
            const auto volume2image = (
                ng::rotate_z(angles[0]) *
                ng::rotate_y(angles[1]) *
                ng::rotate_x(angles[2])
            ).filter_rows(1, 2);
            global_shifts[i] = volume2image * global_shifts[i].push_front(0);
        }
        return mean;
    }
}

namespace qn {
    PairwiseShift::PairwiseShift(
        const Shape4<i64>& shape,
        Device compute_device
    ) {
        const auto allocated_start = Allocator::bytes_currently_allocated(compute_device);

        const auto options = ArrayOption{compute_device, Allocator::ASYNC};
        m_buffer = Array<f32>({3, 1, shape[2], shape[3]}, options);
        m_buffer_rfft = Array<c32>(m_buffer.shape().rfft(), options);
        m_xmap = Array<f32>(m_buffer.shape().set<0>(1), options);
        m_xmap_centered = Array<f32>({1, 1, 64, 64}, {.device = compute_device, .allocator = Allocator::MANAGED});

        const auto allocated = Allocator::bytes_currently_allocated(compute_device) - allocated_start;
        Logger::trace("PairwiseShift(): allocated {:.2f}MB on {} ({})",
                      static_cast<f64>(allocated) * 1e-6, options.device, options.allocator);
    }

    void PairwiseShift::update(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const PairwiseShiftParameters& parameters
    ) {
        if (m_buffer_rfft.is_empty())
            return;

        auto timer = Logger::info_scope_time("Coarse shift alignment");
        Logger::trace(
            "device={}\n"
            "stretching={}\n"
            "fov_mask={}\n"
            "smooth_edge={}%",
            m_xmap.device(),
            parameters.cosine_stretch,
            parameters.area_match,
            parameters.smooth_edge_percent * 100
        );

        // We'll need the images sorted by tilt angles, with the lowest absolute tilt being the pivot point.
        metadata.sort("tilt");
        const i64 pivot = metadata.find_lowest_tilt_index();
        const i64 n_images = metadata.ssize();
        const auto image_shape = stack.shape().filter(2, 3);

        // Iterating a few times is required to get a stable shift.
        auto max_shifts = Vec<f64, 2>{};
        auto first_average_shift = Vec<f64, 2>{};
        auto last_average_shift = Vec<f64, 2>{};
        const bool converge = parameters.update_count < 0;
        const i32 count = converge ? 125 : parameters.update_count;
        auto pair_metadata = Metadata::Stack{};

        i32 i{};
        while (i < count) {
            if (i > 0)
                Logger::s_is_debug = false;

            if (parameters.area_match) {
                // Enforce the common FOV between all the images.
                // This is quite restrictive and removes regions from
                // the higher tilts that are not in the lower tilts.
                m_common_fov.set_geometry(image_shape, metadata);
            }

            // The main processing loop. From the lowest to the highest tilt, find the shifts.
            // These shifts are relative, i.e., between an image and its lower tilt neighbor, and in volume space.
            auto relative_shifts = std::vector<Vec<f64, 2>>{};
            relative_shifts.reserve(static_cast<size_t>(n_images));

            for (i64 idx_target{}; idx_target < n_images; ++idx_target) {
                if (pivot == idx_target) {
                    relative_shifts.push_back({}); // global reference
                    continue;
                }

                // If ith target has:
                //  - a negative tilt, then its reference is at i + 1.
                //  - a positive tilt, then its reference is at i - 1.
                const bool is_negative = idx_target < pivot;
                const i64 idx_reference = idx_target + 1 * is_negative - 1 * !is_negative;

                if (not parameters.area_match) {
                    // The common FOV mask ends up removing significant regions of the high tilts.
                    // When the shifts are not known and large shifts are present, it is best to turn off
                    // the common FOV for the entire stack and instead apply the common FOV only between
                    // the two images that are being compared.
                    pair_metadata.images.clear();
                    pair_metadata.images.push_back(metadata[idx_reference]);
                    pair_metadata.images.push_back(metadata[idx_target]);
                    m_common_fov.set_geometry(image_shape, pair_metadata);
                }

                // Compute the shifts.
                auto relative_shift = find_relative_shifts_(
                    stack, metadata[idx_reference], metadata[idx_target], parameters
                );
                relative_shifts.push_back(relative_shift);
            }

            // Compute the global shifts, i.e., the shift relative to the global reference.
            // This is where the high tilts end up accumulating the errors of the lower tilts.
            const auto [global_shifts, average_shift] = relative2global_shifts_(relative_shifts, metadata, pivot);

            // Logging.
            if (i == 0)
                first_average_shift = average_shift;
            last_average_shift = average_shift;

            // Update the metadata.
            max_shifts = 0;
            for (auto&& [slice, global_shift]: noa::zip(metadata, global_shifts)) {
                slice.shifts += global_shift;
                max_shifts = max(max_shifts, abs(global_shift));
            }

            // Loop logic.
            ++i;
            if (converge and noa::sqrt(noa::dot(average_shift, average_shift)) <= 0.001)
                break;
        }

        Logger::info(
            "first_average_shift={::.3f}, last_average_shift={::.3f}, max_shift={::.3f}, n_iter={}",
            first_average_shift, last_average_shift, max_shifts, i
        );
        save_plot_shifts(metadata, *parameters.output_directory / "coarse_shifts.txt", {.title = "Coarse Shifts"});
    }

    auto PairwiseShift::find_relative_shifts_(
        const View<f32>& stack,
        Metadata::Image reference_image,
        const Metadata::Image& target_image,
        const PairwiseShiftParameters& parameters
    ) const -> Vec2<f64> {
        // To find the shift between the target image and its lower-tilt image (aka the reference),
        // we forward project the target image onto the reference, as if the specimen was perfectly thin.
        // The thicker the specimen actually is, the more inaccurate this method becomes.
        // This is often referred to as "cosine-stretching", but here we correctly account for the 3 stage angles.

        // If parameters.stretch == false, shift the target onto the reference and assume the two images were
        // collected with the same stage angles. This is useful when we don't know the image rotation.
        if (not parameters.cosine_stretch)
            reference_image.angles = target_image.angles;

        const auto target_angles = noa::deg2rad(target_image.angles);
        const auto reference_angles = noa::deg2rad(reference_image.angles);
        const auto image_shape = m_xmap.shape().filter(2, 3);
        const auto image_center = (image_shape.vec / 2).as<f64>();

        // Compute the reference plane-coefficients to get the z image-coordinates of the reference.
        const auto reference_plane_rotation = (
            ng::rotate_z(reference_angles[0]) *
            ng::rotate_y(reference_angles[1]) *
            ng::rotate_x(reference_angles[2])
        );
        const auto reference_plane_coefficients = [&] {
            const auto [c, b, a] = reference_plane_rotation * Vec{1., 0., 0.}; // plane normal
            const auto reference_center = image_center + reference_image.shifts;
            const auto d = b * -reference_center[0] + a * -reference_center[1]; // precompute coordinate - shifts
            return Vec{a, b, c, d};
        }();

        // Compute the transformation to project the target onto the reference.
        const auto reference2target = (
            ng::translate(image_center.push_front(0) + target_image.shifts.push_front(0)) *
            ng::rotate_z<true>(+target_angles[0]) *
            ng::rotate_y<true>(+target_angles[1]) *
            ng::rotate_x<true>(+target_angles[2]) *
            ng::rotate_x<true>(-reference_angles[2]) *
            ng::rotate_y<true>(-reference_angles[1]) *
            ng::rotate_z<true>(-reference_angles[0]) *
            ng::translate(-image_center.push_front(0) - reference_image.shifts.push_front(0))
        );

        // Get the views from the buffer.
        const auto projected_target = m_buffer.view().subregion(0);
        const auto reference = m_buffer.view().subregion(1);
        const auto target = m_buffer.view().subregion(2);

        // Real-space masks to guide the alignment and not compare things that cannot or shouldn't be compared.
        // This is relevant for large shifts between images and high-tilt angles, but also to restrict the
        // alignment to a specific region (in this case, the center of the volume).
        if (parameters.area_match) {
            const auto fov = FOVMaskOptions{.smooth_edge_percent = parameters.smooth_edge_percent, .add_shifts = false};
            m_common_fov.apply_fov(stack.subregion(reference_image.index), reference, reference_image, fov);
            m_common_fov.apply_fov(stack.subregion(target_image.index), target, target_image, fov);
        } else {
            const auto reference_and_target = m_buffer.subregion(ni::Slice{1, 3});
            const auto hw = reference_and_target.shape().pop_front<2>();
            const auto smooth_edge_size = static_cast<f64>(noa::max(hw)) * parameters.smooth_edge_percent;
            const auto indices = std::array{reference_image.index, target_image.index};
            noa::copy_batches(stack, reference_and_target, View(indices.data(), 2));

            const auto mask = ng::Rectangle{
                .center = image_center,
                .radius = image_center - smooth_edge_size,
                .smoothness = smooth_edge_size,
            }.draw<f32>();
            const Mat33<f64> target2reference =
                ng::translate(image_center + reference_image.shifts) *
                ng::rotate<true>(reference_angles[0]) *
                ng::rotate<true>(-target_angles[0]) *
                ng::translate(-image_center - target_image.shifts);
            const auto fwd_target2reference = target2reference.as<f32>();
            const auto inv_target2reference = target2reference.inverse().as<f32>();
            ng::draw(reference_and_target, reference_and_target, mask, fwd_target2reference);
            ng::draw(reference_and_target, reference_and_target, mask, inv_target2reference);
        }

        // Compute the stretched target.
        noa::iwise(image_shape, target.device(), StretchTarget{
            .target = StretchTarget::interpolator_type(target.span_contiguous<const f32, 3, i32>(), image_shape.as<i32>()),
            .target_stretched = projected_target.span_contiguous<f32, 2, i32>(),
            .reference_plane_coefficients = reference_plane_coefficients.as<f32>(),
            .reference2target = reference2target.filter_rows(1, 2).as<f32>(),
        });

        // Get the views from the buffers.
        const auto projected_target_and_reference = m_buffer.view().subregion(ni::Slice{0, 2});
        const auto projected_target_and_reference_rfft = m_buffer_rfft.view().subregion(ni::Slice{0, 2});
        const auto xmap = m_xmap.view();

        if (Logger::is_debug()) {
            auto filename = *parameters.output_directory / fmt::format("projected_target_and_reference_{:0>2}.mrc", target_image.index);
            noa::write_image(projected_target_and_reference, filename, {.dtype = "f16"});
            Logger::debug("{} saved", filename);
        }

        // (Conventional) cross-correlation.
        // Technically, we should zero-pad here to cancel the circular property of the DFT (only the zero-lag is
        // unaffected by it). However, while we do expect significant lags, since we don't care about the actual peak
        // value and that we iterate, as long as we can find the highest peak, it is fine.
        nf::r2c(projected_target_and_reference, projected_target_and_reference_rfft);
        // ns::bandpass<"h2h">(
        //     projected_target_and_reference_rfft,
        //     projected_target_and_reference_rfft,
        //     projected_target_and_reference.shape(),
        //     parameters.bandpass
        // );
        ns::cross_correlation_map<"h2fc">(
            projected_target_and_reference_rfft.subregion(0),
            projected_target_and_reference_rfft.subregion(1),
            xmap, {.mode = ns::Correlation::CONVENTIONAL}
        );

        if (Logger::is_debug()) {
            auto filename = *parameters.output_directory / fmt::format("xmap_{:0>2}.mrc", target_image.index);
            noa::write_image(xmap, filename, {.dtype = "f16"});
            Logger::debug("{} saved", filename);
        }

        // Compute the shift, i.e., by how much the projected target is away from the reference.
        // To align the target onto the reference, we would beed to subtract this shift from it.
        const auto projected_shift = find_peak<"fc2fc">(xmap, m_xmap_centered.view(), {
            .distortion_angle_deg = reference_image.angles[0],
            .max_shift_percent = parameters.max_shift_percent,
        }).first;

        // We should now transform the shift back to the target reference-frame. However, we'll need to compute the
        // global shifts and center them later on. These operations require accumulating the shifts of the lower views
        // up to the global reference. As such, the simplest is to scale all these slice-to-slice shifts directly to
        // the same reference-frame, process everything there, and then go back to each image's reference-frame at
        // the end. For simplicity, we chose this common reference-frame to be the volume reference-frame, which has
        // no rotation, no tilt, no pitch.

        // Compute the z-coordinate at the image shift.
        const auto [c, b, a] = reference_plane_rotation * Vec{1., 0., 0.}; // plane normal
        const auto z = -(a * projected_shift[1] + b * projected_shift[0]) / c;

        // Transform the shifts to volume-space.
        const auto reference2volume = (
            ng::rotate_x<true>(-reference_angles[2]) *
            ng::rotate_y<true>(-reference_angles[1]) *
            ng::rotate_z<true>(-reference_angles[0])
        ).filter_rows(1, 2);
        auto shift = reference2volume * Vec{z, projected_shift[0], projected_shift[1], 1.};

        return shift;
    }

    PairwiseShift2::PairwiseShift2(
        const Shape4<i64>& shape,
        Device device
    ) {
        const auto allocated_start = Allocator::bytes_currently_allocated(device);

        const auto n_total_images = shape[0];
        const auto n_target_images = shape[0] - 1;

        // Allocate 4 times the stack. While we only need 3 times the stack, the forward FFT needed for the
        // cross-correlation is significantly faster when the references and stretched targets are batched
        // into the same array. This is at the cost of a larger workspace.
        const auto buffer_shape = shape.set<0>(n_total_images * 4);

        // Surprisingly, using managed memory is significantly slower than device-only, so try to use device-only.
        const auto bytes_to_allocate = static_cast<size_t>(buffer_shape.rfft().n_elements()) * sizeof(c32);
        const bool has_enough_space = bytes_to_allocate < device.memory_capacity().free;
        m_buffer_rfft = Array<c32>(buffer_shape.rfft(), {
            .device=device,
            .allocator = has_enough_space ? Allocator::ASYNC : Allocator::MANAGED,
        });
        m_buffer = nf::alias_to_real(m_buffer_rfft.view(), buffer_shape);

        const auto options_managed = ArrayOption{device, Allocator::MANAGED};
        m_fov_masks = Array<ParallelogramMask>(n_target_images, options_managed);
        m_plane_coefficients = Array<Vec<f32, 4>>(n_target_images, options_managed);
        m_projection_matrices = Array<Mat<f32, 2, 4>>(n_target_images, options_managed);

        m_xmap_centered = Array<f32>({n_target_images, 1, 64, 64}, options_managed);
        m_peak_shifts = Array<Vec<f32, 2>>(n_target_images, options_managed);
        m_peak_values = Array<f32>(n_target_images, options_managed);
        m_peak_stats = Array<Vec<f32, 5>>(n_target_images, options_managed);

        m_relative_shifts.resize(static_cast<size_t>(n_total_images));
        m_global_shifts.resize(static_cast<size_t>(n_total_images));

        // Prepare FFT plans and set the workspace.
        if (device.is_gpu()) {
            nf::clear_cache(device);
            nf::set_cache_limit(10, device);
            nf::r2c(buffer(0, 2), buffer_rfft(0, 2), {.record_and_share_workspace = true});
            nf::c2r(buffer_rfft(1), buffer(0), {.record_and_share_workspace = true});
            auto n_plans_set = nf::set_workspace(device, m_buffer_rfft.subregion(ni::Offset{2 * n_target_images}));
            if (n_plans_set != 2) {
                Logger::warn(
                    "Failed to set the FFT workspace. An new workspace will be allocated. Please report this. "
                    "shape={}, workspace_left_to_allocate={}bytes, n_plans_set={}",
                    shape, nf::workspace_left_to_allocate(device), n_plans_set);
            }
        }

        const auto allocated = Allocator::bytes_currently_allocated(device) - allocated_start;
        Logger::trace("PairwiseShift(): allocated {:.2f}GB on {} ({})",
                      static_cast<f64>(allocated) * 1e-9, m_buffer.device(), m_buffer.allocator());
    }

    void PairwiseShift2::update(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const PairwiseShiftParameters& parameters
    ) {
        if (m_buffer.is_empty())
            return;

        auto timer = Logger::info_scope_time("Coarse shift alignment");
        Logger::trace(
            "device={}\n"
            "stretching={}\n"
            "fov_mask={}\n"
            "smooth_edge={}%",
            m_buffer.device(),
            parameters.cosine_stretch,
            parameters.area_match,
            parameters.smooth_edge_percent * 100
        );

        // We'll need the images sorted by tilt angles, with the lowest absolute tilt being the pivot point.
        metadata.sort("tilt");

        // Iterating a few times is required to get a stable shift.
        auto max_shifts = Vec<f64, 2>{};
        auto first_average_shift = Vec<f64, 2>{};
        auto last_average_shift = Vec<f64, 2>{};
        const bool converge = parameters.update_count < 0;
        const i32 count = converge ? 125 : parameters.update_count;
        auto pair_metadata = Metadata::Stack{};

        i32 i{};
        while (i < count) {
            auto average_shift = find_relative_shifts_(stack, metadata, parameters).first;

            // Logging.
            if (i == 0)
                first_average_shift = average_shift;
            last_average_shift = average_shift;

            // Save the shifts

            // Update the metadata.
            max_shifts = 0;
            for (auto&& [slice, global_shift]: noa::zip(metadata, m_global_shifts)) {
                slice.shifts += global_shift;
                max_shifts = max(max_shifts, abs(global_shift));
            }

            // Loop logic.
            ++i;
            if (converge and noa::sqrt(noa::dot(average_shift, average_shift)) <= 0.001)
                break;
        }

        Logger::info(
            "first_average_shift={::.3f}, last_average_shift={::.3f}, max_shift={::.3f}, n_iter={}",
            first_average_shift, last_average_shift, max_shifts, i
        );
        save_plot_shifts(metadata, *parameters.output_directory / "coarse_shifts.txt", {.title = "Coarse Shifts"});
    }

    void PairwiseShift2::stage_level(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const PairwiseShiftParameters& parameters
    ) {
        if (m_buffer.is_empty())
            return;

        auto timer = Logger::info_scope_time("Coarse shift alignment");
        Logger::trace(
            "device={}\n"
            "stretching={}\n"
            "fov_mask={}\n"
            "smooth_edge={}%",
            m_buffer.device(),
            parameters.cosine_stretch,
            parameters.area_match,
            parameters.smooth_edge_percent * 100
        );

        // We'll need the images sorted by tilt angles, with the lowest absolute tilt being the pivot point.
        auto metadata_sorted = metadata;
        metadata_sorted.sort("tilt");
        bool stack_is_sorted{true};
        for (i32 expected_index{0}; const auto& image: metadata_sorted) {
            if (expected_index++ != image.index) {
                stack_is_sorted = false;
                break;
            }
        }
        check(stack_is_sorted, "The tilts in the stack should be sorted in ascending order");

        auto eval = [&](u32 n, const f64* p, f64* g) {
            check(n == 2 and g == nullptr);
            auto i_metadata = metadata_sorted;
            i_metadata.add_image_angles({0, p[0], p[1]});
            auto zncc = find_relative_shifts_(stack, i_metadata, parameters).first;

            // TODO do we save the shifts or should we just run one more time after the optimizer?
        };

        auto tilt_range = Vec{-options.tilt_search_range, options.tilt_search_range};
        auto pitch_range = Vec{-options.pitch_search_range, options.pitch_search_range};
        auto parameters = Vec{0., 0.};
        auto n_evaluations = 0;

        // Logger::s_is_debug = true;
        // eval(2, parameters.data(), nullptr);
        // Logger::s_is_debug = false;

        auto optimizer = Optimizer{};
        if (options.tilt_search_range > 10.) {
            optimizer = Optimizer(NLOPT_GN_DIRECT_L, 2);
            optimizer.set_max_number_of_evaluations(100);
            optimizer.set_bounds(tilt_range, pitch_range);
            optimizer.set_max_objective(eval);
            optimizer.optimize(parameters.data());
            n_evaluations += optimizer.n_evaluations();
        }
        optimizer = Optimizer(NLOPT_LN_SBPLX, 2);
        optimizer.set_x_tolerance_abs(0.005);
        optimizer.set_bounds(tilt_range, pitch_range);
        optimizer.set_max_objective(eval);
        const f64 ncc = optimizer.optimize(parameters.data());
        n_evaluations += optimizer.n_evaluations();

        // Logger::s_is_debug = true;
        // eval(2, parameters.data(), nullptr);
        // Logger::s_is_debug = false;

        // Save the offset.
        metadata.add_image_angles(parameters.push_front(0));
        tilt_pitch_offset += parameters;

        Logger::info(
            "stage=[tilt={:.2f}deg ({:+.2f}), pitch={:.2f}deg ({:+.2f})] (ncc={:.4f}, n_evaluations={})",
            tilt_pitch_offset[0], parameters[0], tilt_pitch_offset[1], parameters[1],
            ncc, n_evaluations
        );
    }

    auto PairwiseShift2::find_relative_shifts_(
        View<f32> stack,
        const Metadata::Stack& metadata,
        const PairwiseShiftParameters& parameters
    ) -> Pair<Vec2<f64>, f64> {
        auto tt = Logger::trace_scope_time("run");
        check(metadata.ssize() == stack.shape()[0]);

        const auto device = stack.device();
        const auto pivot = static_cast<i32>(metadata.find_lowest_tilt_index());
        const auto n_total_images = metadata.ssize();
        const auto n_targets = n_total_images - 1;
        const auto image_shape = stack.shape().filter(2, 3);
        const auto image_center = (image_shape.vec / 2).as<f64>();

        // FIXME
        auto& stream = Stream::current(device);
        auto start = Event{};
        auto end = Event{};

        noa::Timer t0;
        t0.start();

        // Whether to enforce the common FOV between all the images.
        // This is quite restrictive and removes regions from the higher tilts that are not in the lower tilts.
        // When the shifts are not known and large shifts are present, it is best to turn off the common FOV.
        const auto common_fov = parameters.area_match ?
            CommonFOV(image_shape, metadata) :
            CommonFOV(image_shape);

        // Prepare for the CreateStacks operator.
        for (i32 i{}; auto&& [reference_plane_coefficients, projection_matrix, fov_mask]: noa::zip(
            m_plane_coefficients.span_1d(),
            m_projection_matrices.span_1d(),
            m_fov_masks.span_1d()
        )) {
            const auto [index_target, index_reference] = get_indices_(i++, pivot);
            auto target = metadata[index_target];
            auto reference = metadata[index_reference];
            if (not parameters.cosine_stretch)
                reference.angles = target.angles;

            // Compute the plane coefficients of the reference.
            const auto reference_angles = noa::deg2rad(reference.angles);
            const auto reference_plane_rotation = (
                ng::rotate_z(reference_angles[0]) *
                ng::rotate_y(reference_angles[1]) *
                ng::rotate_x(reference_angles[2])
            );
            const auto [c, b, a] = reference_plane_rotation * Vec{1., 0., 0.}; // plane normal
            const auto reference_center = image_center + reference.shifts;
            const auto d = b * -reference_center[0] + a * -reference_center[1]; // precompute coordinate - shifts
            reference_plane_coefficients = Vec{a, b, c, d}.as<f32>();

            // Compute the reference->target transformation.
            const auto target_angles = noa::deg2rad(target.angles);
            projection_matrix = (
                ng::translate(image_center.push_front(0) + target.shifts.push_front(0)) *
                ng::rotate_z<true>(+target_angles[0]) *
                ng::rotate_y<true>(+target_angles[1]) *
                ng::rotate_x<true>(+target_angles[2]) *
                ng::rotate_x<true>(-reference_angles[2]) *
                ng::rotate_y<true>(-reference_angles[1]) *
                ng::rotate_z<true>(-reference_angles[0]) *
                ng::translate(-image_center.push_front(0) - reference.shifts.push_front(0))
            ).filter_rows(1, 2).as<f32>(); // project along z-axis

            fov_mask = common_fov.set_fov(reference, {
                .smooth_edge_percent = parameters.smooth_edge_percent,
                .add_shifts = true,
                .add_tilt_and_pitch = parameters.area_match,
            });
        }
        Logger::trace("init took {}", t0.elapsed());

        start.record(stream);

        // Compute the reference and stretched target stacks.
        using interp_t = CreateStacks::interpolator_type;
        auto iwise_shape = image_shape.push_front(n_targets).as<i32>();
        auto create_stacks = CreateStacks2{
            .input = interp_t(stack.span_contiguous<const f32, 3, i32>(), image_shape.as<i32>()),
            .references = buffer(0).span_contiguous<f32, 3, i32>(),
            .stretched_targets = buffer(1).span_contiguous<f32, 3, i32>(),
            .fov_masks = m_fov_masks.span_contiguous<ParallelogramMask, 1, i32>(),
            .reference_plane_coefficients = m_plane_coefficients.span_contiguous<Vec<f32, 4>, 1, i32>(),
            .reference2target = m_projection_matrices.span_contiguous<Mat<f32, 2, 4>, 1, i32>(),
            .pivot = pivot,
        };

        if (parameters.compute_peaks) {
            noa::reduce_axes_iwise(
                iwise_shape, device,
                CreateStacks2::reduce_type{},
                m_peak_stats.view().flat(0),
                create_stacks
            );
        } else {
            noa::iwise(iwise_shape, device, create_stacks);
        }

        end.record(stream);
        end.synchronize();
        Logger::trace("CreateStacks2 took {}", noa::Event::elapsed(start, end));

        // Get the views from the buffers.
        if (Logger::is_debug()) {
            auto filename = *parameters.output_directory / "stretched_targets.mrc";
            noa::write_image(buffer(1), filename, {.dtype = "f16"});
            filename = *parameters.output_directory / "references.mrc";
            noa::write_image(buffer(0), filename, {.dtype = "f16"});
            Logger::debug("{} saved", filename);
        }

        start.record(stream);

        // (Conventional) cross-correlation.
        // Technically, we should zero-pad here to cancel the circular property of the DFT (only the zero-lag is
        // unaffected by it). However, while we do expect significant lags, since we don't care about the actual peak
        // value and that we iterate, as long as we can find the highest peak, it is fine. We don't normalize for the
        // same reason.

        // Batched in-place FFT seems best.
        nf::r2c(buffer(0, 2), buffer_rfft(0, 2));
        noa::iwise(buffer_rfft(0).shape().filter(0, 2, 3), device, CrossCorrelate{
            .references = buffer_rfft(0).span_contiguous<c32, 3, i32>(),
            .stretched_targets = buffer_rfft(1).span_contiguous<c32, 3, i32>(), // output
        });
        nf::c2r(buffer_rfft(1), buffer(0)); // xmap

        end.record(stream);
        end.synchronize();
        Logger::trace("CrossCorrelation took {}", noa::Event::elapsed(start, end));

        if (Logger::is_debug()) {
            auto filename = *parameters.output_directory / "xmap.mrc";
            noa::write_image(buffer(0), filename, {.dtype = "f16"});
            Logger::debug("{} saved", filename);
        }
        // panic();

        start.record(stream);

        // Compute the shift, i.e., by how much the projected target is away from the reference.
        // To align the target onto the reference, we would need to subtract this shift from it.
        find_peaks<"fc2fc">(buffer(0), m_xmap_centered.view(), m_peak_shifts.view(), m_peak_values.view(), {
            .distortion_angle_deg = metadata[0].angles[0],
            .max_shift_percent = parameters.max_shift_percent,
        });
        // TODO then normalize peaks using stats

        end.record(stream);
        end.synchronize();
        Logger::trace("find_shifts took {}", noa::Event::elapsed(start, end));

        // We should now transform the shift back to the target reference-frame. However, we'll need to compute the
        // global shifts and center them later on. These operations require accumulating the shifts of the lower views
        // up to the global reference. As such, the simplest is to scale all these slice-to-slice shifts directly to
        // the same reference-frame, process everything there, and then go back to each image's reference-frame at
        // the end. For simplicity, we chose this common reference-frame to be the volume reference-frame, which has
        // no rotation, no tilt, no pitch.

        f64 average_peak{};
        m_relative_shifts[static_cast<size_t>(pivot)] = {};
        for (i32 i{}; auto&& [peak_shift, peak_value, peak_stats]: zip(
            m_peak_shifts.span_1d(),
            m_peak_values.span_1d(),
            m_peak_stats.span_1d()
        )) {
            const auto [index_target, index_reference] = get_indices_(i++, pivot);
            auto target = metadata[index_target];
            auto reference = metadata[index_reference];
            if (not parameters.cosine_stretch)
                reference.angles = target.angles;

            // Compute the reference-plane coefficients.
            const auto reference_angles = noa::deg2rad(reference.angles);
            const auto reference_plane_rotation = (
                ng::rotate_z(reference_angles[0]) *
                ng::rotate_y(reference_angles[1]) *
                ng::rotate_x(reference_angles[2])
            );

            // Compute the z-coordinate at the image shift.
            const auto projected_shift = peak_shift.as<f64>();
            const auto [c, b, a] = reference_plane_rotation * Vec{1., 0., 0.}; // plane normal
            const auto z = -(a * projected_shift[1] + b * projected_shift[0]) / c;

            // Transform the shifts to volume-space.
            const auto reference2volume = (
                ng::rotate_x<true>(-reference_angles[2]) *
                ng::rotate_y<true>(-reference_angles[1]) *
                ng::rotate_z<true>(-reference_angles[0])
            ).filter_rows(1, 2);
            auto shift = reference2volume * Vec{z, projected_shift[0], projected_shift[1], 1.};

            m_relative_shifts[static_cast<size_t>(index_target)] = shift;

            if (parameters.compute_peaks) {
                // Center and L2 normalize the peak.
                const auto [lhs_sum, rhs_sum, lhs_sum_sqd, rhs_sum_sqd, mask_sum] = peak_stats.as<f64>();
                const auto meanL = lhs_sum / mask_sum;
                const auto meanR = rhs_sum / mask_sum;

                const auto varL = lhs_sum_sqd - mask_sum * meanL * meanL;
                const auto varR = rhs_sum_sqd - mask_sum * meanR * meanR;

                const auto denom = std::sqrt(varL * varR);
                if (denom >= 1e-6) {
                    average_peak += (peak_value - mask_sum * meanL * meanR) / denom;
                    average_peak /= denom;
                }
            }
        }
        average_peak /= static_cast<f64>(n_targets);

        return {relative2global_shifts_(m_relative_shifts, m_global_shifts, metadata, pivot), average_peak};
    }
}
