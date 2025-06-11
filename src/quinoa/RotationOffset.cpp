#include <noa/Array.hpp>
#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>

#include "quinoa/RotationOffset.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/CommonArea.hpp"
#include "quinoa/Utilities.hpp"

namespace qn {
    void RotationOffset::search(
        const View<const f32>& stack,
        MetadataStack& metadata,
        const RotationOffsetParameters& options
    ) {
        auto timer = Logger::info_scope_time("Rotation offset alignment using common-lines");

        const f64 initial_rotation_offset = options.reset_rotation ? 0 : metadata[0].angles[0];
        for (auto& slice: metadata) {
            if (options.reset_rotation) {
                // For the initial search, we usually want to start from 0 and search +-90 deg.
                slice.angles[0] = 0.;
            } else if (not noa::allclose(initial_rotation_offset, slice.angles[0])) {
                slice.angles[0] = initial_rotation_offset;
                Logger::warn_once(
                    "The rotation search algorithm is assuming a fixed tilt-axis, "
                    "but the provided stack has images with different rotation offsets. "
                    "To continue, the existing values will be overwritten with the "
                    "rotation offset of the lowest tilt."
                );
            }
        }

        // To project the stack along any axis, we need to zero-pad by at least sqrt(2) first.
        // Since the projection is done in Fourier space, bring this up to a factor of 2.
        // TODO Experiment with lower oversampling and Lanczos-6|8 interpolation.
        const auto shape = stack.shape();
        const auto padding = max(shape.filter(2, 3));
        const auto shape_padded = shape + Shape<i64, 4>{0, 0, padding, padding};
        const auto center_padded = (shape_padded.vec.filter(2, 3) / 2).as<f32>();
        const auto border_left = (shape_padded / 2 - shape / 2).vec.filter(2, 3);
        const auto border_right = padding - border_left;
        const auto oversampling_factor = static_cast<f64>(shape_padded[3]) / static_cast<f64>(shape[3]);

        const auto angle_range = Vec{-options.angle_range, +options.angle_range};
        const auto rotation_range = initial_rotation_offset + angle_range;
        const auto phi_range = Vec{-options.line_range, +options.line_range};
        const auto n_lines_per_target = static_cast<i64>(std::round(2 * options.line_range / options.line_step)) + 1;
        const auto shape_polar = Shape<i64, 4>{shape[0], 1, n_lines_per_target, shape_padded[3]};
        const auto shape_lines = Shape<i64, 4>{shape[0] * n_lines_per_target, 1, 1, shape_padded[3]};

        const auto device = stack.device();
        Logger::trace(
            "RotationOffset()::search():\n"
            "  reset_rotation={}\n"
            "  device={}\n"
            "  fourier_interp={} (oversampling_factor={:.2f})\n"
            "  angle_range={:.2f}deg (rotation_offset={:.2f})\n"
            "  angle_step={:.3f}deg\n"
            "  line_range={:.2f}deg\n"
            "  line_step={:.2f}deg\n"
            "  lines_size={}",
            options.reset_rotation, device,
            options.interp, oversampling_factor,
            options.angle_range, initial_rotation_offset,
            options.angle_step, options.line_range, options.line_step,
            shape_padded[3]
        );

        // Allocate the padded stack and initialize it with the input stack.
        if (noa::vany(noa::NotEqual{}, shape_padded.rfft(), m_stack_padded_rfft.shape()) or
            device != m_stack_padded_rfft.device())
        {
            m_stack_padded_rfft = noa::Array<c32>(shape_padded.rfft(), {
                .device = device,
                .allocator =  Allocator::DEFAULT_ASYNC,
            });
            Logger::trace(
                "RotationOffset(): allocated {:.2f}GB on {} (shape={}, allocator={})",
                static_cast<f64>(m_stack_padded_rfft.size() * sizeof(c32)) * 1e-9,
                device, m_stack_padded_rfft.shape(), m_stack_padded_rfft.allocator()
            );
        }
        const auto stack_padded_rfft = m_stack_padded_rfft.view();
        const auto stack_padded = noa::fft::alias_to_real(stack_padded_rfft, shape_padded);
        const auto stack_padded_filled = stack_padded.subregion(
            ni::Ellipsis{}, 0,
            ni::Slice{border_left[0], -border_right[0]},
            ni::Slice{border_left[1], -border_right[1]});

        // Allocate dereferenceable buffers.
        const auto options_managed = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
        const auto inverse_matrices = Array<Mat<f32, 2, 3>>(shape[0], options_managed);
        const auto shifts = Array<Vec<f32, 2>>(shape[0], options_managed);
        const auto polar_rfft = Array<c32>(shape_polar.rfft(), options_managed);
        const auto polar = noa::fft::alias_to_real(polar_rfft.view(), shape_polar);

        std::vector<f64> nccs{};
        f64 best_ncc{};
        f64 best_rotation_offset{};
        auto grid_search = GridSearch<f64>({
            .start = rotation_range[0],
            .end = rotation_range[1],
            .step = options.angle_step
        });
        grid_search.for_each([&, metadata](f64 rotation_offset) mutable {
            // Set to the current rotation.
            for (auto& slice: metadata)
                slice.angles[0] = rotation_offset;

            // Zero-pad for the Fourier projection, and enforce a common field-of-view.
            // Note that this is applied to the unaligned stack, so don't correct for the shifts.
            noa::fill(stack_padded, 0);
            auto area = CommonArea();
            area.set_geometry(shape.filter(2, 3), metadata);
            area.compute_inverse_transforms(metadata, inverse_matrices.span_1d_contiguous(), false);
            area.mask(stack, stack_padded_filled, inverse_matrices.view(), 0.1); // TODO check

            // Get the spectra.
            noa::fft::r2c(stack_padded, stack_padded_rfft);

            // Move the real-space signal to the rotation center.
            for (auto span = shifts.span_1d_contiguous(); auto& slice: metadata)
                span[slice.index] = -slice.shifts.as<f32>() - center_padded;
            ns::phase_shift_2d<"h2h">(
                stack_padded_rfft, stack_padded_rfft, shape_padded,
                shifts.reinterpret_as(device.type())
            );

            // Compute the lines, aka real-space projections orthogonal to the searched tilt-axes.
            // The rotation offset is relative to the y-axis, so we need to add 90deg to match the phi-range.
            const auto iphi_range = noa::deg2rad(phi_range + rotation_offset + 90);
            ng::spectrum2polar<"h2fc">(stack_padded_rfft, shape_padded, polar_rfft.view(), {
                .phi_range = {iphi_range[0], iphi_range[1], true},
                .interp = options.interp,
            });

            // Get the final projections in real-space.
            // We need to process these lines in batch, so reshape: (b,1,h,w) -> (b*h,1,1,w).
            const auto lines_rfft = polar_rfft.view().reinterpret_as_cpu().reshape(shape_lines.rfft());
            ns::bandpass<"h2h">(lines_rfft, lines_rfft, shape_lines, options.bandpass);
            ns::phase_shift_1d<"h2h">(lines_rfft, lines_rfft, shape_lines, center_padded.filter(1));
            auto lines = noa::fft::alias_to_real(lines_rfft, shape_lines);
            noa::fft::c2r(lines_rfft, lines);

            auto normalized_cross_correlate = [](auto lhs, auto rhs) -> f64 {
                f64 ncc{}, lhs_ncc{}, rhs_ncc{};
                for (i64 i{}; i < lhs.ssize(); ++i) {
                    ncc += static_cast<f64>(lhs[i] * rhs[i]);
                    lhs_ncc += static_cast<f64>(lhs[i] * lhs[i]);
                    rhs_ncc += static_cast<f64>(rhs[i] * rhs[i]);
                }
                return ncc / (std::sqrt(lhs_ncc) * std::sqrt(rhs_ncc));
            };

            // Cross-correlation.
            const auto index_reference = metadata.find_lowest_tilt_index();
            const auto lines_bhw = polar.span().filter(0, 2, 3).as_contiguous(); // (b,h,w)
            const auto reference = lines_bhw.subregion(index_reference, n_lines_per_target / 2).as_1d();

            f64 ncc{};
            for (i64 b{}; b < lines_bhw.shape()[0]; ++b) {
                if (b == index_reference)
                    continue;

                f64 b_ncc{};
                const auto lines_hw = lines_bhw[b];
                for (i64 h{}; h < lines_hw.shape()[0]; ++h) {
                    const f64 score = normalized_cross_correlate(reference, lines_hw[h]);
                    b_ncc = std::max(b_ncc, score);
                }
                ncc += b_ncc;
            }
            ncc /= static_cast<f64>(lines_bhw.shape()[0] - 1);

            // Maximize the ncc.
            if (ncc > best_ncc) {
                best_ncc = ncc;
                best_rotation_offset = rotation_offset;
            }
            nccs.push_back(ncc);
        });

        const auto increment = best_rotation_offset - initial_rotation_offset;
        Logger::info(
            "rotation_offset={:.3f} degrees (increment={:+.3f}, ncc={:.4f}, n_iter={}), or equivalently {:.3f} degrees",
            best_rotation_offset, increment, best_ncc, grid_search.size(),
            MetadataSlice::to_angle_range(best_rotation_offset + 180)
        );
        save_plot_xy(noa::Arange{rotation_range[0], options.angle_step}, nccs,
            options.output_directory / "rotation_offset.txt", {
                .title = "Rotation offset search",
                .x_name = "Rotation offset (degrees)",
                .y_name = "NCC",
            });

        // Update metadata with the new rotation.
        for (auto& slice: metadata)
            slice.angles[0] = MetadataSlice::to_angle_range(slice.angles[0] + increment);
    }
}
