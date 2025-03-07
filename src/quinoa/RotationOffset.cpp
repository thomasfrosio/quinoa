#include "quinoa/RotationOffset.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Optimizer.hpp"

namespace {
    using namespace qn;

    class RotationOffsetFitter {
    public:
        MetadataStack* metadata;
        const RotationOffsetParameters* parameters;
        Memoizer memoizer{};

        View<f32> slices;
        View<f32> slices_masked;
        View<c32> slices_masked_rfft;

        View<c32> lines{};
        i64 n_targets;
        i64 line_size;
        i64 line_size_rfft;

        Vec2<f64> rho_range;
        Vec2<f64> phi_range{};
        View<Mat23<f32>> inverse_matrices;

    public:
        [[nodiscard]] auto cost(f64 rotation_offset) const -> f64 {
            const auto compute_type = slices.device().type();
            const i64 n_slices = slices_masked.shape()[0];

            // Set to the current rotation. Note that the rotation offset is relative to the y-axis,
            // so we need to add 90deg for the spectrum2polar rho-range.
            for (auto& slice: *metadata)
                slice.angles[0] = rotation_offset;
            const f64 rotation_offset_rad = noa::deg2rad(rotation_offset + 90);

            // Compute and apply the common field-of-view.
            auto common_area = CommonArea();
            common_area.set_geometry(slices.shape().filter(2, 3), *metadata);
            common_area.compute_inverse_transforms(
                *metadata, inverse_matrices.reinterpret_as_cpu().span_1d_contiguous(), true);
            common_area.mask(slices, slices_masked, inverse_matrices.reinterpret_as(compute_type), 0.1);

            // Get the spectra.
            noa::fft::r2c(slices_masked, slices_masked_rfft);

            // Prefetch the lines to the compute device, if not already.
            lines.reinterpret_as(compute_type);
            auto reference_line = lines.subregion(0);
            auto target_lines = lines.subregion(ni::Slice{1}).reshape({n_targets, 1, -1, line_size_rfft});

            // Compute the reference line.
            ng::spectrum2polar<"h2fc">(
                slices_masked_rfft.subregion(0),
                slices_masked.shape().set<0>(1),
                reference_line,
                {
                    .rho_range = rho_range,
                    .rho_endpoint = true,
                    .phi_range = {rotation_offset_rad, rotation_offset_rad}, // one line
                });

            // Compute the target lines.
            ng::spectrum2polar<"h2fc">(
                slices_masked_rfft.subregion(ni::Slice{1, n_slices}),
                slices_masked.shape().set<0>(n_targets),
                target_lines,
                {
                    .rho_range = rho_range,
                    .rho_endpoint = true,
                    .phi_range = phi_range + rotation_offset_rad,
                    .phi_endpoint = true,
                });

            // Weighting and normalize on the CPU.
            auto lines_cpu = lines.reinterpret_as_cpu();
            ns::bandpass<"h2h">(lines_cpu, lines_cpu, lines.shape().set<3>(line_size), parameters->bandpass, {
                .fftfreq_range = rho_range,
                .fftfreq_endpoint = true,
            });
            noa::normalize_per_batch(lines_cpu, lines_cpu, {.mode = noa::Norm::L2});

            // Real-valued cross-correlation score between the complex-valued reference and the target.
            auto cross_correlate = [](auto lhs, auto rhs) -> f64 {
                f64 score{};
                for (i64 i{}; i < lhs.ssize(); ++i)
                    score += static_cast<f64>((lhs[i] * noa::conj(rhs[i])).real);
                return score;
            };

            // Compute the cross-correlation.
            // Take the best target line and sum across all targets.
            auto reference_line_w = reference_line.span_1d_contiguous();
            auto target_lines_bhw = target_lines.span().filter(0, 2, 3).as_contiguous();
            f64 total_score{};
            for (i64 b{}; b < target_lines_bhw.shape()[0]; ++b) {
                auto target_lines_hw = target_lines_bhw[b];
                f64 best_score = std::numeric_limits<f64>::lowest();
                // i64 best_index = -1;
                for (i64 h{}; h < target_lines_bhw.shape()[1]; ++h) {
                    f64 score = cross_correlate(reference_line_w, target_lines_hw[h]);
                    best_score = std::max(best_score, score);
                    // if (score > best_score) {
                    //     best_score = score;
                    //     best_index = h;
                    // }
                }
                total_score += best_score;
            }
            return total_score / static_cast<f64>(n_targets);
        }

        static auto function_to_maximise(
            [[maybe_unused]] u32 n_parameters, const f64* parameters, f64* gradients, void* instance
        ) -> f64 {
            check(n_parameters == 1 and parameters != nullptr);
            auto& self = *static_cast<RotationOffsetFitter*>(instance);
            auto& memoizer = self.memoizer;
            const f64 rotation_offset = *parameters;

            std::optional<f64> memoized_cost = memoizer.find(parameters, gradients, /*epsilon=*/ 1e-8);
            if (memoized_cost.has_value()) {
                auto cost = memoized_cost.value();
                trace(cost, rotation_offset, gradients, true);
                return cost;
            }

            const f64 cost = self.cost(rotation_offset);
            if (gradients) {
                constexpr f64 DELTA = 0.1;
                const f64 f_plus = self.cost(rotation_offset + DELTA);
                const f64 f_minus = self.cost(rotation_offset - DELTA);
                gradients[0] = CentralFiniteDifference::get(f_minus, f_plus, DELTA);
            }

            memoizer.record(parameters, cost, gradients);
            trace(cost, rotation_offset, gradients);
            return cost;
        }

        static void trace(f64 cost, f64 rotation_offset, const f64* gradient, bool memoized = false) {
            Logger::trace(
                "v={:+.8f}, f={:.8f}{}{}",
                rotation_offset, cost,
                gradient ? fmt::format(", g={:.8f}", *gradient) : "",
                memoized ? ", memoized=true" : "");
        }
    };
}

namespace qn {
    RotationOffset::RotationOffset(
        const View<f32>& input,
        const MetadataStack& metadata,
        f64 absolute_max_tilt_difference
    ) {
        m_metadata_sorted = metadata;
        m_metadata_sorted.sort("absolute_tilt");

        // Remove the excluded views.
        const auto reference_tilt_angle = m_metadata_sorted[0].angles[1];
        m_metadata_sorted.exclude([&](const MetadataSlice& slice) {
            const auto absolute_tilt_difference = std::abs(reference_tilt_angle - slice.angles[1]);
            return absolute_tilt_difference > absolute_max_tilt_difference;
        });

        // Allocate the slices.
        const auto stack_shape = Shape4<i64>{m_metadata_sorted.ssize(), 1, input.shape()[2], input.shape()[3]};
        const auto options = ArrayOption{input.device(), Allocator::DEFAULT_ASYNC};
        m_slices = noa::Array<f32>(stack_shape, options);
        m_slices_rfft = noa::Array<c32>(stack_shape.rfft(), options);

        const f64 n_bytes = static_cast<f64>(
            m_slices.size() * sizeof(f32) +
            m_slices_rfft.size() * sizeof(c32)
        ) * 1e-9;
        Logger::trace("RotationOffset(): allocated {:.2f}GB on {} ({})", n_bytes, options.device, options.allocator);
    }

    void RotationOffset::search(
        const View<f32>& input,
        MetadataStack& metadata,
        const RotationOffsetParameters& parameters
    ) {
        if (m_slices_rfft.is_empty())
            return;

        auto timer = Logger::info_scope_time("Rotation offset alignment using common-lines...");
        Logger::trace(
            "compute_device={}\n"
            "interpolation={}\n"
            "reset_rotation={}\n"
            "Global search:\n"
            "  range={:.2f}deg\n"
            "  step={:.2f}deg\n"
            "  line_range={:.2f}deg\n"
            "  line_delta={:.2f}deg\n"
            "Local search:\n"
            "  range={:.2f}deg\n"
            "  use_gradients={}\n"
            "  line_range={:.2f}deg\n"
            "  line_delta={:.2f}deg",
            m_slices_rfft.device(), parameters.interp, parameters.reset_rotation,
            parameters.grid_search_range, parameters.grid_search_step,
            parameters.grid_search_line_range, parameters.grid_search_line_delta,
            parameters.local_search_range, parameters.local_search_using_estimated_gradient,
            parameters.local_search_line_range, parameters.local_search_line_delta
        );

        // Update the metadata.
        for (auto& slice: metadata) {
            for (auto& sorted: m_metadata_sorted) {
                if (slice.index == sorted.index) {
                    sorted = slice;
                    if (parameters.reset_rotation)
                        sorted.angles[0] = 0.;
                }
            }
        }

        // Assume that the rotation is the same for every view.
        f64 rotation_offset = m_metadata_sorted[0].angles[0];

        // Copy the slices to a contiguous buffer and align to the rotation center.
        const auto slices = m_slices.view();
        const i64 n_targets = slices.shape()[0] - 1;
        for (i64 i{}; const auto& slice: m_metadata_sorted) {
            ng::transform_2d(
                input.subregion(slice.index), slices.subregion(i++),
                ng::translate(slice.shifts).pop_back().as<f32>(),
                {.interp = parameters.interp});
        }

        // Set the fftfreq range and line size.
        const auto slice_shape = slices.shape().filter(2, 3);
        const auto rho_range = Vec{0., noa::max(noa::fft::highest_fftfreq<f64>(slice_shape))};
        const i64 line_size = noa::min(slice_shape);
        const i64 line_size_rfft = line_size / 2 + 1;

        // Allocate matrices for the common-area. Keep them dereferenceable.
        auto inverse_matrices = Array<Mat<f32, 2, 3>>(m_slices.shape()[0], {
            .device = slices.device(),
            .allocator = Allocator::MANAGED,
        });

        const auto slices_masked_rfft = m_slices_rfft.view();
        const auto slices_masked = noa::fft::alias_to_real(slices_masked_rfft, slices.shape());

        auto fitter = RotationOffsetFitter{
            .metadata = &m_metadata_sorted,
            .parameters = &parameters,
            .slices = slices,
            .slices_masked = slices_masked,
            .slices_masked_rfft = slices_masked_rfft,
            .n_targets = n_targets,
            .line_size = line_size,
            .line_size_rfft = line_size_rfft,
            .rho_range = rho_range,
            .inverse_matrices = inverse_matrices.view(),
        };

        // TODO 1. rotation is relative to y.
        //      1bis. correct for the case where we do have a rotation offset. The search is relative to this value.
        //      2. check in cpu mode the cost function, making sure everything is okay.
        //      3. check with perfect tilt-series (no shift). We should get near perfect match ncc=1

        {
            // Get the number of lines.
            const i64 n_lines_per_target = static_cast<i64>(
                std::round(parameters.grid_search_line_range * 2 / parameters.grid_search_line_delta)) + 1;
            fitter.phi_range = noa::deg2rad(Vec{
                -parameters.grid_search_line_range,
                +parameters.grid_search_line_range
            });

            // Allocate the lines. Keep them dereferenceable.
            const i64 n_lines = 1 + n_targets * n_lines_per_target;
            auto lines = Array<c32>({n_lines, 1, 1, line_size_rfft}, {
                .device = slices.device(),
                .allocator = Allocator::MANAGED,
            });
            fitter.lines = lines.view();

            Logger::trace("Grid search...");
            auto grid_search = GridSearch<f64>({
                .start=rotation_offset - parameters.grid_search_range,
                .end=rotation_offset + parameters.grid_search_range,
                .step=parameters.grid_search_step
            });
            std::vector<f64> scores{};
            scores.reserve(grid_search.size());
            f64 best_score = std::numeric_limits<f64>::lowest();
            grid_search.for_each([&](f64 rotation) {
                f64 score = RotationOffsetFitter::function_to_maximise(1, &rotation, nullptr, &fitter);
                scores.emplace_back(score);
                if (score > best_score) {
                    best_score = score;
                    rotation_offset = rotation;
                }
            });
            Logger::trace("Grid search... done. rotation_offset={:.3f}, best_score={:.5f}, n_evaluations={}",
                          rotation_offset, best_score, scores.size());

            auto filename = parameters.output_directory / "initial_search_common_lines_ncc.txt";
            noa::write_text(fmt::format("{:.5f}\n", fmt::join(scores, ",")), filename);
            Logger::info("{} saved", filename);
        }

        {
            // Get the number of lines.
            const i64 n_lines_per_target = static_cast<i64>(
                std::round(parameters.local_search_line_range * 2 / parameters.local_search_line_delta)) + 1;
            fitter.phi_range = noa::deg2rad(Vec{
                -parameters.local_search_line_range,
                +parameters.local_search_line_range
            });

            // Allocate the lines. Keep them dereferenceable.
            const i64 n_lines = 1 + n_targets * n_lines_per_target;
            auto lines = Array<c32>({n_lines, 1, 1, line_size_rfft}, {
                .device = slices.device(),
                .allocator = Allocator::MANAGED,
            });
            fitter.lines = lines.view();

            Logger::trace("Local search...");
            fitter.memoizer = Memoizer{1, 8};
            const auto algorithm = parameters.local_search_using_estimated_gradient ? NLOPT_LD_LBFGS : NLOPT_LN_SBPLX;
            const auto optimizer = Optimizer(algorithm, 1);
            optimizer.set_max_objective(RotationOffsetFitter::function_to_maximise, &fitter);
            optimizer.set_x_tolerance_abs(0.01);
            optimizer.set_bounds(
                rotation_offset - parameters.local_search_range,
                rotation_offset + parameters.local_search_range);
            f64 best_score = optimizer.optimize(&rotation_offset);
            Logger::trace("Local search... done. rotation_offset={:.3f}, best_score={:.5f}, n_evaluations={}",
                          rotation_offset, best_score, optimizer.n_evaluations());
        }

        // Update metadata.
        rotation_offset = MetadataSlice::to_angle_range(rotation_offset);
        for (auto& slice: metadata)
            slice.angles[0] = rotation_offset;

        Logger::info("rotation_offset={:.3f} degrees (which at this stage is equivalent to {:3.f} degrees)",
                     rotation_offset, MetadataSlice::to_angle_range(rotation_offset + 180));
    }
}
