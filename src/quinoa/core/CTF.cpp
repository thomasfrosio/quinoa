#include "quinoa/core/CTF.hpp"
#include "quinoa/core/Utilities.h"
#include "quinoa/core/Plot.hpp"

namespace qn {
    CTF::CTF(
            const Shape4<i64>& stack_shape,
            const Shape2<i64>& patch_shape,
            const Vec2<i64>& patch_step,
            Device compute_device,
            Allocator allocator
    ) {
        NOA_CHECK(noa::all(patch_shape % 2 == 0), "Patch size should be even");

        // Patches.
        const i64 patches_in_y = patch_grid_1d_count(
                stack_shape[2], patch_shape[0], patch_step[0]);
        const i64 patches_in_x = patch_grid_1d_count(
                stack_shape[3], patch_shape[1], patch_step[1]);
        const i64 patches_in_slice = patches_in_y * patches_in_x;
        const i64 patches_in_stack = stack_shape[0] * patches_in_slice;
        const auto slice_patches_shape = Shape4<i64>{patches_in_slice, 1, patch_shape[0], patch_shape[1]};
        const auto stack_patches_shape = Shape4<i64>{patches_in_stack, 1, patch_shape[0], patch_shape[1]};

        // The patches are loaded one slice at a time. So allocate enough for one slice.
        auto options = ArrayOption(compute_device, allocator);
        m_slice = noa::memory::empty<f32>(stack_shape.set<0>(1), options);
        m_patches_rfft = noa::memory::empty<c32>(slice_patches_shape.rfft(), options);

        // For the initial fitting, only a few patches are used because patches are reduced.
        // For the global fitting, this holds the patches of the entire stack, that's the biggest array.
        m_patches_rfft_ps = noa::memory::empty<f32>(stack_patches_shape.rfft(), options);

        // Preserve the alignment between the row vectors, so use pitched memory.
        options = options.allocator(Allocator::PITCHED);
        const i64 rotational_average_size = noa::math::min(patch_shape) / 2 + 1;
        const auto rotational_average_shape = Shape4<i64>{patches_in_stack, 1, 1, rotational_average_size};
        m_rotational_averages = noa::memory::empty<f32>(rotational_average_shape, options);
        m_simulated_ctfs = noa::memory::empty<f32>(rotational_average_shape, options);
    }

    void CTF::fit_global(
            StackLoader& stack_loader,
            MetadataStack& metadata,
            Shape2<i64> patch_shape,
            Vec2<i64> patch_step,
            Vec2<f64> delta_z_range_nanometers,
            Vec2<f64> fitting_range,
            bool fit_phase_shift,
            CTFIsotropic64 ctf,
            const Path& debug_directory
    ) {
        // Allocate buffers.

        compute_rotational_average_of_mean_ps_(
                stack_loader, metadata, patch_shape, patch_step,
                delta_z_range_nanometers, debug_directory);
        fit_isotropic_ctf_to_rotational_average_(
                m_rotational_averages.subregion(0), ctf, fitting_range, fit_phase_shift, debug_directory);
    }
}

// Private methods:
namespace qn {
    auto CTF::patch_transformed_coordinate_(
            Shape2<i64> slice_shape,
            Vec2<f64> slice_shifts,
            Vec3<f64> slice_angles,
            Vec2<f64> slice_sampling,
            Vec2<f64> patch_center
    ) -> Vec3<f64> {
        slice_angles = noa::math::deg2rad(slice_angles);

        // By convention, the rotation angle is the additional rotation of the image.
        // Subtracting it aligns the tilt-axis to the y-axis.
        slice_angles[0] *= -1;

        // Switch coordinates from pixels to micrometers.
        const auto scale = slice_sampling * 1e-4;
        const auto slice_center_3d = (slice_shape.vec().as<f64>() * scale).push_front(0) / 2;
        const auto slice_shifts_3d = (slice_shifts * scale).push_front(0);

        // Place the slice into a 3d volume, with the center of the slice at the origin of the volume.
        namespace ng = noa::geometry;
        const Double44 image2microscope_matrix =
                ng::linear2affine(ng::euler2matrix(slice_angles, /*axes=*/ "zyx", /*intrinsic=*/ false)) *
                ng::translate(- slice_center_3d - slice_shifts_3d);

        const auto patch_center_3d = (patch_center * scale).push_front(0).push_back(1);
        const Vec3<f64> patch_center_transformed = (image2microscope_matrix * patch_center_3d).pop_back();
        return patch_center_transformed;
    }

    auto CTF::extract_patches_origins_(
            const Shape2<i64>& slice_shape,
            const MetadataSlice& metadata,
            Vec2<f64> sampling_rate,
            Shape2<i64> patch_shape,
            Vec2<i64> patch_step,
            Vec2<f64> delta_z_range_nanometers
    ) -> std::vector<Vec4<i32>> {
        // Divide 2d grid in patches.
        const std::vector<Vec2<i64>> initial_patches_origins = patch_grid_2d(
                slice_shape, patch_shape, patch_step);

        std::vector<Vec4<i32>> output_patches_origins;
        const Vec2<f64>& slice_shifts = metadata.shifts;
        const Vec3<f64>& slice_angles = metadata.angles;

        for (auto patch_origin: initial_patches_origins) {
            // Get the 3d position of the patch.
            const auto patch_center = (patch_origin + patch_shape.vec() / 2).as<f64>();
            const auto patch_coordinates = patch_transformed_coordinate_(
                    slice_shape, slice_shifts, slice_angles, sampling_rate, patch_center);

            // Filter based on its z position.
            // TODO Filter to remove patches at the corners?
            const auto z_nanometers = patch_coordinates[0] * 1e3; // micro -> nano
            if (z_nanometers < delta_z_range_nanometers[0] ||
                z_nanometers > delta_z_range_nanometers[1])
                continue;

            // Save the patch.
            output_patches_origins.emplace_back(0, 0, patch_origin[0], patch_origin[1]);
        }
        return output_patches_origins;
    }

    void CTF::compute_rotational_average_of_mean_ps_(
            StackLoader& stack_loader,
            const MetadataStack& metadata,
            Shape2<i64> patch_shape,
            Vec2<i64> patch_step,
            Vec2<f64> delta_z_range_nanometers,
            const Path& debug_directory
    ) {
        const auto slice = m_slice.view();
        const auto patches_rfft_ps_average = m_patches_rfft_ps.view().subregion(0);
        const auto patches_rfft_ps_average_tmp = m_patches_rfft_ps.view().subregion(1);

        // Loading every single patch at once can be too much if the compute device is a GPU.
        // Since this function is only called a few times, simply load the patches slice per slice.
        // TODO mask tiles and change patch-grid resolution with tilt to keep constant defocus range withing a patch.

        qn::Logger::trace("Patches (within {}nm range):", delta_z_range_nanometers);
        bool is_first{true};
        i64 index{0};
        size_t total{0};
        for (const auto& slice_metadata: metadata.slices()) {
            const std::vector<Vec4<i32>> patches_origins_vector = extract_patches_origins_(
                    stack_loader.slice_shape(), slice_metadata, stack_loader.stack_spacing(),
                    patch_shape, patch_step, delta_z_range_nanometers);
            const auto patches_origins = View(patches_origins_vector.data(), patches_origins_vector.size())
                    .to(slice.options()); // FIXME

            total += patches_origins_vector.size();
            qn::Logger::debug("index={:>02}, patches={:>03}, total={:>05}",
                              index, patches_origins_vector.size(), total);

            // Prepare the views.
            const auto patches_shape = patch_shape.push_front<2>({patches_origins.size(), 1});
            const auto patches_rfft = m_patches_rfft.view().subregion(noa::indexing::slice_t{0, patches_origins.size()});
            const auto patches = noa::fft::alias_to_real(patches_rfft, patches_shape);

            // Extract the patches. Assume the slice is normalized and edges are tapered.
            stack_loader.read_slice(slice, slice_metadata.index_file);
            noa::memory::extract_subregions(slice, patches, patches_origins);
            noa::math::normalize_per_batch(patches, patches);

            if (!debug_directory.empty())
                noa::io::save(patches, debug_directory / noa::string::format("patches_{:>02}.mrc", index));

            // Compute the average power-spectrum of these tiles.
            noa::fft::r2c(patches, patches_rfft, noa::fft::Norm::NONE);
            noa::math::mean(patches_rfft, patches_rfft_ps_average_tmp, noa::abs_squared_t{});
            if (is_first) {
                patches_rfft_ps_average_tmp.to(patches_rfft_ps_average);
                is_first = false;
            } else {
                noa::ewise_trinary(
                        patches_rfft_ps_average_tmp, patches_rfft_ps_average, 2.f,
                        patches_rfft_ps_average, noa::plus_divide_t{});
            }

            if (!debug_directory.empty()) {
                noa::io::save(noa::ewise_unary(patches_rfft_ps_average_tmp, noa::abs_one_log_t{}),
                              debug_directory / noa::string::format("patches_ps_average_{:>02}.mrc", index));
            }
            ++index;
        }

        if (!debug_directory.empty()) {
            noa::io::save(noa::ewise_unary(patches_rfft_ps_average, noa::abs_one_log_t{}),
                          debug_directory / "patches_ps_average.mrc");
        }

        // Compute the rotational average.
        const auto rotational_average = m_rotational_averages.view().subregion(0);
        noa::geometry::fft::rotational_average<fft::H2H>(
                patches_rfft_ps_average, rotational_average,
                patch_shape.push_front(Vec2<i64>{1}));
    }

    View<f32> trimmed_rotational_average_(
            const View<f32>& rotational_average,
            const Vec2<f64>& fitting_range, // angstrom
            const Vec2<f64>& sampling_rate // angstrom/pixel
    ) {
        // Resolution -> Index.
        const auto start = static_cast<i64>(std::round(fitting_range[0] / sampling_rate[0]));
        const auto stop = static_cast<i64>(std::round(fitting_range[1] / sampling_rate[1]));

        return rotational_average.subregion(noa::indexing::slice_t{start, stop});
    }

    auto CTF::fit_background_and_envelope_(
            const View<const f32>& rotational_average,
            noa::signal::fft::CTFIsotropic<f64> ctf,
            Vec2<f64> fitting_range, // resolution in Angstrom,
            bool gradient_based_optimization
    ) const -> std::pair<CubicSplineGrid<f64, 1>, CubicSplineGrid<f64, 1>> {

        // Simulate ctf slope with high enough sampling to prevent aliasing.
        constexpr i64 SIMULATED_SIZE = 4096;
        constexpr f64 SIMULATED_FREQUENCY_STEP = 1 / static_cast<f64>(SIMULATED_SIZE);

        // Assume patches were even sized. h=(n//2+1) -> n=((h-1)*2)
        const i64 spectrum_size = rotational_average.size();
        const f64 spectrum_logical_size = static_cast<f64>((spectrum_size - 1) * 2);
        const auto fitting_range_normalized_frequency = fitting_range; // FIXME

        // Circular buffer to only sample the ctf once per iteration.
        // Each iteration computes the ith+2 element, so we need to precompute the
        // first two for the first iteration.
        std::array<f64, 3> ctf_values{
                std::abs(ctf.value_at(0 * SIMULATED_FREQUENCY_STEP)),
                std::abs(ctf.value_at(1 * SIMULATED_FREQUENCY_STEP)),
                0
        };

        // Collect data points. [0]=frequency, [1]=value
        const auto rotational_average_accessor = rotational_average.accessor_contiguous_1d();
        std::vector<Vec2<f64>> points_at_zeros;
        std::vector<Vec2<f64>> points_at_peaks;

        size_t c = 2;
        for (i64 i = 0; i < SIMULATED_SIZE - 1; ++i) {
            // Do your part, sample ith+2 and increment circular buffer.
            ctf_values[c] = std::abs(ctf.value_at(static_cast<f64>(i + 2) * SIMULATED_FREQUENCY_STEP));
            c = (c + 1) % 3; // 0,1,2,0,1,2,...

            // Get the corresponding frequency in the experimental spectrum.
            // Here it's ok to round the frequency to the nearest sampled point.
            // We could interpolate, but we don't need that level of precision here.
            const f64 sampled_normalized_frequency = static_cast<f64>(i) * SIMULATED_FREQUENCY_STEP;
            f64 frequency = std::round(sampled_normalized_frequency * spectrum_logical_size);

            // If in the fitting range.
            if (fitting_range_normalized_frequency[0] <= sampled_normalized_frequency &&
                sampled_normalized_frequency <= fitting_range_normalized_frequency[1]) {

                // Compute the simulated ctf slope.
                const f64 ctf_value_0 = ctf_values[(c - 2) % 3];
                const f64 ctf_value_1 = ctf_values[(c - 1) % 3];
                const f64 ctf_value_2 = ctf_values[c];
                const f64 slope_0 = ctf_value_1 - ctf_value_0;
                const f64 slope_1 = ctf_value_2 - ctf_value_1;

                // The fitting range should prevent any OOB situations, but just for safety.
                const auto index = static_cast<i64>(frequency);
                const bool is_valid = index >= 0 && index < spectrum_size;

                // The coordinate system should be between [0,1] for the cubic grid.
                frequency *= 2;

                if (is_valid && slope_0 < 0 && slope_1 >= 0) // zero: negative slope to positive slope.
                    points_at_zeros.emplace_back(frequency, rotational_average_accessor[index]);
                if (is_valid && slope_0 > 0 && slope_1 <= 0) // peak: positive slope to negative slope.
                    points_at_peaks.emplace_back(frequency, rotational_average_accessor[index]);
            }
        }

        // Fit smooth curve.
        constexpr i64 SPLINE_RESOLUTION = 3;
        auto curve_fitting = [](Optimizer& optimizer, const std::vector<Vec2<f64>>& data_points) {
            struct OptimizerData {
                const std::vector<Vec2<f64>>* data_points{};
                CubicSplineGrid<f64, 1> model{};
            };

            auto function_to_minimise = [](
                    u32 n_parameters,
                    [[maybe_unused]] const f64* parameters,
                    f64* gradients,
                    void* instance
            ) {
                auto& data = *static_cast<OptimizerData*>(instance);

                // Compute the least-square score between the experimental points and the model.
                const auto compute_least_square = [&data]() -> f64 {
                    f64 score{0};
                    for (auto data_point: *data.data_points) {
                        const f64 experiment = data_point[1];
                        const f64 predicted = data.model.interpolate(data_point[0]);
                        score += std::pow(experiment - predicted, 2);
                    }
                    return score;
                };

                if (gradients) {
                    f64* model_parameters = data.model.data();
                    NOA_ASSERT(parameters == model_parameters);

                    for (u32 i = 0; i < n_parameters; ++i) {
                        const f64 parameter = model_parameters[i];

                        // TODO Use 32-bits precision?
                        const f64 delta = CentralFiniteDifference::delta(parameter);

                        model_parameters[i] = parameter - delta;
                        const f64 score_minus = compute_least_square();
                        model_parameters[i] = parameter + delta;
                        const f64 score_plus = compute_least_square();

                        gradients[i] = (score_plus - score_minus) / (2 * delta); // central finite difference
                        model_parameters[i] = parameter; // reset to original value
                    }
                }
                return compute_least_square();
            };

            // Some stats about the data points.
            f64 mean{0}, min{data_points[0][1]}, max{data_points[0][1]};
            for (auto data_point: data_points) {
                mean += data_point[1];
                min = std::min(min, data_point[1]);
                max = std::max(max, data_point[1]);
            }
            mean /= static_cast<f64>(data_points.size());

            OptimizerData optimizer_data{&data_points, CubicSplineGrid<f64, 1>(SPLINE_RESOLUTION)};
            optimizer_data.model.set_all_points_to(mean);

            optimizer.set_min_objective(function_to_minimise, &optimizer_data);
            optimizer.set_bounds(min - min * 0.25, max + max * 0.25);
            optimizer.optimize(optimizer_data.model.data());
            // FIXME tolerance
            return optimizer_data.model;
        };

        Optimizer optimizer(gradient_based_optimization ? NLOPT_LD_LBFGS : NLOPT_LN_SBPLX, SPLINE_RESOLUTION);
        CubicSplineGrid<f64, 1> background = curve_fitting(optimizer, points_at_zeros);
        CubicSplineGrid<f64, 1> envelope = curve_fitting(optimizer, points_at_peaks);
        return {background, envelope}; // TODO return first zero coord?
    }

    auto CTF::fit_isotropic_ctf_to_rotational_average_(
            Array<f32> rotational_average,
            CTFIsotropic64 ctf,
            Vec2<f64> fitting_range,
            bool fit_phase_shift,
            const Path& debug_directory
    ) -> std::pair<f64, f64> {
        Timer timer;
        timer.start();

        // Here we do a bunch of computations on small 1d arrays, so do everything on the CPU.
        if (rotational_average.device().is_gpu())
            rotational_average = rotational_average.to_cpu();
        const auto rotational_averages = noa::memory::empty<f32>(rotational_average.shape().set<0>(3));

        struct OptimizerData {
            const CTF* instance{};
            CTFIsotropic64 ctf{};
            Vec2<f64> fitting_range{};
            View<const f32> rotational_average_original{};
            View<f32> rotational_average_smooth{};
            View<f32> rotational_average_current{};
            View<f32> simulated_ctf{};
            const Path* debug_directory{};
        };
        auto optimizer_data = OptimizerData{
                this,
                ctf,
                fitting_range,
                rotational_average.view(),
                rotational_averages.view().subregion(0),
                rotational_averages.view().subregion(1),
                rotational_averages.view().subregion(2),
                &debug_directory
        };

        // To fit background and envelope, use a smoothed version of the rotational average.
        // Also, since we fit a cubic B-spline curves, prefilter the data so that the curves end up interpolating.
        noa::signal::convolve(optimizer_data.rotational_average_original,
                              optimizer_data.rotational_average_smooth,
                              noa::signal::gaussian_window<f32>(7, 1.25, /*normalize=*/ true));
        noa::geometry::cubic_bspline_prefilter(
                optimizer_data.rotational_average_smooth,
                optimizer_data.rotational_average_smooth);

        // Function to maximise.
        const auto function_to_maximise = [](u32 n, const f64* parameters, f64*, void* instance) -> f64 {
            // Retrieve data.
            auto& data = *static_cast<OptimizerData*>(instance);
            auto& self = *data.instance;
            const f64 defocus = parameters[0];
            const f64 phase_shift = n == 2 ? parameters[1] : 0;

            // Update isotropic CTF for these parameters.
            data.ctf.set_defocus(defocus);
            if (n == 2)
                data.ctf.set_phase_shift(phase_shift);

            // Fit background and envelope.
            const auto [background_model, envelope_model] = self.fit_background_and_envelope_(
                    data.rotational_average_smooth, data.ctf, data.fitting_range, true);

            // Subtract background from rotational average.
            apply_cubic_bspline_1d(
                    data.rotational_average_original,
                    data.rotational_average_current,
                    background_model, noa::minus_t{});

            // Simulate isotropic CTF.
            noa::signal::fft::ctf_isotropic<fft::H2H>(
                    {}, data.simulated_ctf, data.simulated_ctf.shape(), data.ctf); // FIXME
            apply_cubic_bspline_1d(
                    data.simulated_ctf, data.simulated_ctf,
                    envelope_model, noa::multiply_t{});

            // TODO Trim to fitting range?

            if (!data.debug_directory->empty()) {
                const auto size = data.simulated_ctf.shape()[3];
                const auto x = noa::memory::arange<f32>(size);
                matplot::plot(Span(x.data(), size), Span(data.rotational_average_current.data(), size));
                matplot::hold(matplot::on);
                matplot::plot(Span(x.data(), size), Span(data.simulated_ctf.data(), size));
                matplot::save((*data.debug_directory / "rotational_average_fitting_i.jpg").string());
            }

            // Normalized cross-correlation coefficient.
            // TODO Move this to noa.
            const f32 lhs_norm = std::sqrt(noa::math::sum(data.rotational_average_current, noa::abs_squared_t{}));
            const f32 rhs_norm = std::sqrt(noa::math::sum(data.simulated_ctf, noa::abs_squared_t{}));
            noa::ewise_binary(data.rotational_average_current, data.simulated_ctf,
                              data.rotational_average_current,
                              [=](f32 lhs, f32 rhs) {
                                  return lhs / lhs_norm + rhs / rhs_norm;
                              });
            const f32 cc = noa::math::sum(data.rotational_average_current);
            return static_cast<f64>(cc);
        };

        std::array<f64, 2> defocus_and_phase_shift{};
        i64 evaluations{0};
        {
            // Global optimization.
            Optimizer optimizer(NLOPT_LN_SBPLX, 1 + fit_phase_shift); // FIXME
            // TODO Bounds
            optimizer.set_max_objective(function_to_maximise, &optimizer_data);
            optimizer.optimize(defocus_and_phase_shift.data());
            evaluations += optimizer.number_of_evaluations();
        }
        {
            // Local optimization to polish to optimum.
            Optimizer optimizer(NLOPT_LN_SBPLX, 1 + fit_phase_shift); // FIXME
            // TODO Bounds
            optimizer.set_max_objective(function_to_maximise, &optimizer_data);
            optimizer.optimize(defocus_and_phase_shift.data());
            evaluations += optimizer.number_of_evaluations();
        }

        const f64 best_defocus = defocus_and_phase_shift[0];
        const f64 best_phase_shift = defocus_and_phase_shift[1];
        qn::Logger::trace("Rotational average fitting: defocus={}, phase_shift={} (evaluations={}, elapsed={}ms)",
                          best_defocus, best_phase_shift, evaluations, timer.elapsed());

        return {best_defocus, best_phase_shift};
    }
}
