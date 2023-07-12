#include "quinoa/core/CTF.hpp"
#include <noa/unified/Allocator.hpp>

namespace {
    using namespace ::qn;

    // The parameters to optimize.
    //  - The parameters are divided in two categories, 1) the global parameters, and 2) the defoci.
    //    The global parameters affect every patch of every slice, while the defocus (of a given slice)
    //    only affects the patches of that slice. This distinction is used to compute the gradients efficiently.
    //  - The parameters are saved contiguously, so we can iterate through them. If a global parameter is not
    //    fitted, it is not included in this continuous buffer.
    class Parameters {
    public:
        Parameters(
                Vec3<bool> fit_angles, bool fit_phase_shift, bool fit_astigmatism,
                i64 n_slices, CTFIsotropic64 average_ctf,
                f64 astigmatism_value = 0,
                f64 astigmatism_angle = 0
        ) :
                m_n_defocus(n_slices),
                m_fit(fit_angles[0], fit_angles[1], fit_angles[2], fit_phase_shift, fit_astigmatism),
                m_buffer(size())
        {
            // Set the indexes.
            i64 count{0};
            for (auto i: irange<size_t>(6)) { // rotation, tilt, elevation, phase_shift, astigmatism x2
                if (m_fit[std::min(i, size_t{5})]) // astigmatism value/angle are both at 5
                    m_indexes[i] = count++;
            }

            // Initialise values.
            if (has_phase_shift())
                set(phase_shift_index(), average_ctf.phase_shift());
            if (has_astigmatism()) {
                set(astigmatism_value_index(), astigmatism_value);
                set(astigmatism_angle_index(), astigmatism_angle);
            }
            const f64 average_defocus = average_ctf.defocus();
            for (auto& defocus: defoci())
                defocus = average_defocus;
        }

        // Sets the (low and high) bounds for every parameter.
        void set_bounds(Vec2<f64> rotation, Vec2<f64> tilt, Vec2<f64> elevation,
                        Vec2<f64> phase_shift, Vec2<f64> astigmatism_value, Vec2<f64> astigmatism_angle,
                        Vec2<f64> defocus) {
            m_lower_bounds.clear();
            m_upper_bounds.clear();
            m_lower_bounds.reserve(size());
            m_upper_bounds.reserve(size());

            const auto push_back = [&](const Vec2<f64>& low_and_high_bounds) {
                m_lower_bounds.push_back(low_and_high_bounds[0]);
                m_upper_bounds.push_back(low_and_high_bounds[1]);
            };

            if (has_rotation())
                push_back(rotation);
            if (has_tilt())
                push_back(tilt);
            if (has_elevation())
                push_back(elevation);
            if (has_phase_shift())
                push_back(phase_shift);
            if (has_astigmatism()) {
                push_back(astigmatism_value);
                push_back(astigmatism_angle);
            }
            for (i64 i = 0; i < n_defocus(); ++i)
                push_back(defocus);
        }

    public:
        [[nodiscard]] constexpr i64 n_globals() const noexcept { return noa::math::sum(m_fit.as<i64>()); }
        [[nodiscard]] constexpr i64 n_defocus() const noexcept { return m_n_defocus; }
        [[nodiscard]] constexpr i64 ssize() const noexcept { return n_globals() + n_defocus(); }
        [[nodiscard]] constexpr size_t size() const noexcept { return static_cast<size_t>(ssize()); }

        void set(i64 index, f64 value) noexcept {
            m_buffer[static_cast<size_t>(index)] = value;
        }
        [[nodiscard]] f64 get(i64 index) const noexcept {
            return m_buffer[static_cast<size_t>(index)];
        }
        [[nodiscard]] f64* data() noexcept {
            return m_buffer.data();
        }

        [[nodiscard]] constexpr bool has_rotation() const noexcept { return m_fit[0]; }
        [[nodiscard]] constexpr bool has_tilt() const noexcept { return m_fit[1]; }
        [[nodiscard]] constexpr bool has_elevation() const noexcept { return m_fit[2]; }
        [[nodiscard]] constexpr bool has_phase_shift() const noexcept { return m_fit[3]; }
        [[nodiscard]] constexpr bool has_astigmatism() const noexcept { return m_fit[4]; }

        [[nodiscard]] constexpr i64 rotation_index() const noexcept { return m_indexes[0]; }
        [[nodiscard]] constexpr i64 tilt_index() const noexcept { return m_indexes[1]; }
        [[nodiscard]] constexpr i64 elevation_index() const noexcept { return m_indexes[2]; }
        [[nodiscard]] constexpr i64 phase_shift_index() const noexcept { return m_indexes[3]; }
        [[nodiscard]] constexpr i64 astigmatism_value_index() const noexcept { return m_indexes[4]; }
        [[nodiscard]] constexpr i64 astigmatism_angle_index() const noexcept { return m_indexes[5]; }
        [[nodiscard]] constexpr i64 defocus_index() const noexcept { return n_globals(); }

        [[nodiscard]] Span<f64> globals() noexcept {
            return {m_buffer.data(), n_globals()};
        }

        [[nodiscard]] Span<f64> defoci() noexcept {
            return {m_buffer.data() + n_globals(), n_defocus()};
        }

        [[nodiscard]] Span<f64> lower_bounds() noexcept {
            return {m_lower_bounds.data(), ssize()};
        }

        [[nodiscard]] Span<f64> upper_bounds() noexcept {
            return {m_upper_bounds.data(), ssize()};
        }

    public: // safe access of the globals, whether they are fitted or not.
        [[nodiscard]] Vec3<f64> angles() const noexcept {
            return {
                    has_rotation() ? get(rotation_index()) : 0,
                    has_tilt() ? get(tilt_index()) : 0,
                    has_elevation() ? get(elevation_index()) : 0
            };
        }

        [[nodiscard]] f64 phase_shift() const noexcept {
            return has_phase_shift() ? get(phase_shift_index()) : 0;
        }

        [[nodiscard]] f64 astigmatism_value() const noexcept {
            return has_astigmatism() ? get(astigmatism_value_index()) : 0;
        }

        [[nodiscard]] f64 astigmatism_angle() const noexcept {
            return has_astigmatism() ? get(astigmatism_angle_index()) : 0;
        }

    private:
        i64 m_n_defocus;
        Vec<bool, 5> m_fit; // rotation, tilt, elevation, phase_shift, astigmatism
        std::vector<f64> m_buffer;
        std::array<i64, 6> m_indexes{}; // rotation, tilt, elevation, phase_shift, astigmatism x2
        std::vector<f64> m_lower_bounds{};
        std::vector<f64> m_upper_bounds{};
    };
}

namespace qn {
    CTF::Patches CTF::compute_patches_rfft_ps_(
            StackLoader& stack_loader,
            const MetadataStack& metadata,
            const FittingRange& fitting_range,
            f64 max_tilt,
            const Path& debug_directory
    ) {
        const auto options = m_slice.options();
        const auto n_patches_per_slice = m_patches_rfft.shape()[0];
        const auto patch_shape = Shape2<i64>(m_patch_size);

        // How many patches do we have?
        i64 n_patches_per_stack{0};
        for (const auto& slice_metadata: metadata.slices()) {
            if (std::abs(slice_metadata.angles[1]) <= max_tilt)
                n_patches_per_stack += n_patches_per_slice;
        }

        // Here, to save memory and computation, we store the Fourier crop spectra directly.
        const auto cropped_patch_shape = fitting_range
                .fourier_cropped_shape()
                .push_front<2>({n_patches_per_stack, 1});

        // This is the big array with all the patches. For safety here, use managed memory
        // in case the compute-device cannot hold the entire thing at once.
        qn::Logger::trace("Patches (max_tilt={} degrees, n_patches={} ({}GB)):",
                          max_tilt, n_patches_per_stack,
                          static_cast<f64>(cropped_patch_shape.rfft().as<size_t>().elements() * sizeof(f32)) * 1e-9);

        Patches cropped_patches{
                /*rfft_ps=*/ noa::memory::empty<f32>(
                        cropped_patch_shape.rfft(),
                        ArrayOption(options).set_allocator(Allocator::MANAGED)),

                /*center_coordinates=*/ extract_patches_centers(
                        stack_loader.slice_shape(),
                        patch_shape, Vec2<i64>(m_patch_step))
        };

        // Loading every single patch at once can be too much if the compute-device is a GPU.
        // Since this function is only called a few times, simply load the patches slice per slice.
        const auto slice = m_slice.view();
        const auto patches_rfft = m_patches_rfft.view();
        const auto patches_rfft_ps = noa::memory::like<f32>(m_patches_rfft);
        const auto patches_origins = noa::memory::empty<Vec4<i32>>(n_patches_per_slice, options);
        i64 index{0};
        i64 total{0};

        for (const auto& slice_metadata: metadata.slices()) {
            if (std::abs(slice_metadata.angles[1]) > max_tilt)
                continue;

            // Get the patch origins.
            const std::vector<Vec4<i32>> patches_origins_vector = extract_patches_origins(
                    stack_loader.slice_shape(), slice_metadata, stack_loader.stack_spacing(),
                    patch_shape, Vec2<i64>(m_patch_step));
            NOA_ASSERT(patches_origins_vector.size() == static_cast<size_t>(n_patches_per_slice));
            View(patches_origins_vector.data(), n_patches_per_slice).to(patches_origins);

            // Prepare the patches for extraction.
            const auto patches = noa::fft::alias_to_real(
                    patches_rfft, patch_shape.push_front<2>({n_patches_per_slice, 1}));

            // Extract the patches. Assume the slice is normalized and edges are tapered.
            stack_loader.read_slice(slice, slice_metadata.index_file);
            noa::memory::extract_subregions(slice, patches, patches_origins);
            noa::math::normalize_per_batch(patches, patches);

            if (!debug_directory.empty())
                noa::io::save(patches, debug_directory / noa::string::format("patches_{:>02}.mrc", index));

            // Compute the power-spectra of these tiles.
            noa::fft::r2c(patches, patches_rfft, noa::fft::Norm::FORWARD);
            noa::ewise_unary(patches_rfft, patches_rfft_ps, noa::abs_squared_t{});

            // Fourier crop to fitting range and store.
            cropped_patches.slices.emplace_back(total, total + n_patches_per_slice);
            const auto slice_cropped_patches_rfft_ps = cropped_patches.patches_from_last_slice();
            noa::fft::resize<fft::H2H>(
                    patches_rfft_ps, patches.shape(),
                    slice_cropped_patches_rfft_ps, cropped_patch_shape.set<0>(n_patches_per_slice));

            ++index;
            total += n_patches_per_slice;
            qn::Logger::debug("index={:>02.2f}, patches={:>03}, total={:>05}",
                              slice_metadata.angles[1], n_patches_per_slice, total);
        }

        if (!debug_directory.empty()) {
            noa::io::save(noa::ewise_unary(cropped_patches.rfft_ps, noa::abs_one_log_t{}),
                          debug_directory / "patches_ps.mrc");
        }
        return cropped_patches;
    }

    void CTF::fit_ctf_to_patches_(
            MetadataStack& metadata,
            const Shape2<i64>& slice_shape,
            const Patches& patches_rfft_ps,
            const FittingRange& fitting_range,
            CTFIsotropic64& average_ctf,
            Vec3<bool> fit_angles,
            bool fit_phase_shift,
            bool fit_astigmatism,
            f64 initial_astigmatism_value,
            f64 initial_astigmatism_angle,
            const Path& debug_directory
    ) {
        // Just make sure the spacing in the ctf object is the same as the patches' spacing.
        average_ctf.set_pixel_size(fitting_range.spacing);

        const i64 n_slices = metadata.ssize();
        const i64 n_patches = patches_rfft_ps.rfft_ps.shape()[0];
        const auto options = patches_rfft_ps.rfft_ps.options();
        const auto options_pitched = ArrayOption(options).set_allocator(Allocator::PITCHED);
        const auto options_managed = ArrayOption(options).set_allocator(Allocator::MANAGED);

        // Prepare rotational averages, simulated ctf and background buffer.
        // These are within the fitting range.
        const auto rotational_averages = noa::memory::empty<f32>({n_patches, 1, 1, fitting_range.size}, options_pitched);
        const auto rotational_weights = noa::memory::like(rotational_averages);
        const auto simulated_ctfs = noa::memory::like(rotational_averages);
        auto background_curve = noa::memory::empty<f32>(fitting_range.size);
        apply_cubic_bspline_1d(background_curve.view(), background_curve.view(), fitting_range.background,
                               [](f32, f32 interpolant) { return interpolant; });
        if (options.device().is_gpu())
            background_curve = background_curve.to(options);

        // Prepare per-patch ctfs. This needs to be dereferenceable.
        // Whether the astigmatism is fitted or not, the isotropic ctfs are needed.
        const auto ctfs_isotropic = noa::memory::empty<CTFIsotropic64>(n_patches, options_managed);
        for (auto& ctf: ctfs_isotropic.span())
            ctf = average_ctf;

        // Astigmatism. In this case, we need an extra array with the anisotropic ctfs.
        Array<CTFAnisotropic64> ctfs_anisotropic;
        if (fit_astigmatism) {
            ctfs_anisotropic = noa::memory::empty<CTFAnisotropic64>(n_patches, options_managed);
            const auto average_anisotropic_ctf = CTFAnisotropic64(average_ctf);
            for (auto& ctf: ctfs_anisotropic.span())
                ctf = average_anisotropic_ctf;
        }

        // Allocate buffers for the cross-correlation. In total, to compute the gradients efficiently,
        // we need a set of 3 NCCs per patch, and these need to be dereferenceable.
        const auto nccs = noa::memory::empty<f32>({3, 1, 1, n_patches}, options_managed);

        // Initialize the parameters with the current estimates.
        auto parameters = Parameters(
                fit_angles, fit_phase_shift, fit_astigmatism, n_slices,
                average_ctf, initial_astigmatism_value, initial_astigmatism_angle);

        struct OptimizerData {
            Shape2<i64> slice_shape;
            const MetadataStack* metadata{};
            const FittingRange* fitting_range{};
            Parameters* parameters{};

            // Patches and their ctfs
            const Patches* patches{};
            View<CTFIsotropic64> ctfs_isotropic;
            View<CTFAnisotropic64> ctfs_anisotropic;
            View<f32> nccs;

            // 1d spectra
            View<f32> rotational_averages;
            View<f32> rotational_weights;
            View<f32> simulated_ctfs;
            View<f32> background_curve;

            // Others
            std::vector<Vec2<f64>> defoci_and_delta{};
            bool are_rotational_averages_ready{false};

            // Cost function
            using compute_cost_ptr = void (*)(OptimizerData&, i64);
            compute_cost_ptr cost_function{};
        };
        auto optimizer_data = OptimizerData{
                slice_shape,
                &metadata,
                &fitting_range,
                &parameters,
                &patches_rfft_ps,
                ctfs_isotropic.view(),
                ctfs_anisotropic.view(),
                nccs.view(),
                rotational_averages.view(),
                rotational_weights.view(),
                simulated_ctfs.view(),
                background_curve.view(),
                std::vector<Vec2<f64>>(static_cast<size_t>(parameters.n_defocus()))
        };

        const auto cost_function = [](OptimizerData& data, i64 nccs_index) {
            // Retrieve data.
            const auto& metadata = *data.metadata;
            auto& patches = *data.patches;
            auto& fitting_range = *data.fitting_range;
            auto& parameters = *data.parameters;
            const i64 n_slices = patches.n_slices();
            const i64 n_patches = patches.n_patches();

            // Update CTFs with these parameters. The defocus of the patches needs to be updated
            // when the stage angles or the defocus of the slice is changed.
            const Span<f64> defoci = parameters.defoci();
            const auto patches_center_coordinates =
                    Span(patches.center_coordinates.data(), patches.center_coordinates.size());
            for (i64 i = 0; i < n_slices; ++i) {
                const Vec2<f64>& slice_shifts = metadata[i].shifts;
                const Vec3<f64> slice_angles = metadata[i].angles + parameters.angles();
                const f64 slice_defocus = defoci[i];

                const auto ctfs_isotropic_of_that_slice = data.ctfs_isotropic
                        .subregion(noa::indexing::Ellipsis{}, patches.slicing_operator(i))
                        .span();

                update_slice_patches_ctfs(
                        patches_center_coordinates, ctfs_isotropic_of_that_slice,
                        data.slice_shape, slice_shifts, slice_angles, slice_defocus,
                        parameters.phase_shift());
            }

            // Compute the rotation averages, if needed.
            if (!data.are_rotational_averages_ready) {
                if (parameters.has_astigmatism()) {
                    // Update the astigmatic ctfs.
                    const auto anisotropic_span = data.ctfs_anisotropic.span();
                    const auto isotropic_span = data.ctfs_isotropic.span();
                    for (i64 i = 0; i < n_patches; ++i) {
                        anisotropic_span[i].set_defocus(
                                {isotropic_span[i].defocus(),
                                 parameters.astigmatism_value(),
                                 parameters.astigmatism_angle()}
                        );
                    }

                    // Compute the astigmatism-corrected rotational average.
                    noa::geometry::fft::rotational_average_anisotropic<fft::H2H>(
                            patches.rfft_ps, patches.logical_shape().push_front<2>({n_patches, 1}),
                            data.ctfs_anisotropic, data.rotational_averages, data.rotational_weights,
                            fitting_range.fourier_cropped_fftfreq_range().as<f32>(), /*endpoint=*/ true);
                } else {
                    // In the isotropic case, the optimization parameters don't influence this step,
                    // so we just need to compute the rotational average once.
                    noa::geometry::fft::rotational_average<fft::H2H>(
                            patches.rfft_ps, patches.logical_shape().push_front<2>({n_patches, 1}),
                            data.rotational_averages, data.rotational_weights,
                            fitting_range.fourier_cropped_fftfreq_range().as<f32>(), /*endpoint=*/ true);
                    data.are_rotational_averages_ready = true;
                }

                // Subtract background and normalize, in-place.
                noa::ewise_binary(
                        data.rotational_averages, data.background_curve,
                        data.rotational_averages, noa::minus_t{});
                noa::math::normalize_per_batch(
                        data.rotational_averages, data.rotational_averages,
                        NormalizationMode::L2_NORM);
            }

            // Simulate ctf^2 within fitting range.
            noa::signal::fft::ctf_isotropic<fft::H2H>(
                    data.simulated_ctfs, Shape4<i64>{n_patches, 1, 1, data.fitting_range->logical_size},
                    data.ctfs_isotropic, /*ctf_abs=*/ false, /*ctf_square=*/ true,
                    data.fitting_range->fftfreq.as<f32>(), /*endpoint=*/ true);
            noa::math::normalize_per_batch(data.simulated_ctfs, data.simulated_ctfs, NormalizationMode::L2_NORM);

            // Normalized cross-correlation.
            // This is the cost for every patch. We need to take the sum of these to get the total cost
            // for the stack, but let the maximization function do that since it needs the individual nccs
            // for the gradients anyway.
            noa::math::dot(data.rotational_averages, data.simulated_ctfs, data.nccs.subregion(nccs_index));
        };
        optimizer_data.cost_function = cost_function;

        const auto function_to_maximise = [](u32, const f64*, f64* gradients, void* buffer) -> f64 {
            Timer timer;
            timer.start();
            auto& data = *static_cast<OptimizerData*>(buffer);
            auto& parameters = *data.parameters;

            auto cost_function_sum = [](OptimizerData& data, i64 nccs_index = 0) -> f64 {
                data.cost_function(data, nccs_index);
                return static_cast<f64>(noa::math::sum(data.nccs.subregion(nccs_index)));
            };

            // Compute the gradients for the global parameters.
            // Changing one of these parameters affects every patch,
            // so we need to recompute everything every time.
            {
                f64* gradient_globals = gradients;
                for (auto& value: parameters.globals()) {
                    const f64 initial_value = value;
                    const f64 delta = CentralFiniteDifference::delta(initial_value);

                    value = initial_value - delta;
                    const f64 fx_minus_delta = cost_function_sum(data);
                    value = initial_value + delta;
                    const f64 fx_plus_delta = cost_function_sum(data);

                    value = initial_value; // back to original value
                    *(gradient_globals++) = CentralFiniteDifference::get(fx_minus_delta, fx_plus_delta, delta);
                }
            }

            // Compute the cost.
            const f64 cost = cost_function_sum(data, 0);

            // Compute the gradients for the defocus parameters.
            // The defocus of a slice only affects the patches of that slice. Instead of computing
            // the gradients for each defocus one by one, we can apply the delta to every defocus
            // first, then compute the partial costs with and without delta for every slice,
            // and then compute the final costs one by one.
            {
                Span defoci = parameters.defoci();
                gradients += parameters.defocus_index();

                // First, save the current defoci and compute the deltas.
                for (size_t i = 0; i < defoci.size(); ++i)
                    data.defoci_and_delta[i] = {defoci[i], CentralFiniteDifference::delta(defoci[i])};

                // Compute the partial costs with minus delta.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] -= data.defoci_and_delta[i][1];
                data.cost_function(data, 1);

                // Compute the partial costs with plus delta.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] = data.defoci_and_delta[i][0] + data.defoci_and_delta[i][1]; // use original value
                data.cost_function(data, 2);

                // Reset to original values.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] = data.defoci_and_delta[i][0];

                // Compute the gradients.
                const auto fx = data.nccs.subregion(0).span();
                const auto fx_minus_delta = data.nccs.subregion(1).span();
                const auto fx_plus_delta = data.nccs.subregion(2).span();

                for (i64 i = 0; i < data.patches->n_slices(); ++i) {
                    const auto [start, end] = data.patches->range(i);
                    f64 cost_minus_delta{0};
                    f64 cost_plus_delta{0};

                    for (i64 j = 0; j < data.patches->n_patches(); ++j) {
                        // Whether the patch belongs to this slice.
                        if (j >= start && j < end) {
                            // This patch is affected by this defocus.
                            cost_minus_delta += static_cast<f64>(fx_minus_delta[j]);
                            cost_plus_delta += static_cast<f64>(fx_plus_delta[j]);
                        } else {
                            // This patch isn't affected by this defocus,
                            // so the delta would have had no effect on the costs,
                            // so we can use the costs from fx.
                            cost_minus_delta += static_cast<f64>(fx[j]);
                            cost_plus_delta += static_cast<f64>(fx[j]);
                        }
                    }

                    gradients[i] = CentralFiniteDifference::get(
                            cost_minus_delta, cost_plus_delta,
                            data.defoci_and_delta[static_cast<size_t>(i)][1]);
                }
            }

            qn::Logger::debug("cost={:.3f}, elapsed={:.2f}ms", cost, timer.elapsed());
            return cost;
        };

        // Optimizer.
        Optimizer optimizer(NLOPT_LD_LBFGS, parameters.ssize());
        optimizer.set_max_objective(function_to_maximise);

        // Set bounds.
        constexpr auto PI = noa::math::Constant<f64>::PI;
        constexpr auto PI_EPSILON = PI / 32;
        parameters.set_bounds(
                /*rotation=*/ {-5, 5},
                /*tilt=*/ {-15, 15},
                /*elevation=*/ {-5, 5},
                /*phase_shift=*/ {0, PI / 6},
                /*astigmatism_value=*/ {-0.5, 0.5},
                /*astigmatism_angle=*/ {0 - PI_EPSILON, 2 * PI + PI_EPSILON},
                /*defocus=*/ {-0.5, 0.5}
        );
        optimizer.set_bounds(parameters.lower_bounds().data(),
                             parameters.upper_bounds().data());
        optimizer.optimize(parameters.data());

        // Update the metadata.
        // TODO
    }
}
