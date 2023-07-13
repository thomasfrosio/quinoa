#include <noa/unified/Indexing.hpp>

#include "quinoa/core/Metadata.h"
#include "quinoa/core/Utilities.h"
#include "quinoa/core/CTF.hpp"

namespace {
    using namespace ::qn;

    // The parameters to optimize.
    //
    //  - The parameters are divided in two categories, 1) the global parameters, and 2) the defoci.
    //    The global parameters affect every patch of every slice. These are the 3 stage-angles (rotation, tilt,
    //    elevation), the phase shift and the defocus astigmatism (value and angle). On the other hand, the defocus
    //    (of a given slice) only affects the patches of that slice. This distinction is used to compute the
    //    gradients efficiently.
    //
    //  - The stage-angles are offsets to whatever angles are saved in the metadata.
    //    The other parameters are the actual values of the CTF.
    //
    //  - The parameters are saved contiguously, so we can iterate through them. If a global parameter is not
    //    fitted, it is not included in this continuous buffer.
    class Parameters {
    public:
        Parameters(
                Vec3<bool> fit_angles, bool fit_phase_shift, bool fit_astigmatism,
                i64 n_slices, CTFAnisotropic64 average_ctf
        ) :
                m_n_defocus(n_slices),
                m_fit(fit_angles[0], fit_angles[1], fit_angles[2], fit_phase_shift, fit_astigmatism),
                m_buffer(size()
        ) {
            // Save the initial values.
            m_initial_values[0] = average_ctf.phase_shift();
            m_initial_values[1] = average_ctf.defocus().astigmatism;
            m_initial_values[2] = average_ctf.defocus().angle;
            m_initial_values[3] = average_ctf.defocus().value;

            // Set the indexes.
            i64 count{0};
            for (auto i: irange<size_t>(6)) { // rotation, tilt, elevation, phase_shift, astigmatism x2
                if (m_fit[std::min(i, size_t{5})]) // astigmatism value/angle are both at 5
                    m_indexes[i] = count++;
            }

            // Initialise the continuous parameter array (the angles offsets are 0).
            if (has_phase_shift())
                set(phase_shift_index(), m_initial_values[0]);
            if (has_astigmatism()) {
                set(astigmatism_value_index(), m_initial_values[1]);
                set(astigmatism_angle_index(), m_initial_values[2]);
            }
            for (auto& defocus: defoci())
                defocus = m_initial_values[3];
        }

        // Sets the (low and high) bounds for every parameter.
        void set_relative_bounds(
                Vec2<f64> rotation,
                Vec2<f64> tilt,
                Vec2<f64> elevation,
                Vec2<f64> phase_shift,
                Vec2<f64> astigmatism_value,
                Vec2<f64> astigmatism_angle,
                Vec2<f64> defocus
        ) {
            m_lower_bounds.clear();
            m_upper_bounds.clear();
            m_lower_bounds.reserve(size());
            m_upper_bounds.reserve(size());

            const auto push_back = [&](i64 index, const Vec2<f64>& low_and_high_bounds) {
                const auto value = get(index); // relative bounds
                m_lower_bounds.push_back(value + low_and_high_bounds[0]);
                m_upper_bounds.push_back(value + low_and_high_bounds[1]);
            };

            if (has_rotation())
                push_back(rotation_index(), rotation);
            if (has_tilt())
                push_back(tilt_index(), tilt);
            if (has_elevation())
                push_back(elevation_index(), elevation);
            if (has_phase_shift())
                push_back(phase_shift_index(), phase_shift);
            if (has_astigmatism()) {
                push_back(astigmatism_value_index(), astigmatism_value);
                push_back(astigmatism_angle_index(), astigmatism_angle);
            }
            for (i64 i = 0; i < n_defocus(); ++i)
                push_back(defocus_index() + i, defocus);
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

    public: // safe access of the parameters, whether they are fitted.
        [[nodiscard]] Vec3<f64> angle_offsets() const noexcept {
            return {
                    has_rotation() ? get(rotation_index()) : 0,
                    has_tilt() ? get(tilt_index()) : 0,
                    has_elevation() ? get(elevation_index()) : 0
            };
        }

        [[nodiscard]] f64 phase_shift() const noexcept {
            return has_phase_shift() ? get(phase_shift_index()) : m_initial_values[0];
        }

        [[nodiscard]] f64 astigmatism_value() const noexcept {
            return has_astigmatism() ? get(astigmatism_value_index()) : m_initial_values[1];
        }

        [[nodiscard]] f64 astigmatism_angle() const noexcept {
            return has_astigmatism() ? get(astigmatism_angle_index()) : m_initial_values[2];
        }

    private:
        i64 m_n_defocus;
        Vec<bool, 5> m_fit; // rotation, tilt, elevation, phase_shift, astigmatism
        std::array<f64, 3> m_initial_values{}; // phase_shift, defocus, astigmatism value, astigmatism angle
        std::vector<f64> m_buffer;
        std::array<i64, 6> m_indexes{}; // rotation, tilt, elevation, phase_shift, astigmatism x2
        std::vector<f64> m_lower_bounds{};
        std::vector<f64> m_upper_bounds{};
    };

    class CTFGlobalFitter {
    public:
        CTFGlobalFitter(
                MetadataStack& metadata,
                const Shape2<i64>& slice_shape,
                const CTF::Patches& patches_rfft_ps,
                const CTF::FittingRange& fitting_range,
                CTFAnisotropic64& average_anisotropic_ctf,
                Vec3<bool> fit_angles,
                bool fit_phase_shift,
                bool fit_astigmatism
        ) :
                m_slice_shape(slice_shape),
                m_metadata(&metadata),
                m_fitting_range(&fitting_range),
                m_parameters(fit_angles, fit_phase_shift, fit_astigmatism, metadata.ssize(), average_anisotropic_ctf),
                m_patches(&patches_rfft_ps)
        {
            // Few checks...
            QN_CHECK(metadata.ssize() == patches_rfft_ps.n_slices(), "");
            QN_CHECK(noa::math::are_almost_equal(
                     fitting_range.spacing,
                     noa::math::sum(average_anisotropic_ctf.pixel_size()) / 2), "");

            const i64 n_patches = patches_rfft_ps.n_patches();
            const auto options = patches_rfft_ps.rfft_ps.options();
            const auto options_pitched = ArrayOption(options).set_allocator(Allocator::PITCHED);
            const auto options_managed = ArrayOption(options).set_allocator(Allocator::MANAGED);

            // Prepare for rotational averages.
            //  - These are within the fitting range, so allocate for exactly that.
            //  - Use pitched layout for performance, since accesses are per row.
            //  - The weights are optional, but don't rely on the device-cache.
            m_rotational_averages = noa::memory::empty<f32>({n_patches, 1, 1, fitting_range.size}, options_pitched);
            m_rotational_weights = noa::memory::like(m_rotational_averages);
            m_simulated_ctfs = noa::memory::like(m_rotational_averages);

            // The background is within the fitting range too. Evaluate once and broadcast early.
            // After this point, the background is on the device and ready to be subtracted.
            m_background_curve = noa::memory::empty<f32>(fitting_range.size);
            apply_cubic_bspline_1d(m_background_curve.view(), m_background_curve.view(), fitting_range.background,
                                   [](f32, f32 interpolant) { return interpolant; });
            if (options.device().is_gpu())
                m_background_curve = m_background_curve.to(options);
            m_background_curve = noa::indexing::broadcast(m_background_curve, m_rotational_averages.shape());

            // Prepare per-patch ctfs. This needs to be dereferenceable.
            // Regardless of whether the astigmatism is fitted, the isotropic ctfs are needed.
            m_ctfs_isotropic = noa::memory::empty<CTFIsotropic64>(n_patches, options_managed);
            const auto average_isotropic_ctf = CTFIsotropic64(average_anisotropic_ctf);
            for (auto& ctf: m_ctfs_isotropic.span())
                ctf = average_isotropic_ctf;

            // Astigmatism. In this case, we need an extra array with the anisotropic ctfs.
            // The defocus is going to be overwritten, but we still need to initialize everything else.
            if (fit_astigmatism) {
                m_ctfs_anisotropic = noa::memory::empty<CTFAnisotropic64>(n_patches, options_managed);
                for (auto& ctf: m_ctfs_anisotropic.span())
                    ctf = average_anisotropic_ctf;
            }

            // Allocate buffers for the cross-correlation. In total, to compute the gradients efficiently,
            // we need a set of 3 NCCs per patch, and these need to be dereferenceable.
            m_nccs = noa::memory::empty<f32>({3, 1, 1, n_patches}, options_managed);
        }

        void update_slice_patches_ctfs_(
                Span<const Vec2<f32>> patches_centers,
                Span<CTFIsotropic64> patches_ctfs,
                const Vec2<f64>& slice_shifts,
                const Vec3<f64>& slice_angles,
                f64 slice_defocus,
                f64 phase_shift
        ) {
            NOA_ASSERT(patches_centers.ssize() == patches_ctfs.ssize());

            for (i64 i = 0; i < patches_ctfs.ssize(); ++i) {
                CTFIsotropic64& patch_ctf = patches_ctfs[i];

                // Get the 3d position of the patch, in micrometers.
                const auto patch_coordinates = CTF::patch_transformed_coordinate(
                        m_slice_shape, slice_shifts, slice_angles,
                        Vec2<f64>{patch_ctf.pixel_size()},
                        patches_centers[i].as<f64>());

                // The defocus at the patch center is simply the slice defocus plus the z offset from the tilt axis.
                patch_ctf.set_defocus(patch_coordinates[0] + slice_defocus);
                patch_ctf.set_phase_shift(phase_shift);
            }
        }

        void cost(i64 nccs_index) {
            const i64 n_slices = m_patches->n_slices();
            const i64 n_patches = m_patches->n_patches();

            // Update CTFs with the current parameters.
            const Span<f64> defoci = m_parameters.defoci();
            const auto patches_center_coordinates = Span(
                    m_patches->center_coordinates.data(), m_patches->center_coordinates.size());

            for (i64 i = 0; i < n_slices; ++i) {
                const MetadataSlice& metadata_slice = (*m_metadata)[i];
                const Vec2<f64>& slice_shifts = metadata_slice.shifts;
                const Vec3<f64> slice_angles = metadata_slice.angles + m_parameters.angle_offsets();
                const f64 slice_defocus = defoci[i];

                // Fetch the CTFs of the patches belonging to the current slice.
                const auto ctfs_isotropic_of_that_slice = m_ctfs_isotropic
                        .subregion(noa::indexing::Ellipsis{}, m_patches->slicing_operator(i))
                        .span();

                update_slice_patches_ctfs_(
                        patches_center_coordinates, ctfs_isotropic_of_that_slice,
                        slice_shifts, slice_angles, slice_defocus,
                        m_parameters.phase_shift());
            }

            // Compute the rotation averages, if needed.
            if (!m_are_rotational_averages_ready) {
                if (m_parameters.has_astigmatism()) {
                    // Update the astigmatic ctfs.
                    const auto anisotropic_ctf = m_ctfs_anisotropic.span();
                    const auto isotropic_ctf = m_ctfs_isotropic.span();
                    for (i64 i = 0; i < n_patches; ++i) {
                        anisotropic_ctf[i].set_defocus(
                                {isotropic_ctf[i].defocus(),
                                 m_parameters.astigmatism_value(),
                                 m_parameters.astigmatism_angle()}
                        );
                    }

                    // Compute the astigmatism-corrected rotational average.
                    noa::geometry::fft::rotational_average_anisotropic<fft::H2H>(
                            m_patches->rfft_ps, m_patches->logical_shape().push_front<2>({n_patches, 1}),
                            m_ctfs_anisotropic, m_rotational_averages, m_rotational_weights,
                            m_fitting_range->fourier_cropped_fftfreq_range().as<f32>(), /*endpoint=*/ true);
                } else {
                    noa::geometry::fft::rotational_average<fft::H2H>(
                            m_patches->rfft_ps, m_patches->logical_shape().push_front<2>({n_patches, 1}),
                            m_rotational_averages, m_rotational_weights,
                            m_fitting_range->fourier_cropped_fftfreq_range().as<f32>(), /*endpoint=*/ true);

                    // In the isotropic case, the optimization parameters don't influence this step,
                    // so we just need to compute the rotational average once.
                    m_are_rotational_averages_ready = true;
                }

                // Subtract background and normalize, in-place.
                noa::ewise_binary(
                        m_rotational_averages, m_background_curve,
                        m_rotational_averages, noa::minus_t{});
                noa::math::normalize_per_batch(
                        m_rotational_averages, m_rotational_averages,
                        NormalizationMode::L2_NORM);
            }

            // Simulate ctf^2 within fitting range.
            noa::signal::fft::ctf_isotropic<fft::H2H>(
                    m_simulated_ctfs, Shape4<i64>{n_patches, 1, 1, m_fitting_range->logical_size},
                    m_ctfs_isotropic, /*ctf_abs=*/ false, /*ctf_square=*/ true,
                    m_fitting_range->fftfreq.as<f32>(), /*endpoint=*/ true);
            noa::math::normalize_per_batch(m_simulated_ctfs, m_simulated_ctfs, NormalizationMode::L2_NORM);

            // Normalized cross-correlation.
            // This is the cost for every patch. We need to take the sum|average of these to get the total cost
            // for the stack, but let the maximization function do that since it needs the individual nccs
            // for the gradients anyway.
            noa::math::dot(m_rotational_averages, m_simulated_ctfs, m_nccs.subregion(nccs_index));
        }

        [[nodiscard]] const CTF::Patches& patches() noexcept {
            return *m_patches;
        }

        [[nodiscard]] View<const f32> nccs() const noexcept {
            return m_nccs.view();
        }

        [[nodiscard]] Parameters& parameters() noexcept {
            return m_parameters;
        }

        static auto function_to_maximise(u32, const f64*, f64* gradients, void* buffer) -> f64 {
            Timer timer;
            timer.start();
            auto& self = *static_cast<CTFGlobalFitter*>(buffer);
            auto& parameters = self.parameters();

            auto cost_mean = [&self](i64 nccs_index = 0) -> f64 {
                self.cost(nccs_index);
                return static_cast<f64>(noa::math::mean(self.nccs().subregion(nccs_index)));
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
                    const f64 fx_minus_delta = cost_mean();
                    value = initial_value + delta;
                    const f64 fx_plus_delta = cost_mean();

                    value = initial_value; // back to original value
                    *(gradient_globals++) = CentralFiniteDifference::get(fx_minus_delta, fx_plus_delta, delta);
                }
            }

            // Compute the cost.
            const f64 cost = cost_mean(0);

            // Compute the gradients for the defocus parameters.
            // The defocus of a slice only affects the patches of that slice. Instead of computing
            // the gradients for each defocus one by one, we can apply the delta to every defocus first,
            // then compute the partial costs with and without delta for every slice, and then compute
            // the final costs one by one.
            {
                Span defoci = parameters.defoci();
                f64* defoci_gradients = gradients + parameters.defocus_index();

                // First, save the current defoci and compute the deltas.
                std::vector<Vec2<f64>> defoci_and_delta;
                defoci_and_delta.reserve(defoci.size());
                for (f64 i : defoci)
                    defoci_and_delta.emplace_back(i, CentralFiniteDifference::delta(i));

                // Compute the partial costs with minus delta.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] -= defoci_and_delta[i][1];
                self.cost(1);

                // Compute the partial costs with plus delta.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] = defoci_and_delta[i][0] + defoci_and_delta[i][1]; // use original value
                self.cost(2);

                // Reset to original values.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] = defoci_and_delta[i][0];

                // Compute the gradients.
                const auto nccs = self.nccs();
                const auto fx = nccs.subregion(0).span();
                const auto fx_minus_delta = nccs.subregion(1).span();
                const auto fx_plus_delta = nccs.subregion(2).span();
                const CTF::Patches& patches = self.patches();
                const auto mean_weight = static_cast<f64>(patches.n_patches());

                for (i64 i = 0; i < patches.n_slices(); ++i) {
                    const auto [index_start, index_end] = patches.range(i);
                    f64 cost_minus_delta{0};
                    f64 cost_plus_delta{0};

                    for (i64 j = 0; j < patches.n_patches(); ++j) {
                        // Whether the patch belongs to this slice.
                        if (j >= index_start && j < index_end) {
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

                    cost_minus_delta /= mean_weight;
                    cost_plus_delta /= mean_weight;

                    defoci_gradients[i] = CentralFiniteDifference::get(
                            cost_minus_delta, cost_plus_delta,
                            defoci_and_delta[static_cast<size_t>(i)][1]);
                }
            }

            qn::Logger::debug("cost={:.3f}, elapsed={:.2f}ms", cost, timer.elapsed());
            return cost;
        }

    private:
        Shape2<i64> m_slice_shape;
        const MetadataStack* m_metadata{};
        const CTF::FittingRange* m_fitting_range{};
        Parameters m_parameters;

        // Patches and their ctfs.
        const CTF::Patches* m_patches{};
        Array<CTFIsotropic64> m_ctfs_isotropic;
        Array<CTFAnisotropic64> m_ctfs_anisotropic;
        Array<f32> m_nccs;

        // 1d spectra.
        Array<f32> m_rotational_averages;
        Array<f32> m_rotational_weights;
        Array<f32> m_simulated_ctfs;
        Array<f32> m_background_curve;
        bool m_are_rotational_averages_ready{false};
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
            CTFAnisotropic64& ctf_anisotropic,
            Vec3<bool> fit_angles,
            bool fit_phase_shift,
            bool fit_astigmatism,
            const Path& debug_directory
    ) {
        auto fitter = CTFGlobalFitter(
                metadata, slice_shape, patches_rfft_ps, fitting_range, ctf_anisotropic,
                fit_angles, fit_phase_shift, fit_astigmatism);
        auto& parameters = fitter.parameters();

        // Optimizer.
        Optimizer optimizer(NLOPT_LD_LBFGS, parameters.ssize());
        optimizer.set_max_objective(CTFGlobalFitter::function_to_maximise);

        // Set bounds.
        constexpr auto PI = noa::math::Constant<f64>::PI;
        constexpr auto PI_EPSILON = PI / 32;
        parameters.set_relative_bounds(
                /*rotation=*/ {-5, 5},
                /*tilt=*/ {-15, 15},
                /*elevation=*/ {-5, 5},
                /*phase_shift=*/ {0, PI / 6},
                /*astigmatism_value=*/ {-0.5, 0.5},
                /*astigmatism_angle=*/ {0 - PI_EPSILON, 2 * PI + PI_EPSILON},
                /*defocus=*/ {-0.5, 0.5}
        );
        optimizer.set_bounds(
                parameters.lower_bounds().data(),
                parameters.upper_bounds().data());
        optimizer.optimize(parameters.data());

        // Actual average defocus.
        f64 average_defocus{0};
        for (f64 defocus: parameters.defoci())
            average_defocus += defocus;
        average_defocus /= static_cast<f64>(parameters.n_defocus());

        // Update the metadata:
        add_global_angles(metadata, parameters.angle_offsets());
        ctf_anisotropic.set_phase_shift(parameters.phase_shift());
        ctf_anisotropic.set_defocus({average_defocus, parameters.astigmatism_value(), parameters.astigmatism_angle()});
    }
}
