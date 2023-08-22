#include <noa/unified/Indexing.hpp>

#include "quinoa/core/Metadata.h"
#include "quinoa/core/Utilities.h"
#include "quinoa/core/CTF.hpp"

namespace {
    using namespace ::qn;

    // TODO Make the defocus fitting optional. The user could provide a vector with the defocus of each slice,
    //      and we can wrap and access it from e.g. Parameters::defocus(i64 slice_index)? Not sure it's really
    //      useful, especially given that fitting the defoci is really cheap in resources.

    // The parameters to optimize.
    //  - The parameters are divided in two categories, 1) the global parameters, and 2) the defoci.
    //    The global parameters affect every patch of every slice. These are the 3 stage-angles (rotation, tilt,
    //    elevation), the phase shift and the defocus astigmatism (value and angle). On the other hand, the defocus
    //    (of a given slice) only affects the patches of that slice. This distinction is used to compute the
    //    gradients efficiently.
    //  - The stage-angles are offsets to whatever angles are saved in the metadata.
    //    The other parameters are the actual values of the CTF.
    //  - The parameters are saved contiguously, so we can iterate through them. If a global parameter is not
    //    fitted, it is not included in this continuous buffer.
    class Parameters {
    private:
        // How many slices, i.e., defocus, do we have?
        i64 m_n_defocus;

        Vec<bool, 5> m_fit_global; // whether to fit the rotation, tilt, elevation, phase_shift, astigmatism
        std::array<f64, 3> m_initial_values{}; // phase_shift, astigmatism value, astigmatism angle
        std::array<i64, 6> m_indexes{}; // rotation, tilt, elevation, phase_shift, astigmatism x2

        // Contiguous buffers, where parameters for the optimizer are saved sequentially.
        std::vector<f64> m_buffer;
        std::vector<f64> m_lower_bounds{};
        std::vector<f64> m_upper_bounds{};
        std::vector<f64> m_abs_tolerance{};

    public:
        Parameters(
                const CTFFitter::GlobalFit& fit,
                const MetadataStack& metadata,
                const CTFAnisotropic64& average_ctf
        ) :
                m_n_defocus(metadata.ssize()),
                m_fit_global(fit.rotation, fit.tilt, fit.elevation, fit.phase_shift, fit.astigmatism),
                m_buffer(size())
        {
            // Save some initial values. This is in case they are not fitted, and we need to get the original values.
            // Assume phase shift and astigmatism are the same for every slice.
            m_initial_values[0] = average_ctf.phase_shift();
            m_initial_values[1] = average_ctf.defocus().astigmatism;
            m_initial_values[2] = average_ctf.defocus().angle;

            // Set the indexes.
            i64 count{0};
            for (auto i: irange<size_t>(6)) { // rotation, tilt, elevation, phase_shift, astigmatism x2
                if (m_fit_global[std::min(i, size_t{4})]) { // astigmatism value/angle are both at 4
                    m_indexes[i] = count;
                    count += 1; // be more explicit than "count++"
                }
            }

            // Initialise the continuous parameter array (the angles offsets are 0).
            if (has_phase_shift())
                set(phase_shift_index(), m_initial_values[0]);
            if (has_astigmatism()) {
                set(astigmatism_value_index(), m_initial_values[1]);
                set(astigmatism_angle_index(), m_initial_values[2]);
            }
            auto defocus_span = defoci();
            for (i64 i = 0; i < n_defocus(); ++i)
                defocus_span[i] = metadata[i].defocus;
        }

        // Sets the (low and high) bounds for every parameter.
        // Angles are in radians.
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

        void set_abs_tolerance(
                f64 angle_tolerance,
                f64 phase_shift_tolerance,
                f64 astigmatism_value_tolerance,
                f64 astigmatism_angle_tolerance,
                f64 defocus_tolerance
        ) {
            m_abs_tolerance = std::vector<f64>(size(), 0);
            if (has_rotation())
                m_abs_tolerance[static_cast<size_t>(rotation_index())] = angle_tolerance;
            if (has_tilt())
                m_abs_tolerance[static_cast<size_t>(tilt_index())] = angle_tolerance;
            if (has_elevation())
                m_abs_tolerance[static_cast<size_t>(elevation_index())] = angle_tolerance;
            if (has_phase_shift())
                m_abs_tolerance[static_cast<size_t>(phase_shift_index())] = (phase_shift_tolerance);
            if (has_astigmatism()) {
                m_abs_tolerance[static_cast<size_t>(astigmatism_value_index())] = astigmatism_value_tolerance;
                m_abs_tolerance[static_cast<size_t>(astigmatism_angle_index())] = astigmatism_angle_tolerance;
            }
            for (i64 i = 0; i < n_defocus(); ++i)
                m_abs_tolerance[static_cast<size_t>(defocus_index() + i)] = defocus_tolerance;
        }

        void update(const f64* parameters) {
            std::copy(parameters, parameters + size(), data());
        }

    public:
        [[nodiscard]] constexpr auto n_globals() const noexcept -> i64 { return noa::math::sum(m_fit_global.as<i64>()); }
        [[nodiscard]] constexpr auto n_defocus() const noexcept -> i64 { return m_n_defocus; }
        [[nodiscard]] constexpr auto ssize() const noexcept -> i64 { return n_globals() + n_defocus(); }
        [[nodiscard]] constexpr auto size() const noexcept -> size_t { return static_cast<size_t>(ssize()); }

        void set(i64 index, f64 value) noexcept { m_buffer[static_cast<size_t>(index)] = value; }
        [[nodiscard]] auto get(i64 index) const noexcept -> f64 { return m_buffer[static_cast<size_t>(index)]; }
        [[nodiscard]] auto data() noexcept -> f64* { return m_buffer.data(); }

        [[nodiscard]] constexpr auto has_rotation() const noexcept -> bool { return m_fit_global[0]; }
        [[nodiscard]] constexpr auto has_tilt() const noexcept -> bool { return m_fit_global[1]; }
        [[nodiscard]] constexpr auto has_elevation() const noexcept -> bool { return m_fit_global[2]; }
        [[nodiscard]] constexpr auto has_phase_shift() const noexcept -> bool { return m_fit_global[3]; }
        [[nodiscard]] constexpr auto has_astigmatism() const noexcept -> bool { return m_fit_global[4]; }

        [[nodiscard]] constexpr auto rotation_index() const noexcept -> i64 { return m_indexes[0]; }
        [[nodiscard]] constexpr auto tilt_index() const noexcept -> i64 { return m_indexes[1]; }
        [[nodiscard]] constexpr auto elevation_index() const noexcept -> i64 { return m_indexes[2]; }
        [[nodiscard]] constexpr auto phase_shift_index() const noexcept -> i64 { return m_indexes[3]; }
        [[nodiscard]] constexpr auto astigmatism_value_index() const noexcept -> i64 { return m_indexes[4]; }
        [[nodiscard]] constexpr auto astigmatism_angle_index() const noexcept -> i64 { return m_indexes[5]; }
        [[nodiscard]] constexpr auto defocus_index() const noexcept -> i64 { return n_globals(); }

    public: // access through spans.
        [[nodiscard]] auto globals() noexcept -> Span<f64> { // can be empty
            return {m_buffer.data(), n_globals()};
        }

        [[nodiscard]] auto defoci() noexcept -> Span<f64> {
            return {m_buffer.data() + defocus_index(), n_defocus()};
        }

        [[nodiscard]] auto parameters() noexcept -> Span<f64> {
            return {m_buffer.data(), ssize()};
        }

        [[nodiscard]] auto lower_bounds() noexcept -> Span<f64> {
            return {m_lower_bounds.data(), ssize()};
        }

        [[nodiscard]] auto upper_bounds() noexcept -> Span<f64> {
            return {m_upper_bounds.data(), ssize()};
        }

        [[nodiscard]] auto abs_tolerance() noexcept -> Span<f64> {
            return {m_abs_tolerance.data(), ssize()};
        }

    public: // safe access of the parameters, whether they are fitted.
        [[nodiscard]] auto angle_offsets() const noexcept -> Vec3<f64> {
            // These are offsets, so the initial value is always 0.
            return {
                    has_rotation() ? get(rotation_index()) : 0,
                    has_tilt() ? get(tilt_index()) : 0,
                    has_elevation() ? get(elevation_index()) : 0
            };
        }

        [[nodiscard]] auto phase_shift() const noexcept -> f64 {
            return has_phase_shift() ? get(phase_shift_index()) : m_initial_values[0];
        }

        [[nodiscard]] auto astigmatism_value() const noexcept -> f64 {
            return has_astigmatism() ? get(astigmatism_value_index()) : m_initial_values[1];
        }

        [[nodiscard]] auto astigmatism_angle() const noexcept -> f64 {
            return has_astigmatism() ? get(astigmatism_angle_index()) : m_initial_values[2];
        }
    };

    class CTFGlobalFitter {
    private:
        const MetadataStack* m_metadata{};
        const CTFFitter::FittingRange* m_fitting_range{};
        const CTFFitter::Grid* m_grid{};
        Parameters m_parameters;
        Memoizer m_memoizer; // must be after m_parameters

        // Patches and their ctfs.
        const CTFFitter::Patches* m_patches{};
        Array<CTFIsotropic64> m_ctfs_isotropic;
        Array<CTFAnisotropic64> m_ctfs_anisotropic;
        Array<f32> m_nccs;

        // 1d spectra.
        Array<f32> m_rotational_averages;
        Array<f32> m_rotational_weights;
        Array<f32> m_simulated_ctfs;
        Array<f32> m_background_curve;
        bool m_are_rotational_averages_ready{false};

    public:
        CTFGlobalFitter(
                const MetadataStack& metadata,
                const CTFAnisotropic64& average_anisotropic_ctf,
                const CTFFitter::Patches& patches_rfft_ps,
                const CTFFitter::FittingRange& fitting_range,
                const CTFFitter::Grid& grid,
                const CTFFitter::GlobalFit& fit
        ) :
                m_metadata(&metadata),
                m_fitting_range(&fitting_range),
                m_grid(&grid),
                m_parameters(fit, metadata, average_anisotropic_ctf),
                m_memoizer(/*n_parameters=*/ m_parameters.ssize(), /*resolution=*/ 1),
                m_patches(&patches_rfft_ps)
        {
            // Few checks...
            QN_CHECK(metadata.ssize() == patches_rfft_ps.n_slices() &&
                     metadata.ssize() == m_parameters.n_defocus(),
                     "Metadata mismatch");
            QN_CHECK(noa::math::are_almost_equal(
                     fitting_range.spacing,
                     noa::math::sum(average_anisotropic_ctf.pixel_size()) / 2),
                     "Spacing mismatch");

            const i64 n_patches = patches_rfft_ps.n_patches_per_stack();
            const auto options = patches_rfft_ps.rfft_ps().options();
            const auto options_pitched = ArrayOption(options).set_allocator(Allocator::PITCHED);
            const auto options_managed = ArrayOption(options).set_allocator(Allocator::MANAGED);

            // Prepare for rotational averages.
            //  - These are within the fitting range, so allocate for exactly that.
            //  - Use pitched layout for performance, since accesses are per row.
            //  - The weights are optional here, but don't rely on the device-cache.
            m_rotational_averages = noa::memory::empty<f32>({n_patches, 1, 1, fitting_range.size}, options_pitched);
            m_rotational_weights = noa::memory::like(m_rotational_averages);
            m_simulated_ctfs = noa::memory::like(m_rotational_averages);

            // The background is within the fitting range too. Evaluate once and broadcast early.
            // After this point, the background is on the device and ready to use.
            m_background_curve = noa::memory::empty<f32>(fitting_range.size);
            apply_cubic_bspline_1d(m_background_curve.view(), m_background_curve.view(), fitting_range.background,
                                   [](f32, f32 interpolant) { return interpolant; });
            if (options.device().is_gpu())
                m_background_curve = m_background_curve.to(options);
            m_background_curve = noa::indexing::broadcast(m_background_curve, m_rotational_averages.shape());

            // Prepare per-patch ctfs. This needs to be dereferenceable.
            // In the case of astigmatism, we need an extra array with the anisotropic ctfs.
            m_ctfs_isotropic = noa::memory::empty<CTFIsotropic64>(n_patches, options_managed);
            m_ctfs_anisotropic =
                    m_parameters.has_astigmatism() ?
                    noa::memory::empty<CTFAnisotropic64>(n_patches, options_managed) :
                    Array<CTFAnisotropic64>{};

            // Initialize ctfs.
            // The defocus is going to be overwritten, but we still need to initialize everything else.
            // Regardless of whether the astigmatism is fitted, the isotropic ctfs are needed.
            auto defocus = average_anisotropic_ctf.defocus();
            for (i64 i = 0; i < patches_rfft_ps.n_slices(); ++i) {
                auto slice_anisotropic_ctf = average_anisotropic_ctf;
                slice_anisotropic_ctf.set_defocus({metadata[i].defocus, defocus.astigmatism, defocus.angle});
                const auto slice_isotropic_ctf = CTFIsotropic64(slice_anisotropic_ctf);

                const auto slice_isotropic_ctfs = m_ctfs_isotropic
                        .subregion(noa::indexing::Ellipsis{}, m_patches->chunk(i));
                for (auto& ctf: slice_isotropic_ctfs.span())
                    ctf = slice_isotropic_ctf;

                if (m_parameters.has_astigmatism()) {
                    const auto slice_anisotropic_ctfs = m_ctfs_anisotropic
                            .subregion(noa::indexing::Ellipsis{}, m_patches->chunk(i));
                    for (auto& ctf: slice_anisotropic_ctfs.span())
                        ctf = slice_anisotropic_ctf;
                }
            }

            // Allocate buffers for the cross-correlation. In total, to compute the gradients efficiently,
            // we need a set of 3 NCCs per patch, and these need to be dereferenceable.
            m_nccs = noa::memory::empty<f32>({3, 1, 1, n_patches}, options_managed);
        }

        void update_slice_patches_ctfs_(
                Span<CTFIsotropic64> patches_ctfs,
                const Vec2<f64>& slice_shifts,
                const Vec3<f64>& slice_angles_radians,
                f64 slice_defocus,
                f64 phase_shift
        ) {
            const auto patches_centers = m_grid->patches_centers();
            NOA_ASSERT(patches_centers.ssize() == patches_ctfs.ssize());

            for (i64 i = 0; i < patches_ctfs.ssize(); ++i) {
                CTFIsotropic64& patch_ctf = patches_ctfs[i];

                // Get the z-offset of the patch.
                const auto patch_z_offset_um = m_grid->patch_z_offset(
                        slice_shifts, slice_angles_radians,
                        Vec2<f64>{patch_ctf.pixel_size()},
                        patches_centers[i]);

                // The defocus at the patch center is simply the slice defocus minus the z offset from the tilt axis.
                // Indeed, the defocus is positive, so a negative z-offset -> below the tilt axis -> further away
                // from the defocus -> has a larger defocus.
                patch_ctf.set_defocus(slice_defocus - patch_z_offset_um);
                patch_ctf.set_phase_shift(phase_shift);
            }
        }

        void cost(i64 nccs_index) {
            const i64 n_slices = m_patches->n_slices();
            const i64 n_patches = m_patches->n_patches_per_stack();
            const auto patches_shape = m_patches->shape().push_front<2>({n_patches, 1});

            {
                // Update CTFs with the current parameters.
                const Vec3<f64> angle_offsets = m_parameters.angle_offsets();
                const f64 phase_shift = m_parameters.phase_shift();
                const Span<f64> defoci = m_parameters.defoci();

                for (i64 i = 0; i < n_slices; ++i) {
                    const MetadataSlice& metadata_slice = (*m_metadata)[i];
                    const Vec2<f64>& slice_shifts = metadata_slice.shifts;
                    const Vec3<f64> slice_angles = noa::math::deg2rad(metadata_slice.angles) + angle_offsets;
                    const f64 slice_defocus = defoci[i];

                    // Fetch the CTFs of the patches belonging to the current slice.
                    const auto slice_patches_ctfs = m_ctfs_isotropic
                            .subregion(noa::indexing::Ellipsis{}, m_patches->chunk(i))
                            .span();

                    update_slice_patches_ctfs_(
                            slice_patches_ctfs, slice_shifts, slice_angles, slice_defocus, phase_shift);
                }
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
                            m_patches->rfft_ps(), patches_shape,
                            m_ctfs_anisotropic, m_rotational_averages, m_rotational_weights,
                            m_fitting_range->fourier_cropped_fftfreq_range().as<f32>(), /*endpoint=*/ true);
                } else {
                    noa::geometry::fft::rotational_average<fft::H2H>(
                            m_patches->rfft_ps(), patches_shape,
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

            // TODO Print per patch?
//            if (qn::Logger::is_debug()) {
//                save_vector_to_text(m_simulated_ctfs.view(), "/home/thomas/Projects/quinoa/tests/ribo_ctf/debug_ctf/simulated_ctfs.txt");
//                save_vector_to_text(m_rotational_averages.view(), "/home/thomas/Projects/quinoa/tests/ribo_ctf/debug_ctf/rotational_averages.txt");
//            }

            // Normalized cross-correlation.
            // This is the cost for every patch. We need to take the sum|average of these to get the total cost
            // for the stack, but let the maximization function do that since it needs the individual nccs
            // for the defocus gradients.
            noa::math::dot(m_rotational_averages, m_simulated_ctfs, m_nccs.subregion(nccs_index));

            // We have to explicitly synchronize here, to make sure the inputs are not modified while this is running.
            m_nccs.eval();
        }

        static auto function_to_maximise(u32, const f64* parameters, f64* gradients, void* buffer) -> f64 {
            Timer timer;
            timer.start();
            auto& self = *static_cast<CTFGlobalFitter*>(buffer);

            // TODO Weighted sum using cos(tilt) to increase weight of the lower tilt (where we have more signal)?
            //      Or use exposure since we don't know the lower tilt...? The issue with that is that the higher
            //      tilts are contributing a lot to the angle offsets because they are more sensible to it.
            auto cost_mean = [&self](i64 nccs_index = 0) -> f64 {
                self.cost(nccs_index);
                return static_cast<f64>(noa::math::mean(self.nccs().subregion(nccs_index)));
            };

            // The optimizer may pass its own array, so update/memcpy our parameters.
            if (parameters != self.parameters().data())
                self.parameters().update(parameters);

            if (!gradients)
                return cost_mean();

            // Memoization. This is only to skip for when the linear search is stuck.
            std::optional<f64> memoized_cost = self.memoizer().find(self.parameters().data(), gradients, 1e-8);
            if (memoized_cost.has_value()) {
                f64 cost = memoized_cost.value();
                qn::Logger::trace("cost={:.4f}, elapsed={:.2f}ms, memoized=true", cost, timer.elapsed());
                return cost;
            }

            // For the finite central difference method, use a delta that is the 4th of the tolerance for that parameter.
            // This is to be small enough for good accuracy, and large enough to make a significant change on the score.
            const Span<f64> abs_tolerance = self.parameters().abs_tolerance();

            // Compute the gradients for the global parameters.
            // Changing one of these parameters affects every patch,
            // so we need to recompute everything every time.
            {
                f64* gradient_globals = gradients;
                i64 i = 0;
                for (auto& value: self.parameters().globals()) {
                    const f64 initial_value = value;
                    const f64 delta = abs_tolerance[i] / 4;

                    value = initial_value - delta;
                    const f64 fx_minus_delta = cost_mean();
                    value = initial_value + delta;
                    const f64 fx_plus_delta = cost_mean();

                    value = initial_value; // back to original value
                    const f64 gradient = CentralFiniteDifference::get(fx_minus_delta, fx_plus_delta, delta);
                    *(gradient_globals++) = gradient;
                    ++i;
                    qn::Logger::trace("global: g={:.8f}, v={:.8f}", gradient, value);
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
                Span defoci = self.parameters().defoci();
                const auto defocus_index = self.parameters().defocus_index();
                f64* defoci_gradients = gradients + defocus_index;

                // First, save the current defoci and the deltas.
                std::vector<Vec2<f64>> defoci_and_delta;
                defoci_and_delta.reserve(defoci.size());
                for (i64 i = 0; i < defoci.ssize(); ++i)
                    defoci_and_delta.emplace_back(defoci[i], abs_tolerance[i + defocus_index] / 4);

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
                const auto nccs = self.nccs().eval(); // make sure to synchronize
                const auto fx = nccs.subregion(0).span();
                const auto fx_minus_delta = nccs.subregion(1).span();
                const auto fx_plus_delta = nccs.subregion(2).span();

                const auto n_slices = self.patches().n_slices();
                const auto n_patches = self.patches().n_patches_per_stack();
                const auto mean_weight = static_cast<f64>(n_patches);

                for (i64 i = 0; i < n_slices; ++i) {
                    const noa::indexing::Slice chunk = self.patches().chunk(i);
                    f64 cost_minus_delta{0};
                    f64 cost_plus_delta{0};

                    for (i64 j = 0; j < n_patches; ++j) {
                        // Whether the patch belongs to this slice.
                        if (j >= chunk.start && j < chunk.end) {
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
                    const auto gradient = CentralFiniteDifference::get(
                            cost_minus_delta, cost_plus_delta,
                            defoci_and_delta[static_cast<size_t>(i)][1]);

                    defoci_gradients[i] = gradient;
                    qn::Logger::trace("defocus {:>02}: g={:.8f}, v={:.8f}", i, gradient, defoci[i]);
                }
            }

            self.memoizer().record(self.parameters().data(), cost, gradients);
            qn::Logger::trace("cost={:.4f}, elapsed={:.2f}ms", cost, timer.elapsed());
            return cost;
        }

        [[nodiscard]] auto patches() noexcept -> const CTFFitter::Patches& {
            return *m_patches;
        }

        [[nodiscard]] auto nccs() const noexcept -> View<const f32> {
            return m_nccs.view();
        }

        [[nodiscard]] auto parameters() noexcept -> Parameters& {
            return m_parameters;
        }

        [[nodiscard]] auto memoizer() noexcept -> Memoizer& {
            return m_memoizer;
        }
    };
}

namespace qn {
    auto CTFFitter::compute_patches_rfft_ps(
            Device compute_device,
            StackLoader& stack_loader,
            const MetadataStack& metadata,
            const FittingRange& fitting_range,
            const Grid& grid,
            const Path& debug_directory
    ) -> CTFFitter::Patches {
        const auto options = ArrayOption(compute_device, Allocator::DEFAULT_ASYNC);
        const auto slice_patches_shape = grid.patch_shape().push_front<2>({grid.n_patches(), 1});

        // The patches are loaded one slice at a time. So allocate enough for one slice.
        const auto slice = noa::memory::empty<f32>(grid.slice_shape().push_front<2>({1, 1}), options);
        const auto patches_rfft = noa::memory::empty<c32>(slice_patches_shape.rfft(), options);
        const auto patches_rfft_ps = noa::memory::like<f32>(patches_rfft);
        const auto patches = noa::fft::alias_to_real(patches_rfft, slice_patches_shape);

        // Create the big array with all the patches.
        // Use managed memory in case it doesn't fit in device memory.
        const auto options_managed = ArrayOption(options).set_allocator(Allocator::MANAGED);
        auto cropped_patches = Patches(grid, fitting_range, metadata.ssize(), options_managed);

        // Prepare the subregion origins, ready for extract_subregions().
        // The grid divides sets the field-of-view in patches, meaning that the patch origins
        // are the same for every slice.
        const std::vector<Vec4<i32>> subregion_origins = grid.compute_subregion_origins();
        const auto patches_origins = View(subregion_origins.data(), subregion_origins.size()).to(options);

        i64 index{0};
        for (const auto& slice_metadata: metadata.slices()) {
            // Extract the patches. Assume the slice is normalized and edges are tapered.
            stack_loader.read_slice(slice.view(), slice_metadata.index_file);
            noa::memory::extract_subregions(slice, patches, patches_origins);
            noa::math::normalize_per_batch(patches, patches);

            if (!debug_directory.empty()) {
                const auto filename = debug_directory / noa::string::format("patches_{:>02}.mrc", index);
                noa::io::save(patches, filename);
                qn::Logger::debug("{} saved", filename);
            }

            // Compute the power-spectra of these tiles.
            noa::fft::r2c(patches, patches_rfft, noa::fft::Norm::FORWARD);
            noa::ewise_unary(patches_rfft, patches_rfft_ps, noa::abs_squared_t{});
            if (!debug_directory.empty()) {
                const auto filename = debug_directory / "patches_ps_full.mrc";
                noa::io::save(noa::ewise_unary(patches_rfft_ps, noa::abs_one_log_t{}), filename);
                qn::Logger::debug("{} saved", filename);
            }

            // Fourier crop to fitting range and store in the output.
            noa::fft::resize<fft::H2H>(
                    patches_rfft_ps, patches.shape(),
                    cropped_patches.rfft_ps(index), cropped_patches.chunk_shape());

            ++index;
            qn::Logger::trace("index={:>+6.2f}", slice_metadata.angles[1]);
        }

        if (!debug_directory.empty()) {
            const auto filename = debug_directory / "patches_ps.mrc";
            noa::io::save(noa::ewise_unary(cropped_patches.rfft_ps().to_cpu(), noa::abs_one_log_t{}), filename);
            qn::Logger::debug("{} saved", filename);
        }
        return cropped_patches;
    }

    auto CTFFitter::fit_ctf_to_patches(
            MetadataStack& metadata, // updated: .angles, .defocus
            CTFAnisotropic64& anisotropic_ctf, // updated: .phase_shift, .defocus
            const Patches& patches_rfft_ps,
            const FittingRange& fitting_range,
            const Grid& grid,
            const GlobalFit& fit,
            const Path& debug_directory
    ) -> Vec3<f64> {
        auto fitter = CTFGlobalFitter(
                metadata, anisotropic_ctf, patches_rfft_ps, fitting_range, grid, fit);
        auto& parameters = fitter.parameters();

        // Set bounds.
        constexpr auto PI = noa::math::Constant<f64>::PI;
        constexpr auto PI_EPSILON = PI / 32;
        parameters.set_relative_bounds(
                /*rotation=*/ noa::math::deg2rad(Vec2<f64>{-5, 5}),
                /*tilt=*/ noa::math::deg2rad(Vec2<f64>{-15, 15}),
                /*elevation=*/ noa::math::deg2rad(Vec2<f64>{-5, 5}),
                /*phase_shift=*/ {0, PI / 6},
                /*astigmatism_value=*/ {-0.5, 0.5},
                /*astigmatism_angle=*/ {0 - PI_EPSILON, 2 * PI + PI_EPSILON},
                /*defocus=*/ {-0.5, 0.5}
        );
        parameters.set_abs_tolerance(
                /*angle_tolerance=*/ noa::math::deg2rad(0.05),
                /*phase_shift_tolerance=*/ noa::math::deg2rad(0.25),
                /*astigmatism_value_tolerance=*/ 5e-4,
                /*astigmatism_angle_tolerance=*/ noa::math::deg2rad(0.1),
                /*defocus_tolerance=*/ 5e-4
        );

        // Optimizer.
        Optimizer optimizer(NLOPT_LD_LBFGS, parameters.ssize());
        optimizer.set_max_objective(CTFGlobalFitter::function_to_maximise, &fitter);
        optimizer.set_bounds(
                parameters.lower_bounds().data(),
                parameters.upper_bounds().data());
        optimizer.set_x_tolerance_abs(parameters.abs_tolerance().data());
        optimizer.optimize(parameters.data());

        // Update metadata and average ctf.
        f64 average_defocus{0};
        const auto defoci = parameters.defoci();
        for (i64 i = 0; i < metadata.ssize(); ++i) {
            const auto defocus = defoci[i];
            average_defocus += defocus;
            metadata[i].defocus = defocus;
        }
        average_defocus /= static_cast<f64>(metadata.ssize());

        anisotropic_ctf.set_phase_shift(parameters.phase_shift());
        anisotropic_ctf.set_defocus({average_defocus, parameters.astigmatism_value(), parameters.astigmatism_angle()});

        const auto angle_offsets = noa::math::rad2deg(parameters.angle_offsets());
        metadata.add_global_angles(angle_offsets);
        return angle_offsets;
    }
}
