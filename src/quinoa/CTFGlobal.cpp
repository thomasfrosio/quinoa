#include <noa/Array.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/CTF.hpp"

namespace {
    using namespace ::qn;
    using namespace ::qn::ctf;

    /// Reduction operator computing the normalized cross-correlation between power spectra and simulated CTFs.
    struct CrossCorrelate {
    private:
        SpanContiguous<const f32, 3> m_power_spectrum;
        SpanContiguous<const f32, 2> m_background;
        SpanContiguous<const ns::CTFAnisotropic<f64>, 1> m_ctfs;

        Shape<i32, 1> m_size;
        Vec<f32, 2> m_fftfreq_bound_sqd;
        f32 m_fftfreq_step;

    public:
        constexpr CrossCorrelate(
            const SpanContiguous<const f32, 3>& power_spectrum,
            const SpanContiguous<const f32, 2>& background,
            const SpanContiguous<const ns::CTFAnisotropic<f64>, 1>& ctfs,
            Vec<f64, 2> fftfreq_range
        ) :
            m_power_spectrum{power_spectrum},
            m_background{background},
            m_ctfs{ctfs},
            m_size{static_cast<i32>(power_spectrum.shape()[1])},
            m_fftfreq_bound_sqd{(fftfreq_range * fftfreq_range).as<f32>()},
            m_fftfreq_step{static_cast<f32>(fftfreq_range[1] / static_cast<f64>(power_spectrum.shape().width() - 1))}
        {}

        constexpr void init(i32 b, i32 i, i32 j, f32& cc, f32& cc_lhs, f32& cc_rhs) const {
            const auto frequency = nf::index2frequency<false, true>(Vec{i, j}, m_size); // rfft, non-centered
            const auto fftfreq_2d = frequency.as<f32>() * m_fftfreq_step;

            // Impose a frequency range.
            const auto fftfreq_sqd = noa::dot(fftfreq_2d, fftfreq_2d);
            if (fftfreq_sqd < m_fftfreq_bound_sqd[0] or fftfreq_sqd > m_fftfreq_bound_sqd[1])
                return;

            // 1. Get the experimental power spectrum.
            const auto lhs = m_power_spectrum(b, i, j) - m_background(i, j);

            // 2. Get the simulated ctf.
            auto rhs = static_cast<f32>(m_ctfs[b].value_at(fftfreq_2d));
            rhs *= rhs; // ctf^2

            // 3. Cross-correlate.
            // To compute the normalized cross-correlation score, we usually L2-normalize the inputs.
            // Here the inputs are generated on-the-fly to save memory, so use the auto cross-correlation instead.
            cc += lhs * rhs;
            cc_lhs += lhs * lhs;
            cc_rhs += rhs * rhs;
        }

        static constexpr void join(
            f32 icc, f32 icc_lhs, f32 icc_rhs,
            f32& cc, f32& cc_lhs, f32& cc_rhs
        ) {
            cc += icc;
            cc_lhs += icc_lhs;
            cc_rhs += icc_rhs;
        }

        // Normalize using autocorrelation.
        using remove_default_final = bool;
        static constexpr void final(f32 cc, f32 cc_lhs, f32 cc_rhs, f32& ncc) {
            const auto energy = noa::sqrt(cc_lhs) * noa::sqrt(cc_rhs);
            ncc = cc / energy;
        }
    };

    /// The parameters to optimize.
    ///
    /// \note The parameters are divided in two categories: the global parameters, and the defoci.
    ///       The global parameters affect every patch of every slice:
    ///         - 3 stage-angle offsets (rotation, tilt, pitch).
    ///         - The phase shift and the defocus astigmatism (value and angle).
    ///           These can be time-resolved, with up to 3 control points.
    ///       On the other hand, the defocus (of a given slice) only affects the patches of that slice.
    ///       This distinction is used to compute the gradients efficiently.
    ///
    /// \note The parameters are saved contiguously, so we can iterate through them.
    ///       If a global parameter is not fitted, it is not included in this continuous buffer passed
    ///       to the optimizer, but a default value can still be provided.
    class Parameters {
    private:
        enum Index : size_t {
            ROTATION = 0,
            TILT,
            PITCH,
            PHASE_SHIFT,
            ASTIGMATISM_VALUE,
            ASTIGMATISM_ANGLE,
            DEFOCUS,
        };
        struct Parameter {
            i32 resolution{};
            i32 index{};
            bool fit{};
        };
        std::array<Parameter, 7> m_parameters;

        // Keep track of the initial/default values in case we don't fit them.
        // Maximum time resolution is 3.
        std::array<f64, 3> m_initial_phase_shift{};
        std::array<f64, 6> m_initial_astigmatism{};

        // Contiguous buffers, where parameters for the optimizer are saved sequentially.
        std::vector<f64> m_buffer{};
        std::vector<f64> m_lower_bounds{};
        std::vector<f64> m_upper_bounds{};
        std::vector<f64> m_abs_tolerance{};

    public:
        Parameters(
            const MetadataStack& metadata,
            const CubicSplineGrid<f64, 1>& initial_phase_shift,
            const CubicSplineGrid<f64, 1>& initial_astigmatism,
            const FitRefineOptions& options
        ) {
            const auto phase_shift_resolution = static_cast<i32>(initial_phase_shift.resolution()[0]);
            const auto astigmatism_resolution = static_cast<i32>(initial_astigmatism.resolution()[0]);
            m_parameters = {
                Parameter{.resolution = 1,                                  .fit = options.fit_rotation},
                Parameter{.resolution = 1,                                  .fit = options.fit_tilt},
                Parameter{.resolution = 1,                                  .fit = options.fit_pitch},
                Parameter{.resolution = phase_shift_resolution,             .fit = options.fit_phase_shift},
                Parameter{.resolution = astigmatism_resolution,             .fit = options.fit_astigmatism}, // value
                Parameter{.resolution = astigmatism_resolution,             .fit = options.fit_astigmatism}, // angle
                Parameter{.resolution = static_cast<i32>(metadata.ssize()), .fit = true},
            };

            i32 index{};
            for (auto& parameter: m_parameters) {
                if (parameter.fit) {
                    parameter.index = index;
                    index += parameter.resolution;
                }
            }
            m_buffer.resize(static_cast<size_t>(index), 0.);

            // Set the parameters.
            // If it is fitted, this sets the buffer. Otherwise, it sets the initial/default values.
            phase_shift().update_channel(0, initial_phase_shift);
            astigmatism().update_channel(0, initial_astigmatism);
            astigmatism().update_channel(1, initial_astigmatism);
            for (auto&& [defocus, slice]: noa::zip(defoci(), metadata))
                defocus = slice.defocus.value;
        }

        template<typename T>
        struct SetOptions {
            T rotation;             // radians
            T tilt;                 // radians
            T pitch;                // radians
            T phase_shift;          // radians
            T defocus;              // um
            T astigmatism_value;    // um
            T astigmatism_angle;    // radians
        };

        // Sets the (low and high) bounds for every parameter.
        void set_relative_bounds(const SetOptions<Vec<f64, 2>>& relative_bounds) {
            m_lower_bounds.resize(size(), 0.);
            m_upper_bounds.resize(size(), 0.);

            const auto set_buffer = [&](const Parameter& parameter, const Vec<f64, 2>& low_and_high_bounds) {
                if (not parameter.fit)
                    return;
                for (i32 i{}; i < parameter.resolution; ++i) {
                    const auto index = static_cast<size_t>(parameter.index + i);
                    const auto value = m_buffer[index];
                    m_lower_bounds[index] = value + low_and_high_bounds[0];
                    m_upper_bounds[index] = value + low_and_high_bounds[1];
                }
            };

            set_buffer(m_parameters[ROTATION], relative_bounds.rotation);
            set_buffer(m_parameters[TILT], relative_bounds.tilt);
            set_buffer(m_parameters[PITCH], relative_bounds.pitch);
            set_buffer(m_parameters[PHASE_SHIFT], relative_bounds.phase_shift);
            set_buffer(m_parameters[ASTIGMATISM_VALUE], relative_bounds.astigmatism_value);
            set_buffer(m_parameters[ASTIGMATISM_ANGLE], relative_bounds.astigmatism_angle);
            set_buffer(m_parameters[DEFOCUS], relative_bounds.defocus);
        }

        void set_abs_tolerance(const SetOptions<f64>& abs_tolerance) {
            m_abs_tolerance.resize(size(), 0.);

            const auto set_buffer = [&](const Parameter& parameter, const f64& tolerance) {
                if (not parameter.fit)
                    return;
                for (i32 i{}; i < parameter.resolution; ++i) {
                    const auto index = static_cast<size_t>(parameter.index + i);
                    const auto value = m_buffer[index];
                    m_abs_tolerance[index] = value + tolerance;
                }
            };

            set_buffer(m_parameters[ROTATION], abs_tolerance.rotation);
            set_buffer(m_parameters[TILT], abs_tolerance.tilt);
            set_buffer(m_parameters[PITCH], abs_tolerance.pitch);
            set_buffer(m_parameters[PHASE_SHIFT], abs_tolerance.phase_shift);
            set_buffer(m_parameters[ASTIGMATISM_VALUE], abs_tolerance.astigmatism_value);
            set_buffer(m_parameters[ASTIGMATISM_ANGLE], abs_tolerance.astigmatism_angle);
            set_buffer(m_parameters[DEFOCUS], abs_tolerance.defocus);
        }

        void update(const f64* parameters) {
            std::copy_n(parameters, size(), data());
        }

    public:
        [[nodiscard]] constexpr auto n_globals() const noexcept -> i64 { return m_parameters[DEFOCUS].index; }
        [[nodiscard]] constexpr auto n_defocus() const noexcept -> i64 { return m_parameters[DEFOCUS].resolution; }
        [[nodiscard]] constexpr auto ssize() const noexcept -> i64 { return n_globals() + n_defocus(); }
        [[nodiscard]] constexpr auto size() const noexcept -> size_t { return static_cast<size_t>(ssize()); }

        [[nodiscard]] auto data() noexcept -> f64* { return m_buffer.data(); }

        [[nodiscard]] constexpr auto has_rotation() const noexcept -> bool {
            return m_parameters[ROTATION].fit;
        }
        [[nodiscard]] constexpr auto has_tilt() const noexcept -> bool {
            return m_parameters[TILT].fit;
        }
        [[nodiscard]] constexpr auto has_pitch() const noexcept -> bool {
            return m_parameters[PITCH].fit;
        }
        [[nodiscard]] constexpr auto has_phase_shift() const noexcept -> bool {
            return m_parameters[PHASE_SHIFT].fit;
        }
        [[nodiscard]] constexpr auto has_astigmatism() const noexcept -> bool {
            return m_parameters[ASTIGMATISM_VALUE].fit;
        }

    public:
        [[nodiscard]] auto globals() noexcept -> SpanContiguous<f64> {
            return SpanContiguous{m_buffer.data(), n_globals()};
        }

        [[nodiscard]] auto angle_offsets() const noexcept -> Vec<f64, 3> {
            return Vec{
                has_rotation() ? m_buffer[static_cast<size_t>(m_parameters[ROTATION].index)] : 0,
                has_tilt()     ? m_buffer[static_cast<size_t>(m_parameters[TILT].index)] : 0,
                has_pitch()    ? m_buffer[static_cast<size_t>(m_parameters[PITCH].index)] : 0
            };
        }

        [[nodiscard]] auto defoci() noexcept -> SpanContiguous<f64> {
            return SpanContiguous{m_buffer.data() + n_globals(), n_defocus()};
        }

        [[nodiscard]] auto phase_shift() noexcept -> CubicSplineGrid<f64, 1> {
            auto ptr = has_phase_shift() ?
                m_buffer.data() + m_parameters[PHASE_SHIFT].index :
                m_initial_phase_shift.data();
            auto data = SpanContiguous(ptr, Shape{1, m_parameters[PHASE_SHIFT].resolution}.as<i64>());
            return CubicSplineGrid<f64, 1>(m_parameters[PHASE_SHIFT].resolution, 1, data);
        }

        [[nodiscard]] auto astigmatism() noexcept -> CubicSplineGrid<f64, 1> {
            auto ptr = has_astigmatism() ?
                m_buffer.data() + m_parameters[ASTIGMATISM_VALUE].index :
                m_initial_astigmatism.data();
            auto data = SpanContiguous(ptr, Shape{2, m_parameters[ASTIGMATISM_VALUE].resolution}.as<i64>());
            return CubicSplineGrid<f64, 1>(m_parameters[ASTIGMATISM_VALUE].resolution, 2, data);
        }

        [[nodiscard]] auto lower_bounds() noexcept -> SpanContiguous<f64> {
            return {m_lower_bounds.data(), ssize()};
        }

        [[nodiscard]] auto upper_bounds() noexcept -> SpanContiguous<f64> {
            return {m_upper_bounds.data(), ssize()};
        }

        [[nodiscard]] auto abs_tolerance() noexcept -> SpanContiguous<f64> {
            return {m_abs_tolerance.data(), ssize()};
        }
    };

    class Fitter {
    private:
        const MetadataStack* m_metadata{};
        const Grid* m_grid{};
        const Patches* m_patches{};

        Array<f32> m_background;
        Array<ns::CTFAnisotropic<f64>> m_ctfs;
        Array<f32> m_nccs;

        Parameters m_parameters;
        Memoizer m_memoizer;
        Vec<f64, 2> m_fftfreq_range;
        f64 m_spacing;

    public:
        Fitter(
            const MetadataStack& metadata,
            const Grid& grid,
            const ns::CTFAnisotropic<f64>& ctf,
            const Patches& patches,
            const Vec<f64, 2>& fftfreq_range,
            const Background& background,
            const CubicSplineGrid<f64, 1>& phase_shift,
            const CubicSplineGrid<f64, 1>& astigmatism,
            const FitRefineOptions& fitting_options
        ) :
            m_metadata{&metadata},
            m_grid{&grid},
            m_patches{&patches},
            m_parameters(metadata, phase_shift, astigmatism, fitting_options),
            m_memoizer(m_parameters.ssize(), 1),
            m_fftfreq_range{fftfreq_range},
            m_spacing{noa::mean(ctf.pixel_size())}
        {
            check(metadata.ssize() == patches.n_slices() and
                  metadata.ssize() == m_parameters.n_defocus());

            const i64 n_patches = patches.n_patches_per_stack();
            const auto options = ArrayOption{
                .device = patches.rfft_ps().device(),
                .allocator = Allocator::MANAGED
            };
            m_background = Array<f32>(patches.rfft_ps().shape().set<0>(1), options);
            background.sample(m_background.view(), m_fftfreq_range);

            // Prepare per-patch CTFs. This needs to be dereferenceable.
            // The anisotropic defocus and phase shift are overwritten, but the rest should be set correctly.
            m_ctfs = Array<ns::CTFAnisotropic<f64>>(n_patches, options);
            for (auto& ictf: m_ctfs.span_1d_contiguous())
                ictf = ctf;

            // Allocate buffers for the cross-correlation. In total, to compute the gradients efficiently,
            // we need a set of 3 NCCs per patch, and these need to be dereferenceable.
            m_nccs = Array<f32>({3, 1, 1, n_patches}, options);
        }

        void update_background(
            Background& background,
            const View<f32>& rotational_average,
            ns::CTFIsotropic<f64>& average_ctf
        ) {
            update_ctfs();
            const i64 n_patches = m_patches->n_patches_per_stack();
            const auto logical_size = m_patches->shape().height();
            const auto spectrum_size = logical_size / 2 + 1;
            const auto spectrum_range = noa::Linspace{m_fftfreq_range[0], m_fftfreq_range[1]};

            // Compute the 1d spectra.
            auto buffer = noa::zeros<f32>({2, n_patches, 1, spectrum_size}, {
                .device = m_patches->rfft_ps().device(),
                .allocator = Allocator::ASYNC,
            });
            auto rotational_averages = buffer.view().subregion(0).permute({1, 0, 2, 3});
            auto rotational_average_weights = buffer.view().subregion(1).permute({1, 0, 2, 3});
            ng::rotational_average_anisotropic<"h2h">(
                m_patches->rfft_ps(), {n_patches, i64{1}, logical_size, logical_size}, m_ctfs,
                rotational_averages, rotational_average_weights, {
                    .input_fftfreq = {0, m_fftfreq_range[1]},
                    .output_fftfreq = spectrum_range,
                    .add_to_output = true,
                });

            // Compute the isotropic CTFs and their average.
            f64 average_defocus{};
            f64 average_phase_shift{};
            auto isotropic_ctfs = noa::like<ns::CTFIsotropic<f64>>(m_ctfs);
            for (auto&& [anisotropic_ctf, isotropic_ctf]:
                noa::zip(m_ctfs.span_1d_contiguous(), isotropic_ctfs.span_1d_contiguous()))
            {
                isotropic_ctf = ns::CTFIsotropic(anisotropic_ctf);
                average_defocus += isotropic_ctf.defocus();
                average_phase_shift += isotropic_ctf.phase_shift();
            }
            average_defocus /= static_cast<f64>(n_patches);
            average_phase_shift /= static_cast<f64>(n_patches);
            average_ctf.set_defocus(average_defocus);
            average_ctf.set_phase_shift(average_phase_shift);

            // Fuse the 1d spectra by their CTF phases.
            ng::fuse_rotational_averages(
                rotational_averages, spectrum_range, isotropic_ctfs.view(),
                rotational_average, spectrum_range, average_ctf
            );

            // Fit the background/envelope using the CTF zeros/peaks.
            background.fit_1d(rotational_average, m_fftfreq_range, average_ctf);
            background.sample(m_background.view(), m_fftfreq_range);
            m_memoizer.reset_cache(); // background has changed
        }

        // Read the current parameters and update the CTF of each patch accordingly.
        // Only the defocus, and optionally the astigmatism and phase shift, are updated.
        // IMPORTANT: if the astigmatism or phase shift are time resolved,
        //            we do expect the metadata to be sorted by order/exposure.
        void update_ctfs() {
            const i64 n_slices = m_patches->n_slices();
            const Vec<f64, 3> angle_offsets = m_parameters.angle_offsets();
            const SpanContiguous<f64> defoci = m_parameters.defoci();

            const CubicSplineGrid<f64, 1> time_resolved_phase_shift = m_parameters.phase_shift();
            const CubicSplineGrid<f64, 1> time_resolved_astigmatism = m_parameters.astigmatism();

            for (i64 i{}; i < n_slices; ++i) {
                // Get the slice angles for this current iteration.
                const MetadataSlice& slice = (*m_metadata)[i];
                const Vec<f64, 3> slice_angles = noa::deg2rad(slice.angles) + angle_offsets;

                // Get the astigmatism and phase shift for this slice,
                // assuming metadata is sorted by exposure.
                const auto time = static_cast<f64>(i) / static_cast<f64>(std::max(n_slices, i64{1}) - 1);
                const f64 phase_shift = time_resolved_phase_shift.interpolate_at(time);
                const auto slice_astigmatism_value = time_resolved_astigmatism.interpolate_at(time, 0);
                const auto slice_astigmatism_angle = time_resolved_astigmatism.interpolate_at(time, 1);

                // Fetch the CTFs of the patches belonging to the current slice.
                const auto ctfs = m_ctfs
                    .subregion(ni::Ellipsis{}, m_patches->chunk_slice(i))
                    .span_1d_contiguous();
                const auto patches_centers = m_grid->patches_centers();
                NOA_ASSERT(patches_centers.ssize() == ctfs.ssize());

                // Update the isotropic CTF of every patch.
                for (i64 j{}; j < ctfs.ssize(); ++j) {
                    // Get the z-offset of the patch.
                    const auto patch_z_offset_um = m_grid->patch_z_offset(
                        slice.shifts, slice_angles, ctfs[j].pixel_size(), patches_centers[j]);

                    // The defocus at the patch center is simply the slice defocus minus
                    // the z-offset from the tilt axis. Indeed, the defocus is positive,
                    // so a negative z-offset means we are below the tilt axis, thus further
                    // away from the defocus, thus have a larger defocus.
                    ctfs[j].set_phase_shift(phase_shift);
                    ctfs[j].set_defocus({
                        .value = defoci[i] - patch_z_offset_um,
                        .astigmatism = slice_astigmatism_value,
                        .angle = slice_astigmatism_angle,
                    });
                }
            }
        }

        void cost(i64 nccs_index) {
            // Update the CTFs of every patch according to the current parameters.
            update_ctfs();

            View<const f32> patches = m_patches->rfft_ps();
            noa::reduce_axes_iwise(
                patches.shape().filter(0, 2, 3), patches.device(),
                noa::wrap(0.f, 0.f, 0.f), m_nccs.subregion(nccs_index),
                CrossCorrelate(
                    patches.span<const f32>().filter(0, 2, 3).as_contiguous(),
                    m_background.span<const f32>().filter(2, 3).as_contiguous(),
                    m_ctfs.span_1d_contiguous(),
                    m_fftfreq_range
                ));

            // We have to explicitly synchronize here,
            // to make sure the inputs are not modified while this is running.
            m_nccs.eval();
        }

        static auto function_to_maximise(u32, const f64* parameters, f64* gradients, void* buffer) -> f64 {
            noa::Timer timer;
            timer.start();
            auto& self = *static_cast<Fitter*>(buffer);

            // TODO Weighted sum using cos(tilt) to increase weight of the lower tilt (where we have more signal)?
            //      Or use exposure since we don't know the lower tilt...? The issue with that is that the higher
            //      tilts are contributing a lot to the angle offsets because they are more sensible to it.
            auto cost_mean = [&self](i64 nccs_index = 0) -> f64 {
                self.cost(nccs_index);
                return noa::mean(self.nccs().subregion(nccs_index));
            };

            // The optimizer may pass its own array, so update/memcpy our parameters.
            if (parameters != self.parameters().data())
                self.parameters().update(parameters);

            if (not gradients)
                return cost_mean();

            // Memoization. This is only to skip for when the linear search within L-BFGS is stuck.
            std::optional<f64> memoized_cost = self.memoizer().find(self.parameters().data(), gradients, 1e-8);
            if (memoized_cost.has_value()) {
                f64 cost = memoized_cost.value();
                Logger::trace("cost={:.4f}, elapsed={:.2f}ms, memoized=true", cost, timer.elapsed().count());
                return cost;
            }

            // For the finite central difference method, use as delta tolerance/4 for that parameter.
            // This is to be small enough for good accuracy, and large enough to make a significant change on the score.
            const Span<f64> abs_tolerance = self.parameters().abs_tolerance();

            // Compute the gradients for the global parameters.
            // Changing one of these parameters affects every patch, so we need to recompute everything for
            // each parameter. With just the stage angles, this is 2*3=6 evaluations. With astigmatism with
            // 3 control points, this is 2*(3+3+3)=18 evaluations. Adding phase-shift at 3 control points,
            // this goes up to 2*(3+3+3+3)=24 evaluations.
            {
                f64* gradient_globals = gradients;
                i64 i = 0;
                for (auto& value: self.parameters().globals()) {
                    const f64 initial_value = value;
                    const f64 delta = noa::deg2rad(0.01);//abs_tolerance[i] / 4; // FIXME

                    value = initial_value - delta;
                    const f64 fx_minus_delta = cost_mean();
                    value = initial_value + delta;
                    const f64 fx_plus_delta = cost_mean();

                    value = initial_value; // back to original value
                    const f64 gradient = CentralFiniteDifference::get(fx_minus_delta, fx_plus_delta, delta);
                    *(gradient_globals++) = gradient;
                    ++i;
                    Logger::trace("global: g={:.8f}, v={:.8f}", gradient, value);
                }
            }

            // Compute the cost.
            const f64 cost = cost_mean(0);

            // Compute the gradients for the defocus parameters. The defocus of a slice only affects
            // the patches of that slice, so instead of computing the gradients for each defocus one by one:
            // 1) apply the delta to every defocus first,
            // 2) then compute the partial costs with and without delta for every slice, and
            // 3) then compute the final costs one by one.
            // As a result, to compute the gradient for every defocus, we only have to evaluate the cost function twice.
            {
                SpanContiguous<f64, 1> defoci = self.parameters().defoci();
                const auto offset = self.parameters().n_globals();
                f64* defoci_gradients = gradients + offset;

                // First, save the current defoci and the deltas.
                std::vector<Pair<f64, f64>> defoci_and_delta;
                defoci_and_delta.reserve(defoci.size());
                for (i64 i = 0; i < defoci.ssize(); ++i)
                    defoci_and_delta.push_back(Pair{defoci[i], abs_tolerance[i + offset] / 4});

                // Compute the partial costs with minus delta.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] -= defoci_and_delta[i].second;
                self.cost(1);

                // Compute the partial costs with plus delta.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] = defoci_and_delta[i].first + defoci_and_delta[i].second;
                self.cost(2);

                // Reset to original values.
                for (size_t i = 0; i < defoci.size(); ++i)
                    defoci[i] = defoci_and_delta[i].first;

                // Compute the gradients.
                const auto nccs = self.nccs().eval(); // make sure to synchronize
                const auto fx = nccs.subregion(0).span_1d_contiguous();
                const auto fx_minus_delta = nccs.subregion(1).span_1d_contiguous();
                const auto fx_plus_delta = nccs.subregion(2).span_1d_contiguous();

                const auto n_slices = self.patches().n_slices();
                const auto n_patches = self.patches().n_patches_per_stack();
                const auto mean_weight = static_cast<f64>(n_patches);

                for (i64 i{}; i < n_slices; ++i) {
                    const ni::Slice chunk = self.patches().chunk_slice(i);
                    f64 cost_minus_delta{0};
                    f64 cost_plus_delta{0};

                    for (i64 j{}; j < n_patches; ++j) {
                        // Whether the patch belongs to this slice.
                        if (j >= chunk.start and j < chunk.end) {
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
                        defoci_and_delta[static_cast<size_t>(i)].second);

                    defoci_gradients[i] = gradient;
                    Logger::trace("defocus {:>02}: g={:.8f}, v={:.8f}", i, gradient, defoci[i]);
                }
            }

            self.memoizer().record(self.parameters().data(), cost, gradients);
            Logger::trace("cost={:.4f}, elapsed={:.2f}ms", cost, timer.elapsed().count());
            return cost;
        }

        [[nodiscard]] auto patches() const noexcept -> const Patches& { return *m_patches; }
        [[nodiscard]] auto nccs() const noexcept -> View<const f32> { return m_nccs.view(); }
        [[nodiscard]] auto parameters() noexcept -> Parameters& { return m_parameters; }
        [[nodiscard]] auto memoizer() noexcept -> Memoizer& { return m_memoizer; }
        [[nodiscard]] auto background() const noexcept -> const CubicSplineGrid<f64, 1>& { return m_background; }
    };
}

namespace qn::ctf {
    void refine_fit_patches_ps(
        MetadataStack& metadata,
        const Grid& grid,
        const Patches& patches_rfft_ps,
        const Vec<f64, 2>& fftfreq_range,
        ns::CTFIsotropic<f64>& isotropic_ctf,
        Background& background,
        CubicSplineGrid<f64, 1>& phase_shift,
        CubicSplineGrid<f64, 1>& astigmatism,
        const FitRefineOptions& options,
        const View<f32>& rotational_average
    ) {
        // Important for time-resolution.
        metadata.sort("exposure");

        auto run_optimization = [&](FitRefineOptions fit) {
            auto fitter = Fitter(
                metadata, grid, isotropic_ctf, patches_rfft_ps, fftfreq_range,
                background, phase_shift, astigmatism, fit
            );
            auto& parameters = fitter.parameters();

            // Set bounds.
            constexpr auto PI = noa::Constant<f64>::PI;
            constexpr auto PI_EPSILON = PI / 32;
            parameters.set_relative_bounds({
                .rotation = deg2rad(Vec{-5., 5.}),
                .tilt = deg2rad(Vec{-15., 15.}),
                .pitch = deg2rad(Vec{-5., 5.}),
                .phase_shift = {0., PI / 6},
                .defocus = {-0.5, 0.5},
                .astigmatism_value = {-0.5, 0.5},
                .astigmatism_angle = {0 - PI_EPSILON, 2 * PI + PI_EPSILON},
            });
            parameters.set_abs_tolerance({
                .rotation = noa::deg2rad(0.001),
                .tilt = noa::deg2rad(0.001),
                .pitch = noa::deg2rad(0.001),
                .phase_shift = noa::deg2rad(0.25),
                .defocus = 5e-4,
                .astigmatism_value = 5e-4,
                .astigmatism_angle = noa::deg2rad(0.1),
            });

            // Optimizer.
            auto optimizer = Optimizer(NLOPT_LD_LBFGS, parameters.ssize());
            optimizer.set_max_objective(Fitter::function_to_maximise, &fitter);
            optimizer.set_bounds(
                parameters.lower_bounds().data(),
                parameters.upper_bounds().data());
            optimizer.set_x_tolerance_abs(parameters.abs_tolerance().data());
            optimizer.optimize(parameters.data());

            // Update the background.
            fitter.update_background(background, rotational_average, isotropic_ctf);

            // Update the splines.
            phase_shift.update_channel(0, parameters.phase_shift());
            astigmatism.update_channel(0, parameters.astigmatism());
            astigmatism.update_channel(1, parameters.astigmatism());

            // Update metadata.
            const auto defoci = parameters.defoci();
            const auto norm = 1 / static_cast<f64>(metadata.ssize() - 1);
            const auto angle_offsets = noa::rad2deg(parameters.angle_offsets());
            for (i64 i{}; i < metadata.ssize(); ++i) {
                auto& slice = metadata[i];
                const auto time = static_cast<f64>(i) * norm;
                slice.angles = MetadataSlice::to_angle_range(slice.angles + angle_offsets);
                slice.phase_shift = phase_shift.interpolate_at(time);
                slice.defocus = {
                    .value = defoci[i],
                    .astigmatism = astigmatism.interpolate_at(time, 0),
                    .angle = astigmatism.interpolate_at(time, 1),
                };
            }
        };

        auto timer = Logger::trace_scope_time("Refine CTF fitting");

        // 1. Fit the angles and per-tilt defocus, with the fixed input phase-shift and astigmatism (we have good
        //    estimates from the coarse alignment). This is the most important step, where the defocus goes per-tilt
        //    and the stage angles are fitted. The other iterations below are just refining with the phase-shift and
        //    astigmatism, which shouldn't change the overall solution too much.
        run_optimization({
            .fit_rotation = options.fit_rotation,
            .fit_tilt = options.fit_tilt,
            .fit_pitch = options.fit_pitch,
            .fit_phase_shift = false,
            .fit_astigmatism = false,
        });

        // 2. Same as 1, but fit phase-shift and astigmatism.
        //    Of course, if neither of these parameters is fitted, we are done.
        if (not options.fit_phase_shift and not options.fit_astigmatism)
            return;
        run_optimization(options);

        // 3. Same as 2, but fit a time-resolved (3 control points) phase-shift and astigmatism.
        //    This is only necessary if the input phase-shift and astigmatism are not already time-resolved.
        if ((not options.fit_phase_shift or phase_shift.resolution()[0] == 3) and
            (not options.fit_astigmatism or astigmatism.resolution()[0] == 3))
            return;
        if (options.fit_phase_shift and phase_shift.resolution()[0] == 1) {
            auto input_phase_shift = phase_shift.span()[0][0];
            phase_shift = CubicSplineGrid<f64, 1>(3);
            for (auto& e: phase_shift.span()[0])
                e = input_phase_shift;
        }
        if (options.fit_astigmatism and astigmatism.resolution()[0] == 1) {
            auto input_astigmatism_value = astigmatism.span()[0][0];
            auto input_astigmatism_angle = astigmatism.span()[1][0];
            astigmatism = CubicSplineGrid<f64, 1>(3, 2);
            for (auto& e: astigmatism.span()[0])
                e = input_astigmatism_value;
            for (auto& e: astigmatism.span()[1])
                e = input_astigmatism_angle;
        }
        run_optimization(options);
    }
}
