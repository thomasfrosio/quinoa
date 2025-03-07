#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>

#include "quinoa/CTF.hpp"
#include "quinoa/CubicGrid.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/Logger.hpp"

namespace {
    using namespace ::qn;
    using namespace ::qn::ctf;

    // Manages the parameters to optimize.
    // The parameters are organized in a contiguous array. When a parameter is not fitted
    // (only for optional parameters), it is excluded from that array. This allows looping
    // through the parameters, while supporting optional parameters.
    class Parameters {
    private:
        bool m_fit_phase_shift{};
        bool m_fit_astigmatism{};

        // phase_shift, astigmatism value, astigmatism angle
        std::array<f64, 3> m_default_values{};

        // defocus, (phase_shift), (astigmatism value, astigmatism angle)
        std::array<f64, 4> m_parameters{};
        std::array<f64, 4> m_lower_bounds{};
        std::array<f64, 4> m_upper_bounds{};
        std::array<f64, 4> m_abs_tolerance{};

    public:
        constexpr Parameters() = default;

        template<typename T>
        struct SetOptions {
            T defocus;
            T phase_shift;
            T astigmatism_value{};
            T astigmatism_angle{};
        };

        void set_fitting_parameters(bool fit_phase_shift, bool fit_astigmatism) {
            m_fit_phase_shift = fit_phase_shift;
            m_fit_astigmatism = fit_astigmatism;
        }

        void set_default_values(f64 defocus, f64 astigmatism_value, f64 astigmatism_angle, f64 phase_shift) {
            set_defocus(defocus);
            set_astigmatism(astigmatism_value, astigmatism_angle);
            set_phase_shift(phase_shift);
        }

        // Sets the (low and high) bounds for every parameter.
        void set_relative_bounds(const SetOptions<Vec<f64, 2>>& bounds) {
            m_lower_bounds[0] = std::max(0., defocus() + bounds.defocus[0]);
            m_upper_bounds[0] = std::max(0., defocus() + bounds.defocus[1]);
            if (has_phase_shift()) {
                m_lower_bounds[1] = std::max(0., phase_shift() + bounds.phase_shift[0]);
                m_upper_bounds[1] = std::max(0., phase_shift() + bounds.phase_shift[1]);
            }
            if (has_astigmatism()) {
                const size_t index = 2 - not has_phase_shift();
                m_lower_bounds[index] = astigmatism_value() + bounds.astigmatism_value[0];
                m_upper_bounds[index] = astigmatism_value() + bounds.astigmatism_value[1];
                m_lower_bounds[index + 1] = astigmatism_angle() + bounds.astigmatism_angle[0];
                m_upper_bounds[index + 1] = astigmatism_angle() + bounds.astigmatism_angle[1];
            }
        }

        void set_abs_tolerance(const SetOptions<f64>& tolerance) {
            m_abs_tolerance[0] = tolerance.defocus;
            if (has_phase_shift())
                m_abs_tolerance[1] = tolerance.phase_shift;
            if (has_astigmatism()) {
                const size_t index = 2 - not has_phase_shift();
                m_abs_tolerance[index] = tolerance.astigmatism_value;
                m_abs_tolerance[index + 1] = tolerance.astigmatism_angle;
            }
        }

        void update(const f64* parameters) {
            std::copy_n(parameters, size(), data());
        }

    public:
        [[nodiscard]] auto has_phase_shift() const noexcept -> bool { return m_fit_phase_shift; }
        [[nodiscard]] auto has_astigmatism() const noexcept -> bool { return m_fit_astigmatism; }

        [[nodiscard]] auto ssize() const noexcept -> i64 { return 1 + has_phase_shift() + 2 * has_astigmatism(); }
        [[nodiscard]] auto size() const noexcept -> size_t { return static_cast<size_t>(ssize()); }
        [[nodiscard]] auto data() noexcept -> f64* { return m_parameters.data(); }

        [[nodiscard]] auto span() noexcept { return SpanContiguous{m_parameters.data(), ssize()}; }
        [[nodiscard]] auto lower_bounds() noexcept { return SpanContiguous{m_lower_bounds.data(), ssize()}; }
        [[nodiscard]] auto upper_bounds() noexcept { return SpanContiguous{m_upper_bounds.data(), ssize()}; }
        [[nodiscard]] auto abs_tolerance() noexcept { return SpanContiguous{m_abs_tolerance.data(), ssize()}; }

    public: // safe access of the globals, whether they are fitted or not.
        [[nodiscard]] auto defocus() const noexcept -> f64 {
            return m_parameters[0];
        }

        [[nodiscard]] auto phase_shift() const noexcept -> f64 {
            return has_phase_shift() ? m_parameters[1] : m_default_values[0];
        }

        [[nodiscard]] auto astigmatism_value() const noexcept -> f64 {
            return has_astigmatism() ? m_parameters[2 - not has_phase_shift()] : m_default_values[1];
        }

        [[nodiscard]] auto astigmatism_angle() const noexcept -> f64 {
            return has_astigmatism() ? m_parameters[3 - not has_phase_shift()] : m_default_values[2];
        }

    public: // Setters for compatibility with grid search.
        void set_defocus(f64 defocus) noexcept {
            m_parameters[0] = defocus;
        }

        void set_phase_shift(f64 phase_shift) noexcept {
            if (has_phase_shift())
                m_parameters[1] = phase_shift;
            else
                m_default_values[0] = phase_shift;
        }

        void set_astigmatism(f64 value, f64 angle) noexcept {
            if (has_astigmatism()) {
                const size_t index = 2 - not has_phase_shift();
                m_parameters[index] = value;
                m_parameters[index + 1] = angle;
            } else {
                m_default_values[1] = value;
                m_default_values[2] = angle;
            }
        }
    };

    class CTFCoarse {
    private:
        View<const f32> m_power_spectrum;
        Array<f32> m_background_and_envelope;

        Vec<f64, 2> m_fftfreq_range;
        ns::CTFAnisotropic<f64> m_ctf{};
        Parameters m_parameters{};
        Memoizer m_memoizer{};


    public:
        CTFCoarse(
            const View<f32>& patch_rfft_ps,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFAnisotropic<f64>& ctf,
            const CTFSplines& splines,
            bool fit_astigmatism,
            bool fit_envelope
        ) :
            m_power_spectrum(patch_rfft_ps),
            m_fftfreq_range(fftfreq_range),
            m_ctf(ctf)
        {
            m_background_and_envelope = Array<f32>(patch_rfft_ps.shape().set<0>(1 + fit_envelope), {
                .device = patch_rfft_ps.device(),
                .allocator = Allocator::MANAGED,
            });

            // Initialize the parameters.
            // If the astigmatism or phase shift is not fitted, we should still use the provided CTF as default.
            m_parameters.set_default_values(
                ctf.defocus().value, ctf.defocus().astigmatism, ctf.defocus().angle, ctf.phase_shift());

            // Sample the background (and envelope) to match the power spectrum.
            splines.sample_splines(m_background_and_envelope.view(), fftfreq_range);

            m_memoizer = Memoizer(m_parameters.ssize(), fit_astigmatism ? 4 : 0);
        }

        auto params() noexcept -> Parameters& { return m_parameters; }
        auto ctf() noexcept -> ns::CTFAnisotropic<f64>& {
            m_ctf.set_defocus({
                .value = m_parameters.defocus(),
                .astigmatism = m_parameters.astigmatism_value(),
                .angle = m_parameters.astigmatism_angle()
            });
            m_ctf.set_phase_shift(m_parameters.phase_shift());
            return m_ctf;
        }

        auto cost() -> f64 {
            f32 ncc{};
            noa::reduce_iwise(
                m_power_spectrum.shape().filter(0, 2, 3), m_power_spectrum.device(),
                noa::wrap(0.f, 0.f, 0.f), ncc, CTFCrossCorrelate(
                    m_power_spectrum.span().filter(0, 2, 3).as_contiguous(),
                    m_background_and_envelope.span().filter(0, 2, 3).as_contiguous(),
                    noa::BatchedParameter{ctf()}, m_fftfreq_range
                ));
            return ncc;
        }

        [[gnu::noinline]] void log(f64 ncc, bool memoized = false) const {
            std::string defocus_str = m_parameters.has_astigmatism() ?
                fmt::format("defocus=(value={:.8f}, astigmatism={:.8f}, angle={:.8f})",
                            m_parameters.defocus(),
                            m_parameters.astigmatism_value(),
                            noa::rad2deg(m_parameters.astigmatism_angle())) :
                fmt::format("defocus={:.8f}", m_parameters.defocus());

            std::string phase_shift = m_parameters.has_phase_shift() ?
                fmt::format("phase_shift={:.8f}, ", noa::rad2deg(m_parameters.phase_shift())) : "";

            Logger::trace(
                "{}, {}ncc={:.8f}{}",
                defocus_str, phase_shift,
                ncc, memoized ? ", memoized=true" : ""
            );
        }

        static auto function_to_maximise(
            u32 n_parameters,
            const f64* parameters,
            f64* gradients,
            void* instance
        ) -> f64 {
            auto& self = *static_cast<CTFCoarse*>(instance);
            check(n_parameters == self.m_parameters.size(),
                  "The parameters of the fitter and the optimizer don't seem to match");

            // The optimizer may pass its own array, so update/memcpy our parameters.
            if (parameters != self.m_parameters.data())
                self.m_parameters.update(parameters);

            // Check if this function was called with the same parameters.
            std::optional<f64> memoized_score = self.m_memoizer.find(self.m_parameters.data(), gradients, 1e-8);
            if (memoized_score.has_value()) {
                self.log(memoized_score.value(), true);
                return memoized_score.value();
            }

            if (gradients) {
                for (i32 i{}; auto& value: self.m_parameters.span()) {
                    const f32 initial_value = static_cast<f32>(value);
                    const f32 delta = CentralFiniteDifference::delta(initial_value);

                    value = static_cast<f64>(initial_value - delta);
                    const f64 fx_minus_delta = self.cost();
                    value = static_cast<f64>(initial_value + delta);
                    const f64 fx_plus_delta = self.cost();

                    value = static_cast<f64>(initial_value); // back to original value
                    f64 gradient = CentralFiniteDifference::get(
                        fx_minus_delta, fx_plus_delta, static_cast<f64>(delta));

                    Logger::trace("g={:.8f}", gradient);
                    gradients[i++] = gradient;
                }
            }

            const f64 cost = self.cost();
            self.log(cost);
            self.m_memoizer.record(parameters, cost, gradients);
            return cost;
        }
    };

    struct AddPowerSpectrum {
        using remove_default_final = bool;
        f32 norm;

        constexpr void operator()(c32 input, f32& reduced) const {
            auto ps = noa::AbsSquared{}(input);
            ps *= norm;
            reduced += ps;
        }
        static constexpr void join(f32 i_reduced, f32& reduced) {
            reduced += i_reduced;
        }
        static constexpr void final(f32 i_reduced, f32& output) {
            output += i_reduced;
        }
    };

    auto compute_average_patch(
        StackLoader& stack_loader,
        const MetadataStack& metadata,
        const Grid& grid,
        i64 fourier_cropped_size,
        Vec<f64, 2> delta_z_range_nanometers,
        f64 max_tilt_for_average,
        const Path& debug_directory
    ) -> Array<f32> {
        // Loading every single patch at once can be too much if using discrete GPU memory.
        // Since this function is only called 3 times, load the patches slice per slice.
        const auto options = ArrayOption(stack_loader.compute_device(), Allocator::DEFAULT_ASYNC);
        const auto n_patches_max = grid.n_patches();
        const auto patch_shape = grid.patch_shape().push_front<2>(1);
        const auto all_patch_shape = patch_shape.set<0>(n_patches_max);

        const auto fft_scale = 1 / static_cast<f32>(grid.patch_size() * grid.patch_size());
        const auto average_op = AddPowerSpectrum{fft_scale};

        const auto slice = Array<f32>(grid.slice_shape().push_front<2>(1), options);
        const auto all_patches_rfft = Array<c32>(all_patch_shape.rfft(), options);
        const auto all_patches_origins = Array<Vec<i32, 4>>(n_patches_max, options);

        // 1. Anything higher resolution than the fitting range isn't used, so remove it.
        // 2. Small pixel sizes can cramp up the Thon rings to a small number of pixels in the spectrum.
        // As such, Fourier crop the patches to the target resolution, and go back to real space
        // to zero-pad back to the original patch size, effectively increasing the sampling and
        // stretching the Thon rings as much as possible.

        const auto patch_cropped_shape = Shape{fourier_cropped_size, fourier_cropped_size}.push_front<2>(1);
        const auto zero_padding = (patch_shape - patch_cropped_shape).vec;
        const auto all_patch_cropped_shape = patch_cropped_shape.set<0>(n_patches_max);
        const auto all_patches_cropped_rfft = Array<c32>(all_patch_cropped_shape.rfft(), options);

        // Average power-spectrum, initialized to zero since we add directly to it.
        auto patches_rfft_ps_average = noa::zeros<f32>(patch_shape.rfft(), options);

        Logger::trace(
            "Creating average power spectrum from patches:\n"
            "  z_range={}nm\n"
            "  max_tilt={} degrees\n"
            "  patch_size={}\n"
            "  fourier_cropped_size={}",
            delta_z_range_nanometers, max_tilt_for_average,
            patch_shape[3], patch_cropped_shape[3]
        );

        i64 total{};
        for (const auto& slice_metadata: metadata) {
            if (std::abs(slice_metadata.angles[1]) > max_tilt_for_average or slice_metadata.angles[1] > 0) // FIXME
                continue;

            // Filter out the patches that are not within the desired z-range.
            const std::vector patches_origins_vector = grid.compute_subregion_origins(
                    slice_metadata, stack_loader.stack_spacing(), delta_z_range_nanometers);
            const auto n_patches = std::ssize(patches_origins_vector);
            total += n_patches;
            Logger::trace("tilt={:>+6.2f}, n_patches_added={:0>3}, n_patches_total={:0>5}",
                          slice_metadata.angles[1], n_patches, total);
            if (n_patches < 2)
                continue;

            const auto patches_origins = all_patches_origins.view().subregion(ni::Ellipsis{}, ni::Slice{0, n_patches});
            View(patches_origins_vector.data(), n_patches).to(patches_origins);

            // Take the patches you need for this extraction.
            const auto patches_rfft = all_patches_rfft.view().subregion(ni::Slice{0, n_patches});
            const auto patches_cropped_rfft = all_patches_cropped_rfft.view().subregion(ni::Slice{0, n_patches});
            const auto patches = nf::alias_to_real(patches_rfft, patch_shape.set<0>(n_patches));
            const auto patches_cropped = nf::alias_to_real(patches_cropped_rfft, patch_cropped_shape.set<0>(n_patches));

            // Extract the patches. Assume the slice is normalized and edges are tapered.
            // We run this function 3 times in a row, so cache to not have the reload and preprocess.
            stack_loader.read_slice(slice.view(), slice_metadata.index_file, /*cache=*/ true);
            noa::extract_subregions(slice.view(), patches, patches_origins);

            // Crop to the maximum frequency and oversample back to the original size
            // to nicely stretch the Thon rings to counteract small pixel sizes and high defoci.
            nf::r2c(patches, patches_rfft);
            nf::resize<"h2h">(patches_rfft, patches.shape(), patches_cropped_rfft, patches_cropped.shape());
            nf::c2r(patches_cropped_rfft, patches_cropped);
            // TODO smooth edges?
            noa::normalize_per_batch(patches_cropped, patches_cropped);
            noa::resize(patches_cropped, patches, {}, zero_padding);

            // Compute the average power-spectrum of these tiles and add it to the average.
            nf::r2c(patches, patches_rfft, {.norm = nf::Norm::NONE});
            noa::reduce_axes_ewise(patches_rfft, f32{0}, patches_rfft_ps_average.view(), average_op);

            // Not necessary, but I prefer to sync here...
            patches_rfft_ps_average.eval();
        }

        // Importantly, take the mean, for the background subtraction to make sense.
        const auto norm = 1 / static_cast<f32>(total);
        noa::ewise({}, patches_rfft_ps_average, [norm]NOA_HD(f32& output) {
            output *= norm;
        });

        if (not debug_directory.empty()) {
            // Save the average power spectrum.
            auto average_patch = noa::fft::remap("h2hc", patches_rfft_ps_average, patch_shape);
            auto filename = debug_directory / "average_patch.mrc";
            noa::write(std::move(average_patch), filename);
            Logger::debug("{} saved", filename);

            // Save the rotational average.
            auto rotational_average = Array<f32>(patch_shape[3] / 2 + 1, options);
            ng::rotational_average<"h2h">(patches_rfft_ps_average, patch_shape, rotational_average);
            filename.replace_filename("rotational_average.txt");
            save_vector_to_text(rotational_average.view(), filename);
            Logger::debug("{} saved", filename);
        }

        return patches_rfft_ps_average;
    }

    auto initial_search(
        const View<f32>& patch_rfft_ps,
        const Vec<f64, 2>& fftfreq_range,
        ns::CTFAnisotropic<f64>& ctf,
        bool fit_phase_shift
    ) {
        auto timer = Logger::trace_scope_time("Initial 1d defocus search");

        // With a patch shape of 512x512 and a fftfreq range of [40,6] angstrom,
        // the rfft_size (512/2+1=257) is enough sampling for a spacing of 1 and defocus 6.
        // So for now, do this, but we may want to increase the patch size in the future,
        // or simply increase the size of the 1d rotational average.
        const auto logical_size = patch_rfft_ps.shape()[2];
        const auto spectrum_size = patch_rfft_ps.shape()[3];

        // Everything is expressed in the original spacing, so keep track of the fftfreq range.
        // The patches are already fourier cropped to the max frequency (fftfreq_range[1]).
        // Exclude the low frequencies (fftfreq_range[0]) from the 1d rotational average and simulated CTF.
        const auto patches_fftfreq_range = noa::Linspace{
            .start = 0.,
            .stop = fftfreq_range[1],
            .endpoint = true,
        };
        const auto spectrum_fftfreq_range = noa::Linspace{
            .start = fftfreq_range[0],
            .stop = fftfreq_range[1],
            .endpoint = true,
        };

        // Compute the l2-normalized rotation average.
        // If the patch is on the GPU, do the rotational average on the GPU as well.
        const auto options = ArrayOption{.device = patch_rfft_ps.device(), .allocator = Allocator::MANAGED};
        auto rotational_average = Array<f32>(spectrum_size, options);
        ng::rotational_average<"h2h">(
            patch_rfft_ps, {1, 1, logical_size, logical_size},
            rotational_average, {}, {
                .input_fftfreq = patches_fftfreq_range,
                .output_fftfreq = spectrum_fftfreq_range,
            });
        rotational_average = std::move(rotational_average).reinterpret_as_cpu();

        // Compute and subtract the background.
        // Get initial background fitting. This is just a rough approximation of the background and simply fits
        // a smooth spline through the rotational average, cutting through the Thon rings. Similar results
        // could be achieved by subtracting a local mean or by applying a strong gaussian blur onto the spectrum.
        CubicSplineGrid<f64, 1> background_spline = fit_coarse_background_1d(rotational_average.view(), 3);
        const f64 norm = 1 / static_cast<f64>(spectrum_size - 1);
        for (i64 i{}; auto& e: rotational_average.span_1d_contiguous()) {
            const f64 coordinate = static_cast<f64>(i++) * norm; // [0,1]
            e -= static_cast<f32>(background_spline.interpolate_at(coordinate));
        }

        //
        noa::normalize(rotational_average, rotational_average, {.mode = noa::Norm::L2});

        const auto max_phase_shift = fit_phase_shift ? noa::Constant<f64>::PI / 6 : 0.;
        const auto grid_search = GridSearch<f64, f64>(
            {.start = 0., .end = max_phase_shift, .step = 0.05}, // phase shift
            {.start = 0.4, .end = 7.5, .step = 0.02} // defocus
        );
        Logger::trace("grid_search:shape=:", grid_search.shape());

        auto simulated_ctf = noa::Array<f32>(spectrum_size);
        auto ictf = ns::CTFIsotropic(ctf);

        f64 best_ncc{};
        Vec<f64, 2> best_values{};
        grid_search.for_each([&](f64 phase_shift, f64 defocus) mutable {
            ictf.set_defocus(defocus);
            ictf.set_phase_shift(phase_shift);
            ns::ctf_isotropic<"h2h">(
                simulated_ctf, Shape{logical_size}.push_front<3>(1), ictf, {
                    .fftfreq_range = spectrum_fftfreq_range,
                    .ctf_squared = true
                });

            f64 ncc{};
            f64 ncc_rhs{};
            const auto lhs = rotational_average.span_1d_contiguous();
            const auto rhs = simulated_ctf.span_1d_contiguous();
            for (i64 i{}; i < lhs.ssize(); ++i) {
                auto lhs_i = static_cast<f64>(lhs[i]);
                auto rhs_i = static_cast<f64>(rhs[i]);
                ncc += lhs_i * rhs_i;
                ncc_rhs += rhs_i * rhs_i;
            }
            ncc /= std::sqrt(ncc_rhs);

            const bool new_best = ncc > best_ncc;
            if (new_best) {
                best_values = {defocus, phase_shift};
                best_ncc = ncc;
            }
            Logger::trace(
                "defocus={:.3f}, phase_shift={}:.3f, ncc={:.4f}{}",
                defocus, phase_shift, ncc, new_best ? "(+)" : ""
            );
        });

        // Update the CTF with the defocus estimate.
        ctf.set_defocus({
            .value = best_values[0],
            .astigmatism = ctf.defocus().astigmatism,
            .angle = ctf.defocus().angle
        });
    }

    auto fit_ctf_to_patch(
        const View<f32>& patch_rfft_ps,
        const Vec<f64, 2>& fftfreq_range,
        ns::CTFAnisotropic<f64>& ctf_anisotropic, // updated
        CTFSplines& splines, // updated
        bool fit_background,
        bool fit_envelope,
        bool fit_phase_shift,
        bool fit_astigmatism,
        const Path& output_directory
    ) {
        f64 ncc{};
        i64 n_evaluations{};
        for (i32 i: noa::irange(2 + fit_astigmatism)) {
            //
            if (fit_background)
                splines.fit_from_2d(patch_rfft_ps, fftfreq_range, ctf_anisotropic, fit_envelope);

            // Set up.
            const bool has_astigmatism = i == 0 ? false : fit_astigmatism;
            auto fitter = CTFCoarse(
                patch_rfft_ps, fftfreq_range, ctf_anisotropic,
                splines, has_astigmatism, fit_envelope
            );

            // Local optimization to polish to optimum and search for astigmatism.
            constexpr auto PI = noa::Constant<f64>::PI;
            constexpr auto PI_EPSILON = PI / 32;
            fitter.params().set_fitting_parameters(fit_phase_shift, has_astigmatism);
            fitter.params().set_relative_bounds({
                .defocus = {-0.25, 0.25},
                .phase_shift = {-PI / 6, PI / 6},
                .astigmatism_value = {-0.3, 0.3},
                .astigmatism_angle = {-PI / 2 - PI_EPSILON, PI / 2 + PI_EPSILON},
            });
            fitter.params().set_abs_tolerance({
                .defocus = 5e-4,
                .phase_shift = noa::deg2rad(0.25),
                .astigmatism_value = 5e-5,
                .astigmatism_angle = noa::deg2rad(0.1),
            });

            auto optimizer = Optimizer(has_astigmatism ? NLOPT_LD_LBFGS : NLOPT_LN_SBPLX, fitter.params().ssize());
            optimizer.set_max_objective(CTFCoarse::function_to_maximise, &fitter);
            optimizer.set_bounds(fitter.params().lower_bounds().data(),
                                 fitter.params().upper_bounds().data());
            optimizer.set_x_tolerance_abs(fitter.params().abs_tolerance().data());

            ncc = optimizer.optimize(fitter.params().data());
            n_evaluations += optimizer.n_evaluations();
            fitter.log(ncc);

            ctf_anisotropic = fitter.ctf();
        }
        Logger::trace("n_evaluations={}", n_evaluations);

        if (fit_background)
            splines.fit_from_2d(patch_rfft_ps, fftfreq_range, ctf_anisotropic, fit_envelope);

        const i64 logical_size = patch_rfft_ps.shape()[2];
        const i64 spectrum_size = patch_rfft_ps.shape()[3];

        // Log
        auto rotational_average = Array<f32>(spectrum_size, {patch_rfft_ps.device()});
        ng::rotational_average_anisotropic<"h2h">(
            patch_rfft_ps, {1, 1, logical_size, logical_size}, ctf_anisotropic,
            rotational_average, {}, {
                .input_fftfreq = {0, fftfreq_range[1]},
                .output_fftfreq = {fftfreq_range[0], fftfreq_range[1]},
                .add_to_output = false,
            });
        rotational_average = rotational_average.to_cpu();
        save_vector_to_text(rotational_average.view(), output_directory / "rotational_average.txt");

        auto background = Array<f32>({1, 1, 1, spectrum_size});
        splines.sample_spectrum_1d(background.view(), fftfreq_range);
        save_vector_to_text(background.view(), output_directory / "background.txt");

        for (auto&& [p, b]: noa::zip(rotational_average.span_1d(), background.span()[0].as_1d())) // , background.span()[1].as_1d())
            p = p - b;
        noa::normalize(rotational_average, rotational_average, {.mode = noa::Norm::L2});
        save_vector_to_text(rotational_average.view(), output_directory / "rotational_average_bs.txt");

        auto simulated_ctf = Array<f32>(spectrum_size);
        ns::ctf_isotropic<"h2h">(
            simulated_ctf, Shape{logical_size}.push_front<3>(1), ns::CTFIsotropic(ctf_anisotropic), {
                .fftfreq_range = {fftfreq_range[0], fftfreq_range[1]},
                .ctf_squared = true
            });
        noa::normalize(simulated_ctf, simulated_ctf, {.mode = noa::Norm::L2});
        save_vector_to_text(simulated_ctf.view(), output_directory / "simulated_ctf.txt");

        return ncc;
    }
}

namespace qn::ctf {
    auto coarse_fit_average_ps(
        StackLoader& stack_loader,
        const Grid& grid,
        const MetadataStack& metadata,
        const ns::CTFAnisotropic<f64>& initial_ctf,
        const FitCoarseOptions& options
    ) -> FitCoarseResults {
        constexpr auto directory = std::array{"at_eucentric", "below_eucentric", "above_eucentric"};
        const auto max_tilt_for_averages = std::array{options.max_tilt_for_average, 90., 90.};
        const auto delta_z_ranges = std::array{
            options.delta_z_range_nanometers - std::abs(options.delta_z_shift_nanometers),
            options.delta_z_range_nanometers ,
            options.delta_z_range_nanometers + std::abs(options.delta_z_shift_nanometers)
        };

        FitCoarseResults results{};

        // Resolution to fftfreq range.
        // The patch is Fourier cropped to the integer frequency closest to the resolution.
        const auto spacing = mean(stack_loader.stack_spacing()); // assume isotropic spacing by this point
        const auto [fourier_cropped_size, fourier_cropped_fftfreq] = fourier_crop_to_resolution(
            grid.patch_size(), spacing, options.resolution_range[1]
        );
        results.fourier_crop_size = fourier_cropped_size;
        results.fftfreq_range = Vec{
            resolution_to_fftfreq(spacing, options.resolution_range[0]),
            fourier_cropped_fftfreq,
        };
        Logger::info(
            "Coarse CTF fitting frequency range:\n"
            "  resolution_range={::.3f}A\n"
            "  fftfreq_range={::.5f}",
            fftfreq_to_resolution(spacing, results.fftfreq_range),
            results.fftfreq_range
        );

        // The best spectrum is at_eucentric, do the full search there, then refine and fit the background.
        // For the two spectra above and below the tilt-axis, only refine the defocus, using the same astigmatism
        // and background (+ phase-shift and envelope). These spectra are likely worse than the first spectrum
        // at the tilt-axis, and since all of these are averages over the tilt-series, we do indeed expect
        // to have the same astigmatism etc.
        results.average_ctf = initial_ctf;
        for (auto i: noa::irange<size_t>(3)) {
            auto timer = Logger::info_scope_time("Coarse CTF fitting on average power spectrum: {}", directory[i]);

            const auto average_patch_rfft_ps = compute_average_patch(
                stack_loader, metadata, grid,
                fourier_cropped_size, delta_z_ranges[i], max_tilt_for_averages[i],
                options.debug_directory / directory[i]
            );

            // Initial defocus estimate. This is only done for the first spectrum at the tilt-axis.
            // TODO Allow the user to provide an initial estimate (or a range).
            auto i_ctf = results.average_ctf;
            if (i == 0) {
                initial_search(
                    average_patch_rfft_ps.view(), results.fftfreq_range,
                    i_ctf, options.fit_phase_shift
                );
            }

            const bool fit_background = i == 0;
            const f64 ncc = fit_ctf_to_patch(
                average_patch_rfft_ps.view(), results.fftfreq_range, i_ctf, results.splines,
                fit_background, options.fit_envelope, options.fit_phase_shift, options.fit_astigmatism,
                options.debug_directory / directory[i]
            );

            results.defocus[i] = i_ctf.defocus().value;
            results.ncc[i] = ncc;
            if (i == 0)
                results.average_ctf = i_ctf;
        }

        return results;
    }
}
