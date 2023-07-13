#include <noa/Math.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>

#include "quinoa/core/CTF.hpp"
#include "quinoa/core/CubicGrid.hpp"
#include "quinoa/core/GridSearch1D.hpp"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/Utilities.h"
#include "quinoa/io/Logging.h"

namespace {
    using namespace ::qn;

    class Background {
    private:
        Array<f32> m_full_rotational_average;
        Array<f32> m_full_rotational_average_smooth;
        View<const f32> m_patch_rfft_ps; // if astigmatism, we need it
        i64 m_patch_size;

    public:
        struct FetchedCTFValues {
            using frequency_and_value = Vec2<f64>; // [0]=frequency between [0,1], [1]=value
            std::vector<frequency_and_value> points_at_zeros;
            std::vector<frequency_and_value> points_at_peaks;
        };

    public:
        Background(const View<const f32>& patch_rfft_ps, i64 patch_size)
                : m_patch_rfft_ps(patch_rfft_ps), m_patch_size(patch_size) {
            // Full range rotational average.
            // For the accurate background fitting, we first smooth out the curve, and we do this on the
            // full-range average to not see the edges of the convolution withing the fitting range.
            m_full_rotational_average = noa::memory::empty<f32>(m_patch_size / 2 + 1);
            m_full_rotational_average_smooth = noa::memory::zeros<f32>(
                    m_patch_size / 2 + 1); // noa::memory::like(m_full_rotational_average);

            noa::geometry::fft::rotational_average<fft::H2H>(
                    m_patch_rfft_ps, {1, 1, m_patch_size, m_patch_size},
                    m_full_rotational_average.view(),
                    m_full_rotational_average_smooth.view());

            noa::signal::convolve(
                    m_full_rotational_average.view(),
                    m_full_rotational_average_smooth.view(),
                    noa::signal::gaussian_window<f32>(7, 1.25, /*normalize=*/ true));
        }

    public:
        void fit_spline(CTF::FittingRange& fitting_range, i64 spline_resolution = 3) {
            const auto rotational_average = m_full_rotational_average
                    .view().subregion(noa::indexing::Ellipsis{}, fitting_range.slice);
            fitting_range.background = smooth_curve_fitting_(rotational_average.span(), spline_resolution);
        }

        void fit_accurate(
                CTF::FittingRange& fitting_range,
                const CTFIsotropic64& ctf,
                bool fit_astigmatism,
                f64 astigmatism_value,
                f64 astigmatism_angle,
                i64 spline_resolution = 3
        ) {
            if (fit_astigmatism) {
                // In this case, we need to recompute the rotational average,
                // correcting for the given astigmatism.
                noa::geometry::fft::rotational_average_anisotropic<fft::H2H>(
                        m_patch_rfft_ps, {1, 1, m_patch_size, m_patch_size},
                        CTFAnisotropic64(ctf, astigmatism_value, astigmatism_angle),
                        m_full_rotational_average.view(),
                        m_full_rotational_average_smooth.view());

                noa::signal::convolve(
                        m_full_rotational_average.view(),
                        m_full_rotational_average_smooth.view(),
                        noa::signal::gaussian_window<f32>(7, 1.25, /*normalize=*/ true));
            }

            // The background is within the fitting range.
            // When fetching ctf values at the ctf zeros/peaks, use the smooth spectrum.
            const auto rotational_average = m_full_rotational_average_smooth
                    .view().subregion(noa::indexing::Ellipsis{}, fitting_range.slice);
            auto [points_at_zeros, points_at_peaks] = fetch_ctf_values(
                    rotational_average, fitting_range.fftfreq, ctf);

            // If we are too close to the focus (or fitting range is too small), we may not have enough points to
            // accurately fit the background using the ctf zeros, so ignore this and continue using the initial fitting.
            if (points_at_zeros.size() >= 2 || points_at_peaks.size() >= 2) {
                fitting_range.background = smooth_curve_fitting_(points_at_zeros, spline_resolution);
//                if (false) {
//                    // Subtract the background from the peaks.
//                    for (auto& [frequency, peak]: points_at_peaks)
//                        peak -= fitting_range.background.interpolate(frequency);
//                    fitting_range.envelope = smooth_curve_fitting_(points_at_peaks);
//                }
            } else {
                fitting_range.background = smooth_curve_fitting_(rotational_average.span(), spline_resolution);
            }
        }

    private:
        [[nodiscard]] static auto smooth_curve_fitting_( // uniform
                const Span<const f32>& values,
                i64 spline_resolution
        ) -> CubicSplineGrid<f64, 1> {
            struct OptimizerData {
                noa::Span<const f32> values{};
                CubicSplineGrid<f64, 1> model{};
            };

            auto cost = [](
                    [[maybe_unused]] u32 n_parameters,
                    [[maybe_unused]] const f64* parameters,
                    [[maybe_unused]] f64* gradients,
                    void* instance
            ) {
                auto& data = *static_cast<OptimizerData*>(instance);
                NOA_ASSERT(n_parameters == data.model.resolution()[0]);
                NOA_ASSERT(gradients == nullptr);
                NOA_ASSERT(parameters == data.model.data());

                // Compute the least-square score between the experimental points and the model.
                f64 score{0};
                f64 norm = 1 / static_cast<f64>(data.values.ssize());
                for (i64 i = 0; i < data.values.ssize(); ++i) {
                    const auto coordinate = static_cast<f64>(i) * norm; // uniform spacing
                    const auto experiment = static_cast<f64>(data.values[i]);
                    const f64 predicted = data.model.interpolate(coordinate);
                    score += std::pow(experiment - predicted, 2);
                }
                return score;
            };

            // Some stats about the data points.
            f64 mean{0}, min{noa::math::Limits<f64>::max()}, max{noa::math::Limits<f64>::lowest()};
            for (auto e: values) {
                const auto value = static_cast<f64>(e);
                mean += value;
                min = std::min(min, value);
                max = std::max(max, value);
            }
            mean /= static_cast<f64>(values.ssize());
            const auto value_range = max - min;

            OptimizerData optimizer_data{values, CubicSplineGrid<f64, 1>(spline_resolution)};
            optimizer_data.model.set_all_points_to(mean);

            // Here use a local derivative-less algorithm, since we assume
            // the spline resolution is < 5 and the cost is cheap to compute.
            Optimizer optimizer(NLOPT_LN_SBPLX, spline_resolution);
            optimizer.set_min_objective(cost, &optimizer_data);
            optimizer.set_bounds(min - value_range * 1.5, max + value_range * 1.5);
            optimizer.optimize(optimizer_data.model.data());
            // TODO tolerance
            return optimizer_data.model;
        }

        [[nodiscard]] static auto smooth_curve_fitting_( // non-uniform
                const std::vector<Vec2<f64>>& data_points,
                i64 spline_resolution
        ) -> CubicSplineGrid<f64, 1> {
            struct OptimizerData {
                const std::vector<Vec2<f64>>* data_points{};
                CubicSplineGrid<f64, 1> model{};
            };

            auto cost = [](
                    [[maybe_unused]] u32 n_parameters,
                    [[maybe_unused]] const f64* parameters,
                    [[maybe_unused]] f64* gradients,
                    void* instance
            ) {
                auto& data = *static_cast<OptimizerData*>(instance);
                NOA_ASSERT(n_parameters == data.model.resolution()[0]);
                NOA_ASSERT(gradients == nullptr);
                NOA_ASSERT(parameters == data.model.data());

                // Compute the least-square score between the experimental points and the model.
                f64 score{0};
                for (auto data_point: *data.data_points) {
                    const f64 experiment = data_point[1];
                    const f64 predicted = data.model.interpolate(data_point[0]);
                    score += std::pow(experiment - predicted, 2);
                }
                return score;
            };

            // Some stats about the data points.
            f64 mean{0}, min{data_points[0][1]}, max{data_points[0][1]};
            for (const auto& data_point: data_points) {
                mean += data_point[1];
                min = std::min(min, data_point[1]);
                max = std::max(max, data_point[1]);
            }
            mean /= static_cast<f64>(data_points.size());

            OptimizerData optimizer_data{&data_points, CubicSplineGrid<f64, 1>(spline_resolution)};
            optimizer_data.model.set_all_points_to(mean);

            // Here use a local derivative-less algorithm, since we assume
            // the spline resolution is <= 5 and the cost is cheap to compute.
            Optimizer optimizer(NLOPT_LN_SBPLX, spline_resolution);
            optimizer.set_min_objective(cost, &optimizer_data);
            optimizer.set_bounds(min - min * 0.25, max + max * 0.25);
            optimizer.optimize(optimizer_data.model.data());
            // TODO tolerance
            return optimizer_data.model;
        }

        [[nodiscard]] static auto fetch_ctf_values(
                const View<const f32>& trimmed_rotational_average,
                Vec2<f64> range_fftfreq,
                const CTFIsotropic64& ctf,
                bool fit_envelope = false
        ) -> FetchedCTFValues {
            // Simulate ctf slope with high enough sampling to prevent aliasing for high defoci.
            constexpr i64 SIMULATED_SIZE = 4096;
            constexpr f64 SIMULATED_FREQUENCY_STEP = 1 / static_cast<f64>(SIMULATED_SIZE);

            // Only evaluates the frequencies that are within the trimmed range.
            const auto range_index = noa::math::round(range_fftfreq * SIMULATED_SIZE).as<i64>();
            i64 simulated_start = std::max(range_index[0] - 3, i64{0});
            i64 simulated_end = std::min(range_index[1] + 3, SIMULATED_SIZE - 1);

            // Go from the index in the simulated spectrum, to the index in the trimmed spectrum.
            const auto n_samples_in_range = trimmed_rotational_average.ssize();
            const auto range_fftfreq_step =
                    (range_fftfreq[1] - range_fftfreq[0]) /
                    static_cast<f64>(n_samples_in_range - 1);

            // We use the change in the sign of the slope to detect the zeros and peaks of the ctf^2 function.
            // To compute this change at index i, we need the slope at i and i+1, which requires 3 ctf evaluations
            // (at i, i+1 and i+2). Instead, use a circular buffer to only sample the ctf once per iteration.
            // Each iteration computes the i+2 element, so we need to precompute the first two for the first iteration.
            constexpr auto index_circular_buffer = [](size_t index, i64 step) {
                if (step == 0)
                    return index;
                const i64 value = static_cast<i64>(index) + step;
                return static_cast<size_t>(value < 0 ? (value + 3) : value);
            };
            size_t circular_count = 2;
            std::array<f64, 3> ctf_values{
                    std::abs(ctf.value_at(static_cast<f64>(simulated_start + 0) * SIMULATED_FREQUENCY_STEP)),
                    std::abs(ctf.value_at(static_cast<f64>(simulated_start + 1) * SIMULATED_FREQUENCY_STEP)),
                    0
            };

            const auto trimmed_rotational_average_a = trimmed_rotational_average.accessor_contiguous_1d();
            const auto coord_norm = 1 / static_cast<f64>(trimmed_rotational_average.ssize() - 1); // -1 for inclusive
            FetchedCTFValues output;

            // Collect zeros and peaks.
            for (i64 i = simulated_start; i < simulated_end; ++i) {
                // Do your part, sample ith+2.
                ctf_values[circular_count] = std::abs(ctf.value_at(static_cast<f64>(i + 2) * SIMULATED_FREQUENCY_STEP));

                // Get the corresponding index in the experimental spectrum.
                // Here we want the frequency in the middle of the window, so at i + 1.
                const auto fftfreq = static_cast<f64>(i + 1) * SIMULATED_FREQUENCY_STEP;
                const auto corrected_frequency = (fftfreq - range_fftfreq[0]) / range_fftfreq_step;
                const i64 index = static_cast<i64>(std::round(corrected_frequency));

                if (index >= 0 && index < trimmed_rotational_average.ssize()) {
                    const f64 spline_coordinate = static_cast<f64>(index) * coord_norm; // [0,n-1] -> [0,1]

                    // Compute the simulated ctf slope.
                    // Based on the slope, we could lerp the index and value, but nearest is good enough here.
                    const f64 ctf_value_0 = ctf_values[index_circular_buffer(circular_count, -2)];
                    const f64 ctf_value_1 = ctf_values[index_circular_buffer(circular_count, -1)];
                    const f64 ctf_value_2 = ctf_values[index_circular_buffer(circular_count, 0)];
                    const f64 slope_0 = ctf_value_1 - ctf_value_0;
                    const f64 slope_1 = ctf_value_2 - ctf_value_1;

                    if (slope_0 < 0 && slope_1 >= 0) { // zero: negative slope to positive slope.
                        output.points_at_zeros.emplace_back(
                                spline_coordinate, trimmed_rotational_average_a[index]);
                    }
                    if (fit_envelope && slope_0 > 0 && slope_1 <= 0) { // peak: positive slope to negative slope.
                        output.points_at_peaks.emplace_back(
                                spline_coordinate, trimmed_rotational_average_a[index]);
                    }
                }

                // Increment circular buffer.
                circular_count = (circular_count + 1) % 3; // 0,1,2,0,1,2,...
            }
            return output;
        }
    };

    // Manages the parameters to optimize.
    // The parameters are organized in a contiguous array. When a parameter is not fitted
    // (only for optional parameters), it is excluded from that array. This allows to loop
    // through the parameters and while allowing optional parameters.
    class Parameters {
    public:
        constexpr Parameters() = default;

        Parameters(
                bool fit_phase_shift, bool fit_astigmatism,
                const CTFIsotropic64& average_ctf,
                f64 astigmatism_value = 0,
                f64 astigmatism_angle = 0
        ) :
                m_fit_phase_shift(fit_phase_shift), m_fit_astigmatism(fit_astigmatism) {
            // Initialise values.
            m_parameters[0] = average_ctf.defocus();
            if (has_phase_shift())
                m_parameters[1] = average_ctf.phase_shift();
            if (has_astigmatism()) {
                const size_t index = 2 - !has_phase_shift();
                m_parameters[index] = astigmatism_value;
                m_parameters[index + 1] = astigmatism_angle;
            }
        }

        // Sets the (low and high) bounds for every parameter.
        void set_relative_bounds(
                Vec2<f64> defocus_bounds,
                Vec2<f64> phase_shift_bounds,
                Vec2<f64> astigmatism_value_bounds,
                Vec2<f64> astigmatism_angle_bounds
        ) {
            m_lower_bounds[0] = std::max(0., defocus() + defocus_bounds[0]);
            m_upper_bounds[0] = std::max(0., defocus() + defocus_bounds[1]);
            if (has_phase_shift()) {
                m_lower_bounds[1] = std::max(0., phase_shift() + phase_shift_bounds[0]);
                m_upper_bounds[1] = std::max(0., phase_shift() + phase_shift_bounds[1]);
            }
            if (has_astigmatism()) {
                const size_t index = 2 - !has_phase_shift();
                m_lower_bounds[index] = std::max(0., astigmatism_value() + astigmatism_value_bounds[0]);
                m_upper_bounds[index] = std::max(0., astigmatism_value() + astigmatism_value_bounds[1]);
                m_lower_bounds[index + 1] = astigmatism_angle() + astigmatism_angle_bounds[0];
                m_upper_bounds[index + 1] = astigmatism_angle() + astigmatism_angle_bounds[1];
            }
        }

        void set_abs_tolerance(
                f64 defocus_tolerance,
                f64 phase_shift_tolerance,
                f64 astigmatism_value_tolerance,
                f64 astigmatism_angle_tolerance
        ) {
            m_abs_tolerance[0] = defocus_tolerance;
            if (has_phase_shift())
                m_abs_tolerance[1] = phase_shift_tolerance;
            if (has_astigmatism()) {
                const size_t index = 2 - !has_phase_shift();
                m_abs_tolerance[index] = astigmatism_value_tolerance;
                m_abs_tolerance[index + 1] = astigmatism_angle_tolerance;
            }
        }

    public:
        [[nodiscard]] constexpr bool has_phase_shift() const noexcept { return m_fit_phase_shift; }
        [[nodiscard]] constexpr bool has_astigmatism() const noexcept { return m_fit_astigmatism; }

        [[nodiscard]] constexpr i64 ssize() const noexcept { return 1 + has_phase_shift() + 2 * has_astigmatism(); }
        [[nodiscard]] constexpr size_t size() const noexcept { return static_cast<size_t>(ssize()); }
        [[nodiscard]] constexpr f64* data() noexcept { return m_parameters.data(); }

        [[nodiscard]] constexpr Span<f64> span() noexcept { return {m_parameters.data(), ssize()}; }
        [[nodiscard]] constexpr Span<f64> lower_bounds() noexcept { return {m_lower_bounds.data(), ssize()}; }
        [[nodiscard]] constexpr Span<f64> upper_bounds() noexcept { return {m_upper_bounds.data(), ssize()}; }
        [[nodiscard]] constexpr Span<f64> abs_tolerance() noexcept { return {m_abs_tolerance.data(), ssize()}; }

    public: // safe access of the globals, whether they are fitted or not.
        [[nodiscard]] constexpr f64 defocus() const noexcept {
            return m_parameters[0];
        }

        [[nodiscard]] constexpr f64 phase_shift() const noexcept {
            return has_phase_shift() ? m_parameters[1] : 0;
        }

        [[nodiscard]] constexpr f64 astigmatism_value() const noexcept {
            return has_astigmatism() ? m_parameters[2 - !has_phase_shift()] : 0;
        }

        [[nodiscard]] constexpr f64 astigmatism_angle() const noexcept {
            return has_astigmatism() ? m_parameters[3 - !has_phase_shift()] : 0;
        }

    public: // Setters for compatibility with grid search.
        constexpr void set_defocus(f64 defocus) noexcept {
            m_parameters[0] = defocus;
        }

        constexpr void set_phase_shift(f64 phase_shift) noexcept {
            if (has_phase_shift())
                m_parameters[1] = phase_shift;
        }

    private:
        bool m_fit_phase_shift{};
        bool m_fit_astigmatism{};

        // defocus, (phase_shift), (astigmatism value, astigmatism angle)
        std::array<f64, 4> m_parameters{};
        std::array<f64, 4> m_lower_bounds{};
        std::array<f64, 4> m_upper_bounds{};
        std::array<f64, 4> m_abs_tolerance{};
    };

    class Helper {
    public:
        // Initialized by the constructor.
        View<const f32> patch_rfft_ps;
        Shape4<i64> patch_shape;
        const CTF::FittingRange* fitting_range{};
        CTFIsotropic64* ctf_iso;
        const Path* debug_directory{};

        Array<f32> buffer;
        View<f32> rotational_average;
        View<f32> rotational_average_weights;
        View<f32> simulated_ctf;
        CTFAnisotropic64 ctf_aniso{};
        bool is_rotational_average_ready{false};

        // Must be set before calling cost().
        Parameters parameters{};

    public:
        Helper(const View<f32>& patch_rfft_ps_,
               const CTF::FittingRange& fitting_range_,
               CTFIsotropic64& ctf_,
               const Path& debug_directory_
        ) :
                patch_rfft_ps(patch_rfft_ps_),
                patch_shape(Shape2<i64>(fitting_range_.original_logical_size).push_front<2>({1, 1})),
                fitting_range(&fitting_range_),
                ctf_iso(&ctf_),
                debug_directory(&debug_directory_) {

            buffer = noa::memory::empty<f32>(Shape4<i64>{3, 1, 1, fitting_range_.size});
            rotational_average = buffer.view().subregion(0);
            rotational_average_weights = buffer.view().subregion(1);
            simulated_ctf = buffer.view().subregion(2);
        }

        f64 cost(bool print = true) {
            const f64 defocus = parameters.defocus();
            const f64 phase_shift = parameters.phase_shift();

            // Update isotropic CTF with these parameters.
            ctf_iso->set_defocus(defocus);
            if (parameters.has_phase_shift())
                ctf_iso->set_phase_shift(phase_shift);

            // Compute the rotational average, if necessary.
            if (!is_rotational_average_ready) {
                if (parameters.has_astigmatism()) {
                    ctf_aniso = CTFAnisotropic64(
                            *ctf_iso,
                            parameters.astigmatism_value(),
                            parameters.astigmatism_angle());

                    noa::geometry::fft::rotational_average_anisotropic<fft::H2H>(
                            patch_rfft_ps, patch_shape, ctf_aniso,
                            rotational_average, rotational_average_weights,
                            /*frequency_range=*/ fitting_range->fftfreq.as<f32>(),
                            /*frequency_endpoint=*/ true);
                } else {
                    noa::geometry::fft::rotational_average<fft::H2H>(
                            patch_rfft_ps, patch_shape,
                            rotational_average, rotational_average_weights,
                            /*frequency_range=*/ fitting_range->fftfreq.as<f32>(),
                            /*frequency_endpoint=*/ true);

                    is_rotational_average_ready = true;
                }

                apply_cubic_bspline_1d(
                        rotational_average, rotational_average,
                        fitting_range->background, noa::minus_t{});
                noa::math::normalize(rotational_average, rotational_average, NormalizationMode::L2_NORM);
            }

            // Simulate isotropic ctf^2 within the fitting range.
            noa::signal::fft::ctf_isotropic<fft::H2H>(
                    simulated_ctf, Shape4<i64>{1, 1, 1, fitting_range->logical_size},
                    *ctf_iso, /*ctf_abs=*/ false, /*ctf_square=*/ true,
                    fitting_range->fftfreq.as<f32>(), /*endpoint=*/ true);
            noa::math::normalize(simulated_ctf, simulated_ctf, NormalizationMode::L2_NORM);

            if (!debug_directory->empty()) {
                save_vector_to_text(rotational_average, *debug_directory / "fit_average_rotational_average.txt");
                save_vector_to_text(simulated_ctf, *debug_directory / "fit_average_simulated_ctf.txt");
            }

            // Normalized cross-correlation.
            const f64 ncc = noa::reduce_binary( // or dot product
                    rotational_average, simulated_ctf,
                    f64{}, [](f32 lhs, f32 rhs) {
                        return static_cast<f64>(lhs) * static_cast<f64>(rhs);
                    }, noa::plus_t{}, {});

            if (print)
                log(ncc);
            return ncc;
        }

        void log(f64 ncc) const {
            const auto defocus = ctf_iso->defocus();
            const auto phase_shift = ctf_iso->phase_shift();

            using namespace noa::string;
            qn::Logger::trace(
                    "CTF average fitting: {}, {}ncc={:.8f}",

                    // Defocus.
                    parameters.has_astigmatism() ?
                    format("defocus=(value={:.4f}, astigmatism={:.4f}, angle={:.4f})",
                           defocus, parameters.astigmatism_value(),
                           parameters.astigmatism_angle()) :
                    format("defocus={:.8f}", defocus),

                    // Phase shift.
                    parameters.has_phase_shift() ? format("phase_shift={:.4f}, ", phase_shift) : "",

                    ncc);
        }
    };

    auto function_to_maximise_(u32, const f64*, f64* gradients, void* instance) -> f64 {
        auto& helper = *static_cast<Helper*>(instance);

        if (gradients) {
            for (auto& value: helper.parameters.span()) {
                const f32 initial_value = static_cast<f32>(value);
                const f32 delta = CentralFiniteDifference::delta(initial_value);

                value = static_cast<f64>(initial_value - delta);
                const f64 fx_minus_delta = helper.cost(false);
                value = static_cast<f64>(initial_value + delta);
                const f64 fx_plus_delta = helper.cost(false);

                value = static_cast<f64>(initial_value); // back to original value
                f64 gradient = CentralFiniteDifference::get(
                        fx_minus_delta, fx_plus_delta, static_cast<f64>(delta));

                if (noa::math::abs(gradient) < 1e-6) {
                    qn::Logger::trace("g={:.8f} -> 0", gradient);
                    gradient = 0;
                } else {
                    qn::Logger::trace("g={:.8f}", gradient);
                }

                *(gradients++) = gradient;
            }
        }

        return helper.cost();
    };
}

namespace qn {
    Array<f32> CTF::compute_average_patch_rfft_ps_(
            StackLoader& stack_loader,
            const MetadataStack& metadata,
            Vec2<f64> delta_z_range_nanometers,
            f64 max_tilt_for_average,
            const Path& debug_directory
    ) {
        const auto options = m_slice.options();
        const auto n_patches_max = m_patches_rfft.shape()[0];
        const auto patch_shape = Shape4<i64>{1, 1, m_patch_size, m_patch_size};

        // Load the slice and the patches of the slice in these arrays.
        const auto slice = m_slice.view();
        const auto all_patches_rfft = m_patches_rfft.view();
        const auto all_patches_origins = noa::memory::empty<Vec4<i32>>(n_patches_max, options);

        // Average power-spectrum.
        const auto patches_rfft_ps_average = noa::memory::empty<f32>(patch_shape.rfft(), options);
        const auto patches_rfft_ps_average_tmp = noa::memory::like(patches_rfft_ps_average);

        // The delta-z range should prevent regions with a too wide range of defocus to be averaged.
        // However, it seems that 1) even if the center is within the allowed z-range, the defocus ramp within
        // a single tilted patch can be bigger than the allowed z-range, 2) slices have different defoci,
        // so we don't want to average too much of them, 3) after a few images, the exposure is too big,
        // and it's not worth it. As such, it is useful to limit the average to low tilts.
        qn::Logger::trace("Creating average power spectrum from patches (z_range={}-nm, max_tilt={}-degrees):",
                          delta_z_range_nanometers, max_tilt_for_average);

        // Loading every single patch at once can be too much if the compute-device is a GPU.
        // Since this function is only called a few times, simply load the patches slice per slice.
        bool is_first{true};
        i64 index{0};
        i64 total{0};
        namespace ni = noa::indexing;

        for (const auto& slice_metadata: metadata.slices()) {
            if (std::abs(slice_metadata.angles[1]) > max_tilt_for_average)
                continue;

            // Get the patch origins.
            const std::vector<Vec4<i32>> patches_origins_vector = extract_patches_origins(
                    stack_loader.slice_shape(), slice_metadata, stack_loader.stack_spacing(),
                    Shape2<i64>(m_patch_size), Vec2<i64>(m_patch_step), delta_z_range_nanometers);
            if (patches_origins_vector.empty())
                continue;

            const auto n_patches = static_cast<i64>(patches_origins_vector.size());
            const auto patches_origins = all_patches_origins.subregion(ni::Ellipsis{}, ni::Slice(0, n_patches));
            View(patches_origins_vector.data(), n_patches).to(patches_origins);

            total += n_patches;
            qn::Logger::debug("index={:0>2.2f}, patches={:0>3}, total={:0>5}",
                              slice_metadata.angles[1], n_patches, total);

            // Prepare the patches for extraction.
            const auto patches_rfft = all_patches_rfft.subregion(ni::Slice(0, n_patches));
            const auto patches = noa::fft::alias_to_real(patches_rfft, patch_shape.set<0>(n_patches));

            // Extract the patches. Assume the slice is normalized and edges are tapered.
            stack_loader.read_slice(slice, slice_metadata.index_file);
            noa::memory::extract_subregions(slice, patches, patches_origins);
            noa::math::normalize_per_batch(patches, patches);

            if (!debug_directory.empty())
                noa::io::save(patches, debug_directory / noa::string::format("patches_{:>02}.mrc", index));

            // Compute the average power-spectrum of these tiles.
            noa::fft::r2c(patches, patches_rfft, noa::fft::Norm::FORWARD);
            noa::math::mean(patches_rfft, patches_rfft_ps_average_tmp, noa::abs_squared_t{});

            // Add the average to the other averages.
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
        return patches_rfft_ps_average;
    }

    void CTF::fit_ctf_to_patch_(
            Array<f32> patch_rfft_ps,
            FittingRange& fitting_range,
            CTFAnisotropic64& ctf_anisotropic,
            bool fit_phase_shift,
            bool fit_astigmatism,
            const Path& debug_directory
    ) {
        Timer timer;
        timer.start();

        // This entire function is CPU only.
        if (!patch_rfft_ps.device().is_cpu())
            patch_rfft_ps = patch_rfft_ps.to_cpu();

        // Check that these two match.
        NOA_ASSERT(patch_rfft_ps.shape()[2] == fitting_range.original_logical_size &&
                   patch_rfft_ps.shape()[3] == fitting_range.original_logical_size / 2 + 1);

        // Extract CTF to isotropic + astigmatism value and angle. This is just simpler to deal with.
        auto ctf_isotropic = CTFIsotropic64(ctf_anisotropic);
        auto astigmatism_value = ctf_anisotropic.defocus().astigmatism;
        auto astigmatism_angle = ctf_anisotropic.defocus().angle;
        NOA_ASSERT(ctf_isotropic.pixel_size() == fitting_range.spacing);

        // Set up.
        auto helper = Helper(patch_rfft_ps.view(), fitting_range, ctf_isotropic, debug_directory);
        auto background = Background(patch_rfft_ps.view(), fitting_range.original_logical_size);

        i64 evaluations{0};
        {
            // First (grid) search. No astigmatism.
            helper.parameters = Parameters(fit_phase_shift, /*fit_astigmatism=*/ false, ctf_isotropic);

            // Get initial background fitting. This is just an approximation of the background and simply fits
            // a smooth spline through the rotational average, cutting through the Thon rings. Similar results
            // could be achieved by subtracting a local mean or a strong gaussian blur version of the spectrum.
            // TODO If background exists, use it instead?
            background.fit_spline(fitting_range);

            const f64 max_phase_shift = fit_phase_shift ? noa::math::Constant<f64>::PI / 2 : 0;
            f64 score = noa::math::Limits<f64>::lowest();
            Vec2<f64> best_values;

            GridSearch2D grid_search(
                    /* phase shift: */ 0., max_phase_shift, 0.05,
                    /* defocus: */     0.4, 8., 0.02);

            grid_search.for_each([&](f64 phase_shift, f64 defocus) {
                auto& parameters = helper.parameters;
                parameters.set_phase_shift(phase_shift);
                parameters.set_defocus(defocus);
                const f64 ncc = function_to_maximise_(0, parameters.data(), nullptr, &helper);
                if (ncc > score) {
                    best_values = {defocus, phase_shift};
                    score = ncc;
                }
                ++evaluations;
            });

            // Update ctf for next optimization.
            ctf_isotropic.set_defocus(best_values[0]);
            ctf_isotropic.set_phase_shift(best_values[1]);
        }

        f64 ncc{};
        {
            // Local optimization to polish to optimum and astigmatism.
            constexpr auto PI = noa::math::Constant<f64>::PI;
            constexpr auto PI_EPSILON = PI / 32;
            helper.parameters = Parameters(
                    fit_phase_shift, fit_astigmatism, ctf_isotropic, astigmatism_value, astigmatism_angle);
            helper.parameters.set_relative_bounds(
                    /*defocus_bounds=*/ {-0.25, 0.25},
                    /*phase_shift_bounds=*/ {-PI / 6, PI / 6},
                    /*astigmatism_value_bounds=*/ {0, 0.2},
                    /*astigmatism_angle_bounds=*/ {-PI / 2 - PI_EPSILON, PI / 2 + PI_EPSILON});
            helper.parameters.set_abs_tolerance(
                    /*defocus_tolerance=*/ 5e-4,
                    /*phase_shift_tolerance=*/ noa::math::deg2rad(0.25),
                    /*astigmatism_value_tolerance=*/ 5e-4,
                    /*astigmatism_angle_tolerance=*/ noa::math::deg2rad(0.25));

            Optimizer optimizer(NLOPT_LN_SBPLX, helper.parameters.ssize());
            optimizer.set_max_objective(function_to_maximise_, &helper);
            optimizer.set_bounds(helper.parameters.lower_bounds().data(),
                                 helper.parameters.upper_bounds().data());
            optimizer.set_x_tolerance_abs(helper.parameters.abs_tolerance().data());

            for ([[maybe_unused]] auto _: irange(2 + fit_astigmatism)) {
                // Use a more accurate background now that we have a good idea of the ctf parameters.
                background.fit_accurate(
                        fitting_range, ctf_isotropic, fit_astigmatism, astigmatism_value, astigmatism_angle);

                // We updated the background, so recompute the rotational average.
                helper.is_rotational_average_ready = false;

                ncc = optimizer.optimize(helper.parameters.data());
                evaluations += optimizer.number_of_evaluations();

                // Update outputs (note that the fitting range has its background already updated).
                ctf_isotropic.set_defocus(helper.parameters.defocus());
                ctf_isotropic.set_phase_shift(helper.parameters.phase_shift());
                if (fit_astigmatism) {
                    astigmatism_value = helper.parameters.astigmatism_value();
                    astigmatism_angle = helper.parameters.astigmatism_angle();
                }
            }
        }
        helper.log(ncc);

        // Update output ctf.
        // If astigmatism isn't fitted, these are just the input values.
        ctf_anisotropic = CTFAnisotropic64(ctf_isotropic, astigmatism_value, astigmatism_angle);
    }
}
