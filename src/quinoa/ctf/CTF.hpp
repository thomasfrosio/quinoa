#pragma once

#include "quinoa/Metadata.hpp"
#include "quinoa/Types.hpp"
#include "quinoa/ctf/Baseline.hpp"
#include "quinoa/ctf/Grid.hpp"
#include "quinoa/ctf/Patches.hpp"

namespace qn::ctf {
    /// Range-like type designed to iterate through a |CTF| curve and efficiently analyze its gradient.
    /// This is used when we need to know where the |CTF| vertexes (peaks and zeros) are and sample a spectrum
    /// at these locations.
    class Simulate {
    public:
        constexpr static usize SIMULATED_LOGICAL_SIZE = 8192;
        constexpr static f64 SIMULATED_FREQ_STEP = 1 / static_cast<f64>(SIMULATED_LOGICAL_SIZE);

    public:
        /// Sample the spectrum at the given fftfreq.
        template<nt::real T>
        [[nodiscard]] static auto sample_at(
            SpanContiguous<T> spectrum,
            const Vec<f64, 2>& fftfreq_range,
            f64 fftfreq
        ) -> f64 {
            const auto spectrum_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
            const auto spectrum_frequency = (fftfreq - fftfreq_range[0]) / spectrum_step;
            const auto floored_f64 = std::floor(spectrum_frequency);
            const auto floored_i64 = static_cast<isize>(floored_f64);
            const auto fraction = spectrum_frequency - floored_f64;

            // Lerp.
            const auto index_0 = noa::index_at<noa::Border::REFLECT>(floored_i64 + 0, spectrum.ssize());
            const auto index_1 = noa::index_at<noa::Border::REFLECT>(floored_i64 + 1, spectrum.ssize());
            const auto interpolated =
                static_cast<f64>(spectrum[index_0]) * (1 - fraction) +
                static_cast<f64>(spectrum[index_1]) * fraction;
            return interpolated;
        }

    public:
        constexpr explicit Simulate(
            const CTFIsotropic64& ctf,
            const Vec<f64, 2>& fftfreq_range
        ) :
            m_ctf{&ctf},
            m_simulated_range_index{noa::round(fftfreq_range * SIMULATED_LOGICAL_SIZE).as<isize>()}
        {}

    public: // range-for loop support
        struct Iterator {
        public:
            [[nodiscard]] constexpr auto fftfreq() const -> f64 {
                return static_cast<f64>(m_index) * SIMULATED_FREQ_STEP;
            }
            [[nodiscard]] constexpr auto fftfreq_step() const -> f64 {
                return SIMULATED_FREQ_STEP;
            }

            /// Given the current position i, retrieve the slope [i-1,i] and [i,i+1].
            [[nodiscard]] constexpr auto slopes() const -> Vec<f64, 2> {
                const f64 ctf_value_0 = circular_buffer_get_(-2); // i - 1
                const f64 ctf_value_1 = circular_buffer_get_(-1); // i
                const f64 ctf_value_2 = circular_buffer_get_( 0); // i + 1
                return {
                    ctf_value_1 - ctf_value_0,
                    ctf_value_2 - ctf_value_1
                };
            }

            [[nodiscard]] constexpr auto is_ctf_zero() const -> bool {
                auto [slope_0, slope_1] = slopes();
                return slope_0 < 0 and slope_1 >= 0;
            }
            [[nodiscard]] constexpr auto is_ctf_peak() const -> bool {
                auto [slope_0, slope_1] = slopes();
                return slope_0 > 0 and slope_1 <= 0;
            }
            [[nodiscard]] constexpr auto is_ctf_vertex() const -> bool {
                auto [slope_0, slope_1] = slopes();
                return slope_0 * slope_1 < 0.; // if different sign
            }

        public: // minimal range-for support
            constexpr explicit Iterator(const Simulate* parent, isize index) noexcept: m_parent{parent}, m_index{index} {
                circular_buffer_next_(std::max(isize{0}, m_index - 1)); // prevent negative frequencies
                circular_buffer_next_(m_index);
                circular_buffer_next_(m_index + 1);
            }
            constexpr bool operator!=(const isize& end) const noexcept { return m_index != end; }
            constexpr auto operator*() const noexcept -> const Iterator& { return *this; }
            constexpr Iterator& operator++() noexcept {
                ++m_index;
                // Sample one ahead, so that when this returns, we have:
                // m_index-1 at circular_buffer_get_(offset: -2)
                // m_index   at circular_buffer_get_(offset: -1)
                // m_index+1 at circular_buffer_get_(offset:  0)
                circular_buffer_next_(m_index + 1);
                return *this;
            }

        private:
            const Simulate* m_parent;
            isize m_index;

            // Circular buffer.
            constexpr static isize CIRCULAR_BUFFER_SIZE = 3;
            Vec<f64, CIRCULAR_BUFFER_SIZE> m_circular_buffer{};
            isize m_circular_index{};

            constexpr void circular_buffer_next_(isize simulated_index) {
                const auto fftfreq = static_cast<f64>(simulated_index) * SIMULATED_FREQ_STEP;
                m_circular_index = (m_circular_index + 1) % CIRCULAR_BUFFER_SIZE;
                m_circular_buffer[static_cast<usize>(m_circular_index)] = std::abs(m_parent->m_ctf->value_at(fftfreq));
            }
            [[nodiscard]] constexpr auto circular_buffer_get_(isize offset) const -> f64 {
                auto current = (m_circular_index + CIRCULAR_BUFFER_SIZE + offset) % CIRCULAR_BUFFER_SIZE;
                return m_circular_buffer[static_cast<usize>(current)];
            }
        };

        [[nodiscard]] constexpr auto begin() const -> Iterator { return Iterator(this, m_simulated_range_index[0]); }
        [[nodiscard]] constexpr auto end() const -> isize { return m_simulated_range_index[1]; }

    private:
        const CTFIsotropic64* m_ctf{};
        Vec<isize, 2> m_simulated_range_index{};
    };

    /// Thickness modulation step function, from -1 to 1.
    /// The smoothness controls the smoothness of the steps. The on/off difference is not that significant,
    /// but the logic is that for autotunning purposes, we increase the smoothness to exclude a larger range near
    /// the sinc nodes, and for the scoring function, use sharper steps to better judge where the phase-flip occurs.
    template<bool SMOOTH>
    struct ThicknessModulation {
        f64 wavelength{};
        f64 spacing{};
        f64 thickness{};

        /// Samples the thickness-modulation curve at that frequency. Between nodes, this is 1 or -1, meaning that the
        /// CTF oscillations are either unchanged or flipped. The transition regions, aka nodes, are controlled by a
        /// sin to essentially smooth out this transition.
        [[nodiscard]] auto sample_at(f64 fftfreq) const -> f64 {
            constexpr f64 PI = noa::Constant<f64>::PI;
            constexpr f64 SIN_PI_10 = 0.3090169944; // std::sin(PI / 10.)
            constexpr f64 SIN_PI_22 = 0.1423148383; // std::sin(PI / 22.)
            constexpr f64 FACTOR = SMOOTH ? 5. : 11;
            constexpr f64 SIN_PI_FACTOR = SMOOTH ? SIN_PI_10 : SIN_PI_22;

            fftfreq /= spacing;
            const auto c = PI * thickness * wavelength;
            const auto p = c * fftfreq * fftfreq;
            if (p < PI / 2)
                return 1.; // thickness==0 goes here, low frequencies before sin decay to zero

            const auto s = std::sin(p);
            if (std::abs(s) > SIN_PI_FACTOR)
                return s >= 0 ? 1. : -1.; // between nodes
            return std::sin(p * FACTOR); // smooth transition to a node
        }

        template<i32 N_NODES = 5>
        [[nodiscard]] auto is_fftfreq_range_containing_node(const Vec<f64, 2>& fftfreq_range) const -> bool {
            for (i32 i{1}; i < N_NODES + 1; ++i) {
                const auto k = static_cast<f64>(i);
                const auto node_fftfreq = spacing * std::sqrt(k / (wavelength * thickness));
                if (fftfreq_range[0] <= node_fftfreq and node_fftfreq <= fftfreq_range[1])
                    return true;
            }
            return false;
        }

        /// Samples the background on 1d spectrum.
        void sample(
            SpanContiguous<f32> spectrum,
            const Vec<f64, 2>& fftfreq_range
        ) const {
            const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.ssize() - 1);
            for (i64 i{}; i < spectrum.ssize(); ++i) {
                const auto fftfreq = static_cast<f64>(i) * fftfreq_step + fftfreq_range[0];
                spectrum[i] = static_cast<f32>(sample_at(fftfreq));
            }
        }

        void sample(
            const View<f32>& spectrum,
            const Vec<f64, 2>& fftfreq_range
        ) const {
            auto [b, d, h, w] = spectrum.shape();
            check(b == 1 and d == 1 and h == 1);
            sample(spectrum.reinterpret_as_cpu().span_1d(), fftfreq_range);
        }
    };

    /// Computes the minimum logical size necessary for the Thon-rings to not alias.
    constexpr auto aliasing_free_size(
        const CTFIsotropic64& ctf,
        const Vec<f64, 2>& fftfreq_range,
        f64 minimum_pixels_between_vertexes = 1.8
    ) -> i64 {
        // Warp has a similar function, but it uses the gradient of the curve to compute the minimum size.
        // Honestly, I don't understand it, so I'm using a more intuitive approach (at least to me).
        // By default, it is slightly more generous than what Warp gives you, but this can be controlled with
        // minimum_pixels_between_vertexes. I think CTFFIND uses a similar approach, but with a
        // minimum_pixels_between_vertexes of 4, which seems too much.

        // Collect the smallest distance between two vertexes.
        f64 fftfreq_distance{1};
        f64 previous_vertex{};
        for (auto& e: Simulate(ctf, fftfreq_range)) {
            if (e.is_ctf_vertex()) {
                auto current_fftfreq = e.fftfreq();
                fftfreq_distance = std::min(fftfreq_distance, current_fftfreq - previous_vertex);
                previous_vertex = current_fftfreq;
            }
        }

        // Deduce the logical size where this distance ends up being at least the given number of pixels.
        const auto range = fftfreq_range[1] - fftfreq_range[0];
        const auto minimum_spectrum_size = minimum_pixels_between_vertexes * range / fftfreq_distance;
        auto minimum_logical_size = static_cast<i64>(std::ceil(minimum_spectrum_size)) * 2 + 1;
        if (noa::is_odd(minimum_logical_size))
            minimum_logical_size += 1;
        return minimum_logical_size;
    }

    template<nt::almost_any_of<f32, f64> T, typename B = Empty, typename M = Empty>
    constexpr auto zero_normalized_cross_correlation(
        SpanContiguous<T> spectrum,
        const CTFIsotropic64& ctf,
        const Vec<f64, 2>& fftfreq_range,
        const Vec<f64, 2>& fitting_range,
        const B& baseline = B{},
        const M& thickness_modulation = M{}
    ) -> f64 {
        const auto n = spectrum.ssize();
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(n - 1);

        // Only loop through the fitting range.
        const auto indices = noa::round((fitting_range - fftfreq_range[0]) / fftfreq_step).as<isize>();
        const auto start = std::max(indices[0], isize{});
        const auto end = std::min(indices[1] + 1, n);

        // Single-pass ZNCC.
        f64 sum_lhs = 0.0;
        f64 sum_rhs = 0.0;
        f64 sum_lhs_lhs = 0.0;
        f64 sum_rhs_rhs = 0.0;
        f64 sum_lhs_rhs = 0.0;

        for (isize i = start; i < end; ++i) {
            const auto fftfreq = fftfreq_range[0] + static_cast<f64>(i) * fftfreq_step;

            // Get the simulated (CTF * envelope)^2.
            // The baseline goes through the Thon rings, so the CTF should be centered on zero.
            // Note that the envelope is already applied by ctf.value_at(), but here we subtract it to center
            // the rings at zero. If the B-factor is 0, this simply subtracts 0.5.
            // Now that the curve is zero-centered, we can simply multiply the thickness modulation.
            auto lhs = ctf.value_at(fftfreq); // FIXME use phase_at and envelope_at, don't square the bfactor
            lhs *= lhs;
            auto envelope = ctf.envelope_at(fftfreq);
            envelope *= envelope;
            lhs -= envelope / 2; // [0,1] -> [-0.5, 0.5]

            // Thickness modulation.
            // We do not use the cos-weighted modulation curve from McMullan et al. 2015. Instead, we scale the sinc
            // oscillations between [-1,1] to not downweight the CTF curve as the frequency increases. This idea is
            // taken from CTFFIND5, but we directly multiply the modulation curve to the classic zero-centered CTF^2
            // curve (which is equivalent to the "rounded-square" mode in CTFFIND5). The resulting curve is identical
            // to the classic CTF^2 curve, except after every other node where the CTF^2 oscillations are simply
            // out-of-phase compared to the classic curve. Note that the B-factor/envelope is still applied and has
            // the same effect as with the classic model, so we can still downweight higher frequencies if we want to.
            f64 modulation;
            if constexpr (nt::span_nd<M, 1>)
                modulation = static_cast<f64>(thickness_modulation[i]);
            else if constexpr (nt::any_of<M, ThicknessModulation<true>, ThicknessModulation<false>>)
                modulation = thickness_modulation.sample_at(fftfreq);
            else
                modulation = 1; // the thickness of the sample has no effect

            // The thickness modulation curve implies that certain regions have no Thon rings for us to fit.
            // CTFFIND5 excludes these regions (|modulation| < 0.9), but I don't think this would help the optimization.
            // Granted, I haven't tested it, but if the experimental spectrum has a Thon ring where a node is, this
            // would mean that the thickness is probably wrong, therefore, we should pay the cost.
            // if (std::abs(modulation) < 0.9)
            //     continue;
            lhs *= modulation;

            // Get the baseline-subtracted (i.e. zero-centered) spectrum.
            auto rhs = static_cast<f64>(spectrum[i]);
            if constexpr (nt::any_of<B, Baseline>)
                rhs -= baseline.sample_at(fftfreq); // baseline is sampled on-the-fly
            else if constexpr (nt::span_nd<B, 1>)
                rhs -= static_cast<f64>(baseline[i]); // baseline is already sampled
            else if constexpr (not nt::empty<B>) // if empty, the spectrum is assumed to be already corrected
                static_assert(nt::always_false<B>, "Unknown baseline type");

            // ZNCC.
            sum_lhs += lhs;
            sum_rhs += rhs;
            sum_lhs_lhs += lhs * lhs;
            sum_rhs_rhs += rhs * rhs;
            sum_lhs_rhs += lhs * rhs;
        }

        const f64 count = static_cast<f64>(end - start);
        const f64 denominator_lhs = sum_lhs_lhs - sum_lhs * sum_lhs / count;
        const f64 denominator_rhs = sum_rhs_rhs - sum_rhs * sum_rhs / count;
        f64 denominator = denominator_lhs * denominator_rhs;
        if (denominator <= 0.0)
            return 0.0;
        const f64 numerator = sum_lhs_rhs - sum_lhs * sum_rhs / count;
        return numerator / std::sqrt(denominator);
    }
}

namespace qn::ctf {
    struct FitInitialOptions {
        i64 n_slices_to_average;
        bool fit_phase_shift{};
        Path output_directory{};
    };
    struct FitInitialResults {
        f64 defocus;
        f64 phase_shift;
        Vec<f64, 2> fitting_range;
    };
    auto initial_fit(
        const Metadata& metadata,
        const Grid& grid,
        const Patches& patches,
        const FitInitialOptions& options
    ) -> FitInitialResults;

    struct FitCoarseOptions {
        Vec<f64, 2> initial_fitting_range;
        bool exclude_bad_images{};
        bool first_image_has_higher_exposure{};
        bool fit_phase_shift{};
        bool check_defocus_gradient{};
        Path output_directory{};
    };
    void coarse_fit(
        Metadata& metadata,
        const Grid& grid,
        const Patches& patches,
        const FitCoarseOptions& options
    );

    struct FitSettings {
        Device compute_device;
        Path output_directory;

        f64 patch_size_ang;
        isize patch_size_min_pix;
        isize nb_images_in_initial_average;
        isize max_nb_high_resolution_recovery;
        Vec<isize, 2> astigmatism_tilt_resolution;
        Vec<isize, 2> phase_shift_time_resolution;
        Vec<f64, 2> resolution_range;
        bool fit_phase_shift;
        bool fit_astigmatism;
        bool fit_thickness;
        bool check_defocus_gradient;

        // Refine:
        bool fit_rotation;
        bool fit_tilt;
        bool fit_pitch;
    };
    void fit(
        const Path& stack_filename,
        Metadata& metadata,
        const FitSettings& settings
    );
}
