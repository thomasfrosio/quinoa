#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/CTFBaseline.hpp"
#include "quinoa/CTFGrid.hpp"
#include "quinoa/CTFPatches.hpp"
#include "quinoa/CTFSimulate.hpp"
#include "quinoa/Thickness.hpp"

namespace qn::ctf {
    /// Computes the minimum logical size necessary for the Thon-rings to not alias.
    constexpr auto aliasing_free_size(
        const ns::CTFIsotropic<f64>& ctf,
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

    constexpr auto power_spectrum_bfactor_at(
        ns::CTFIsotropic<f64> ctf,
        f64 fftfreq,
        f64 weight
    ) {
        f64 bfactor{};
        f64 step = weight < 1. ? -1. : 1.;
        for (i32 i{}; i < 500; ++i) {
            ctf.set_bfactor(bfactor);
            auto iweight = ctf.envelope_at(fftfreq);
            iweight *= iweight;
            if (iweight <= weight)
                break;
            bfactor += step;
        }
        return bfactor;
    }

    template<nt::almost_any_of<f32, f64> T, typename B = Empty, typename M = Empty>
    constexpr auto zero_normalized_cross_correlation(
        SpanContiguous<T> spectrum,
        const ns::CTFIsotropic<f64>& ctf,
        const Vec<f64, 2>& fftfreq_range,
        const Vec<f64, 2>& fitting_range,
        const B& baseline = B{},
        const M& thickness_modulation = M{}
    ) -> f64 {
        const auto n = spectrum.ssize();
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(n - 1);

        // Only loop through the fitting range.
        const auto indices = noa::round((fitting_range - fftfreq_range[0]) / fftfreq_step).as<i64>();
        const auto start = std::max(indices[0], i64{});
        const auto end = std::min(indices[1] + 1, n);

        // Single-pass ZNCC.
        f64 sum_lhs = 0.0;
        f64 sum_rhs = 0.0;
        f64 sum_lhs_lhs = 0.0;
        f64 sum_rhs_rhs = 0.0;
        f64 sum_lhs_rhs = 0.0;

        for (i64 i = start; i < end; ++i) {
            const auto fftfreq = fftfreq_range[0] + static_cast<f64>(i) * fftfreq_step;

            // Get the simulated (CTF * envelope)^2.
            // The baseline goes through the Thon rings, so the CTF should be centered on zero.
            // Note that the envelope is already applied by ctf.value_at(), but here we subtract it to center
            // the rings at zero. If the B-factor is 0, this simply subtracts 0.5.
            // Now that the curve is zero-centered, we can simply multiply the thickness modulation.
            auto lhs = ctf.value_at(fftfreq);
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
            else if constexpr (nt::same_as<M, ThicknessModulation>)
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
    auto initial_fit(
        const Grid& grid,
        const Patches& patches,
        const MetadataStack& metadata,
        ns::CTFIsotropic<f64>& ctf, // .defocus and .phase_shift
        const FitInitialOptions& options
    ) -> Vec<f64, 2>;

    struct FitCoarseOptions {
        Vec<f64, 2> initial_fitting_range;
        bool exclude_bad_images{};
        bool first_image_has_higher_exposure;
        bool fit_phase_shift{};
        bool has_user_rotation;
        Path output_directory{};
    };
    void coarse_fit(
        const Grid& grid,
        Patches patches, // images can be removed
        ns::CTFIsotropic<f64>& ctf, // .defocus (and .phase_shift) updated
        MetadataStack& metadata, // images can be removed, .defocus (and .phase_shift) updated, angles[0] may be flipped
        const FitCoarseOptions& options
    );

    struct FitRefineOptions {
        bool fit_rotation{};
        bool fit_tilt{};
        bool fit_pitch{};
        bool fit_phase_shift{};
        bool fit_astigmatism{};
        f64 thickness{};
        Path output_directory{};
    };
    void refine_fit(
        MetadataStack& metadata,
        const Grid& grid,
        const Patches& patches,
        ns::CTFIsotropic<f64>& isotropic_ctf,
        const FitRefineOptions& fit
    );
}
