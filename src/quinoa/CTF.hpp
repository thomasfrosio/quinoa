#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/CTFBackground.hpp"
#include "quinoa/CTFGrid.hpp"
#include "quinoa/CTFPatches.hpp"

namespace qn::ctf {
    constexpr auto aliasing_free_size(
        const ns::CTFIsotropic<f64>& ctf,
        const Vec<f64, 2>& fftfreq_range
    ) -> i64 {
        constexpr i64 SIMULATED_SIZE = 1000;
        const f64 fftfreq_step = noa::Linspace<f64>::from_vec(fftfreq_range).for_size(SIMULATED_SIZE).step;

        f64 previous_phase = ctf.phase_at(fftfreq_range[0]);
        f64 max_gradient{};
        for (i64 i{}; i < SIMULATED_SIZE - 1; ++i) {
            const f64 current_fftfreq = static_cast<f64>(i + 1) * fftfreq_step + fftfreq_range[0];
            const f64 current_phase = ctf.phase_at(current_fftfreq);
            const f64 gradient = current_phase - previous_phase;
            max_gradient = std::max(max_gradient, std::abs(gradient));
            previous_phase = current_phase;
        }
        // This is taken from Warp...
        return static_cast<i64>(std::ceil(max_gradient / 0.5 * static_cast<f64>(SIMULATED_SIZE) * 1.1));
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

    /// Normalized cross-correlation using auto-correlation.
    template<bool COARSE_BACKGROUND = false, typename B = Empty>
    constexpr auto normalized_cross_correlation(
        SpanContiguous<const f32> spectrum,
        const ns::CTFIsotropic<f64>& ctf,
        const Vec<f64, 2>& fftfreq_range,
        const Vec<f64, 2>& fitting_range,
        const B& background = B{}
    ) -> f64 {
        const auto n = spectrum.ssize();
        const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(n - 1);

        // Only loop through the fitting range.
        const auto indices = noa::round((fitting_range - fftfreq_range[0]) / fftfreq_step).as<i64>();
        const auto start = noa::max(indices[0], i64{});
        const auto end = noa::min(indices[1] + 1, n);

        f64 ncc{};     // cross-correlation
        f64 ncc_lhs{}; // auto-correlation
        f64 ncc_rhs{}; // auto-correlation

        for (i64 i = start; i < end; ++i) {
            const auto fftfreq = fftfreq_range[0] + static_cast<f64>(i) * fftfreq_step;

            auto lhs = static_cast<f64>(spectrum[i]);
            if constexpr (nt::same_as<Background, B>)
                lhs -= background.sample_at(fftfreq); // sample background on-the-fly
            else if constexpr (nt::span_nd<B, 1>)
                lhs -= static_cast<f64>(background[i]); // background is already sampled
            else if constexpr (not nt::empty<B>)
                static_assert(nt::always_false<B>); // background is already subtracted

            auto rhs = ctf.value_at(fftfreq);
            rhs *= rhs;
            if constexpr (COARSE_BACKGROUND) {
                // The coarse baseline goes through the Thon rings,
                // so apply the equivalent on the simulated CTF.
                auto envelope = ctf.envelope_at(fftfreq);
                envelope *= envelope;
                rhs -= envelope;
            }

            ncc += lhs * rhs;
            ncc_lhs += lhs * lhs;
            ncc_rhs += rhs * rhs;
        }
        ncc /= std::sqrt(ncc_lhs) * std::sqrt(ncc_rhs);
        return ncc;
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
