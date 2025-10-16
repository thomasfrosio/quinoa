#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct EstimateSampleThicknessParameters {
        f64 resolution;
        Device compute_device;
        Allocator allocator;
        Path output_directory;
    };

    auto estimate_sample_thickness(
        const Path& stack_filename,
        MetadataStack& metadata, // updated: .shifts
        const EstimateSampleThicknessParameters& parameters
    ) -> f64; // nm

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

            fftfreq /= spacing;
            const auto c = PI * thickness * wavelength;
            const auto p = c * fftfreq * fftfreq;
            if (p < PI / 2)
                return 1.; // thickness==0 goes here, low frequencies before sin decay to zero

            const auto s = std::sin(p);
            if (std::abs(s) > SIN_PI_10)
                return s >= 0 ? 1. : -1.; // between nodes
            return std::sin(p * 5); // smooth transition to a node
        }

        /// Whether the thickness modulation varies (below a relative threshold) within the given frequency range.
        /// This is used for the tuning of the fitting range.
        [[nodiscard]] auto is_frequency_range_within_node_transition(
            const Vec<f64, 2>& fftfreq_range,
            f64 relative_threshold = 0.9
        ) const -> bool {
            (void)fftfreq_range;
            (void)relative_threshold;
            return false; // FIXME
        }

        /// Samples the background on 1d spectrum.
        void sample(
            SpanContiguous<f32> spectrum,
            const Vec<f64, 2>& fftfreq_range
        ) const;
        void sample(
            const View<f32>& spectrum,
            const Vec<f64, 2>& fftfreq_range
        ) const;
    };
}
