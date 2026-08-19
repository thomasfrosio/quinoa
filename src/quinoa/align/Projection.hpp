#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct ProjectionMatchingParameters {
        bool correct_ctf{false};
        bool update_metadata{true};
        bool compute_score{true};

        f64 shift_tolerance{0.001};
        f64 max_tilt_difference{21};

        f64 smooth_edge_percent{0.1};

        nx::WindowedSinc insertion_sinc{};
        nx::WindowedSinc extraction_sinc{};
    };

    class ProjectionMatcher {
    public:
        ProjectionMatcher() = default;

        ProjectionMatcher(
            isize n_slices,
            const Shape2& shape,
            Device device
        );

        auto update_shifts(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            const ProjectionMatchingParameters& parameters
        ) const -> f64;

        [[nodiscard]] auto spectrum_size() const -> isize;

        ~ProjectionMatcher();
    };
}
