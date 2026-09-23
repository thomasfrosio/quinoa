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
    };

    class ProjectionMatcher {
    public:
        ProjectionMatcher() = default;
        ProjectionMatcher(isize n_slices, const Shape2& shape, f64 max_tilt_difference);

        [[nodiscard]] auto shared_buffer_bytes() const -> isize;
        void set_shared_buffer(const View<std::byte>& shared_buffer);

        auto update_shifts(
            const View<f32>& stack,
            Metadata& metadata,
            const ProjectionMatchingParameters& parameters
        ) const -> f64;

        ~ProjectionMatcher();
    };
}
