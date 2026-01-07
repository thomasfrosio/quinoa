#pragma once

#include <noa/Runtime.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct ProjectionMatchingParameters {
        bool correct_ctf{false}; // TODO

        f64 shift_tolerance{0.001};
        f64 max_tilt_difference{21};

        f64 smooth_edge_percent{0.1};

        nx::WindowedSinc insertion_sinc{};
        nx::WindowedSinc extraction_sinc{};

        ns::Bandpass bandpass;
        Path debug_directory;
    };

    class ProjectionMatcher {
    public:
        ProjectionMatcher(
            isize n_slices,
            const Shape2& shape,
            Device device
        );

        void update_shifts(
            const View<f32>& stack,
            Metadata::Stack& metadata,
            const ProjectionMatchingParameters& parameters
        ) const;

        [[nodiscard]] auto spectrum_size() const { return m_references_padded_rfft.shape().height(); }
        [[nodiscard]] auto spectrum_shape() const { return Shape{spectrum_size(), spectrum_size()}; }

    private:
        Array<f32> m_reference_padded;
        Array<c32> m_references_padded_rfft;
        Array<c32> m_projected_padded_rfft;
        Array<f32> m_weights_padded_rfft;

        Array<f32> m_target_and_projected;
        Array<c32> m_target_and_projected_rfft;

        Array<f32> m_image_buffer;
        Array<f32> m_xmap;
        Array<f32> m_xmap_centered;
    };

    void update_shifts2(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const ProjectionMatchingParameters& parameters
    );
}
