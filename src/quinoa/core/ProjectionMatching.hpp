#pragma once

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/CommonArea.hpp"
#include "quinoa/core/CubicGrid.hpp"

// The projection-matching alignment
// One "iteration" of projection matching consists in an entire pass through the stack, where every slice (excluding
// the global reference) is aligned against its projected-reference. As such, the projected-reference of the last
// slice contains every other slice in the stack (the target is never included in its projected-reference).
// The order in which the slices are aligned is set by the order of the slices in the metadata (e.g., absolute tilt
// or exposure order).
//
// One "slice-alignment" consists of computing the cross-correlation (CC) of a projected-reference and its target.
// As such, an iteration contains as many slice-alignments as there are slices to align. The CC peak shift from the
// correlation is used for the next slice-alignment: it is added to the target, which is added to the list of
// references for the next slice-alignment. The normalized CC peak values of every slice-alignment are accumulated, giving a
// "score" for each iteration. This score is the score passed to the optimizer in charge of finding the best
// rotation for the stack. When the optimizer finds the best score (i.e., the best iteration), we can then simply
// retrieve the corresponding shifts (from the slice-alignments) that were used during this iteration.


namespace qn {
    struct ProjectionMatchingParameters {
        f64 smooth_edge_percent{0.1};
        bool use_estimated_gradients{true};

        f64 fftfreq_sinc = -1;
        f64 fftfreq_blackman = -1;
        f64 fftfreq_z_sinc = -1;
        f64 fftfreq_z_blackman = -1;

        Vec2<f32> highpass_filter{};
        Vec2<f32> lowpass_filter{};

        Path debug_directory;
    };

    class ProjectionMatching {
    public: // user interface
        ProjectionMatching() = default;

        ProjectionMatching(
                i64 n_slices,
                const Shape2<i64>& shape,
                Device compute_device,
                Allocator allocator = Allocator::DEFAULT_ASYNC
        );

        void update(
                const View<f32>& stack,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters,
                const CTFAnisotropic64& average_ctf,
                MetadataStack& metadata // is updated
        );

    private:
        Array<c32> m_slices_padded_rfft;
        Array<f32> m_weights_padded_rfft;
        Array<f32> m_two_slices;
    };
}
