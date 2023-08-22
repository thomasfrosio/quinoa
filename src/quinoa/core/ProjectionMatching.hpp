#pragma once

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/CommonArea.hpp"
#include "quinoa/core/CubicGrid.hpp"


// One "iteration" of projection matching consists in an entire pass through the stack, where every slice (excluding
// the global reference) is aligned against its projected reference. As such, the projected reference of the last
// slice contains every other slice in the stack (the target is never included in its projected-reference).
// The order in which the slices are aligned is set by the order of the slices in the metadata (e.g., absolute tilt
// or exposure order).
//
// One "slice-alignment" consists of computing the cross-correlation (CC) of a projected-reference and its target.
// As such, an iteration contains as many slice-alignments as there are slices to align. The CC peak shift from the
// correlation is used for the next slice-alignment: it is added to the target, which is added to the list of
// references for the next slice-alignment. The CC peak values of every slice-alignment are accumulated, giving a
// "score" for each iteration. This score is the score passed to the optimizer in charge of finding the best
// rotation for the stack. When the optimizer finds the best score (i.e., the best iteration), we can then simply
// retrieve the corresponding shifts (from the slice-alignments) that were used during this iteration.
//
// One important optimization regarding the references. Once aligned, a given slice can be added to the list of
// references for the next targets. Therefore, we can simply add the contribution of the new reference to the old
// one using the "add_to_output" option of the Fourier insertion. As such, for each slice-alignment, we only need
// to Fourier insert one slice.

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

    public: // "private" interface
        void reset_() const;

        [[nodiscard]] auto align_next_slice_(
                const View<const f32>& stack,
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) const -> std::pair<Vec2<f64>, f64>;

    private:
        [[nodiscard]] constexpr i64 size_padded_() const noexcept {
            return m_reference_and_target_padded_rfft.shape()[2];
        }

        [[nodiscard]] constexpr Shape2<i64> shape_padded_() const noexcept {
            return {size_padded_(), size_padded_()};
        }

        void compute_next_projection_(
                const View<const f32>& stack,
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata,
                const CommonArea& common_area,
                const ProjectionMatchingParameters& parameters
        ) const;

        // Compute the sampling function of both the new reference view and the target.
        // The current sampling function simply contains the exposure filter and the average CTF
        // of the slice. The average CTF is not ideal (it's not local), but it should be a first
        // good approximation, especially given that the field-of-view is restrained to the common-
        // area. Also note that the CTF is multiplied once with the slice, but since the microscope
        // already multiplies by the CTF, the sampling function ends up with CTF^2.
        void compute_and_apply_weights_(
                const View<c32>& reference_and_target_rfft,
                const View<f32>& reference_and_target_weights_rfft,
                const MetadataSlice& reference_metadata,
                const MetadataSlice& target_metadata,
                const Shape4<i64>& slice_shape
        ) const;

        [[nodiscard]] auto cross_correlate_(
                const MetadataSlice& target_metadata,
                const ProjectionMatchingParameters& parameters
        ) const -> std::pair<Vec2<f64>, f64>;

    private:
        Array<f32> m_reference_and_target;
        Array<c32> m_reference_and_target_rfft;
        Array<c32> m_reference_and_target_padded_rfft;
        Array<f32> m_reference_and_target_weights_padded_rfft;
        Array<c32> m_projected_reference_padded_rfft;
        Array<f32> m_projected_weights_padded_rfft;
        Array<f32> m_projected_multiplicity_padded_rfft;
        CTFAnisotropic64 m_average_ctf;
    };
}
