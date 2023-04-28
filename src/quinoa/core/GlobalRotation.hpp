#pragma once

#include <noa/Array.hpp>
#include <noa/Memory.hpp>
#include <noa/Math.hpp>
#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>
#include <noa/Signal.hpp>
#include <noa/core/utils/Timer.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/io/Logging.h"

#include "noa/IO.hpp"

namespace qn {
    struct GlobalRotationParameters {
        Vec2<f32> highpass_filter{0.1f, 0.08f};
        Vec2<f32> lowpass_filter{0.4f, 0.05f};
        f64 absolute_max_tilt_difference{40};
        bool solve_using_estimated_gradient{false};
        noa::InterpMode interpolation_mode = noa::InterpMode::LINEAR_FAST;
    };

    /// Solve for the global rotation, i.e. the average tilt-axis rotation.
    /// \details A reference image is selected (the lowest tilt), as well as a set of target images (the
    ///          neighbouring views of the reference). These images are transformed and tilt-stretched onto the
    ///          reference image using different tilt-axes. The goal is to find the rotation that gives the overall
    ///          maximum normalized cross-correlation between the reference and the stretched target images.
    class GlobalRotation {
    public:
        GlobalRotation() = default;

        GlobalRotation(const Array<f32>& stack,
                        const MetadataStack& metadata,
                        Device compute_device,
                        const GlobalRotationParameters& parameters);

        // This initializes the rotation by looking at the 360deg range.
        // We do expect two symmetric roots (separated by 180 deg). The info we have doesn't allow us to select the
        // correct root (the defocus gradient could be a solution to that). Therefore, try to find the two peaks and
        // select the one with the highest score (which doesn't mean it's the correct one since the difference is likely
        // to be non-significant).
        void initialize(MetadataStack& metadata,
                        const GlobalRotationParameters& parameters);

        void update(MetadataStack& metadata,
                    const GlobalRotationParameters& parameters,
                    f64 bound);

    private:
        [[nodiscard]] f32 max_objective_fx_(
                f32 rotation_offset,
                const MetadataStack& metadata,
                const GlobalRotationParameters& parameters) const;

        [[nodiscard]] f32 max_objective_gx_(
                f32 rotation_offset,
                const MetadataStack& metadata,
                const GlobalRotationParameters& parameters) const;

        void update_stretching_matrices_(const MetadataStack& metadata, f32 rotation_offset) const;

    private:
        noa::Texture<f32> m_targets;
        noa::Array<c32> m_targets_stretched_rfft;
        noa::Array<c32> m_reference_rfft;

        Array<Float33> m_inv_stretching_matrices;
        Array<f32> m_xcorr_coefficients;

        std::vector<i32> m_target_indexes;
        i32 m_reference_index{};
    };
}
