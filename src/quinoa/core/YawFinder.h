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
    struct GlobalYawOffsetParameters {
        Vec2<f32> highpass_filter{0.1f, 0.08f};
        Vec2<f32> lowpass_filter{0.4f, 0.05f};
        f32 absolute_max_tilt_difference{40.f};
        bool solve_using_estimated_gradient{false};
        noa::InterpMode interpolation_mode = noa::InterpMode::LINEAR_FAST;
    };

    /// Solve for the global yaw, i.e. the average tilt-axis rotation.
    /// \details A reference image is selected (the lowest tilt), as well as a set of target images (the
    ///          neighbouring views of the reference). These images are transformed and stretched onto the
    ///          reference image using different yaws. The goal is to find the yaw that gives the overall
    ///          maximum normalized cross-correlation between the reference and the stretched target images.
    class GlobalYawSolver {
    public:
        /// Allocates the buffers and sets up the texture.
        /// \param[in] stack
        /// \param metadata
        /// \param compute_device
        /// \param parameters
        GlobalYawSolver(const Array<f32>& stack,
                        const MetadataStack& metadata,
                        Device compute_device,
                        const GlobalYawOffsetParameters& parameters);

        void initialize_yaw(MetadataStack& metadata,
                            const GlobalYawOffsetParameters& parameters);

        void update_yaw(MetadataStack& metadata,
                        const GlobalYawOffsetParameters& parameters,
                        f32 low_bound, f32 high_bound);

    private:
        void update_stretching_matrices_(const MetadataStack& metadata, f32 yaw_offset);
        f32 max_objective_fx_(f32 yaw_offset);
        f32 max_objective_gx_(f32 yaw_offset);

    private:
        noa::Texture<f32> m_targets;
        noa::Array<f32> m_targets_stretched;
        noa::Array<c32> m_targets_stretched_fft;
        noa::Array<f32> m_reference;
        noa::Array<c32> m_reference_fft;

        std::vector<i32> m_target_indexes;
        i32 m_reference_index;

        Array<Float33> m_inv_stretching_matrices;
        Array<f32> m_xcorr_coefficients;
        const MetadataStack* m_metadata{};
        Vec2<f32> m_highpass_filter;
        Vec2<f32> m_lowpass_filter;
    };
}
