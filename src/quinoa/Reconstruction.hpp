#pragma once

#include <noa/Array.hpp>
#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    enum class ReconstructionMode { FOURIER, REAL };
    enum class ReconstructionWeighting { FOURIER, RADIAL, SIRT };

    struct TomogramReconstructionParameters{
        f64 sample_thickness_nm;
        std::string mode;
        std::string weighting;
        f64 z_padding_percent;
        i64 cube_size;
        Path debug_directory;
    };

    /// Tomogram reconstruction using direct Fourier insertion of small cubes (ala Warp).
    /// The full tomogram is subdivided into small cubes. Cubes are reconstructed by back-projecting their
    /// corresponding tiles. These tiles are twice as large as the final cubes to remove wrap-around and interpolation
    /// effects from the rotation and interpolation in Fourier space.
    auto tomogram_reconstruction(
        const View<f32>& stack,
        const Vec<f64, 2>& stack_spacing,
        const MetadataStack& metadata,
        const TomogramReconstructionParameters& parameters
    ) -> Array<f32>;
}
