#pragma once

#include <noa/Array.hpp>
#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct ReconstructionRadial {

    };
    struct ReconstructionLikeSIRT {

    };
    struct ReconstructionWiener {

    };
    struct ReconstructionPhaseFlip {

    };

    struct TomogramReconstructionParameters{
        f64 sample_thickness_nm;
        f64 z_padding_percent;

        bool correct_ctf;
        f64 defocus_step_nm;

        noa::Interp interp;
        Path output_directory;
    };

    /// Tomogram reconstruction using direct Fourier insertion of small cubes (ala Warp).
    /// The full tomogram is subdivided into small cubes. Cubes are reconstructed by back-projecting their
    /// corresponding tiles. These tiles are twice as large as the final cubes to remove wrap-around and interpolation
    /// effects from the rotation and interpolation in Fourier space.
    auto tomogram_reconstruction(
        const View<f32>& stack,
        const MetadataStack& metadata,
        const ns::CTFIsotropic<f64>& ctf,
        const TomogramReconstructionParameters& parameters
    ) -> Array<f32>;
}
