#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Stack.hpp"
#include "quinoa/core/Ewise.hpp"

namespace qn {
    struct FourierTiledReconstructionParameters{
        Device compute_device;
        f64 resolution;
        f64 sample_thickness_nm;
        i64 cube_size;
        bool use_rasterization{false};
        Path debug_directory;
        bool save_aligned_stack{false};
    };

    void fourier_tiled_reconstruction(
            const Path& stack_path,
            const MetadataStack& metadata,
            const Path& output_directory,
            const FourierTiledReconstructionParameters& parameters
    );
}
