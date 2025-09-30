#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct PostProcessingParameters{
        Device compute_device{};
        f64 target_resolution{10};
        i64 min_size{512};
        Path output_directory{};
    };

    struct PostProcessingStackParameters{
        bool save_aligned_stack{};
        bool correct_rotation{};
        noa::Interp interp{};
        noa::io::Encoding::Type dtype{};
    };

    struct PostProcessingTomogramParameters{
        bool save_tomogram{};

        bool correct_ctf{};
        i64 phase_flip_strength{};
        f64 voltage{};
        f64 amplitude{};
        f64 cs{};
        f64 defocus_step_nm{};

        f64 sample_thickness_nm{};
        f64 z_padding_percent{};

        bool correct_rotation{};
        bool oversample{};
        noa::Interp interp{};
        noa::io::Encoding::Type dtype{};
    };

    void post_processing(
        const Path& input_stack,
        const MetadataStack& metadata,
        const PostProcessingParameters& parameters,
        const PostProcessingStackParameters& stack_parameters,
        const PostProcessingTomogramParameters& tomogram_parameters
    );
}
