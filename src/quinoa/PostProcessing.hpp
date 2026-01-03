#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct PostProcessingParameters{
        Device compute_device{};
        f64 target_resolution{10};
        isize min_size{512};
        Path output_directory{};
    };

    struct PostProcessingStackParameters{
        bool save_aligned_stack{};
        bool correct_rotation{};
        nx::Interp interp{};
        noa::io::DataType dtype{};
    };

    struct PostProcessingTomogramParameters{
        bool save_tomogram{};

        bool correct_ctf{};
        i32 phase_flip_strength{};
        f64 defocus_step_nm{};

        f64 z_padding_percent{};

        bool correct_rotation{};
        bool oversample{};
        nx::Interp interp{};
        noa::io::DataType dtype{};
    };

    void post_processing(
        const Path& input_stack,
        const Metadata& metadata,
        const PostProcessingParameters& parameters,
        const PostProcessingStackParameters& stack_parameters,
        const PostProcessingTomogramParameters& tomogram_parameters
    );
}
