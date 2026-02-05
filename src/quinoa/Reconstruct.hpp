#pragma once

#include "quinoa/Stack.hpp"

namespace qn {
    struct FilterStackSettings{
        bool ramp_filter{};
        bool correct_ctf{};
        f64 phase_flip_strength{};
        f64 defocus_step_nm{};
        f64 bfactor{};
    };

    auto filter_stack(
        StackLoader&& stack,
        Metadata& metadata,
        const FilterStackSettings& settings
    ) -> Array<f32>;

    struct ReconstructTomogramSettings{
        std::string algorithm{};
        f64 z_padding_percent{};
        bool correct_rotation{};
        i32 oversampling_factor{};
        nx::Interp interp{};
    };

    auto reconstruct_tomogram(
        StackLoader&& stack,
        Metadata& metadata,
        const FilterStackSettings& filter_settings,
        const ReconstructTomogramSettings& settings
    ) -> Array<f32>;

    struct PostProcessingSettings{
        Device compute_device{};
        f64 target_resolution{10};
        isize min_size{512};
        Path output_directory{};

        bool save_aligned_stack{};
        noa::io::DataType stack_dtype{};
        bool stack_correct_rotation{};
        nx::Interp stack_interp{};

        bool save_tomogram{};
        noa::io::DataType tomogram_dtype{};
    };

    void post_processing(
        const Path& input_stack,
        const Metadata& metadata,
        const PostProcessingSettings& settings,
        const FilterStackSettings& filter_settings,
        const ReconstructTomogramSettings& reconstruct_settings
    );
}
