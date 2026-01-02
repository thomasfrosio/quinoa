#pragma once

#include <noa/Session.hpp>

#include "quinoa/Types.hpp"

namespace qn {
    // Settings
    class Settings {
    public:
        Settings() = default;
        auto parse(int argc, const char* const* argv) -> bool;

        struct Files {
            Path stack_file{};
            Path mdoc_file{};
            Path star_file{};
            Path frames_directory{};
            Path output_directory{};
        } files;

        struct Experiment {
            f64 tilt_axis{};
            f64 add_specimen_tilt{};
            f64 add_specimen_pitch{};
            f64 voltage{};
            f64 amplitude{};
            f64 cs{};
            f64 phase_shift{};
            f64 thickness{};
        } experiment;

        struct Preprocessing {
            bool run{};
            bool exclude_blank_views{};
            std::vector<i64> exclude_stack_indices{};
        } preprocessing;

        struct Alignment {
            bool coarse_run{};
            bool coarse_check_rotation{};
            bool coarse_fit_rotation{};
            bool coarse_fit_tilt{};
            bool coarse_fit_pitch{};

            bool ctf_run{};
            bool ctf_check_rotation{};
            bool ctf_fit_rotation{};
            bool ctf_fit_tilt{};
            bool ctf_fit_pitch{};
            bool ctf_fit_phase_shift{};
            bool ctf_fit_astigmatism{};
            bool ctf_fit_thickness{};

            bool refine_run{};
            bool refine_fit_thickness{};
        } alignment;

        struct PostProcessing {
            bool run{};
            f64 resolution{};

            bool stack_run{};
            bool stack_correct_rotation{};
            noa::Interp stack_interpolation{};
            noa::io::DataType stack_dtype{};

            bool tomogram_run{};
            bool tomogram_correct_rotation{};
            noa::Interp tomogram_interpolation{};
            noa::io::DataType tomogram_dtype{};
            bool tomogram_oversample{};
            bool tomogram_correct_ctf{};
            f64 tomogram_z_padding_percent{};
            i64 tomogram_phase_flip_strength{};
        } postprocessing;

        struct Compute {
            Device device{};
            i64 n_threads{};
            bool register_stack{};
            std::string log_level{};
        } compute;
    };
}
