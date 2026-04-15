#pragma once

#include <noa/Core.hpp>
#include <noa/Session.hpp>

#include "quinoa/Types.hpp"

namespace qn {
    struct Series {
        Path mdoc_file{};
        Path stack_file{};
        Path rawtlt_file{};
        Path star_file{};
        Path frames_directory{};
        Path output_directory{};

        /// Get the stem: stem(.*).mdoc.
        static auto stem(const Path& path) -> Path {
            auto stem = path.stem();
            while (not stem.extension().empty())
                stem = stem.stem();
            return stem;
        }

        auto stem() const -> Path {
            return stem(mdoc_file);
        }

        auto info() const -> std::string {
            return fmt::format(
                "{}:\n"
                "  mdoc={}\n"
                "  stack={}\n"
                "  rawtlt={}\n"
                "  output={}",
                stem(mdoc_file), mdoc_file, stack_file,
                rawtlt_file.empty() ? "<none>" : rawtlt_file.native(),
                output_directory
            );
        }
    };

    // Settings
    class Settings {
    public:
        Settings() = default;
        auto parse(int argc, const char* const* argv) -> std::vector<Series>;

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
            std::vector<isize> exclude_stack_indices{};
        } preprocessing;

        struct Alignment {
            bool coarse_run{};
            bool coarse_check_rotation{};
            bool coarse_fit_rotation{};
            bool coarse_fit_tilt{};
            bool coarse_fit_pitch{};

            bool ctf_run{};
            f64 ctf_patch_size_ang{};
            i32 ctf_patch_size_min_pix{};
            Vec<f64, 2> ctf_resolution_range{};
            i32 ctf_n_images_in_initial_average{};
            bool ctf_check_defocus_gradient{};
            bool ctf_fit_rotation{};
            bool ctf_fit_tilt{};
            bool ctf_fit_pitch{};
            bool ctf_fit_phase_shift{};
            bool ctf_fit_astigmatism{};
            bool ctf_fit_thickness{};

            bool refine_run{};
            bool refine_correct_ctf{};
            f64 refine_phase_flip_strength{};
            bool refine_fit_rotation{};
            bool refine_fit_tilt{};
            bool refine_fit_pitch{};
            bool refine_fit_thickness{};
        } alignment;

        struct PostProcessing {
            bool run{};
            f64 resolution{};

            bool stack_run{};
            bool stack_correct_rotation{};
            nx::Interp stack_interpolation{};
            noa::io::DataType stack_dtype{};

            bool tomogram_run{};
            bool tomogram_correct_rotation{};
            nx::Interp tomogram_interpolation{};
            noa::io::DataType tomogram_dtype{};
            std::string tomogram_algorithm{};
            i32 tomogram_oversampling_factor{};
            bool tomogram_ramp_filter{};
            bool tomogram_correct_ctf{};
            f64 tomogram_z_padding_percent{};
            f64 tomogram_phase_flip_strength{};
        } postprocessing;

        struct Compute {
            std::vector<Device> devices{};
            i32 n_threads{};
            bool register_stack{};
            std::string log_level{};
            bool dry{};
            bool stop_at_first_error{};
        } compute;
    };
}
