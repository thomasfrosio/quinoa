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
            struct Coarse {
                bool run{};
                f64 resolution{};
                isize min_size_pix{};
                isize max_size_pix{};
                Bandpass bandpass{};
                bool check_rotation{};
                bool allow_90_and_flip_rotation_from_mdoc{};
                bool is_tilt_axis_from_mdoc{};
                bool fit_rotation{};
                bool fit_tilt{};
                bool fit_pitch{};
            } coarse;

            struct Refine {
                bool run{};
                f64 resolution{};
                isize min_size_pix{};
                isize max_size_pix{};
                Bandpass bandpass{};
                bool correct_ctf{};
                f64 ctf_phase_flip_strength{};
                bool fit_rotation{};
                bool fit_tilt{};
                bool fit_pitch{};
                bool fit_thickness{};
            } refine;
        } alignment;

        struct CTF {
            bool run{};
            bool check_defocus_gradient{};
            f64 patch_size_ang{};
            isize patch_size_min_pix{};
            Vec<f64, 2> resolution_range{};
            i32 nb_images_in_initial_average{};
            i32 max_nb_high_resolution_recovery{};
            Vec<isize, 2> astigmatism_tilt_resolution{};
            Vec<isize, 2> phase_shift_time_resolution{};
            bool fit_rotation{};
            bool fit_tilt{};
            bool fit_pitch{};
            bool fit_phase_shift{};
            bool fit_astigmatism{};
            bool fit_thickness{};
        } ctf;

        struct PostProcessing {
            bool run{};
            f64 resolution{};
            isize min_size_pix{};
            isize max_size_pix{};
            Bandpass bandpass{};

            struct Stack {
                bool run{};
                noa::io::DataType dtype{};
                bool correct_rotation{};
                bool correct_shift{};
                nx::Interp interpolation{};
                i32 fake_sirt_iterations{};
            } stack;

            struct Tomogram {
                bool run{};
                noa::io::DataType dtype{};
                bool correct_rotation{};
                bool correct_ctf{};
                f64 ctf_phase_flip_strength{};
                f64 ctf_defocus_step_nm{};
                f64 ctf_bfactor{};

                f64 z_padding_percent{};
                i32 fake_sirt_iterations{};
                std::string algorithm{};

                struct Real {
                    i32 oversampling_factor{};
                    bool prealign_stack{};
                    nx::Interp prealign_stack_interpolation{};
                    bool ramp_filter{};
                    nx::Interp interpolation{};
                } real;

                struct Fourier {
                    i32 oversampling_factor{};
                    bool prealign_stack{};
                    nx::Interp prealign_stack_interpolation{};
                    nx::Interp interpolation{};
                } fourier;
            } tomogram;
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
