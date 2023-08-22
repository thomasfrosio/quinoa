#pragma once

#include <optional>
#include "quinoa/Types.h"

namespace qn {
    // Options for the program.
    class Options {
    public:
        Options(int argc, char* argv[]);

        struct Files {
            Path input_stack{};
            Path input_tlt{};
            Path input_exposure{};
            Path output_directory{};
        } files;

        struct TiltScheme {
            struct Order {
                f64 starting_angle{};
                i64 starting_direction{};
                f64 angle_increment{};
                i64 group{};
                bool exclude_start{};
                f64 per_view_exposure{};
            };

            std::optional<Order> order{};
            f64 rotation_offset{};
            f64 tilt_offset{};
            f64 elevation_offset{};
            f64 voltage{};
            f64 amplitude{};
            f64 cs{};
            f64 phase_shift{};
            f64 astigmatism_value{};
            f64 astigmatism_angle{};
        } tilt_scheme;

        struct Preprocessing {
            bool run{};
            bool exclude_blank_views{};
            std::vector<i64> exclude_view_indexes{};
        } preprocessing;

        struct Alignment {
            bool run{};
            bool fit_rotation_offset{};
            bool fit_tilt_offset{};
            bool fit_elevation_offset{};
            bool fit_phase_shift{};
            bool fit_astigmatism{};
            bool use_initial_pairwise_alignment{};
            bool use_ctf_estimate{};
            bool use_projection_matching{};
        } alignment;

        struct Compute {
            Device device{};
            i64 n_cpu_threads{};
            bool register_input_stack{};
            std::string log_level{};
        } compute;
    };
}
