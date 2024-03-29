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
            Path input_csv{};
            Path output_directory{};
        } files;

        struct Experiment {
            struct CollectionOrder {
                f64 starting_angle{};
                i64 starting_direction{};
                f64 tilt_increment{};
                i64 group_of{};
                bool exclude_start{};
                f64 per_view_exposure{};
            };

            std::optional<CollectionOrder> collection_order{};
            f64 rotation_offset{};
            f64 tilt_offset{};
            f64 elevation_offset{};
            f64 voltage{};
            f64 amplitude{};
            f64 cs{};
            f64 phase_shift{};
            f64 astigmatism_value{};
            f64 astigmatism_angle{};
            f64 thickness{};
        } experiment;

        struct Preprocessing {
            bool run{};
            bool exclude_blank_views{};
            std::vector<i64> exclude_view_indexes{};
            bool use_existing_alignment{};
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
            bool use_thickness_estimate{};
            bool use_projection_matching{};
            i64 rotation_spline_resolution{};
        } alignment;

        struct PostProcessing {
            bool run{};
            bool reconstruct_tomogram{};
            f64 reconstruct_tomogram_resolution{};
            i64 reconstruct_cube_size{};
        } postprocessing;

        struct Compute {
            Device device{};
            i64 n_cpu_threads{};
            bool register_input_stack{};
            std::string log_level{};
        } compute;
    };
}
