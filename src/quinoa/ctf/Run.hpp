#pragma once

#include "quinoa/ctf/CTF.hpp"
#include "quinoa/ctf/Grid.hpp"
#include "quinoa/ctf/Patches.hpp"

namespace qn::ctf {
    struct FitInitialOptions {
        i64 n_slices_to_average;
        bool fit_phase_shift{};
        Path output_directory{};
    };
    struct FitInitialResults {
        f64 defocus;
        f64 phase_shift;
        Vec<f64, 2> fitting_range;
    };
    auto initial_fit(
        const Metadata& metadata,
        const Grid& grid,
        const Patches& patches,
        const FitInitialOptions& options
    ) -> FitInitialResults;

    struct FitCoarseOptions {
        Vec<f64, 2> initial_fitting_range;
        bool exclude_bad_images{};
        bool first_image_has_higher_exposure{};
        bool fit_phase_shift{};
        bool check_defocus_gradient{};
        Path output_directory{};
    };
    void coarse_fit(
        Metadata& metadata,
        const Grid& grid,
        const Patches& patches,
        const FitCoarseOptions& options
    );

    void full_fit(
        const Path& stack_filename,
        Metadata& metadata,
        Device device,
        const Settings::CTF& settings,
        const Path& output_directory
    );
}
