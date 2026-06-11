#pragma once

#include <noa/Runtime.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/ctf/Grid.hpp"
#include "quinoa/ctf/Patches.hpp"

namespace qn::ctf {
    template<typename T>
    struct RefineFittingParameters {
        T rotation{}; // radians
        T tilt{}; // radians
        T pitch{}; // radians
        T thickness{}; // um
        T phase_shift{}; // radians
        T defocus{}; // um
        T astigmatism_value{}; // um
        T astigmatism_angle{}; // radians
    };

    class RefineFitting {
    public:
        RefineFitting(
            Metadata& metadata,
            const Grid& grid,
            const Patches& patches,
            isize initial_phase_shift_resolution,
            isize initial_astigmatism_resolution
        ) :
            m_metadata{metadata},
            m_grid{grid},
            m_patches{patches},
            m_phase_shift{Array<f64>(initial_phase_shift_resolution)},
            m_astigmatism_value{Array<f64>(initial_astigmatism_resolution)},
            m_astigmatism_angle{Array<f64>(initial_astigmatism_resolution)},
            m_fitting_ranges{Array<Vec<f64, 2>>(patches.n_images())},
            m_angle_offsets{Vec<f64, 3>{}}
        {
            // TODO Look at metadata if it's already estimated?
            for (auto& s: m_phase_shift.span_1d())       s = metadata.stack[0].phase_shift; // radians
            for (auto& s: m_astigmatism_value.span_1d()) s = 0.0;
            for (auto& s: m_astigmatism_angle.span_1d()) s = noa::deg2rad(45.);
        }

        void run(
            nlopt_algorithm algorithm,
            i32 max_number_of_evaluations,
            const RefineFittingParameters<Vec<f64, 2>>& relative_bounds
        );

        void plot_diagnostics(const Path& diagnostics_directory) const;

        auto increase_phase_shift_resolution(isize new_resolution) -> bool;
        auto increase_astigmatism_resolution(isize new_resolution) -> bool;

        [[nodiscard]] auto fitting_ranges() const -> SpanContiguous<const Vec<f64, 2>> {
            return m_fitting_ranges.span_1d();
        }

    private:
        Metadata& m_metadata;
        const Grid& m_grid;
        const Patches& m_patches;

        Array<f64> m_phase_shift{};
        Array<f64> m_astigmatism_value{};
        Array<f64> m_astigmatism_angle{};
        Array<Vec<f64, 2>> m_fitting_ranges{};
        Vec<f64, 3> m_angle_offsets{};
    };
}
