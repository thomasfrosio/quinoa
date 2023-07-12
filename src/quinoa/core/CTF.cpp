#include "quinoa/core/CTF.hpp"
#include "CubicGrid.hpp"
#include "Metadata.h"
#include "Optimizer.hpp"
#include "nlopt.h"
#include "quinoa/core/Utilities.h"
//#include "quinoa/core/Plot.hpp"
#include "quinoa/core/GridSearch1D.hpp"
#include "quinoa/io/Logging.h"

#include <noa/core/math/Constant.hpp>
#include <noa/core/math/Enums.hpp>
#include <noa/core/signal/fft/CTF.hpp>
#include <noa/core/types/Shape.hpp>
#include <noa/core/utils/Indexing.hpp>
#include <noa/gpu/cuda/fft/Plan.hpp>
#include <noa/unified/Allocator.hpp>
#include <noa/unified/ArrayOption.hpp>
#include <noa/unified/Ewise.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/fft/Factory.hpp>
#include <noa/unified/geometry/fft/Polar.hpp>
#include <noa/unified/math/Blas.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/Signal.hpp>

namespace qn {
    CTF::CTF(
            const Shape2<i64>& slice_shape,
            i64 patch_size,
            i64 patch_step,
            Device compute_device
    ) : m_patch_size(patch_size), m_patch_step(patch_step) {
        // Patches.
        const i64 patches_in_y = patch_grid_1d_count(slice_shape[0], m_patch_size, m_patch_step);
        const i64 patches_in_x = patch_grid_1d_count(slice_shape[1], m_patch_size, m_patch_step);
        const i64 patches_in_slice = patches_in_y * patches_in_x;

        // The patches are loaded one slice at a time. So allocate enough for one slice.
        auto options = ArrayOption(compute_device, Allocator::DEFAULT_ASYNC);
        const auto slice_patches_shape = Shape4<i64>{patches_in_slice, 1, m_patch_size, m_patch_size};
        m_slice = noa::memory::empty<f32>(slice_shape.push_front<2>({1, 1}), options);
        m_patches_rfft = noa::memory::empty<c32>(slice_patches_shape.rfft(), options);

        // Preserve the alignment between the row vectors, so use pitched memory.
        // TODO options = options.allocator(Allocator::PITCHED);
    }

    void CTF::fit_average(
            StackLoader& stack_loader,
            MetadataStack& metadata,
            FittingRange& fitting_range,
            CTFIsotropic64& ctf,

            Vec2<f64> delta_z_range_nanometers,
            f64 delta_z_shift_nanometers,
            f64 max_tilt_for_average,
            bool fit_phase_shift,
            bool fit_astigmatism,
            f64& astigmatism_value,
            f64& astigmatism_angle,
            bool flip_rotation_to_match_defocus_ramp,
            const Path& debug_directory
    ) {
        // Initial fit below, above and at the eucentric height.
        std::array defoci{0., 0., 0.};
        std::array max_tilt_for_averages{max_tilt_for_average, 90., 90.};
        std::array delta_z_ranges{
                delta_z_range_nanometers,
                delta_z_range_nanometers - std::abs(delta_z_shift_nanometers),
                delta_z_range_nanometers + std::abs(delta_z_shift_nanometers)
        };

        FittingRange i_fitting_range = fitting_range;
        CTFIsotropic64 i_ctf = ctf;

        for (auto i: noa::irange<size_t>(3)) {
            const auto average_patch_rfft_ps = compute_average_patch_rfft_ps_(
                    stack_loader, metadata, delta_z_ranges[i], max_tilt_for_averages[i], debug_directory);

            fit_ctf_to_patch_(
                    average_patch_rfft_ps, i_fitting_range, i_ctf, fit_phase_shift,
                    fit_astigmatism, astigmatism_value, astigmatism_angle,
                    debug_directory);

            defoci[i] = i_ctf.defocus();
            if (i == 0) {
                fitting_range = i_fitting_range;
                ctf = i_ctf;
            }
        }

        // Check that the defocus ramp matches what we would expect from the rotation and tilt angles.
        std::swap(defoci[0], defoci[1]); // below, at, above eucentric height
        const auto region_below_eucentric_has_higher_defocus = defoci[0] > defoci[1];
        const auto region_above_eucentric_has_lower_defocus = defoci[0] > defoci[1];

        if (region_below_eucentric_has_higher_defocus & region_above_eucentric_has_lower_defocus) {
            qn::Logger::info("Defocus ramp matches the angles. All good! "
                             "defocus={::.3f} (below, at, and above eucentric height)",
                             defoci);

        } else if (!region_below_eucentric_has_higher_defocus && !region_above_eucentric_has_lower_defocus) {
            if (flip_rotation_to_match_defocus_ramp) {
                qn::Logger::info("Defocus ramp is reversed, so flipping rotation by 180 degrees. "
                                 "defocus={::.3f} (below, at, and above eucentric height)",
                                 defoci);
                add_global_rotation(metadata, 180.);
            } else {
                qn::Logger::warn("Defocus ramp is reversed. defocus={::.3f} (below, at, and above eucentric height). "
                                 "This could be a really bad sign. Check that the rotation offset and tilt angles "
                                 "are correct, and make sure the images were not flipped",
                                 defoci);
            }
        } else {
            qn::Logger::warn("Defocus ramp isn't conclusive. defocus={::.3f} (below, at, and above eucentric height). "
                             "This could be due to a lack of signal, but note that this isn't really expected, "
                             "so please check your data and results carefully before proceeding");
        }
    }
}
