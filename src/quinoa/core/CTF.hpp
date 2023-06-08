#pragma once

#include <algorithm>

#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Memory.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Utilities.h"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/CubicGrid.hpp"
#include "quinoa/core/Stack.hpp"

namespace qn {
    using CTFIsotropic64 = noa::signal::fft::CTFIsotropic<f64>;

    class CTF {
    public:

        CTF(const Shape4<i64>& stack_shape,
            const Shape2<i64>& patch_shape,
            const Vec2<i64>& patch_step,
            Device compute_device,
            Allocator allocator = noa::Allocator::DEFAULT_ASYNC
        );

        void fit_global(
                StackLoader& stack_loader,
                MetadataStack& metadata,
                Shape2<i64> patch_shape,
                Vec2<i64> patch_step,
                Vec2<f64> delta_z_range_nanometers,
                Vec2<f64> fitting_range,
                bool fit_phase_shift,
                CTFIsotropic64 ctf,
                const Path& debug_directory
        );

        void update(
                const View<const f32>& stack,
                MetadataStack& metadata,
                const Path& debug_directory
        ) {


            // Global optimisation:
            // maximisation (phase_shift + 3 angles + per-view defocus):
            // 1) simulate 1d ctf for every tile, apply envelope.
            // 2) summed ZNCC between rotational average and simulated ctf.

        }

    private:
        static auto patch_transformed_coordinate_(
                Shape2<i64> slice_shape,
                Vec2<f64> slice_shifts,
                Vec3<f64> slice_angles,
                Vec2<f64> slice_sampling,
                Vec2<f64> patch_center
        ) -> Vec3<f64>;

        static auto extract_patches_origins_(
                const Shape2<i64>& slice_shape,
                const MetadataSlice& metadata,
                Vec2<f64> sampling_rate,
                Shape2<i64> patch_shape,
                Vec2<i64> patch_step,
                Vec2<f64> delta_z_range_nanometers
        ) -> std::vector<Vec4<i32>>;

        void compute_rotational_average_of_mean_ps_(
                StackLoader& stack_loader,
                const MetadataStack& metadata,
                Shape2<i64> patch_shape,
                Vec2<i64> patch_step,
                Vec2<f64> delta_z_range_nanometers,
                const Path& debug_directory);

        View<f32> trimmed_rotational_average_(
                const View<f32>& rotational_average,
                const Vec2<f64>& fitting_range, // angstrom
                const Vec2<f64>& sampling_rate // angstrom/pixel
                ) {
            // Resolution -> Index.
            const auto start = static_cast<i64>(std::round(fitting_range[0] / sampling_rate[0]));
            const auto stop = static_cast<i64>(std::round(fitting_range[1] / sampling_rate[1]));

            return rotational_average.subregion(noa::indexing::slice_t{start, stop});
        }

        auto fit_background_and_envelope_(
                const View<const f32>& rotational_average,
                noa::signal::fft::CTFIsotropic<f64> ctf,
                Vec2<f64> fitting_range,
                bool gradient_based_optimization
        ) const -> std::pair<CubicSplineGrid<f64, 1>, CubicSplineGrid<f64, 1>>;

        template<typename Functor>
        static void apply_cubic_bspline_1d(
                const View<const f32>& input,
                const View<f32>& output,
                const CubicSplineGrid<f64, 1>& spline,
                Functor functor
        ) {
            const auto input_1d = input.accessor_contiguous_1d();
            const auto output_1d = output.accessor_contiguous_1d();
            const f64 norm = static_cast<f64>(output.size() - 1);
            for (i64 i = 0; i < output.size(); ++i) {
                const f64 coordinate = static_cast<f64>(i) * norm; // [0,1]
                output_1d[i] = functor(input_1d[i], static_cast<f32>(spline.interpolate(coordinate)));
            }
        }

        auto fit_isotropic_ctf_to_rotational_average_(
                Array<f32> rotational_average,
                noa::signal::fft::CTFIsotropic<f64> ctf,
                Vec2<f64> fitting_range,
                bool fit_phase_shift,
                const Path& debug_directory
        ) -> std::pair<f64, f64>;

        [[nodiscard]] Shape2<i64> patch_shape() const noexcept {
            // We assume the patches are even sized, so we can compute the logical size from the rfft size.
            return {m_patches_rfft.shape()[2], (m_patches_rfft.shape()[3] - 1) * 2};
        }

    private:
        Array<f32> m_slice;
        Array<c32> m_patches_rfft; // (p, 1, h, w/2+1)
        Array<f32> m_patches_rfft_ps; // (p, 1, h, w/2+1)
        Array<f32> m_rotational_averages; // (p, 1, 1, w)
        Array<f32> m_simulated_ctfs; // (p, 1, 1, w)
    };
}
