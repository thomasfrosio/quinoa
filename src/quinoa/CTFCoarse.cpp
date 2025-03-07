#include "quinoa/CTF.hpp"
#include "quinoa/GridSearch.hpp"

namespace {
    using namespace qn;
    using namespace qn::ctf;

    struct AverageSpectrum {
        SpanContiguous<const f32, 2> input;
        SpanContiguous<const i32> indices;
        i32 chunk_size;
        f32 norm;

        constexpr void init(i32 b, i32 i, i32 x, f32& isum) const {
            isum += input(b * chunk_size + indices[i], x);
        }
        static constexpr void join(f32& isum, f32& sum) {
            sum += isum;
        }
        using remove_default_final = bool;
        constexpr void final(f32 sum, f32& average) const {
            average = sum * norm;
        }
    };

    constexpr std::string_view PATH_ =
        "/home/thomas/Projects/datasets/EMPIAR-11830/11830/data/chlamy_visual_proteomics/01122021_BrnoKrios_arctis_lam2_pos14/quinoa";

    auto initial_search_along_tilt_axis(
        const Grid& grid,
        const View<const f32>& rotational_average,
        const Vec<f64, 2>& fftfreq_range,
        const ns::CTFIsotropic<f64>& ctf,
        bool fit_phase_shift
    ) -> Pair<Background, ns::CTFIsotropic<f64>> {

        // Compute the rotational average of that patch.
        const auto logical_size = grid.patch_size();
        const auto spectrum_size = logical_size / 2 + 1;

        // Fit a cubic B-spline with 3 uniform control points and subtract it from the average.
        // This is a first estimate of the background until we get a defocus estimate.
        auto background_spline = Background::fit_coarse_background_1d(rotational_average, 3);
        auto rotational_average_bs = noa::like(rotational_average);
        const auto spline_norm = 1 / static_cast<f64>(spectrum_size - 1);
        for (i64 i{}; auto&& [ra, ra_bs]:
             noa::zip(rotational_average.span_1d_contiguous(),
                      rotational_average_bs.span_1d_contiguous())) {
            ra_bs = ra - background_spline.interpolate_at(i++ * spline_norm);
        }
        noa::normalize(rotational_average_bs, rotational_average_bs, {.mode = noa::Norm::L2});

        save_vector_to_text(rotational_average_bs.view(), Path(PATH_) / "rotational_average_bs.txt");

        // Defocus (+ phase-shift) search.
        auto background = Background{};
        auto simulated_ctf = noa::Array<f32>(spectrum_size);
        auto ictf = ns::CTFIsotropic(ctf);

        for (i32 i: noa::irange(3)) {
            const auto max_phase_shift = fit_phase_shift ? noa::Constant<f64>::PI / 6 : 0.;
            const auto grid_search = GridSearch<f64, f64>(
                {.start = 0., .end = max_phase_shift, .step = 0.05}, // phase shift
                {.start = 0.4, .end = 6., .step = 0.02} // defocus
            );
            Logger::trace("grid_search:shape={}", grid_search.shape());

            f64 best_ncc{};
            Vec<f64, 2> best_values{};
            grid_search.for_each([&](f64 phase_shift, f64 defocus) mutable {
                ictf.set_defocus(defocus);
                ictf.set_phase_shift(phase_shift);
                ns::ctf_isotropic<"h2h">(
                    simulated_ctf, Shape{logical_size}.push_front<3>(1), ictf, {
                        .fftfreq_range = {fftfreq_range[0], fftfreq_range[1]},
                        .ctf_squared = true
                    });

                f64 ncc{};
                f64 ncc_rhs{};
                const auto lhs = rotational_average_bs.span_1d_contiguous(); // already normalized
                const auto rhs = simulated_ctf.span_1d_contiguous();
                for (i64 j{}; j < lhs.ssize(); ++j) {
                    auto lhs_j = static_cast<f64>(lhs[j]);
                    auto rhs_j = static_cast<f64>(rhs[j]);
                    ncc += lhs_j * rhs_j;
                    ncc_rhs += rhs_j * rhs_j;
                }
                ncc /= std::sqrt(ncc_rhs);

                const bool new_best = ncc > best_ncc;
                if (new_best) {
                    best_values = {defocus, phase_shift};
                    best_ncc = ncc;
                }
                Logger::trace(
                    "defocus={:.3f}, phase_shift={:.3f}, ncc={:.4f}{}",
                    defocus, phase_shift, ncc, new_best ? "(+)" : ""
                );
            });
            ictf.set_defocus(best_values[0]);
            ictf.set_phase_shift(best_values[1]); // FIXME

            // Update the background.
            background.fit_1d(rotational_average, fftfreq_range, ictf);
            background.subtract(rotational_average, rotational_average_bs.view(), fftfreq_range);
            noa::normalize(rotational_average_bs, rotational_average_bs, {.mode = noa::Norm::L2});

            save_vector_to_text(rotational_average_bs.view(), Path(PATH_) / "rotational_average_bs2.txt");
        }
        Logger::trace("defocus={:.3f}", ictf.defocus());

        return {std::move(background), ictf};
    }

    auto get_rotational_averages_along_tilt_axis(
        const Grid& grid,
        const View<const f32>& all_rotational_averages,
        const MetadataStack& metadata,
        i32 chunk_size
    ) -> Array<f32> {
        // Get the patches that are along the tilt-axis.
        const auto image_rotation = ng::rotate(noa::deg2rad(-metadata[0].angles[0]));
        const auto image_center = (grid.slice_shape().vec / 2).as<f64>();
        const auto tile_radius = noa::mean(image_center) * 0.25;

        std::vector<i32> indices;
        for (i32 i{}; const Vec<f64, 2>& patch_center: grid.patches_centers()) {
            const f64 x_coordinate = (image_rotation * (patch_center - image_center))[1];
            if (std::abs(x_coordinate) <= tile_radius)
                indices.push_back(i);
            ++i;
        }
        auto indices_ = View(indices.data(), std::ssize(indices)).to(all_rotational_averages.options());

        // Compute the rotational average of that patch.
        const auto logical_size = grid.patch_size();
        const auto spectrum_size = logical_size / 2 + 1;

        // Compute the average spectrum along the tilt-axis.
        auto rotational_averages = Array<f32>(
            all_rotational_averages.shape().set<0>(metadata.ssize()),
            all_rotational_averages.options()
        );
        noa::reduce_axes_iwise(
            Shape{metadata.ssize(), std::ssize(indices), spectrum_size}, rotational_averages.device(),
            f32{0}, rotational_averages.permute({1, 0, 2, 3}), AverageSpectrum{
                .input = all_rotational_averages.span().filter(0, 3).as_contiguous(),
                .indices = indices_.span_1d_contiguous(),
                .chunk_size = chunk_size,
                .norm = 1 / static_cast<f32>(std::ssize(indices)),
            });
        rotational_averages = rotational_averages.reinterpret_as_cpu(); // FIXME
        return rotational_averages;
    }

    template<typename CTF>
    struct CTFCrossCorrelate {
    private:
        SpanContiguous<const f32, 2> m_power_spectrum;
        SpanContiguous<const f32, 1> m_background;
        noa::BatchedParameter<CTF> m_ctfs;
        f32 m_fftfreq_start;
        f32 m_fftfreq_step;

    public:
        constexpr CTFCrossCorrelate(
            const SpanContiguous<const f32, 2>& power_spectrum,
            const SpanContiguous<const f32, 1>& background,
            const noa::BatchedParameter<CTF>& ctfs,
            Vec<f64, 2> fftfreq_range
        ) :
            m_power_spectrum{power_spectrum},
            m_background{background},
            m_ctfs{ctfs},
            m_fftfreq_start{static_cast<f32>(fftfreq_range[0])},
            m_fftfreq_step{
                static_cast<f32>(
                    (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(power_spectrum.shape()[1] - 1)
                )
            }
        {}

        constexpr void init(i32 b, i32 i, f32& cc, f32& cc_lhs, f32& cc_rhs) const {
            const auto fftfreq = static_cast<f32>(i) * m_fftfreq_step + m_fftfreq_start;

            // 1. Get the experimental power spectrum.
            auto lhs = m_power_spectrum(b, i);
            lhs -= m_background(i);

            // 2. Get the simulated ctf.
            auto rhs = static_cast<f32>(m_ctfs[b].value_at(fftfreq));
            rhs *= rhs; // ctf^2

            // 3. Cross-correlate.
            // To compute the normalized cross-correlation score, we usually L2-normalize the inputs.
            // Here the inputs are generated on-the-fly to save memory, so use the auto cross-correlation instead.
            cc += lhs * rhs;
            cc_lhs += lhs * lhs;
            cc_rhs += rhs * rhs;
        }

        static constexpr void join(
            f32 icc, f32 icc_lhs, f32 icc_rhs,
            f32& cc, f32& cc_lhs, f32& cc_rhs
        ) {
            cc += icc;
            cc_lhs += icc_lhs;
            cc_rhs += icc_rhs;
        }

        using remove_defaulted_final = bool;
        static constexpr void final(f32 cc, f32 cc_lhs, f32 cc_rhs, f32& ncc) {
            // Normalize using autocorrelation.
            const auto energy = noa::sqrt(cc_lhs) * noa::sqrt(cc_rhs);
            ncc = cc / energy;
        }
    };

    class DefocusRamp {
    private:
        View<const f32> m_power_spectra;
        Array<f32> m_background;

        Array<ns::CTFIsotropic<f64>> m_ctfs;
        const Grid* m_grid;

        Vec<f64, 2> m_fftfreq_range;
        Memoizer m_memoizer{};

        f64 m_rotation;
        Array<f32> m_nccs;

    public:
        Vec<f64, 3> m_parameters;

        DefocusRamp(
            const View<const f32>& power_spectra,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFIsotropic<f64>& ctf,
            const Background& background,
            const Grid& grid,
            f64 rotation
        ) :
            m_power_spectra(power_spectra),
            m_grid(&grid),
            m_fftfreq_range(fftfreq_range),
            m_rotation(noa::deg2rad(rotation))
        {
            m_background = Array<f32>(power_spectra.shape().set<0>(1), {
                .device = power_spectra.device(),
                .allocator = Allocator::MANAGED,
            });
            background.sample(m_background.view(), fftfreq_range);

            m_ctfs = Array<ns::CTFIsotropic<f64>>(power_spectra.shape()[0]);
            for (auto& ictf: m_ctfs.span_1d_contiguous())
                ictf = ctf;
            m_ctfs = std::move(m_ctfs).to(power_spectra.options());

            m_parameters[0] = ctf.defocus();
            m_parameters[1] = 0; // tilt
            m_parameters[2] = 0; // pitch

            m_memoizer = Memoizer(3, 4);

            m_nccs = Array<f32>({1, 1, power_spectra.shape()[0], 1}, m_background.options());
        }

        auto update_ctfs() noexcept {
            const Vec<f64, 3> slice_angles = Vec{m_rotation, m_parameters[1], m_parameters[2]}; // rads

            // Fetch the CTFs of the patches belonging to the current slice.
            const auto ctfs = m_ctfs.span_1d_contiguous();
            const auto patches_centers = m_grid->patches_centers();
            NOA_ASSERT(patches_centers.ssize() == ctfs.ssize());

            // Update the isotropic CTF of every patch.
            for (i64 i{}; i < ctfs.ssize(); ++i) {
                // Get the z-offset of the patch.
                const auto spacing = Vec<f64, 2>::from_value(ctfs[i].pixel_size());
                const auto patch_z_offset_um = m_grid->patch_z_offset(
                    Vec2<f64>{}, slice_angles, spacing, patches_centers[i]);

                // The defocus at the patch center is simply the slice defocus minus
                // the z-offset from the tilt axis. Indeed, the defocus is positive,
                // so a negative z-offset means we are below the tilt axis, thus further
                // away from the defocus, thus have a larger defocus.
                ctfs[i].set_defocus(ctfs[i].defocus() - patch_z_offset_um);
            }
        }

        auto cost() -> f64 {
            update_ctfs();
            noa::reduce_axes_iwise(
                m_power_spectra.shape().filter(0, 3), m_power_spectra.device(),
                noa::wrap(0.f, 0.f, 0.f), m_nccs, CTFCrossCorrelate(
                    m_power_spectra.span().filter(0, 3).as_contiguous(),
                    m_background.span_1d_contiguous(),
                    noa::BatchedParameter{m_ctfs.data()},
                    m_fftfreq_range
                ));

            f32 ncc{};
            for (f32 e: m_nccs.eval().span_1d())
                ncc += e;
            return ncc / static_cast<f32>(m_nccs.ssize());
        }

        static auto function_to_maximise(
            u32 n_parameters,
            const f64* parameters,
            f64* gradients,
            void* instance
        ) -> f64 {
            auto& self = *static_cast<DefocusRamp*>(instance);

            // The optimizer may pass its own array, so update/memcpy our parameters.
            if (parameters != self.m_parameters.data())
                self.m_parameters = Vec<f64, 3>::from_pointer(parameters);
            Logger::trace("defocus={:.4f}, tilt={:.3f}, pitch={:.3f}",
                self.m_parameters[0], noa::rad2deg(self.m_parameters[1]), noa::rad2deg(self.m_parameters[2]));

            // Check if this function was called with the same parameters.
            std::optional<f64> memoized_score = self.m_memoizer.find(self.m_parameters.data(), gradients, 1e-8);
            if (memoized_score.has_value()) {
                return memoized_score.value();
            }

            if (gradients) {
                for (i32 i{}; auto& value: self.m_parameters) {
                    const f32 initial_value = static_cast<f32>(value);
                    const f32 delta = noa::deg2rad(0.1); // CentralFiniteDifference::delta(initial_value);

                    value = static_cast<f64>(initial_value - delta);
                    const f64 fx_minus_delta = self.cost();
                    value = static_cast<f64>(initial_value + delta);
                    const f64 fx_plus_delta = self.cost();

                    value = static_cast<f64>(initial_value); // back to original value
                    f64 gradient = CentralFiniteDifference::get(
                        fx_minus_delta, fx_plus_delta, static_cast<f64>(delta));

                    Logger::trace("g={:.8f}", gradient);
                    gradients[i++] = gradient;
                }
            }

            const f64 cost = self.cost();
            Logger::trace("f={:.8f}", cost);
            self.m_memoizer.record(parameters, cost, gradients);
            return cost;
        }
    };

    auto fit_tilt(
        const Grid& grid,
        const View<const f32>& rotational_averages,
        const MetadataSlice& metadata,
        ns::CTFIsotropic<f64>& ctf,
        const Background& background,
        const Vec<f64, 2>& fftfreq_range
    ) -> f64 {
        // auto sampled_background = Array<f32>(rotational_averages.shape().set<0>(1), {
        //     .device = rotational_averages.device(),
        //     .allocator = Allocator::MANAGED,
        // });
        // background.sample(sampled_background.view(), fftfreq_range);
        // save_vector_to_text(rotational_averages, Path(PATH_) / "rotational_average_slice.txt");
        // save_vector_to_text(sampled_background.view(), Path(PATH_) / "rotational_average_slice_b.txt");

        i64 n_evaluations{};

        // Set up.
        // auto fitter = DefocusRamp(
        //     rotational_averages, fftfreq_range, ctf,
        //     background, grid, metadata.angles[0]
        // );

        // Local optimization to polish to optimum and search for astigmatism.
        // const auto relative_bounds_low = Vec{
        //     ctf.defocus() + -0.15,
        //     noa::deg2rad(-70.),
        //     noa::deg2rad(-15.),
        // };
        // const auto relative_bounds_high = Vec{
        //     ctf.defocus() + 0.15,
        //     noa::deg2rad(70.),
        //     noa::deg2rad(15.),
        // };
        // const auto abs_tolerance = Vec{
        //     5e-4,
        //     noa::deg2rad(0.1),
        //     noa::deg2rad(0.1),
        // };
        //
        // auto optimizer = Optimizer(NLOPT_LD_LBFGS, 3);
        // optimizer.set_max_objective(DefocusRamp::function_to_maximise, &fitter);
        // // optimizer.set_bounds(relative_bounds_low.data(),
        // //                      relative_bounds_high.data());
        // // optimizer.set_x_tolerance_abs(abs_tolerance.data());
        //
        // f64 ncc = optimizer.optimize(fitter.m_parameters.data());
        // n_evaluations += optimizer.n_evaluations();


        // {
        //     const Vec<f64, 3> slice_angles = noa::deg2rad(Vec{0., 40., 0.}); // rads
        //     const auto patches_centers = grid.patches_centers();
        //
        //     // Update the isotropic CTF of every patch.
        //     for (i64 i{}; i < grid.n_patches(); ++i) {
        //         // Get the z-offset of the patch.
        //         const auto spacing = Vec<f64, 2>::from_value(ctf.pixel_size());
        //         const auto patch_z_offset_um = grid.patch_z_offset(
        //             Vec2<f64>{}, slice_angles, spacing, patches_centers[i]);
        //
        //         // The defocus at the patch center is simply the slice defocus minus
        //         // the z-offset from the tilt axis. Indeed, the defocus is positive,
        //         // so a negative z-offset means we are below the tilt axis, thus further
        //         // away from the defocus, thus have a larger defocus.
        //         fmt::println("center={}, offset_z={}", patches_centers[i], patch_z_offset_um);
        //     }
        // }

        const auto grid_search = GridSearch<f64>(
            {.start = -90, .end = 90, .step = 2}
        );
        // Logger::trace("grid_search:shape={}", grid_search.shape());

        auto rotational_averages_ = rotational_averages.to_cpu();
        background.subtract(rotational_averages_.view(), rotational_averages_.view(), fftfreq_range);
        noa::normalize_per_batch(rotational_averages_, rotational_averages_, {.mode = noa::Norm::L2});
        // save_vector_to_text(rotational_averages_.view(), Path(PATH_) / "rotational_average_slice.txt");

        auto simulated_ctf = noa::like(rotational_averages_);
        auto ctfs = Array<ns::CTFIsotropic<f64>>(rotational_averages_.shape()[0]);
        for (auto& ictf: ctfs.span_1d_contiguous())
            ictf = ctf;

        f64 best_ncc{};
        f64 best_tilt{};
        grid_search.for_each([&](f64 tilt) mutable {
            const Vec<f64, 3> slice_angles = noa::deg2rad(Vec{metadata.angles[0], tilt, 0.}); // rads
            const auto ctfs_ = ctfs.span_1d_contiguous();
            const auto patches_centers = grid.patches_centers();

            // Update the isotropic CTF of every patch.
            for (i64 i{}; i < ctfs_.ssize(); ++i) {
                // Get the z-offset of the patch.
                const auto spacing = Vec<f64, 2>::from_value(ctfs_[i].pixel_size());
                const auto patch_z_offset_um = grid.patch_z_offset(
                    Vec2<f64>{}, slice_angles, spacing, patches_centers[i]);

                // The defocus at the patch center is simply the slice defocus minus
                // the z-offset from the tilt axis. Indeed, the defocus is positive,
                // so a negative z-offset means we are below the tilt axis, thus further
                // away from the defocus, thus have a larger defocus.
                ctfs_[i].set_defocus(ctf.defocus() - patch_z_offset_um);
            }

            ns::ctf_isotropic<"h2h">(
                simulated_ctf, Shape<i64, 4>{ctfs_.ssize(), 1, 1, 512}, ctfs, {
                    .fftfreq_range = {fftfreq_range[0], fftfreq_range[1]},
                    .ctf_squared = true
                });
            noa::normalize_per_batch(simulated_ctf, simulated_ctf, {.mode = noa::Norm::L2});
            // save_vector_to_text(simulated_ctf.view(), Path(PATH_) / "simulated_ctf_slice.txt");

            f64 ncc{};
            const auto lhs = rotational_averages_.span_1d_contiguous(); // already normalized
            const auto rhs = simulated_ctf.span_1d_contiguous();
            for (i64 i{}; i < lhs.ssize(); ++i) {
                auto lhs_j = static_cast<f64>(lhs[i]);
                auto rhs_j = static_cast<f64>(rhs[i]);
                ncc += lhs_j * rhs_j;
            }
            ncc /= static_cast<f64>(ctfs_.ssize());

            const bool new_best = ncc > best_ncc;
            if (new_best) {
                best_tilt = tilt;
                best_ncc = ncc;
            }
            // Logger::trace(
            //     "tilt={:.3f}, ncc={:.3f}", tilt, ncc
            // );
        });

        Logger::trace("tilt={:.3f}, ncc={:.3f}", best_tilt, best_ncc);

        // Logger::trace("n_evaluations={}", n_evaluations);
        return best_tilt;
    }
}

namespace qn::ctf {
    auto coarse_fit(
        const Grid& grid,
        const Patches& patches,
        const ns::CTFIsotropic<f64>& ctf,
        MetadataStack& metadata, // .defocus updated, angles[0] may be flipped
        const FitCoarseOptions& options
    ) -> Background {
        // Compute the rotational average of every patch.
        const auto logical_size = grid.patch_size();
        const auto spectrum_size = logical_size / 2 + 1;
        auto rotational_averages = noa::zeros<f32>({2, patches.n_patches_per_stack(), 1, spectrum_size}, {
            .device = patches.rfft_ps().device(),
            .allocator = Allocator::MANAGED,
        });
        auto rotational_average = rotational_averages.view().subregion(0).permute({1, 0, 2, 3});
        auto rotational_average_weights = rotational_averages.view().subregion(1).permute({1, 0, 2, 3});
        ng::rotational_average<"h2h">(
            patches.rfft_ps(), {patches.n_patches_per_stack(), 1, logical_size, logical_size},
            rotational_average, rotational_average_weights, {
                .input_fftfreq = {0, options.fftfreq_range[1]},
                .output_fftfreq = {options.fftfreq_range[0], options.fftfreq_range[1]},
                .add_to_output = true,
            });

        auto ra_along_axis = get_rotational_averages_along_tilt_axis(
            grid, rotational_average, metadata, patches.chunk_shape()[0]);
        save_vector_to_text(ra_along_axis.view(), Path(PATH_) / "rotational_average_all.txt");

        Array<ns::CTFIsotropic<f64>> ctfs(metadata.ssize());
        f64 average_defocus{};
        for (i64 i{}; auto& slice: metadata) {
            auto [background_i, ctfi] = initial_search_along_tilt_axis(
                grid, ra_along_axis.view().subregion(i),
                options.fftfreq_range, ctf, options.fit_phase_shift
            );
            ctfs.get()[i] = ctfi;
            average_defocus += ctfi.defocus();
            ++i;
        }
        average_defocus /= static_cast<f64>(metadata.ssize());
        auto average_ctf = ctfs.get()[0];
        average_ctf.set_defocus(average_defocus);

        auto fused_ra_along_axis = Array<f32>(spectrum_size);
        ng::fuse_rotational_averages(
            ra_along_axis, {options.fftfreq_range[0], options.fftfreq_range[1]}, ctfs,
            fused_ra_along_axis, {options.fftfreq_range[0], options.fftfreq_range[1]}, average_ctf
        );
        Background background{};
        background.fit_1d(fused_ra_along_axis.view(), options.fftfreq_range, average_ctf);
        auto background_sampled = noa::like(fused_ra_along_axis);
        background.sample(background_sampled.view(), options.fftfreq_range);

        save_vector_to_text(background_sampled.view(), Path(PATH_) / "background.txt");
        save_vector_to_text(fused_ra_along_axis.view(), Path(PATH_) / "rotational_average_fused.txt");

        // 3. Fit patches 1d, varying defocus, tilt and pitch (starting from 0).
        // let the tilt move freely to fit the defocus ramp.
        // allow some pitch
        std::vector<f64> tilts;
        for (i64 i{}; auto& slice: metadata) {
            f64 tilt = fit_tilt(grid, rotational_average.subregion(patches.chunk_slice(i)),
                     metadata[i], ctfs.get()[i], background, options.fftfreq_range
            );
            ++i;
            tilts.push_back(tilt);
        }
        noa::write_text(fmt::format("{}", fmt::join(tilts, ",")), Path(PATH_) / "tilts.txt");
        std::terminate();

        return Background{};

        // fuse the power spectra.
    }
}
