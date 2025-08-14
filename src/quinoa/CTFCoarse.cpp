#include <noa/Geometry.hpp>

#include "quinoa/CTF.hpp"
#include "quinoa/GridSearch.hpp"
#include "quinoa/Plot.hpp"

namespace {
    using namespace qn;
    using namespace qn::ctf;

    struct AverageSpectrum {
        SpanContiguous<const f32, 3> input;
        SpanContiguous<const i32> indices;
        f32 norm;

        constexpr void init(i32 b, i32 i, i32 x, f32& isum) const {
            isum += input(b, indices[i], x);
        }
        static constexpr void join(f32& isum, f32& sum) {
            sum += isum;
        }
        using remove_default_final = bool;
        constexpr void final(f32 sum, f32& average) const {
            average = sum * norm;
        }
    };

    /// For each slice, collect the tiles near the tilt-axis, then average them.
    auto compute_spectra_near_tilt_axis(
        const Grid& grid,
        const View<const f32>& spectra, // (n,p,1,w)
        f64 rotation_offset,
        f64 fraction
    ) -> Array<f32> { // (n,1,1,w)
        const auto device = spectra.device();

        // Get the patches that are near the tilt-axis.
        // 1. We assume the rotation is the same for all slices.
        // 2. The tilt angles and shifts are not used.
        // As such, we select the same patches for every slice in the stack.
        Array<i32> indices = [&] {
            const auto image_rotation = ng::rotate(noa::deg2rad(-rotation_offset));
            const auto image_shape = grid.slice_shape().vec;
            const auto image_center = (image_shape / 2).as<f64>();
            const auto image_size = noa::mean(image_shape.as<f64>());
            const auto maximum_distance = image_size * fraction;

            std::vector<i32> tmp;
            for (i32 i{}; const Vec<f64, 2>& patch_center: grid.patches_centers()) {
                // Given the center of a patch in the original image,
                // compute the corresponding distance orthogonal to the tilt-axis.
                const f64 distance = (image_rotation * (patch_center - image_center))[1];
                if (std::abs(distance) <= maximum_distance)
                    tmp.push_back(i);
                ++i;
            }

            Logger::trace(
                "Selecting patches near the tilt-axis:\n"
                "  maximum_distance={:.2f} (size={})\n"
                "  n_selected={}",
                maximum_distance, image_size, tmp.size()
            );
            return View(std::data(tmp), std::ssize(tmp))
                   .to({.device = device, .allocator = Allocator::ASYNC})
                   .eval();
        }();

        // Reduce: (n,p,w) -> (n,1,w).
        const i64 n = spectra.shape()[0];
        const i64 p = indices.ssize();
        const i64 w = spectra.shape()[3];

        // For every slice, average the 1d spectra that are near the tilt-axis.
        auto spectra_average = Array<f32>({n, 1, 1, w}, {.device = device, .allocator = Allocator::MANAGED});
        noa::reduce_axes_iwise(
            Shape{n, p, w}, device,
            f32{0}, spectra_average.permute({2, 0, 1, 3}), AverageSpectrum{
                .input = spectra.span().filter(0, 1, 3).as_contiguous(),
                .indices = indices.span_1d(),
                .norm = 1 / static_cast<f32>(std::ssize(indices)),
            });

        // The rest of the fitting is on the CPU, so sync and prefetch.
        return spectra_average.reinterpret_as_cpu();
    }

    template<nt::almost_any_of<Baseline, Empty> B = Empty>
    auto coarse_grid_search(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const Vec<f64, 2>& fitting_range,
        ns::CTFIsotropic<f64>& ctf,
        const Vec<f64, 3>& phase_shift_range,
        const Vec<f64, 3>& defocus_range,
        B&& baseline = B{}
    ) -> f64 {
        f64 best_ncc{-1};
        Vec<f64, 2> best_values{};
        GridSearch(phase_shift_range, defocus_range)
            .for_each([&](f64 phase_shift, f64 defocus) {
                ctf.set_defocus(defocus);
                ctf.set_phase_shift(phase_shift);
                const f64 ncc = zero_normalized_cross_correlation(spectrum, ctf, fftfreq_range, fitting_range, baseline);
                if (ncc > best_ncc) {
                    best_values = {defocus, phase_shift};
                    best_ncc = ncc;
                }
            });
        ctf.set_defocus(best_values[0]);
        ctf.set_phase_shift(best_values[1]);
        return best_ncc;
    }

    /// Refines the defocus (and phase-shift) of the input CTF with the fitting range.
    /// When this function returns, the CTF is updated, as well as the fitting range.
    auto refine_grid_search(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        Vec<f64, 2>& fitting_range,
        ns::CTFIsotropic<f64>& ctf,
        f64 smallest_defocus,
        f64 initial_defocus_range,
        f64 initial_phase_shift_range
    ) -> f64 {
        Baseline baseline{};
        baseline.fit(spectrum, fftfreq_range, ctf);

        const auto defocus_offset = Vec{initial_defocus_range, 0.1};
        const auto defocus_step = Vec{0.02, 0.001};
        const auto phase_shift_offset = Vec{initial_phase_shift_range, std::min(initial_phase_shift_range, 10.)};

        f64 best_ncc{};
        for (i32 i: noa::irange(2)) {
            const auto defocus = ctf.defocus();
            const auto defocus_range = Vec{
                std::max(smallest_defocus, defocus - defocus_offset[i]),
                defocus + defocus_offset[i],
                defocus_step[i]
            };
            const auto phase_shift = noa::rad2deg(ctf.phase_shift());
            const auto phase_shift_range = noa::deg2rad(Vec{
                std::max(0., phase_shift - phase_shift_offset[i]),
                phase_shift + phase_shift_offset[i],
                1.
            });

            best_ncc = coarse_grid_search(
                spectrum, fftfreq_range, fitting_range, ctf,
                phase_shift_range, defocus_range, baseline
            );
            fitting_range = baseline.fit_and_tune_fitting_range(spectrum, fftfreq_range, ctf);
        }
        return best_ncc;
    }

    auto fit_spectrum_from_scratch(
        const SpanContiguous<const f32, 1>& spectrum,
        const Vec<f64, 2>& fftfreq_range,
        ns::CTFIsotropic<f64>& ctf,
        bool fit_phase_shift,
        const Path& baseline_plot_filename
    ) -> Vec<f64, 2> {
        auto spectrum_bs = Array<f32>(spectrum.shape().width());

        // Fit and subtract the baseline.
        auto baseline = Baseline{};
        baseline.fit(spectrum, fftfreq_range, {0.07, 0.45});
        baseline.sample(spectrum_bs.span_1d(), fftfreq_range);
        save_plot_xy(noa::Linspace<f64>::from_vec(fftfreq_range), spectrum_bs, baseline_plot_filename, {.label = "baseline"});
        baseline.subtract(spectrum, spectrum_bs.span_1d(), fftfreq_range);

        // For the initial fitting, prioritize the low-frequency rings.
        const f64 original_bfactor = ctf.bfactor();
        const f64 desired_bfactor = power_spectrum_bfactor_at(ctf, 0.4, 0.1);
        ctf.set_bfactor(desired_bfactor);
        auto fitting_range = Vec{fftfreq_range[0], 0.4};

        // Do the full range search.
        const auto defocus_range = Vec{0.6, 10., 0.02}; // start, stop, step
        const auto phase_shift_range = Vec{0., fit_phase_shift ? 120. : 0., 2.}; // start, stop, step
        coarse_grid_search(
            spectrum_bs.span_1d(), fftfreq_range, fitting_range, ctf,
            noa::deg2rad(phase_shift_range), defocus_range
        );
        const f64 initial_defocus = ctf.defocus();
        const f64 initial_phase_shift = noa::rad2deg(ctf.phase_shift());

        // Refine using a more appropriate fitting range.
        ctf.set_bfactor(original_bfactor);
        fitting_range = baseline.fit_and_tune_fitting_range(spectrum, fftfreq_range, ctf, {
            .keep_first_nth_peaks = 3,
            .n_extra_peaks_to_append = 1,
            .n_recoveries_allowed = 1,
        });
        const f64 defocus_offset = 0.75;
        const f64 phase_shift_offset = fit_phase_shift ? 50. : 0.;
        refine_grid_search(
            spectrum, fftfreq_range, fitting_range, ctf, defocus_range[0],
            defocus_offset, phase_shift_offset
        );

        Logger::trace(
            "Spectrum fit:\n"
            "  defocus={:.3f}um (initial={:.3f}um, range={::.2f}um, refine_range=+-{:.2f}um)\n"
            "  phase_shift={:.3f}deg (initial={:.3f}deg, range={::.2f}deg, refine_range=+-{:.2f}deg)\n"
            "  fitting_range={::.3f}cpp ({::.2f}A)",
            ctf.defocus(), initial_defocus, defocus_range, defocus_offset,
            noa::rad2deg(ctf.phase_shift()), initial_phase_shift, phase_shift_range, phase_shift_offset,
            fitting_range, fftfreq_to_resolution(ctf.pixel_size(), fitting_range)
        );

        return fitting_range;
    }

    auto rotation_check(
        MetadataStack& metadata,
        const ns::CTFIsotropic<f64>& ctf,
        const Grid& grid,
        const View<const f32>& spectra, // (n,p,1,w)
        const Vec<f64, 2>& fftfreq_range,
        const Path& output_directory
    ) -> bool {
        auto timer = Logger::info_scope_time("Rotation check");
        const auto [n, p, w] = spectra.shape().filter(0, 1, 3);
        const auto fftfreq_linspace = noa::Linspace<f64>::from_vec(fftfreq_range);

        const auto options_managed = ArrayOption{.device = spectra.device(), .allocator = Allocator::MANAGED};
        const auto buffer = Array<f32>({n + 2, 1, 1, w}, options_managed);
        const auto ctfs_per_patch = Array<ns::CTFIsotropic<f64>>(p, options_managed);
        const auto ctfs_per_image = Array<ns::CTFIsotropic<f64>>(n);
        const auto spacing = Vec<f64, 2>::from_value(ctf.pixel_size());

        auto baseline = Baseline{};
        auto run = [&](f64 rotation) {
            auto spectrum = buffer.view().subregion(0);
            auto spectrum_weights = buffer.view().subregion(1);
            auto spectra_n = buffer.view().subregion(ni::Offset(2));

            f64 ncc{};
            for (auto&& [slice, ictf]: noa::zip(metadata, ctfs_per_image.span_1d())) {
                // Save the CTF of the image and compute the CTF for every patch.
                ictf = ctf;
                ictf.set_defocus(slice.defocus.value);
                ictf.set_phase_shift(slice.phase_shift);
                const auto angles = noa::deg2rad(Vec{rotation, slice.angles[1], slice.angles[2]});
                for (auto&& [pctf, patch_center]: noa::zip(ctfs_per_patch.span_1d(), grid.patches_centers())) {
                    const auto patch_z_offset_um = grid.patch_z_offset(angles, spacing, patch_center);
                    pctf = ictf;
                    pctf.set_defocus(ictf.defocus() - patch_z_offset_um);
                }

                // Compute the average spectrum of the image to get the baseline.
                // Note that we do not want to tune the frequency range and exclude tuned out frequencies from
                // the average. For a fair comparison, we simply want to scale the spectrum to the same (expected)
                // phase and average them together.
                const auto image_spectra = spectra.subregion(slice.index).permute({1, 0, 2, 3}); // (n,p,1,w) -> (p,1,1,w)
                const auto image_spectra_pw = image_spectra.span().filter(0, 3).as_contiguous();
                const auto image_spectrum = spectra_n.subregion(slice.index);
                const auto image_spectrum_w = image_spectrum.span_1d();
                ng::fuse_spectra( // (p,1,1,w) -> (1,1,1,w)
                    image_spectra, fftfreq_linspace, ctfs_per_patch,
                    image_spectrum, fftfreq_linspace, ictf, spectrum_weights
                );
                image_spectrum.eval();

                // Only tune the low frequencies for the normalization and keep the higher frequencies.
                auto fitting_range = baseline.fit_and_tune_fitting_range(image_spectrum_w, fftfreq_range, ictf);
                fitting_range[1] = fftfreq_range[1];

                // The final NCC is a weighted average of the per-image NCC. We want to measure the effect of the
                // tilt-axis, so downweight the very low tilts (the zero should essentially be excluded since it is
                // not affected by the tilt-axis). Sigmoid curve: https://www.desmos.com/calculator/elmw9ptuwc
                const auto weight = 1. / (1. + std::exp(-(std::abs(slice.angles[1]) - 15) / 3.5));

                // NCC between spectrum and simulated CTF of every patch.
                f64 incc{};
                for (i64 b{}; auto& pctf: ctfs_per_patch.span_1d())
                    incc += zero_normalized_cross_correlation(
                        image_spectra_pw[b++], pctf, fftfreq_range, fitting_range, baseline);
                ncc += (incc / static_cast<f64>(p) * weight);

                // For diagnostics, we plot the average spectrum of the stack.
                // So subtract the baseline so we can fuse this spectrum with the others.
                baseline.subtract(image_spectrum_w, image_spectrum_w, fftfreq_range);

                // Set the weight of this image for fuse_spectra.
                ictf.set_scale(weight);
            }
            ncc /= static_cast<f64>(n);

            // Move everything to the CPU.
            auto buffer_cpu = buffer.view().reinterpret_as_cpu();
            spectrum = buffer_cpu.view().subregion(0);
            spectrum_weights = buffer_cpu.view().subregion(1);
            spectra_n = buffer_cpu.view().subregion(ni::Offset(2));

            // Fuse the baseline-subtracted spectrum of every image into a single spectrum.
            ng::fuse_spectra( // (n,1,1,w) -> (1,1,1,w)
                spectra_n, fftfreq_linspace, ctfs_per_image,
                spectrum, fftfreq_linspace, ctf,
                spectrum_weights
            );

            // Subtract the baseline and normalize.
            baseline.fit(spectrum.span_1d(), fftfreq_range, ctf);
            baseline.subtract(spectrum, spectrum, fftfreq_range);
            noa::normalize(spectrum, spectrum, {.mode = noa::Norm::L2});

            // Tune low-frequency range and plot for diagnostics.
            auto fitting_range = baseline.tune_fitting_range(spectrum.span_1d(), fftfreq_range, ctf);
            auto [start_index, start_fftfreq] = nearest_integer_fftfreq(w, fftfreq_range, fitting_range[0]);
            auto new_spectrum = spectrum.subregion(ni::Ellipsis{}, ni::Slice{start_index});
            save_plot_xy(
                noa::Linspace{start_fftfreq, fftfreq_range[1], true}, new_spectrum,
                output_directory / "rotation_check.txt", {
                    .title = "Tilt-weighted average spectrum",
                    .x_name = "fftfreq",
                    .label = fmt::format("tilt-axis={:+.2f}deg", rotation),
                });

            return ncc;
        };

        const auto rotation = metadata[0].angles[0];
        const f64 rotation_flipped = MetadataSlice::to_angle_range(rotation + 180);
        auto ncc = run(rotation);
        auto ncc_flipped = run(rotation_flipped);
        Logger::trace(
            "rotation={:+.2f}: ncc={:.4f}\n"
            "rotation={:+.2f}: ncc={:.4f}\n"
            "ratio={:.4f}",
            rotation, ncc, rotation_flipped, ncc_flipped,
            std::max(ncc, ncc_flipped) / std::min(ncc, ncc_flipped)
        );

        return ncc > ncc_flipped;
    }

    // Reduce the height/phi dimension to get the rotational average.
    auto compute_rotational_averages(const Patches& patches) -> Array<f32> {
        auto spectra = Array<f32>(patches.view().shape().set<2>(1), {
            .device = patches.view().device(),
            .allocator = Allocator::PITCHED_MANAGED,
        });
        const f32 n_patches_per_image = static_cast<f32>(patches.height());
        noa::reduce_axes_ewise(patches.view(), f32{0}, spectra, noa::ReduceMean{.size=n_patches_per_image});
        return spectra;
    }
}

namespace qn::ctf {
    auto initial_fit(
        const Grid& grid,
        const Patches& patches,
        const MetadataStack& metadata,
        ns::CTFIsotropic<f64>& ctf, // .defocus and .phase_shift updated
        const FitInitialOptions& options
    ) -> Vec<f64, 2> {
        auto timer = Logger::info_scope_time("Initial fitting");

        // Compute the per-image spectra.
        const auto spectra_per_patch = compute_rotational_averages(patches);
        const auto spectra_per_image = compute_spectra_near_tilt_axis(
            grid, spectra_per_patch.view(), metadata[0].angles[0], 0.2
        );

        // Get the average spectrum.
        const auto average_spectrum = noa::zeros<f32>(spectra_per_image.shape().width());
        const auto average_spectrum_w = average_spectrum.span_1d();
        const auto spectra_per_image_bw = spectra_per_image.span().filter(0, 3).as_contiguous();
        const auto norm = 1 / static_cast<f32>(options.n_slices_to_average);
        for (i64 i{}; i < options.n_slices_to_average; ++i) {
            for (i64 j{}; f32 e: spectra_per_image_bw[i])
                average_spectrum_w[j++] += e * norm;
        }

        const auto plot_filename = options.output_directory / "initial_power_spectrum.txt";
        save_plot_xy(patches.rho(), average_spectrum_w, plot_filename, {
            .title = fmt::format("initial spectrum (n_images={})", options.n_slices_to_average),
            .x_name = "fftfreq",
            .label = "spectrum",
        });

        auto fitting_range = fit_spectrum_from_scratch(
            average_spectrum_w, patches.rho_vec(),
            ctf, options.fit_phase_shift, plot_filename
        );

        return fitting_range;
    }

    void coarse_fit(
        const Grid& grid,
        Patches patches,
        ns::CTFIsotropic<f64>& ctf,
        MetadataStack& metadata, // defocus is updated, rotation may be flipped
        const FitCoarseOptions& options
    ) {
        auto timer = Logger::info_scope_time("Coarse fitting");

        // Compute the per-image spectrum.
        const auto spectra_per_patch = compute_rotational_averages(patches);
        const auto spectra_per_image = compute_spectra_near_tilt_axis(
            grid, spectra_per_patch.view(), metadata[0].angles[0], 0.2);
        const auto spectra = spectra_per_image.span().filter(0, 3).as_contiguous();
        save_plot_xy(patches.rho(), spectra, options.output_directory / "coarse_spectra.txt", {
            .title = "Per-image coarse spectra (sorted by collection order)",
            .x_name = "fftfreq",
        });

        auto metadata_fit = metadata;
        auto fitting_ranges = std::vector<Vec<f64, 2>>(metadata.size()); // for diagnostics

        // In hybrid mode, exclude the first image from the next steps since it may have a very different
        // defocus from the rest of the stack. It is also excluded from the rotation check since it won't affect it.
        if (options.first_image_has_higher_exposure) {
            Logger::trace("Fitting higher exposure image:");
            const auto plot_filename = options.output_directory / "hybrid_power_spectrum.txt";
            save_plot_xy(patches.rho(), spectra[0], plot_filename, {
                .title = "Power spectrum of the first image of the stack",
                .x_name = "fftfreq",
                .label = "spectrum",
            });
            auto ictf = ctf;
            auto ifitting_range = fit_spectrum_from_scratch(spectra[0], patches.rho_vec(), ictf, options.fit_phase_shift, plot_filename);
            fitting_ranges[0] = ifitting_range;
            metadata[0].defocus = {ictf.defocus(), 0., 0.};
            metadata[0].phase_shift = ictf.phase_shift();

            Logger::trace("Excluding higher exposure image from the rest of the coarse alignment");
            metadata_fit.exclude_if([](auto& s) { return s.index == 0; });
        }

        {
            // Fit the CTF of every image.
            // We'll use the lower tilt neighbor as a first estimate for the defocus and the fitting range.
            // We don't fit the phase-shift, leave this to the CTF-refine that can model a time-resolved shift.
            // Instead, use the phase-shift from the initial reference as first approximation.
            // TODO Maybe fitting the phase-shift could be useful here?
            metadata_fit.sort("tilt");
            const auto smallest_defocus = 0.5;
            auto fit_neighbor_image = [&](i64 i, auto& ifitting_range, auto& ictf) {
                refine_grid_search(
                    spectra[metadata_fit[i].index], patches.rho_vec(),
                    ifitting_range, ictf, smallest_defocus, 0.6, 0
                );
                metadata_fit[i].defocus = {ictf.defocus(), 0., 0.};
                metadata_fit[i].phase_shift = ctf.phase_shift();

                // Save in the same order as the original metadata.
                const auto index = static_cast<size_t>(metadata_fit[i].index);
                fitting_ranges[index] = ifitting_range;
            };

            const auto pivot = metadata_fit.find_lowest_tilt_index();
            auto pool = noa::ThreadPool(2);
            auto _0 = pool.enqueue([&] () mutable {
                auto ictf = ctf;
                auto ifitting_range = options.initial_fitting_range;
                for (i64 i = pivot; i < metadata_fit.ssize(); ++i) // towards positive tilts
                    fit_neighbor_image(i, ifitting_range, ictf);
            });
            auto _1 = pool.enqueue([&] () mutable {
                auto ictf = ctf;
                auto ifitting_range = options.initial_fitting_range;
                for (i64 i = pivot - 1; i >= 0; --i) // towards negative tilts
                    fit_neighbor_image(i, ifitting_range, ictf);
            });
        }

        metadata.update_from(metadata_fit, {.update_defocus = true, .update_phase_shift = true});
        save_plot_xy(
            metadata | stdv::transform([](auto& s) { return s.index_file; }),
            metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
            options.output_directory / "defocus_fit.txt", {
                .title = "Per-tilt defocus",
                .x_name = "Image index (as saved in the stack)",
                .y_name = "Defocus (um)",
                .label = "Coarse fitting",
            });

        save_plot_xy(
            metadata | stdv::transform([](auto& s) { return s.index_file; }),
            fitting_ranges | stdv::transform([&](const auto& v) {
                return fftfreq_to_resolution(ctf.pixel_size(), v[1]);
            }),
            options.output_directory / "fitting_ranges.txt", {
                .title = "Resolution cutoff for CTF fitting",
                .x_name = "Image index (as saved in the file)",
                .y_name = "Resolution (A)",
                .label = "Coarse fitting",
            });

        // Remove hybrid from the average.
        if (options.first_image_has_higher_exposure)
            fitting_ranges.erase(fitting_ranges.begin());

        // Compute the average CTF.
        f64 min_defocus{20};
        f64 max_defocus{};
        f64 average_defocus{};
        f64 average_phase_shift{};
        f64 min_fitting_fftfreq{1};
        f64 max_fitting_fftfreq{};
        auto average_fitting_range = Vec<f64, 2>{};
        for (auto&& [slice, fitting_range]: noa::zip(metadata_fit, fitting_ranges)) {
            min_defocus = std::min(min_defocus, slice.defocus.value);
            max_defocus = std::max(max_defocus, slice.defocus.value);
            average_defocus += slice.defocus.value;
            average_phase_shift += slice.phase_shift;
            min_fitting_fftfreq = std::min(min_fitting_fftfreq, fitting_range[1]);
            max_fitting_fftfreq = std::max(max_fitting_fftfreq, fitting_range[1]);
            average_fitting_range += fitting_range;
        }
        const auto n_images = static_cast<f64>(metadata_fit.size());
        ctf.set_defocus(average_defocus / n_images);
        ctf.set_phase_shift(average_phase_shift / n_images);
        average_fitting_range /= n_images;
        Logger::info(
            "phase_shift={:.2f}deg\n"
            "average_defocus={:.4f}um (min={:.4f}um, max={:.4f}um)\n"
            "average_fitting_range={::.3f}cpp ({::.2f}A, cutoff_min={:.3f}cpp, cutoff_max={:.3f}cpp)",
            noa::rad2deg(ctf.phase_shift()), ctf.defocus(), min_defocus, max_defocus,
            average_fitting_range, fftfreq_to_resolution(ctf.pixel_size(), average_fitting_range),
            min_fitting_fftfreq, max_fitting_fftfreq
        );

        // Check if the rotation is flipped.
        // If it was computed using the common-line search, there's a 50/50 chance we have the opposite tilt-axis,
        // so use the defocus gradients in the images to get the correct rotation. Note that rotating the tilt-axis
        // by +/-180deg is equivalent to negating the tilt angles.
        const bool is_ok = rotation_check(
            metadata_fit, ctf, grid, spectra_per_patch.view(),
            patches.rho_vec(), options.output_directory
        );
        if (is_ok) {
            Logger::info("The defocus ramp matches the tilt-axis and tilt angles.");
        } else {
            if (not options.has_user_rotation) {
                metadata.add_image_angles({180, 0, 0});
                Logger::info(
                    "The defocus ramp is reversed.\n"
                    "Not to worry, this is expected as the tilt-axis was computed using the common-line method.\n"
                    "Rotating the tilt-axis by 180 degrees to match the CTF."
                );
            } else {
                Logger::warn(
                    "The defocus ramp is reversed. This is a bad sign!\n"
                    "Check that the rotation angle and tilt angles are correct, "
                    "and make sure the images were not flipped along one axis."
                );
            }
        }
    }
}
