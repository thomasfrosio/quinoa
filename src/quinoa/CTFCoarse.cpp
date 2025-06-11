#include <noa/Geometry.hpp>

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

    /// For each slice, collect the tiles near the tilt-axis, then average them.
    // FIXME do we load patches within a delta z? It should be close enough since we have an estimate of the tilt.
    //       and the rotation +-180 has no effect.
    auto compute_spectra_near_tilt_axis(
        const Grid& grid,
        const View<const f32>& spectra,
        const MetadataStack& metadata,
        i64 chunk_size,
        f64 fraction
    ) -> Array<f32> {
        const auto options_async = ArrayOption{.device = spectra.device(), .allocator = Allocator::ASYNC};

        // Get the patches that are near the tilt-axis.
        // 1. We assume the rotation is the same for all slices.
        // 2. The tilt angles and shifts are not used.
        // As such, we select the same patches for every slice in the stack.
        Array<i32> indices = [&] {
            const auto image_rotation = ng::rotate(noa::deg2rad(-metadata[0].angles[0]));
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
            return View(std::data(tmp), std::ssize(tmp)).to(options_async).eval();
        }();

        // Reduce: (n,p,w) -> (n,1,w).
        const i64 n = metadata.ssize();
        const i64 p = indices.ssize();
        const i64 w = spectra.shape()[3];

        // For every slice, average the 1d spectra that are near the tilt-axis.
        auto spectra_average = Array<f32>({n, 1, 1, w}, options_async);
        noa::reduce_axes_iwise(
            Shape{n, p, w}, spectra_average.device(),
            f32{0}, spectra_average.permute({1, 0, 2, 3}), AverageSpectrum{
                .input = spectra.span().filter(0, 3).as_contiguous(),
                .indices = indices.span_1d(),
                .chunk_size = static_cast<i32>(chunk_size),
                .norm = 1 / static_cast<f32>(std::ssize(indices)),
            });

        // We expect a good amount of noise, so smooth it out.
        // Use a gaussian kernel small enough to not destroy the peaks.
        auto spectra_average_smoothed = Array<f32>({n, 1, 1, w}, spectra.options());
        ns::convolve(
           std::move(spectra_average), spectra_average_smoothed.view(),
           ns::window_gaussian<f32>(7, 1.25, {.normalize = true}).to(options_async), // TODO better heuristic?
           {.border = noa::Border::REFLECT}
       );

        // The rest of the fitting is on the CPU, so sync and prefetch.
        return spectra_average_smoothed.reinterpret_as_cpu();
    }

    template<bool COARSE_BACKGROUND = true, nt::almost_any_of<Background, Empty> B = Empty>
    auto coarse_grid_search(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        const Vec<f64, 2>& fitting_range,
        ns::CTFIsotropic<f64>& ctf,
        const Vec<f64, 3>& phase_shift_range,
        const Vec<f64, 3>& defocus_range,
        B&& background = B{}
    ) -> f64 {
        f64 best_ncc{-1};
        Vec<f64, 2> best_values{};
        GridSearch(phase_shift_range, defocus_range)
            .for_each([&](f64 phase_shift, f64 defocus) {
                ctf.set_defocus(defocus);
                ctf.set_phase_shift(phase_shift);
                if constexpr (not nt::empty<B>)
                    background.fit(spectrum, fftfreq_range, ctf);
                const f64 ncc = normalized_cross_correlation<COARSE_BACKGROUND>(
                    spectrum, ctf, fftfreq_range, fitting_range, background);
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
    /// When this function returns, the ctf is updated, as well as the fitting range.
    auto refine_grid_search(
        SpanContiguous<const f32> spectrum,
        const Vec<f64, 2>& fftfreq_range,
        Vec<f64, 2>& fitting_range,
        ns::CTFIsotropic<f64>& ctf,
        f64 smallest_defocus,
        f64 initial_defocus_range,
        f64 initial_phase_shift_range
    ) {
        Background background{};

        const auto defocus_offset = Vec{initial_defocus_range, 0.1};
        const auto defocus_step = Vec{0.02, 0.001};
        const auto phase_shift_offset = Vec{initial_phase_shift_range, std::min(initial_phase_shift_range, 10.)};

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

            coarse_grid_search<false>(
                spectrum, fftfreq_range, fitting_range, ctf, phase_shift_range, defocus_range, background);
            fitting_range = background.fit_and_tune_fitting_range(spectrum, fftfreq_range, ctf);
        }
    }

    auto fit_spectrum_from_scratch(
        const SpanContiguous<const f32, 1>& spectrum,
        const Vec<f64, 2>& fftfreq_range,
        ns::CTFIsotropic<f64>& ctf,
        bool fit_phase_shift,
        const Path& baseline_plot_filename
    ) -> Vec<f64, 2> {
        const auto w = spectrum.shape().width();
        auto buffer = noa::zeros<f32>({2, 1, 1, w});
        auto spectrum0 = buffer.span().subregion(0).as_1d();
        auto spectrum1 = buffer.span().subregion(1).as_1d();

        // TODO Fit baseline within [0.1,0.4]

        // Get the baseline.
        // Use strong smooth gaussian-smoothing to get the baseline. The spectrum size can vary significantly,
        // so make the gaussian size and stddev about 1/8th the size of the spectrum size. This may not be able
        // to erase the Thon rings in case of a large spacing and low defocus, but applying the smoothing
        // iteratively should be enough to cover most if not all cases.
        const auto kernel_stddev = static_cast<f64>(w) / 8.;
        auto kernel_size = static_cast<i64>(std::round(kernel_stddev));
        if (noa::is_even(kernel_size))
            kernel_size += 1;
        const auto kernel = ns::window_gaussian<f32>(kernel_size, kernel_stddev, {.normalize = true});
        ns::convolve(View(spectrum), View(spectrum0), kernel, {.border = noa::Border::REFLECT});
        ns::convolve(View(spectrum0), View(spectrum1), kernel, {.border = noa::Border::REFLECT});
        ns::convolve(View(spectrum1), View(spectrum0), kernel, {.border = noa::Border::REFLECT});
        ns::convolve(View(spectrum0), View(spectrum1), kernel, {.border = noa::Border::REFLECT});
        save_plot_xy(noa::Linspace<f64>::from_vec(fftfreq_range), spectrum1, baseline_plot_filename, {.label = "baseline"});

        // Subtract the baseline from the spectrum.
        for (auto&& [i, baseline, o]: noa::zip(spectrum, spectrum1, spectrum0))
            o = i - baseline;

        // For the initial fitting, prioritize the low-frequency rings.
        // Anything close to Nyquist is useless at this stage, so ignore it.
        const f64 original_bfactor = ctf.bfactor();
        const f64 desired_bfactor = power_spectrum_bfactor_at(ctf, 0.4, 0.1);
        ctf.set_bfactor(desired_bfactor);
        auto fitting_range = Vec{fftfreq_range[0], 0.4};

        // Do the full range search. While we could use the "varying" background method (see refine step below),
        // for this initial full-range search, a fixed background is more stable at low and high defocus.
        const auto initial_defocus_range = Vec{0.6, 10., 0.02}; // start, stop, step
        const auto initial_phase_shift_range = Vec{0., fit_phase_shift ? 120. : 0., 2.}; // start, stop, step
        coarse_grid_search(
            spectrum0, fftfreq_range, fitting_range, ctf,
            noa::deg2rad(initial_phase_shift_range), initial_defocus_range
        );

        const f64 initial_defocus = ctf.defocus();
        const f64 initial_phase_shift = noa::rad2deg(ctf.phase_shift());
        Logger::trace(
            "  initial_defocus={:.2f}um (range={::.3f}um)\n"
            "  initial_phase_shift={:.2f}deg (range={::.2f}deg)",
            initial_defocus, initial_defocus_range,
            initial_phase_shift, initial_phase_shift_range
        );

        // This is a current limitation of the background fitting, but this should only be an issue if the
        // fftfreq_range is unreasonably too small. Now that we have the fitting_range, this is even less
        // of an issue. We can set the fftfreq_range to a large range (e.g. [40,4]A), and then rely on the
        // auto-tuning to remove the regions of the spectrum that are not contributing positively to the CTF fit.
        const auto smallest_defocus = Background::smallest_defocus_for_fitting(fftfreq_range, ctf, 3);
        if (initial_defocus < smallest_defocus) {
            panic("The CTF fitting currently requires a minimum number of 3 CTF zeros in the fitting range. "
                  "Please report this issue. fftfreq_range={::.4f}, defocus={:.4f}um, phase_shift={:.4f}deg, pixel_size={:.3f}Apix",
                  fftfreq_range, initial_defocus, initial_phase_shift, ctf.pixel_size());
        }

        // Refine using more accurate background and fitting range.
        ctf.set_bfactor(original_bfactor);
        fitting_range = Background{}.fit_and_tune_fitting_range(spectrum, fftfreq_range, ctf);
        const f64 defocus_offset = 0.75;
        const f64 phase_shift_offset = fit_phase_shift ? 50. : 0.;
        refine_grid_search(
            spectrum, fftfreq_range, fitting_range, ctf, smallest_defocus,
            defocus_offset, phase_shift_offset
        );

        Logger::trace(
            "  final_defocus={:.3f}um (min={:.3f}, range=+-{:.2f}um)\n"
            "  final_phase_shift={:.3f}deg (min=0, range=+-{}deg)\n"
            "  final_fitting_range={::.3f} ({::.2f}A)\n",
            ctf.defocus(), smallest_defocus, defocus_offset,
            initial_phase_shift, phase_shift_offset,
            fitting_range, fftfreq_to_resolution(ctf.pixel_size(), fitting_range)
        );

        return fitting_range;
    }

    auto rotation_check(
        MetadataStack& metadata,
        const ns::CTFIsotropic<f64>& ctf,
        const Grid& grid,
        const Patches& patches,
        const View<const f32>& spectra,
        const Vec<f64, 2>& fftfreq_range,
        const Path& output_directory
    ) -> bool {
        auto timer = Logger::info_scope_time("Rotation check");
        const i64 n_images = patches.n_slices();
        const i64 n_patches_per_image = grid.n_patches();
        const i64 width = spectra.shape()[3];

        auto background = Background{};
        const auto options_managed = ArrayOption{.device = spectra.device(), .allocator = Allocator::MANAGED};
        const auto ctfs_per_patch = Array<ns::CTFIsotropic<f64>>(n_patches_per_image, options_managed);
        const auto ctfs_per_image = Array<ns::CTFIsotropic<f64>>(n_images);

        const auto spacing = Vec<f64, 2>::from_value(ctf.pixel_size());
        const auto fftfreq_linspace = noa::Linspace<f64>::from_vec(fftfreq_range);

        const auto image_spectrum = noa::Array<f32>(width, options_managed);
        const auto image_weights = noa::Array<f32>(width, options_managed);

        const auto buffer = noa::Array<f32>({n_images + 2, 1, 1, width});
        const auto spectrum = buffer.view().subregion(0);
        const auto spectrum_weights = buffer.view().subregion(1);
        const auto spectra_smoothed = buffer.view().subregion(ni::Slice{2});

        // The final NCC is a weighted average of the per-image NCC. We want to measure the effect of the
        // tilt-axis, so downweight the very low tilts (the zero should essentially be excluded since it is
        // not affected by the tilt-axis). Sigmoid curve: https://www.desmos.com/calculator/elmw9ptuwc
        auto weight = [](f64 tilt) { return 1. / (1. + std::exp((-(std::abs(tilt) - 15) / 3.5))); };

        auto run = [&](f64 rotation) {
            f64 ncc{};
            for (auto&& [slice, ictf]: noa::zip(metadata, ctfs_per_image.span_1d())) {
                const auto angles = noa::deg2rad(Vec{rotation, slice.angles[1], slice.angles[2]});
                const auto iweight = weight(slice.angles[1]);

                // Save the CTF of the image and compute the CTF for every patch.
                ictf = ctf;
                ictf.set_defocus(slice.defocus.value);
                ictf.set_phase_shift(slice.phase_shift);
                for (auto&& [pctf, patch_center]: noa::zip(ctfs_per_patch.span_1d(), grid.patches_centers())) {
                    const auto patch_z_offset_um = grid.patch_z_offset(angles, spacing, patch_center);
                    pctf = ictf;
                    pctf.set_defocus(ictf.defocus() - patch_z_offset_um);
                }

                // Compute the average spectrum of the image to get the background.
                // Note that we do not want to tune the frequency range and exclude tuned out frequencies from
                // the average. For a fair comparison, we simply want to scale the spectrum to the same (expected)
                // phase and average them together.
                const auto spectrum_smooth = spectra_smoothed.span().subregion(slice.index).as_1d();
                const auto image_spectra = spectra.subregion(patches.chunk_slice(slice.index));
                const auto image_spectra_bw = image_spectra.span().filter(0, 3).as_contiguous();

                ng::fuse_spectra( // (p,1,1,w) -> (1,1,1,w)
                    image_spectra, fftfreq_linspace, ctfs_per_patch,
                    image_spectrum.view(), fftfreq_linspace, ictf, image_weights.view()
                );
                ns::convolve(
                    image_spectrum.view().reinterpret_as_cpu(), View(spectrum_smooth),
                    ns::window_gaussian<f32>(11, 1.5, {.normalize = true}),
                    {.border = noa::Border::REFLECT}
                );

                // Only tune the low frequencies for the normalization and keep the higher frequencies.
                auto fitting_range = background.fit_and_tune_fitting_range(spectrum_smooth, fftfreq_range, ictf);
                fitting_range[1] = fftfreq_range[1];

                // NCC between spectrum and simulated CTF of every patch.
                f64 incc{};
                for (i64 b{}; auto& pctf: ctfs_per_patch.span_1d())
                    incc += normalized_cross_correlation(
                        image_spectra_bw[b++], pctf, fftfreq_range, fitting_range, background);
                ncc += (incc / static_cast<f64>(n_patches_per_image) * iweight);

                // For diagnostics, we plot the average spectrum of the stack.
                // So subtract the background so we can fuse this spectrum with the others.
                background.subtract(spectrum_smooth, spectrum_smooth, fftfreq_range);

                // Set the weight of this image for fuse_spectra.
                ictf.set_scale(iweight);
            }
            ncc /= static_cast<f64>(n_images);

            // Fuse every background subtracted spectrum to the average CTF and plot for diagnostics.
            ng::fuse_spectra( // (n,1,1,w) -> (1,1,1,w)
                spectra_smoothed, fftfreq_linspace, ctfs_per_image,
                spectrum, fftfreq_linspace, ctf,
                spectrum_weights
            );
            background.fit(spectrum.span_1d(), fftfreq_range, ctf);
            background.subtract(spectrum, spectrum, fftfreq_range);
            noa::normalize(spectrum, spectrum, {.mode = noa::Norm::L2});
            save_plot_xy(fftfreq_linspace, spectrum, output_directory / "rotation_check.txt", {
                .title = "Weighted average spectrum",
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

    auto compute_rotational_averages(
        const Patches& patches,
        const Vec<f64, 2>& fftfreq_range
    ) -> Array<f32> {
        // Compute the rotational average of every patch.
        const auto logical_size = patches.shape().height();
        const auto spectrum_size = logical_size / 2 + 1;
        auto buffer = noa::zeros<f32>({2, patches.n_patches_per_stack(), 1, spectrum_size}, {
            .device = patches.rfft_ps().device(),
            .allocator = Allocator::MANAGED, // TODO PITCHED_MANAGED
        });
        auto spectra = buffer.subregion(0).permute({1, 0, 2, 3});
        auto spectra_weights = buffer.view().subregion(1).permute({1, 0, 2, 3});
        ng::rotational_average<"h2h">(
            patches.rfft_ps(), {patches.n_patches_per_stack(), 1, logical_size, logical_size},
            spectra.view(), spectra_weights, {
                .input_fftfreq = {0, fftfreq_range[1]},
                .output_fftfreq = {fftfreq_range[0], fftfreq_range[1]},
                .add_to_output = true,
            });
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
        const auto spectra_per_patch = compute_rotational_averages(patches, options.fftfreq_range);
        const auto spectra_per_image = compute_spectra_near_tilt_axis(
            grid, spectra_per_patch.view(), metadata, patches.n_patches_per_slice(), 0.2
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
        save_plot_xy(noa::Linspace<f64>::from_vec(options.fftfreq_range), average_spectrum_w, plot_filename, {
            .title = fmt::format("initial spectrum (n_images={})", options.n_slices_to_average),
            .x_name = "fftfreq",
            .label = "spectrum",
        });

        auto fitting_range = fit_spectrum_from_scratch(
            average_spectrum_w, options.fftfreq_range,
            ctf, options.fit_phase_shift, plot_filename
        );

        return fitting_range;
    }

    void coarse_fit(
        const Grid& grid,
        const Patches& patches,
        ns::CTFIsotropic<f64>& ctf,
        MetadataStack& metadata, // defocus is updated, rotation may be flipped
        const FitCoarseOptions& options
    ) {
        auto timer = Logger::info_scope_time("Coarse fitting");

        // Compute the per-image spectrum.
        const auto spectra_per_patch = compute_rotational_averages(patches, options.fftfreq_range);
        const auto spectra_per_image = compute_spectra_near_tilt_axis(
            grid, spectra_per_patch.view(), metadata, patches.n_patches_per_slice(), 0.2);
        const auto spectra = spectra_per_image.span().filter(0, 3).as_contiguous();

        {
            // FIXME
            save_plot_xy(noa::Linspace<f64>::from_vec(options.fftfreq_range),
                spectra.as_strided(), options.output_directory / "coarse_spectra.txt");
        }

        auto metadata_fit = metadata;

        // In hybrid mode, exclude the first image from the next steps since it may have a very different
        // defocus from the rest of the stack. It is also excluded from the rotation check since it won't affect it.
        if (options.first_image_has_higher_exposure) {
            Logger::trace("Fitting higher exposure image:");
            const auto plot_filename = options.output_directory / "hybrid_power_spectrum.txt";
            save_plot_xy(noa::Linspace<f64>::from_vec(options.fftfreq_range), spectra[0], plot_filename, {
                .title = "Power spectrum of the first image of the stack",
                .x_name = "fftfreq",
                .label = "spectrum",
            });
            auto ictf = ctf;
            fit_spectrum_from_scratch(spectra[0], options.fftfreq_range, ictf, options.fit_phase_shift, plot_filename);
            metadata[0].defocus = {ictf.defocus(), 0., 0.};
            metadata[0].phase_shift = ictf.phase_shift();

            metadata_fit.exclude_if([](auto& s) { return s.index == 0; });
        }

        {
            // Fit the CTF of every image.
            // We'll use the lower tilt neighbor as a first estimate for the defocus and the fitting range.
            // We don't fit the phase-shift, leave this to the CTF-refine that can model a time-resolved shift.
            // Instead, use the phase-shift from the initial reference as first approximation.
            // TODO Maybe fitting the phase-shift could be useful here?
            metadata_fit.sort("tilt");
            const auto smallest_defocus = Background::smallest_defocus_for_fitting(options.fftfreq_range, ctf, 3);
            auto fit_neighbor_image = [&](i64 i, auto& ifitting_range, auto& ictf) {
                refine_grid_search(
                    spectra[metadata_fit[i].index], options.fftfreq_range,
                    ifitting_range, ictf, smallest_defocus, 0.6, 0
                );
                metadata_fit[i].defocus = {ictf.defocus(), 0., 0.};
                metadata_fit[i].phase_shift = ctf.phase_shift();
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
            metadata | stdv::transform([](auto& s) { return s.angles[1]; }),
            metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
            options.output_directory / "defocus_fit.txt", {
                .title = "Per-tilt defocus",
                .x_name = "Tilts (degrees)",
                .y_name = "Defocus (μm)",
                .label = "Defocus - Coarse fit",
            });

        // Compute the average CTF.
        f64 min_defocus{20};
        f64 max_defocus{};
        f64 average_defocus{};
        f64 average_phase_shift{};
        for (const auto& slice: metadata_fit) {
            min_defocus = std::min(min_defocus, slice.defocus.value);
            max_defocus = std::max(max_defocus, slice.defocus.value);
            average_defocus += slice.defocus.value;
            average_phase_shift += slice.phase_shift;
        }
        const auto n_images = static_cast<f64>(metadata_fit.size());
        ctf.set_defocus(average_defocus / n_images);
        ctf.set_phase_shift(average_phase_shift / n_images);
        Logger::info("Average defocus={:.4f}μm [min={:.4f}, max={:.4f}]μm, phase-shift={:.2f}deg",
                     ctf.defocus(), min_defocus, max_defocus, noa::rad2deg(ctf.phase_shift()));

        // Check if the rotation is flipped.
        // If it was computed using the common-line search, there's a 50/50 chance we have the opposite tilt-axis,
        // so use the defocus gradients in the images to get the correct rotation. Note that rotating the tilt-axis
        // by +/-180deg is equivalent to negating the tilt angles.
        const bool is_ok = rotation_check(
            metadata_fit, ctf, grid, patches, spectra_per_patch.view(),
            options.fftfreq_range, options.output_directory
        );
        if (is_ok) {
            Logger::info("The defocus ramp matches the tilt-axis and tilt angles.");
        } else {
            if (not options.has_user_rotation) {
                metadata.add_global_angles({180, 0, 0});
                Logger::info(
                    "The defocus ramp is reversed.\n"
                    "Not to worry, this is expected as the tilt-axis was computed using the common-line method.\n"
                    "Rotating the tilt-axis by 180 degrees to match the CTF.");
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
