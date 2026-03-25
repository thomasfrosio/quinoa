#include "quinoa/ctf/Utilities.hpp"

namespace qn::ctf {
    void rotation_check(
        Metadata::Stack& metadata,
        const CTFIsotropic64& average_ctf,
        const Grid& grid,
        const View<const f32>& spectra, // (n,p,1,w)
        const Vec<f64, 2>& fftfreq_range,
        const Path& output_directory
    ) {
        auto timer = Logger::info_scope_time<false>("Rotation check");
        const auto [n, p, w] = spectra.shape().filter(0, 1, 3);
        const auto fftfreq_linspace = noa::Linspace<f64>::from_vec(fftfreq_range);

        const auto options_managed = ArrayOption{.device = spectra.device(), .allocator = Allocator::MANAGED};
        const auto buffer = Array<f32>({n + 2, 1, 1, w}, options_managed);
        const auto ctfs_per_patch = Array<CTFIsotropic64>(p, options_managed);
        const auto ctfs_per_image = Array<CTFIsotropic64>(n);
        const auto spacing = Vec<f64, 2>::from_value(average_ctf.pixel_size());

        auto baseline = Baseline{};
        auto run = [&](f64 rotation) {
            auto spectrum = buffer.view().subregion(0);
            auto spectrum_weights = buffer.view().subregion(1);
            auto spectra_n = buffer.view().subregion(Offset(2));

            f64 ncc{};
            f64 ncc2{};
            for (auto&& [image, ictf]: noa::zip(metadata, ctfs_per_image.span_1d())) {
                // Save the CTF of the image and compute the CTF for every patch.
                ictf = average_ctf;
                ictf.set_defocus(image.defocus.value);
                ictf.set_phase_shift(image.phase_shift);
                const auto angles = noa::deg2rad(Vec{rotation, image.angles[1], image.angles[2]});
                for (auto&& [pctf, patch_center]: noa::zip(ctfs_per_patch.span_1d(), grid.patches_centers())) {
                    const auto patch_z_offset_um = grid.patch_z_offset(angles, spacing, patch_center);
                    pctf = ictf;
                    pctf.set_defocus(ictf.defocus() - patch_z_offset_um);
                }

                // Compute the average spectrum of the image to get the baseline.
                // Note that we do not want to tune the frequency range and exclude tuned out frequencies from
                // the average. For a fair comparison, we simply want to scale the spectrum to the same (expected)
                // phase and average them together.
                const auto image_spectra = spectra.subregion(image.index).permute({1, 0, 2, 3}); // (n,p,1,w) -> (p,1,1,w)
                const auto image_spectra_pw = image_spectra.span().filter(0, 3).as_contiguous();
                const auto image_spectrum = spectra_n.subregion(image.index);
                const auto image_spectrum_w = image_spectrum.span_1d();
                nx::fuse_spectra( // (p,1,1,w) -> (1,1,1,w)
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
                const auto weight = 1. / (1. + std::exp(-(std::abs(image.angles[1]) - 15) / 3.5));

                {
                    ncc2 += zero_normalized_cross_correlation(image_spectrum_w, ictf, fftfreq_range, fitting_range, baseline);
                }

                // NCC between spectrum and simulated CTF of every patch.
                // Note: This should be noiser than per-image CC, but per-patch leads to better ratios.
                //       This must be because the EPA is working with imprecise defoci, right?
                f64 incc{};
                for (isize b{}; auto& pctf: ctfs_per_patch.span_1d())
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
            ncc2 /= static_cast<f64>(n);
            fmt::println("ncc2={}", ncc2);

            // Move everything to the CPU.
            auto buffer_cpu = buffer.view().reinterpret_as_cpu();
            spectrum = buffer_cpu.view().subregion(0);
            spectrum_weights = buffer_cpu.view().subregion(1);
            spectra_n = buffer_cpu.view().subregion(Offset(2));

            // Fuse the baseline-subtracted spectrum of every image into a single spectrum.
            nx::fuse_spectra( // (n,1,1,w) -> (1,1,1,w)
                spectra_n, fftfreq_linspace, ctfs_per_image,
                spectrum, fftfreq_linspace, average_ctf,
                spectrum_weights
            );

            // Subtract the baseline and normalize.
            baseline.fit(spectrum.span_1d(), fftfreq_range, average_ctf);
            baseline.subtract(spectrum, spectrum, fftfreq_range);
            noa::normalize(spectrum, spectrum, {.mode = noa::Norm::L2});

            // Tune low-frequency range and plot for diagnostics.
            auto fitting_range = baseline.tune_fitting_range(spectrum.span_1d(), fftfreq_range, average_ctf);
            auto [start_index, start_fftfreq] = nearest_integer_fftfreq(w, fftfreq_range, fitting_range[0]);
            auto new_spectrum = spectrum.subregion(Ellipsis{}, Slice{start_index});
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
        const f64 rotation_flipped = Metadata::Image::to_angle_range(rotation + 180);
        auto ncc = run(rotation);
        auto ncc_flipped = run(rotation_flipped);
        Logger::trace(
            "rotation={:+.2f}: ncc={:.4f}\n"
            "rotation={:+.2f}: ncc={:.4f}\n"
            "ratio={:.4f}",
            rotation, ncc, rotation_flipped, ncc_flipped,
            std::max(ncc, ncc_flipped) / std::min(ncc, ncc_flipped)
        );

        if (ncc > ncc_flipped) {
            Logger::info("The defocus ramp matches the tilt-axis and tilt angles.");
        } else {
            panic(
                "The defocus ramp is reversed. This is a bad sign!\n"
                "Check that the rotation angle and tilt angles are correct, "
                "and make sure the images were not flipped along one axis."
            );
        }
    }

}
