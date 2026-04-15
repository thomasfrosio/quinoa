#include <noa/Xform.hpp>
#include <noa/FFT.hpp>

#include "quinoa/Logger.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/ctf/CTF.hpp"

namespace {
    struct PatchMeanVariance {
        using remove_default_post = bool;
        using reduced_type = Vec<f64, 2>;

        SpanContiguous<const f32, 3, i32> patches;
        Shape<i32, 2> grid_shape{};
        Vec<f32, 2> overflow;
        f32 patch_size;

        constexpr void operator()(const Vec<i32, 3>& indices, reduced_type& reduced) const {
            const auto v = static_cast<f64>(patches(indices));
            reduced[0] += v;
            reduced[1] += v * v;
        }

        static constexpr void join(const reduced_type& reduced, reduced_type& joined) {
            joined += reduced;
        }

        template<size_t N>
        constexpr void post(const reduced_type& joined, Vec<f32, 2>& mean_stddev, const Vec<i32, N>& output_indices) const {
            const auto grid_indices = noa::offset2index(output_indices[0], grid_shape[1]);

            // Compute the actual patch size by removing the region outside the image.
            auto shape = Vec{patch_size, patch_size};
            for (i32 i{}; i < 2; ++i)
                if (grid_indices[i] == 0 or grid_indices[i] == grid_shape[i] - 1) // patch at the edge
                    shape[i] -= overflow[i];
            const auto size = product(shape.as<f64>());

            const auto& [sum, sum_sqd] = joined;
            mean_stddev[0] = static_cast<f32>(sum / size);
            mean_stddev[1] = static_cast<f32>(sqrt((sum_sqd - (sum * sum) / size) / size));
        }
    };

    struct PatchMask {
        SpanContiguous<f32, 3, i32> patches{};
        SpanContiguous<const Vec<f32, 2>, 1, i32> mean_stddev{};
        Shape<i32, 2> grid_shape{};
        Vec<f32, 2> patch_overflow;
        i32 patch_center;
        f32 patch_size;
        f32 taper_radius{};
        f32 taper_smoothness{};

        constexpr void apply_mask(f32 centered_coordinate, f32 radius, f32& mask) const {
            if (centered_coordinate > radius + taper_smoothness) {
                mask = 0;
            } else if (radius < centered_coordinate and centered_coordinate <= radius + taper_smoothness) {
                constexpr auto PI = noa::Constant<f32>::PI;
                const auto distance = (centered_coordinate - radius) / taper_smoothness;
                mask *= (f32{1} + cos(PI * distance)) * f32{0.5};
            }
        }

        constexpr void operator()(i32 b, i32 h, i32 w) {
            // Normalize the patch.
            const auto [mean, stddev] = mean_stddev[b];
            auto& value = patches(b, h, w);
            value -= mean;
            value /= stddev;

            // Smooth edges.
            f32 mask{1};
            const auto grid_indices = noa::offset2index(b, grid_shape[1]);
            const auto is_at_left_edge = grid_indices.cmp_eq(0);
            const auto is_at_right_edge = grid_indices.cmp_eq(grid_shape.vec - 1);

            const auto coordinates = Vec{h, w};
            for (i32 i{}; i < 2; ++i) {
                if (not is_at_left_edge[i] and not is_at_right_edge[i]) {
                    // The patch is fully contained within this image axis.
                    const auto centered_coordinates = abs(static_cast<f32>(coordinates[i] - patch_center));
                    apply_mask(centered_coordinates, taper_radius, mask);
                } else {
                    // The patch is at the edge of the grid and may be outside the image,
                    // so adjust the coordinate to keep the taper at the edge of the signal.
                    const auto new_size = patch_size - patch_overflow[i];
                    const auto new_center = new_size / 2;
                    const auto new_radius = noa::max(0.f, new_center - taper_smoothness);
                    auto centered_coordinates = static_cast<f32>(coordinates[i]) - new_center;
                    if (is_at_left_edge[i])
                        centered_coordinates -= patch_overflow[i];
                    apply_mask(abs(centered_coordinates), new_radius, mask);
                }
            }
            value *= mask;
        }
    };

    struct EquiphaseBin {
        using reduce_type = Vec<f32, 2>;
        using remove_default_post = bool;

        SpanContiguous<const f32, 4, i32> polar{}; // (p,b,h,w)
        SpanContiguous<const f32, 1, i32> defoci{}; // (p)

        ns::CTFIsotropic<f32> isotropic_ctf; // with correct phase-shift
        ns::CTFAnisotropic<f32> anisotropic_ctf; // with correct phase-shift, astigmatism value and angle

        f32 phi_start{};
        f32 phi_bin_step{};
        f32 phi_row_step{};
        f32 rho_start{};
        f32 rho_step{};
        f32 rho_range{};

        NOA_HD void operator()(isize patch, isize bin, isize row, isize col, reduce_type& r) {
            // Get the polar coordinates within the original polar (p,h,w) array.
            const auto phi_bin_offset = static_cast<f32>(bin) * phi_bin_step;
            const auto phi_row_offset = static_cast<f32>(row) * phi_row_step;
            const auto phi = phi_start + phi_bin_offset + phi_row_offset; // radians
            const auto rho = static_cast<f32>(col) * rho_step + rho_start; // fftfreq

            // Set the astigmatic CTF of the patch.
            const auto& patch_defocus = defoci[patch];
            auto astigmatic_defocus = anisotropic_ctf.defocus();
            astigmatic_defocus.value = patch_defocus;
            anisotropic_ctf.set_defocus(astigmatic_defocus);

            // Get the phase at the central line of the bin.
            // This is assuming the bin height starts at zero; otherwise it should be phi_bin_start + phi_bin_offset
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi_bin_offset));
            // isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi_bin_offset-noa::Constant<f32>::PI/2)); // FIXME
            const auto phase = isotropic_ctf.phase_at(rho);

            // Get the corresponding fftfreq within the astigmatic field.
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            const auto fftfreq = isotropic_ctf.fftfreq_at(phase);
            if (not noa::is_finite(fftfreq))
                return;

            // Scale back to unnormalized frequency.
            const auto width = polar.shape().width();
            const auto frequency = static_cast<f32>(width - 1) * (fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = frequency - floored;
            const auto index = static_cast<isize>(floored);

            f32 v0{}, w0{}, v1{}, w1{};
            if (index >= 0 and index < width) {
                v0 = polar(patch, bin, row, index);
                w0 = 1;
            }
            if (index + 1 >= 0 and index + 1 < width) {
                v1 = polar(patch, bin, row, index + 1);
                w1 = 1;
            }
            r[0] += v0 * (1 - fraction) + v1 * fraction;
            r[1] += w0 * (1 - fraction) + w1 * fraction;
        }

        static constexpr void join(reduce_type r, reduce_type& j) {
            j += r;
        }

        template<typename T>
        static constexpr void post(reduce_type j, T& f) {
            f = static_cast<T>(j[1] > 1 ? j[0] / j[1] : j[0]);
        }
    };

    struct PolarTransform {
        isize n_wedges{};
        f64 wedge_step{};
        f64 wedge_half_step{};

        noa::Linspace<f64> wedges_phi_range{};
        Array<f32> patches_polar{};
        View<f32> patches_polar_bin{};

        CTFAnisotropic64 ctf{};
        Array<f32> defoci{};
        Array<f32> defoci_device{};
        f32 phi_row_step{};
        SpanContiguous<const Vec<f64, 2>> patch_centers{};

        PolarTransform(
            const CTFIsotropic64& empty_ctf,
            const ctf::Grid& grid,
            isize n_patches,
            isize polar_width,
            f64 target_bin_angle,
            isize target_phi_size,
            ArrayOption options
        ) {
            // We can average phi-lines together, effectively computing rotational averages of the spectra over wedges
            // of a given angle. If there is some astigmatism in the spectra, this results in a loss of information.
            // However, astigmatism doesn't change that quickly with phi, so we can reduce the polar height by taking
            // wedges of 2-to-5 degrees without losing too much information. Additionally, this can be corrected if the
            // astigmatism is estimated, in which case an EPA is used to bin.
            n_wedges = static_cast<isize>(std::round(180. / target_bin_angle));
            wedge_step = noa::deg2rad(180. / static_cast<f64>(n_wedges));
            wedge_half_step = wedge_step / 2;

            if (n_wedges == 1) {
                // No astigmatism is to be fitted, so compute and save the rotational averages.
                wedges_phi_range = noa::Linspace{0., noa::Constant<f64>::PI, false};
                patches_polar = Array<f32>({n_patches, 1, target_phi_size, polar_width}, options);
                patches_polar_bin = patches_polar.view();
            } else {
                // We want to divide the [0,180) range into n wedges of equal size. Before converting to polar space,
                // we want to know the phi-range to compute so that we can easily bin the wedges. In polar space, we
                // make the height of the bins odd, and the first bin needs to be centered on 0 deg. The bin centered on
                // 180 deg is the same as the one centered on 0 deg, so we don't include it. Thus, the resulting linspace
                // of the patch phi we need to render is:
                //      linspace(start=-wedge_half_angle, stop=180-wedge_half_angle, size=patch_phi+1, endpoint=true)
                wedges_phi_range = noa::Linspace{-wedge_half_step, noa::Constant<f64>::PI - wedge_half_step, true};

                // Make the polar height divisible by the number of bins.
                auto patch_phi = noa::next_multiple_of(target_phi_size, n_wedges);
                const auto wedge_phi = patch_phi / n_wedges + 1;
                patch_phi += 1;

                // Allocate the per-image polar patches.
                const auto polar_shape = Shape{n_patches, isize{1}, patch_phi, polar_width};
                patches_polar = Array<f32>(polar_shape, options);

                // For instance, wedge_step=3 and patch_size=120:
                // 1. The linspace is [-1.5,0.,1.5,3.,4.5,6.,7.5,9.,10.5, 12.,13.5,15.,16.5,...,178.5] degrees.
                //     This is the patch phi range, described above, that we need to compute during the polar transform.
                // 2. The corresponding wedge centers are at [0., 3., 6., 9., 12., 15., ..., 177] degrees.
                //    This is the phi linspace of the binned output.
                // As such, the wedges in the input would be covering 0=[-1.5, 0., 1.5], 1=[1.5, 3., 4.5], 2=[4.5, 6., 7.5]...
                // Notice the necessary overlap between each wedge. To compute the reduction from 1. to 2., we reshape the
                // polar spectrum to expose the wedge phi into a distinct dimension. The stride of this new dimension is
                // then subtracted by one, effectively duplicating the phi line between each wedge, creating the overlap.
                const auto polar_reduce_shape = Shape{n_patches, n_wedges, wedge_phi, polar_width};
                auto polar_reduce_strides = patches_polar.strides();
                polar_reduce_strides[1] = (wedge_phi - 1) * polar_reduce_strides[2]; // -1 to overlap
                check(noa::offset_at(patches_polar.strides(), polar_shape.vec - 1) ==
                      noa::offset_at(polar_reduce_strides, polar_reduce_shape.vec - 1));
                patches_polar_bin = View(patches_polar.get(), polar_reduce_shape, polar_reduce_strides, options);
            }

            // Prepare a few things for the (eventual) equiphase-binning.
            ctf = CTFAnisotropic64(empty_ctf);
            defoci = Array<f32>(n_patches);
            defoci_device = Array<f32>(n_patches, options);
            phi_row_step = static_cast<f32>(wedges_phi_range.for_size(patches_polar.shape()[2]).step);
            patch_centers = grid.patches_centers();
        }

        auto run(
            const Metadata::Image& metadata,
            const ctf::Grid& grid,
            const ctf::Patches& patches,
            const View<f32>& patches_padded_rfft_ps,
            const Shape4& patches_padded_shape,
            const Vec<f64, 2>& fftfreq_range,
            isize polar_width,
            nx::Interp polar_interp
        ) {
            // Transform the power-spectra to polar space. This will allow us to efficiently compute
            // (astigmatism-corrected) rotational averages by a simple reduction along the height.
            if (polar_interp == nx::Interp::CUBIC_BSPLINE)
                nx::cubic_bspline_prefilter(patches_padded_rfft_ps, patches_padded_rfft_ps);
            nx::spectrum2polar<"h2fc">(
                    patches_padded_rfft_ps, patches_padded_shape, patches_polar.view(), {
                        .spectrum_fftfreq = noa::Linspace{0., fftfreq_range[1], true},
                        .rho_range = patches.rho(),
                        .phi_range = wedges_phi_range,
                        .interp = polar_interp,
                });

            // Bin the wedges.
            auto output_bin =
                patches.patches(metadata.index)
                .reshape({-1, n_wedges, 1, polar_width}); // (p,b,1,w)

            if (metadata.defocus.astigmatism < 0.005) {
                noa::reduce_axes_ewise( // (p,b,h,w)->(p,b,1,w)
                    patches_polar_bin, f32{0}, output_bin,
                    noa::ReduceMean{.size = static_cast<f32>(patches_polar_bin.shape()[2])}
                );
            } else {
                // Set the CTF of each patch.
                const auto image_angles = noa::deg2rad(metadata.angles);
                const auto image_spacing = ctf.pixel_size();
                ctf.set_phase_shift(metadata.phase_shift);
                ctf.set_defocus(metadata.defocus);
                for (auto&& [patch_defocus, patch_center]: noa::zip(defoci.span_1d(), patch_centers)) {
                    const auto patch_z_offset_um = grid.patch_z_offset(image_angles, image_spacing, patch_center);
                    patch_defocus = static_cast<f32>(metadata.defocus.value - patch_z_offset_um); // underfocus negative
                }
                defoci.to(defoci_device);

                // Equiphase-binning.
                auto equiphase_bin = EquiphaseBin{
                    .polar = patches_polar_bin.span_contiguous<const f32, 4, i32>(), // (p,b,h,w)
                    .defoci = defoci_device.span_contiguous<const f32, 1, i32>(), // (p)
                    .isotropic_ctf = ns::CTFIsotropic(ctf).as<f32>(),
                    .anisotropic_ctf = ctf.as<f32>(),
                    .phi_start = static_cast<f32>(wedges_phi_range.start),
                    .phi_bin_step = static_cast<f32>(wedge_step),
                    .phi_row_step = phi_row_step,
                    .rho_start = static_cast<f32>(patches.rho().start),
                    .rho_step = static_cast<f32>(patches.rho_step()),
                    .rho_range = static_cast<f32>(patches.rho().stop - patches.rho().start), // assumes endpoint=true
                };
                noa::reduce_axes_iwise( // (p,b,h,w)->(p,b,1,w)
                    patches_polar_bin.shape().as<i32>(), patches_polar_bin.device(),
                    EquiphaseBin::reduce_type{}, output_bin, equiphase_bin
                );
            }
        }
    };

    // struct PolarTransform2 {
    //     isize n_wedges{};
    //     f64 wedge_step{};
    //     f64 wedge_half_step{};
    //
    //     noa::Linspace<f64> wedges_phi_range{};
    //     Array<f32> patches_polar{};
    //     View<f32> patches_polar_bin{};
    //
    //     CTFAnisotropic64 ctf{};
    //     Array<f32> defoci{};
    //     Array<f32> defoci_device{};
    //     f32 phi_row_step{};
    //
    //     PolarTransform2(
    //         const CTFIsotropic64& empty_ctf,
    //         isize n_patches,
    //         isize polar_width,
    //         f64 target_bin_angle,
    //         isize target_phi_size,
    //         ArrayOption options
    //     ) {
    //         // We can average phi-lines together, effectively computing rotational averages of the spectra over wedges
    //         // of a given angle. If there is some astigmatism in the spectra, this results in a loss of information.
    //         // However, astigmatism doesn't change that quickly with phi, so we can reduce the polar height by taking
    //         // wedges of 2-to-5 degrees without losing too much information. Additionally, this can be corrected if the
    //         // astigmatism is estimated, in which case an EPA is used to bin.
    //         n_wedges = static_cast<isize>(std::round(180. / target_bin_angle));
    //         wedge_step = noa::deg2rad(180. / static_cast<f64>(n_wedges));
    //         wedge_half_step = wedge_step / 2;
    //
    //         if (n_wedges == 1) {
    //             // No astigmatism is to be fitted, so compute and save the rotational averages.
    //             wedges_phi_range = noa::Linspace{-noa::Constant<f64>::PI/2, noa::Constant<f64>::PI/2, false};
    //             patches_polar = Array<f32>({n_patches, 1, target_phi_size, polar_width}, options);
    //             patches_polar_bin = patches_polar.view();
    //         } else {
    //             // We want to divide the [0,180) range into n wedges of equal size. Before converting to polar space,
    //             // we want to know the phi-range to compute so that we can easily bin the wedges. In polar space, we
    //             // make the height of the bins odd, and the first bin needs to be centered on 0 deg. The bin centered on
    //             // 180 deg is the same as the one centered on 0 deg, so we don't include it. Thus, the resulting linspace
    //             // of the patch phi we need to render is:
    //             //      linspace(start=-wedge_half_angle, stop=180-wedge_half_angle, size=patch_phi+1, endpoint=true)
    //             wedges_phi_range = noa::Linspace{-noa::Constant<f64>::PI/2 - wedge_half_step,
    //                                               noa::Constant<f64>::PI/2 - wedge_half_step, true};
    //
    //             // Make the polar height divisible by the number of bins.
    //             auto patch_phi = noa::next_multiple_of(target_phi_size, n_wedges);
    //             const auto wedge_phi = patch_phi / n_wedges + 1;
    //             patch_phi += 1;
    //
    //             // Allocate the per-image polar patches.
    //             const auto polar_shape = Shape{n_patches, isize{1}, patch_phi, polar_width};
    //             patches_polar = Array<f32>(polar_shape, options);
    //
    //             // For instance, wedge_step=3 and patch_size=120:
    //             // 1. The linspace is [-1.5,0.,1.5,3.,4.5,6.,7.5,9.,10.5, 12.,13.5,15.,16.5,...,178.5] degrees.
    //             //     This is the patch phi range, described above, that we need to compute during the polar transform.
    //             // 2. The corresponding wedge centers are at [0., 3., 6., 9., 12., 15., ..., 177] degrees.
    //             //    This is the phi linspace of the binned output.
    //             // As such, the wedges in the input would be covering 0=[-1.5, 0., 1.5], 1=[1.5, 3., 4.5], 2=[4.5, 6., 7.5]...
    //             // Notice the necessary overlap between each wedge. To compute the reduction from 1. to 2., we reshape the
    //             // polar spectrum to expose the wedge phi into a distinct dimension. The stride of this new dimension is
    //             // then subtracted by one, effectively duplicating the phi line between each wedge, creating the overlap.
    //             const auto polar_reduce_shape = Shape{n_patches, n_wedges, wedge_phi, polar_width};
    //             auto polar_reduce_strides = patches_polar.strides();
    //             polar_reduce_strides[1] = (wedge_phi - 1) * polar_reduce_strides[2]; // -1 to overlap
    //             check(noa::offset_at(patches_polar.strides(), polar_shape.vec - 1) ==
    //                   noa::offset_at(polar_reduce_strides, polar_reduce_shape.vec - 1));
    //             patches_polar_bin = View(patches_polar.get(), polar_reduce_shape, polar_reduce_strides, options);
    //         }
    //
    //         // Prepare a few things for the (eventual) equiphase-binning.
    //         ctf = CTFAnisotropic64(empty_ctf);
    //         defoci = Array<f32>(n_patches);
    //         defoci_device = Array<f32>(n_patches, options);
    //         phi_row_step = static_cast<f32>(wedges_phi_range.for_size(patches_polar.shape()[2]).step);
    //     }
    //
    //     auto run(
    //         const Metadata::Image& metadata,
    //         const ctf::Patches& patches,
    //         const View<f32>& patches_padded_rfft_ps,
    //         const Shape4& patches_padded_shape,
    //         const Vec<f64, 2>& fftfreq_range,
    //         isize polar_width,
    //         nx::Interp polar_interp
    //     ) {
    //         // Transform the power-spectra to polar space. This will allow us to efficiently compute
    //         // (astigmatism-corrected) rotational averages by a simple reduction along the height.
    //         if (polar_interp == nx::Interp::CUBIC_BSPLINE)
    //             nx::cubic_bspline_prefilter(patches_padded_rfft_ps, patches_padded_rfft_ps);
    //         nx::spectrum2polar<"hc2fc">(
    //                 patches_padded_rfft_ps, patches_padded_shape, patches_polar.view(), {
    //                     .spectrum_fftfreq = noa::Linspace{0., fftfreq_range[1], true},
    //                     .rho_range = patches.rho(),
    //                     .phi_range = wedges_phi_range,
    //                     .interp = polar_interp,
    //             });
    //         noa::write_image(patches_polar, "~/Tmp/quinoa/figures/02/astig02/patches_polar.mrc");
    //         noa::write_image(patches_polar_bin, "~/Tmp/quinoa/figures/02/astig02/patches_polar_bin.mrc");
    //
    //         // Bin the wedges.
    //         auto output_bin =
    //             patches.patches(metadata.index)
    //             .reshape({-1, n_wedges, 1, polar_width}); // (p,b,1,w)
    //
    //         // if (metadata.defocus.astigmatism < 0.005) {
    //             noa::reduce_axes_ewise( // (p,b,h,w)->(p,b,1,w)
    //                 patches_polar_bin, f32{0}, output_bin,
    //                 noa::ReduceMean{.size = static_cast<f32>(patches_polar_bin.shape()[2])}
    //             );
    //         noa::write_image(output_bin.permute({0,2,1,3}), "~/Tmp/quinoa/figures/02/astig02/output_bin1.mrc");
    //         // } else {
    //             // Set the CTF of each patch.
    //             const auto image_angles = noa::deg2rad(metadata.angles);
    //             const auto image_spacing = ctf.pixel_size();
    //             ctf.set_phase_shift(metadata.phase_shift);
    //             ctf.set_defocus(metadata.defocus);
    //             const auto patch_center = (patches_padded_shape.filter(2, 3).vec / 2).as<f64>();
    //             for (auto& patch_defocus: defoci.span_1d()) {
    //                 const auto patch_z_offset_um = ctf::Grid::patch_z_offset(
    //                     patch_center, image_angles, image_spacing, patch_center);
    //                 patch_defocus = static_cast<f32>(metadata.defocus.value - patch_z_offset_um); // underfocus negative
    //             }
    //             defoci.to(defoci_device);
    //
    //             // Equiphase-binning.
    //             auto equiphase_bin = EquiphaseBin{
    //                 .polar = patches_polar_bin.span_contiguous<const f32, 4, i32>(), // (p,b,h,w)
    //                 .defoci = defoci_device.span_contiguous<const f32, 1, i32>(), // (p)
    //                 .isotropic_ctf = ns::CTFIsotropic(ctf).as<f32>(),
    //                 .anisotropic_ctf = ctf.as<f32>(),
    //                 .phi_start = static_cast<f32>(wedges_phi_range.start),
    //                 .phi_bin_step = static_cast<f32>(wedge_step),
    //                 .phi_row_step = phi_row_step,
    //                 .rho_start = static_cast<f32>(patches.rho().start),
    //                 .rho_step = static_cast<f32>(patches.rho_step()),
    //                 .rho_range = static_cast<f32>(patches.rho().stop - patches.rho().start), // assumes endpoint=true
    //             };
    //             noa::reduce_axes_iwise( // (p,b,h,w)->(p,b,1,w)
    //                 patches_polar_bin.shape().as<i32>(), patches_polar_bin.device(),
    //                 EquiphaseBin::reduce_type{}, output_bin, equiphase_bin
    //             );
    //         noa::write_image(output_bin.permute({0,2,1,3}), "~/Tmp/quinoa/figures/02/astig02/output_bin2.mrc");
    //         // }
    //     }
    // };
}

namespace qn::ctf {
    // void test05(
    //     const CTFAnisotropic64& ctf,
    //     isize polar_width,
    //     f64 target_bin_angle,
    //     isize target_phi_size,
    //     nx::Interp polar_interp,
    //     noa::Linspace<f64> rho_range,
    //     noa::Linspace<f64> phi_range,
    //     View<f32> patches_padded_rfft_ps,
    //     Shape4 patches_padded_shape,
    //     const Vec<f64, 2>& fftfreq_range
    // ) {
    //     constexpr isize n_patches = 1;
    //     auto polar_transform = PolarTransform2(
    //         CTFIsotropic64(ctf), n_patches, polar_width, target_bin_angle, target_phi_size, {});
    //
    //     auto output = Patches{};
    //     output.m_rho_range = rho_range;
    //     output.m_phi_range = noa::Linspace<f64>{noa::deg2rad(-90.), noa::deg2rad(90.), false};
    //     output.m_polar = Array<f16>({1, n_patches, polar_transform.n_wedges, polar_width});
    //
    //     auto image_metadata = Metadata::Image{
    //         .index = 0,
    //         .index_file = 1,
    //         .angles = {},
    //         .shifts = {},
    //         .exposure = {},
    //         .phase_shift = 0,
    //         .defocus = ctf.defocus(),
    //         .time = 0,
    //         .frames = nullptr,
    //     };
    //     polar_transform.run(
    //         image_metadata, output,
    //         patches_padded_rfft_ps.view(), patches_padded_shape,
    //         fftfreq_range, polar_width, polar_interp
    //     );
    //     noa::write_image(output.m_polar, "~/Tmp/quinoa/figures/02/astig02/polar_binned.mrc");
    // }

    auto Patches::from_stack(
        StackLoader& stack_loader,
        const Metadata& metadata,
        const Grid& grid,
        const Vec<f64, 2>& resolution_range,
        isize patch_size,
        isize patch_padded_size,
        f64 target_bin_angle,
        isize target_phi_size,
        nx::Interp polar_interp
    ) -> Patches {
        auto timer = Logger::info_scope_time("Loading patches");

        const auto options = ArrayOption{stack_loader.compute_device(), Allocator::ASYNC};
        const auto image = Array<f32>(grid.slice_shape().push_front<2>(1), options);
        const auto n_patches = grid.n_patches();
        check(grid.patch_size() == patch_size);

        const auto allocated_start = Allocator::bytes_currently_allocated(options.device);

        // The patches are loaded one image at a time. So allocate enough for one image.
        const auto patches_shape = grid.patch_shape().push_front(Vec{n_patches, isize{1}});
        const auto patches_rfft = Array<c32>(patches_shape.rfft(), options);
        const auto patches = nf::alias_to_real(patches_rfft.view(), patches_shape);

        const auto spacing = mean(stack_loader.stack_spacing()); // assume isotropic
        check(noa::allclose(spacing, metadata.spacing));

        // Prepare to Fourier-crop the patches to the integer frequency closest to the target end resolution.
        // Don't allocate if there's no need to crop, just point to the original patches.
        const auto [cropped_size, fftfreq_end] = fourier_crop_to_resolution(patch_size, spacing, resolution_range[1], true);
        const bool has_cropping = patch_size > cropped_size;
        const auto downscaling_factor = static_cast<f64>(patch_size) / static_cast<f64>(cropped_size);
        const auto patches_cropped_shape = Shape{n_patches, isize{1}, cropped_size, cropped_size};
        const auto patches_cropped_rfft = has_cropping ? Array<c32>(patches_cropped_shape.rfft(), options) : patches_rfft;
        const auto patches_cropped = nf::alias_to_real(patches_cropped_rfft.view(), patches_cropped_shape);

        // Prepare to zero-pad the (possibly cropped) patches to optimize the sampling for the fitting.
        // Don't allocate if there's no padding or if we pad back to the original size, just point to corresponding patches.
        patch_padded_size = std::max(patch_padded_size, cropped_size);
        const auto patches_padded_shape = Shape{n_patches, isize{1}, patch_padded_size, patch_padded_size};
        const auto zero_padding = (patches_padded_shape - patches_cropped_shape).vec;
        const bool has_padding = zero_padding != 0;
        const bool pad_to_original_size = patch_padded_size == patch_size;
        auto patches_padded_rfft = Array<c32>{};
        if (not has_padding)
            patches_padded_rfft = patches_cropped_rfft;
        else if (pad_to_original_size)
            patches_padded_rfft = patches_rfft;
        else
            patches_padded_rfft = Array<c32>(patches_padded_shape.rfft(), options);
        const auto patches_padded = nf::alias_to_real(patches_padded_rfft.view(), patches_padded_shape);
        const auto patches_padded_rfft_ps = noa::like<f32>(patches_padded_rfft);

        // Then the (possibly oversampled) patches will be transformed to polar space.
        // The polar transformation can and should remove the low frequencies outside the resolution range.
        // Select the starting fftfreq to also be at the nearest integer frequency and round up to a nice size.
        auto polar_width = patch_padded_size / 2 + 1;
        const auto [rho_index, fftfreq_start] = nearest_integer_fftfreq(
            polar_width, Vec{0., fftfreq_end}, resolution_to_fftfreq(spacing, resolution_range[0]));
        polar_width -= rho_index;
        polar_width = noa::next_multiple_of(polar_width, 16);

        const auto fftfreq_range = Vec{fftfreq_start, fftfreq_end}; // rho range
        Logger::info(
            "Patch maximum frequency range:\n"
            "  resolution_range={::.3f}A (target={::.3f}A\n"
            "  fftfreq_range={::.5f}cpp (target={::.5f}cpp)",
            fftfreq_to_resolution(spacing, fftfreq_range), resolution_range,
            fftfreq_range, resolution_to_fftfreq(spacing, resolution_range)
        );
        Logger::info(
            "Oversampling the patches:\n"
            "  original_size={}\n"
            "  fourier_cropped_size={}\n"
            "  padded_size={}",
            patch_size, cropped_size, patch_padded_size
        );

        // Prepare the polar transform and equiphase-binning.
        target_phi_size = std::max(target_phi_size, patch_padded_size);
        auto polar_transform = PolarTransform(
            metadata.empty_ctf(), grid, n_patches, polar_width, target_bin_angle, target_phi_size, options);

        // Prepare patch normalization and smooth edge mask.
        // While this can be done at the image level using the StackLoader, this is significantly faster because it
        // allows to skip the FFTs for the highpass. Here the patches are small enough that we can safely assume that
        // the ice gradient within each patch is negligible, so we can center the mean and apply the smooth edge taper
        // without a highpass. This way, the stack loader can load the images without any expensive preprocessing.
        const auto cropped_overflow = (grid.overflow().first.as<f64>() / downscaling_factor).as<f32>();
        const auto mean_stddev = Array<Vec<f32, 2>>(n_patches, options);
        const auto taper_radius = static_cast<f64>(cropped_size) * (2. / 4.) * 0.5;
        const auto taper_smoothness = static_cast<f64>(cropped_size) * (2. / 4.) * 0.5;
        const auto patch_mean_variance = PatchMeanVariance{
            .patches = patches_cropped.span_contiguous<const f32, 3, i32>(),
            .grid_shape = grid.shape().as<i32>(),
            .overflow = cropped_overflow,
            .patch_size = static_cast<f32>(cropped_size),
        };
        const auto patch_mask = PatchMask{
            .patches = patches_cropped.span_contiguous<f32, 3, i32>(),
            .mean_stddev = mean_stddev.span_1d<const Vec<f32, 2>, i32>(),
            .grid_shape = grid.shape().as<i32>(),
            .patch_overflow = cropped_overflow,
            .patch_center = static_cast<i32>(cropped_size / 2),
            .patch_size = static_cast<f32>(cropped_size),
            .taper_radius = static_cast<f32>(taper_radius),
            .taper_smoothness = static_cast<f32>(taper_smoothness),
        };

        // Optimize the FFT plans.
        nf::set_cache_limit(10, options.device);
        stack_loader.record_fft();
        nf::r2c(patches, patches_rfft, {.record_and_share_workspace = true});
        nf::c2r(patches_cropped_rfft, patches_cropped, {.record_and_share_workspace = true});
        nf::r2c(patches_padded, patches_padded_rfft, {.record_and_share_workspace = true});
        nf::allocate_workspace(options.device, options.allocator);

        const auto allocated_stop = Allocator::bytes_currently_allocated(options.device);
        Logger::trace(
            "Allocated: {:.2f}GB on {} ({})",
            static_cast<f64>(allocated_stop - allocated_start) * 1e-9, options.device, options.allocator);

        // Create the big array with all the patches in polar space.
        auto output = Patches{};
        output.m_rho_range = noa::Linspace{fftfreq_range[0], fftfreq_range[1], true};
        output.m_phi_range = noa::Linspace{0., noa::Constant<f64>::PI, false};
        output.m_polar = Array<value_type>({metadata.stack.ssize(), n_patches, polar_transform.n_wedges, polar_width}, {
            .device = options.device,
            .allocator = Allocator::PITCHED_MANAGED // keep lines aligned
        });

        const auto n_allocated = output.m_polar.shape().set<3>(output.m_polar.strides()[2]).as<size_t>().n_elements();
        Logger::info(
            "Polar patches:\n"
            "  interp={}\n"
            "  n_lines={} (initial_n_lines={}, wedge=[{:.2f}deg, n_lines={}])\n"
            "  shape=[n_images={}, n_patches={}, phi={}, rho={}]\n"
            "  memory={:.0f}MB on {} ({}, dtype={})",
            polar_interp,
            polar_transform.n_wedges, polar_transform.patches_polar.shape()[2],
            noa::rad2deg(polar_transform.wedge_step), polar_transform.patches_polar_bin.shape()[2],
            output.n_images(), output.n_patches_per_image(), output.height(), output.width(),
            static_cast<f64>(n_allocated * sizeof(value_type)) * 1e-6,
            options.device, output.m_polar.allocator(), noa::details::stringify<value_type>()
        );

        // Prepare the subregion origins, ready for extract_subregions.
        const auto patches_origins = grid.compute_subregion_origins().to(options);

        // Load the images in the same order as saved in the metadata.
        auto metadata_sorted = metadata.stack;
        metadata_sorted.reset_indices();

        for (const auto& image_metadata: metadata_sorted) {
            // Extract the patches. Assume the slice is normalized and edges are tapered.
            stack_loader.read_slice(image.view(), image_metadata.index_file);
            noa::extract_subregions(image.view(), patches, patches_origins.view());

            // Crop to the maximum frequency.
            if (has_cropping) {
                nf::r2c(patches, patches_rfft);
                nf::resize<"h">(patches_rfft, patches.shape(), patches_cropped_rfft, patches_cropped.shape(), {.correct_nyquist = true});
                nf::c2r(patches_cropped_rfft, patches_cropped);
            }

            // Smooth edges and normalize.
            noa::reduce_axes_iwise(
                patch_mean_variance.patches.shape(), options.device,
                Vec{0., 0.}, mean_stddev.flat(1), patch_mean_variance
            );
            noa::iwise(patch_mask.patches.shape(), options.device, patch_mask);

            // Add real-space padding to increase the sampling, if necessary.
            if (has_padding)
                noa::resize(patches_cropped, patches_padded, {}, zero_padding);

            // Compute the power-spectra, making sure to normalize the FFT now since we will not be calling c2r.
            nf::r2c(patches_padded, patches_padded_rfft.view(), {.norm = nf::Norm::NONE});
            const auto fft_scale = 1 / static_cast<f32>(patches_padded.shape().filter(2, 3).n_elements());
            noa::ewise(patches_padded_rfft.view(), patches_padded_rfft_ps, [=]NOA_HD(const c32& i, f32& o) {
                o = noa::abs_squared(i) * fft_scale;
            });

            // Polar transform and equiphase-binning.
            polar_transform.run(
                image_metadata, grid, output,
                patches_padded_rfft_ps.view(), patches_padded_shape,
                fftfreq_range, polar_width, polar_interp
            );
        }
        output.view().eval();
        Logger::trace("Loaded {} images (tilt_range={::+.2f})", metadata_sorted.size(), metadata_sorted.tilt_range());
        nf::clear_cache(options.device);

        return output;
    }
}
