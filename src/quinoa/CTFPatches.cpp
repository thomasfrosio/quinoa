#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>

#include "quinoa/CTF.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Plot.hpp"

namespace qn::ctf {
    auto Patches::from_stack(
        StackLoader& stack_loader,
        const MetadataStack& metadata,
        const Grid& grid,
        const Vec<f64, 2>& resolution_range,
        i64 patch_size,
        i64 patch_padded_size,
        f64 target_bin_angle,
        i64 target_phi_size,
        noa::Interp polar_interp
    ) -> Patches {
        auto timer = Logger::info_scope_time("Loading patches");

        // The patches are loaded one image at a time. So allocate enough for one image.
        const auto options = ArrayOption{stack_loader.compute_device(), Allocator::ASYNC};
        const auto image = Array<f32>(grid.slice_shape().push_front<2>(1), options);
        const auto patches_shape = grid.patch_shape().push_front(Vec{grid.n_patches(), i64{1}});
        const auto patches_rfft = Array<c32>(patches_shape.rfft(), options);
        const auto patches = nf::alias_to_real(patches_rfft.view(), patches_shape);

        // For below one pixel-sizes (e.g. super-resolution cases), the resolution limit (4.5 at the time of writing)
        // becomes is too low resolution, and we end up cropping too much of the original spectrum. This limits the
        // Fourier cropping up to 0.2 fftfreq (so for a resolution limit of 4.5, anything above 0.9A/pix is unaffected)
        // TODO I don't think this is necessary. Check if this improves the spectrum.
        const auto spacing = mean(stack_loader.stack_spacing()); // assume isotropic
        const auto maximum_resolution = std::min(resolution_range[1], fftfreq_to_resolution(spacing, 0.2));

        // Fourier-crop the patches to the integer frequency closest to the target end resolution.
        const auto [cropped_size, fftfreq_end] = fourier_crop_to_resolution(patch_size, spacing, maximum_resolution, true);
        const auto patches_cropped_shape = Shape{grid.n_patches(), i64{1}, cropped_size, cropped_size};
        const auto patches_cropped_rfft = Array<c32>(patches_cropped_shape.rfft(), options);
        const auto patches_cropped = nf::alias_to_real(patches_cropped_rfft.view(), patches_cropped_shape);

        // Then go back to real space to zero-pad, effectively increasing the sampling and stretching the Thon rings.
        // In case we zero-pad back to the original patch size, don't allocate and reuse the patches instead.
        const auto patches_padded_shape = Shape{grid.n_patches(), i64{1}, patch_padded_size, patch_padded_size};
        const auto zero_padding = (patches_padded_shape - patches_cropped_shape).vec;
        const bool has_padding = noa::any(zero_padding != 0);
        const auto buffer_padding = has_padding ? Array<c32>(patches_padded_shape.rfft(), options) : Array<c32>{};
        const auto patches_padded_rfft = has_padding ? buffer_padding.view() : patches_rfft.view();
        const auto patches_padded = has_padding ? nf::alias_to_real(patches_padded_rfft, patches_padded_shape) : patches;
        const auto patches_padded_rfft_ps = noa::like<f32>(patches_padded_rfft);

        // Then the zero-padded (oversampled) patch is transformed to polar space.
        // The polar transformation can and should remove the low frequencies outside the resolution range.
        // To reduce interpolation, select the starting fftfreq to also be at the nearest integer frequency.
        auto polar_width = patch_padded_size / 2 + 1;
        const auto [rho_index, fftfreq_start] = nearest_integer_fftfreq(
            polar_width, Vec{0., fftfreq_end}, resolution_to_fftfreq(spacing, resolution_range[0]));
        polar_width -= rho_index;

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

        // We can average phi-lines together, effectively computing rotational averages of the spectra over wedges
        // of a given angle. If there is some astigmatism in the spectra, this results in a loss of information.
        // However, astigmatism doesn't change that quickly with phi, so we can reduce the polar height by taking
        // wedges of 2-to-5 degrees without losing much information.
        const bool use_wedges = target_bin_angle > 0;
        auto n_wedges = target_phi_size;
        auto wedges_phi_range = noa::Linspace{};
        auto wedges_fmt = std::string{};
        auto patches_polar = Array<f32>{};
        auto patches_polar_bin = View<f32>{};
        if (use_wedges) {
            n_wedges = static_cast<i64>(std::round(180. / target_bin_angle));
            const auto wedge_step = 180. / static_cast<f64>(n_wedges);
            const auto wedge_half_step = noa::deg2rad(wedge_step / 2);

            // We need to make sure wedge centered match exactly the [0,180) phi-step of the output.
            // Furthermore, the angular range of the wedges should be centered. This can be done by generating
            // the following linspace range.
            auto phi_size = noa::next_multiple_of(target_phi_size, n_wedges);
            wedges_phi_range = noa::Linspace{-wedge_half_step, noa::Constant<f64>::PI - wedge_half_step, true};
            auto wedge_size = phi_size / n_wedges + 1;
            phi_size += 1;

            const auto polar_shape = Shape{grid.n_patches(), i64{1}, phi_size, polar_width};
            patches_polar = Array<f32>(polar_shape, options);

            // For wedge_step=3, phi_size=120: [1.5,0.,1.5,3.,4.5,6.,7.5,9.,10.5, 12.,13.5,15.,16.5,...]
            // The [0,180) phi-step of the output would be: [0., 3., 6., 9., 12., 15., ..., 177].
            // And the wedges would be 0=[-1.5, 0., 1.5], 1=[1.5, 3., 4.5], 2=[4.5, 6., 7.5]...
            // As such, to keep them correctly centered, they need to overlap by one.
            const auto polar_reduce_shape = Shape{grid.n_patches(), n_wedges, wedge_size, polar_width};
            auto polar_reduce_strides = patches_polar.strides();
            polar_reduce_strides[1] = (wedge_size - 1) * polar_reduce_strides[2]; // -1 to overlap
            check(ni::offset_at(patches_polar.strides(), polar_shape.vec - 1) ==
                  ni::offset_at(polar_reduce_strides, polar_reduce_shape.vec - 1));
            patches_polar_bin = View(patches_polar.get(), polar_reduce_shape, polar_reduce_strides, options);

            wedges_fmt = not use_wedges ? "" : fmt::format(
                "  phi={} (initial={}, wedges=[{:.1}deg, size={}])\n",
                n_wedges, phi_size, wedge_step, wedge_size
            );
        }

        // Create the big array with all the patches in polar space.
        auto output = Patches{};
        output.m_rho_range = noa::Linspace{fftfreq_range[0], fftfreq_range[1], true};
        output.m_phi_range = noa::Linspace{0., noa::Constant<f64>::PI, false};
        output.m_polar = Array<value_type>({metadata.ssize(), grid.n_patches(), n_wedges, polar_width}, {
            .device = options.device,
            .allocator = Allocator::PITCHED_MANAGED
        });

        const auto n_allocated = output.m_polar.shape().set<3>(output.m_polar.strides()[2]).as<size_t>().n_elements();
        Logger::info(
            "Polar patches:\n"
            "  interp={}\n{}"
            "  shape=[n_images={}, n_patches={}, phi={}, rho={}]\n"
            "  allocated={:.2f}GB on {} ({}, dtype={})",
            polar_interp, std::move(wedges_fmt),
            output.n_images(), output.n_patches_per_image(), output.height(), output.width(),
            static_cast<f64>(n_allocated * sizeof(value_type)) * 1e-9,
            options.device, output.m_polar.allocator(), noa::string::stringify<value_type>()
        );

        // Prepare the subregion origins, ready for extract_subregions.
        const auto patches_origins = grid.compute_subregion_origins().to(options);

        // Load the images in the same order as saved in the metadata.
        auto metadata_sorted = metadata;
        metadata_sorted.reset_indices();
        for (const auto& slice_metadata: metadata_sorted) {
            // Extract the patches. Assume the slice is normalized and edges are tapered.
            stack_loader.read_slice(image.view(), slice_metadata.index_file);
            noa::extract_subregions(image.view(), patches, patches_origins.view());

            // Crop to the maximum frequency and oversample back to the original size
            // to nicely stretch the Thon rings, thus counteracting small pixel sizes and high defoci.
            nf::r2c(patches, patches_rfft);
            nf::resize<"h2h">(patches_rfft, patches.shape(), patches_cropped_rfft, patches_cropped.shape());
            nf::c2r(patches_cropped_rfft, patches_cropped);
            noa::normalize_per_batch(patches_cropped, patches_cropped);
            noa::resize(patches_cropped, patches_padded, {}, zero_padding);

            // Compute the power-spectra of these tiles.
            // Also, make sure to normalize the FFT since we will not be calling c2r.
            nf::r2c(patches_padded, patches_padded_rfft, {.norm = nf::Norm::NONE});
            const auto fft_scale = 1 / static_cast<f32>(patches_padded.shape().filter(2, 3).n_elements());
            noa::ewise(patches_padded_rfft, patches_padded_rfft_ps, [=]NOA_HD(const c32& i, f32& o) {
                o = noa::abs_squared(i) * fft_scale;
            });

            // Transform the power-spectra to polar space. This will allow us to efficiently compute
            // (astigmatism-corrected) rotational averages by a simple reduction along the height.
            // Note that we need to offset the phi start here to have the wedges centered on the [0,pi) range.
            if (polar_interp == noa::Interp::CUBIC_BSPLINE)
                noa::cubic_bspline_prefilter(patches_padded_rfft_ps, patches_padded_rfft_ps);

            auto output_patches = output.patches(slice_metadata.index);
            if (use_wedges) {
                ng::spectrum2polar<"h2fc">(
                    patches_padded_rfft_ps, patches_padded_shape, patches_polar.view(), {
                        .spectrum_fftfreq = noa::Linspace{0., fftfreq_range[1], true},
                        .rho_range = output.rho(),
                        .phi_range = wedges_phi_range,
                        .interp = polar_interp,
                });
                noa::reduce_axes_ewise( // (n,1,phi,rho)->(n,1,n_wedges,rho)
                    patches_polar_bin, f32{},
                    output_patches.reshape({-1, n_wedges, 1, polar_width}),
                    noa::ReduceMean{.size = static_cast<f32>(patches_polar_bin.shape()[2])}
                );
            } else {
                ng::spectrum2polar<"h2fc">(
                    patches_padded_rfft_ps, patches_padded_shape, output_patches, {
                        .spectrum_fftfreq = noa::Linspace{0., fftfreq_range[1], true},
                        .rho_range = output.rho(),
                        .phi_range = output.phi(),
                        .interp = polar_interp,
                });
            }

            Logger::trace("tilt={:>+6.2f}", slice_metadata.angles[1]);
        }
        output.view().eval();
        return output;
    }
}
