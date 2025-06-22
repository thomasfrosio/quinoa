#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>

#include "quinoa/CTF.hpp"

namespace qn::ctf {
    auto Patches::from_stack(
        StackLoader& stack_loader,
        const MetadataStack& metadata,
        const Grid& grid,
        const Vec<f64, 2>& resolution_range,
        i64 patch_size,
        i64 patch_padded_size,
        i64 phi_size,
        noa::Interp polar_interp
    ) -> Patches {
        // The patches are loaded one image at a time. So allocate enough for one image.
        const auto options = ArrayOption{stack_loader.compute_device(), Allocator::ASYNC};
        const auto image = Array<f32>(grid.slice_shape().push_front<2>(1), options);
        const auto patches_shape = grid.patch_shape().push_front(Vec{grid.n_patches(), i64{1}});
        const auto patches_rfft = Array<c32>(patches_shape.rfft(), options);
        const auto patches = noa::fft::alias_to_real(patches_rfft.view(), patches_shape);

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
        const auto patches_cropped = noa::fft::alias_to_real(patches_cropped_rfft.view(), patches_cropped_shape);

        // Then go back to real space to zero-pad, effectively increasing the sampling and stretching the Thon rings.
        // In case we zero-pad back to the original patch size, don't allocate and reuse the patches instead.
        const auto patches_padded_shape = Shape{grid.n_patches(), i64{1}, patch_padded_size, patch_padded_size};
        const auto zero_padding = (patches_padded_shape - patches_cropped_shape).vec;
        const bool has_padding = noa::any(zero_padding != 0);
        const auto buffer_padding = has_padding ? Array<c32>(patches_padded_shape.rfft(), options) : Array<c32>{};
        const auto patches_padded_rfft = has_padding ? buffer_padding.view() : patches_rfft.view();
        const auto patches_padded = has_padding ? noa::fft::alias_to_real(patches_padded_rfft, patches_padded_shape) : patches;
        const auto patches_padded_rfft_ps = noa::like<f32>(patches_padded_rfft);

        // Then the zero-padded (oversampled) patch is transformed to polar space.
        // The polar transformation can and should remove the low frequencies outside the resolution range.
        // To reduce interpolation, select the starting fftfreq to also be at the nearest integer frequency.
        auto polar_width = patch_padded_size / 2 + 1;
        const auto [rho_index, fftfreq_start] = nearest_integer_fftfreq(
            polar_width, Vec{0., fftfreq_end}, resolution_to_fftfreq(spacing, resolution_range[0]));
        polar_width -= rho_index;
        const auto polar_shape = Shape{metadata.ssize(), grid.n_patches(), phi_size, polar_width};
        const auto fftfreq_range = Vec{fftfreq_start, fftfreq_end}; // rho range

        Logger::info(
            "Patch maximum frequency range:\n"
            "  resolution_range={::.3f}A (target={::.3f}A\n"
            "  fftfreq_range={::.5f} (target={::.5f})",
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
        Logger::info(
            "Polar patches:\n"
            "  allocated={:.2f}GB on {} ({}, dtype={})\n"
            "  shape=[n_images:{}, n_patches:{}, phi={}, rho={}]",
            static_cast<f64>(polar_shape.as<size_t>().n_elements() * sizeof(value_type)) * 1e-9,
            options.device, options.allocator, noa::string::stringify<value_type>(),
           polar_shape[0], polar_shape[1], polar_shape[2], polar_shape[3]
        );

        // Create the big array with all the patches in polar space.
        // Importantly, use managed memory in case it doesn't fit in the GPU discrete memory.
        auto all_patches = Patches{};
        all_patches.m_rho_range = noa::Linspace{fftfreq_range[0], fftfreq_range[1], true};
        all_patches.m_phi_range = noa::Linspace{0., noa::Constant<f64>::PI, false};
        all_patches.m_polar = Array<value_type>(polar_shape, {
            .device = options.device,
            .allocator = Allocator::MANAGED
        });

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
            // TODO smooth edges?
            noa::normalize_per_batch(patches_cropped, patches_cropped);
            noa::resize(patches_cropped, patches_padded, {}, zero_padding);

            // Compute the power-spectra of these tiles,
            // but also make sure to normalize the FFT since we will not be calling c2r.
            noa::fft::r2c(patches_padded, patches_padded_rfft, {.norm = nf::Norm::NONE});
            const auto fft_scale = 1 / static_cast<f32>(patches_padded.shape().filter(2, 3).n_elements());
            noa::ewise(patches_padded_rfft, patches_padded_rfft_ps, [=]NOA_HD(const c32& i, f32& o) {
                o = noa::abs_squared(i) * fft_scale;
            });

            // Transform the power-spectra to polar space. This will allow us to efficiently compute
            // (astigmatism-corrected) rotational averages by a simple reduction along the height.
            if (polar_interp == noa::Interp::CUBIC_BSPLINE)
                noa::cubic_bspline_prefilter(patches_padded_rfft_ps, patches_padded_rfft_ps);
            ng::spectrum2polar<"h2fc">(
                patches_padded_rfft_ps, patches_padded_shape, all_patches.patches(slice_metadata.index), {
                    .spectrum_fftfreq = noa::Linspace{0., fftfreq_range[1], true},
                    .rho_range = all_patches.rho(),
                    .phi_range = all_patches.phi(),
                    .interp = polar_interp,
            });

            Logger::trace("tilt={:>+6.2f}", slice_metadata.angles[1]);
        }
        all_patches.view().eval();
        return all_patches;
    }

    void Patches::exclude_views(SpanContiguous<const i64> indices) {
        auto is_excluded = [&](i64 i) { return stdr::find(indices, i) != indices.end(); };

        i64 p{};
        for (i64 i{}; i < n_images(); ++i) {
            if (not is_excluded(i)) {
                if (i != p)
                    view().subregion(i).to(view().subregion(p));
                ++p;
            }
        }

        // Reshape only, the extra memory is kept,
        // which is better than having to reallocate the entire array.
        m_polar = m_polar.subregion(ni::Slice{0, p});
    }
}
