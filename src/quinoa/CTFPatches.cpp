#include "quinoa/CTF.hpp"

namespace qn::ctf {
    auto Patches::from_stack(
        StackLoader& stack_loader,
        MetadataStack& metadata,
        const Grid& grid,
        i64 fourier_cropped_size
    ) -> Patches {
        const auto patch_shape_2d = grid.patch_shape();
        const auto patches_shape = patch_shape_2d.push_front(Vec{grid.n_patches(), i64{1}});
        const auto fft_scale = 1 / static_cast<f32>(patch_shape_2d.n_elements());

        // The patches are loaded one slice at a time. So allocate enough for one slice.
        const auto options = ArrayOption{stack_loader.compute_device(), Allocator::ASYNC};
        const auto slice = Array<f32>(grid.slice_shape().push_front<2>(1), options);
        const auto patches_rfft = Array<c32>(patches_shape.rfft(), options);
        const auto patches = noa::fft::alias_to_real(patches_rfft.view(), patches_shape);

        // Fourier crop the patches to the target max frequency, and go back to real space to zero-pad back to the
        // original patch size, effectively increasing the sampling and stretching the Thon rings as much as possible.
        const auto patches_cropped_shape = Shape{grid.n_patches(), i64{1}, fourier_cropped_size, fourier_cropped_size};
        const auto patches_cropped_rfft = Array<c32>(patches_cropped_shape.rfft(), options);
        const auto patches_cropped = noa::fft::alias_to_real(patches_cropped_rfft.view(), patches_cropped_shape);
        const auto zero_padding = (patches_shape - patches_cropped_shape).vec;

        Logger::info(
            "Creating patches:\n"
            "  n_slices={}\n"
            "  n_patches={} (per_slice={})\n"
            "  patch_size={}\n"
            "  fourier_cropped_size={}\n",
            metadata.ssize(), grid.n_patches() * metadata.ssize(), grid.n_patches(),
            grid.patch_size(), fourier_cropped_size
        );

        // Prepare the subregion origins, ready for extract_subregions.
        const std::vector subregion_origins = grid.compute_subregion_origins();
        const auto patches_origins = View(subregion_origins.data(), std::ssize(subregion_origins)).to(options);

        // Create the big array with all the patches.
        // Importantly, use managed memory in case it doesn't fit in the GPU discrete memory.
        auto all_patches = Patches(grid.patch_size(), grid.n_patches(), metadata.ssize(), {
            .device = options.device,
            .allocator = Allocator::MANAGED,
        });

        metadata.sort("exposure").reset_indices();
        for (const auto& slice_metadata: metadata) {
            // Extract the patches. Assume the slice is normalized and edges are tapered.
            stack_loader.read_slice(slice.view(), slice_metadata.index_file);
            noa::extract_subregions(slice, patches, patches_origins);

            // Crop to the maximum frequency and oversample back to the original size
            // to nicely stretch the Thon rings to counteract small pixel sizes and high defoci.
            nf::r2c(patches, patches_rfft);
            nf::resize<"h2h">(patches_rfft, patches.shape(), patches_cropped_rfft, patches_cropped.shape());
            nf::c2r(patches_cropped_rfft, patches_cropped);
            // TODO smooth edges?
            noa::normalize_per_batch(patches_cropped, patches_cropped);
            noa::resize(patches_cropped, patches, {}, zero_padding);

            // Compute the power-spectra of these tiles and save it into the main array.
            noa::fft::r2c(patches, patches_rfft, {.norm = nf::Norm::NONE});
            noa::ewise(patches_rfft, all_patches.rfft_ps(slice_metadata.index), [=]NOA_HD(const c32& i, f32& o) {
                o = noa::abs_squared(i) * fft_scale;
            });

            Logger::trace("tilt={:>+6.2f}", slice_metadata.angles[1]);
        }

        return all_patches;
    }

    Patches::Patches(
        i64 patch_size,
        i64 n_patch_per_slice,
        i64 n_slices,
        ArrayOption options
    ) :
        m_n_slices(n_slices),
        m_n_patches_per_slice(n_patch_per_slice)
    {
        const auto patch_shape = Shape{m_n_patches_per_slice * m_n_slices, i64{1}, patch_size, patch_size}.rfft();
        const size_t n_bytes = patch_shape.as<size_t>().n_elements() * sizeof(f32);
        Logger::trace(
            "Patches(): allocating {:.2f}GB on {} ({}), shape={}",
            static_cast<f64>(n_bytes) * 1e-9, options.device, options.allocator, patch_shape
        );

        // This is the big array with all the patches.
        m_rfft_ps = noa::Array<f32>(patch_shape, options);
    }
}
