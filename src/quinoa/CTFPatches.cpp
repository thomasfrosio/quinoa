#include <noa/Geometry.hpp>
#include <noa/FFT.hpp>

#include "quinoa/CTF.hpp"

namespace qn::ctf {
    auto Patches::from_stack(
        StackLoader& stack_loader,
        const MetadataStack& metadata,
        const Grid& grid,
        i64 fourier_cropped_size,
        i64 final_size
    ) -> Patches {
        auto timer = Logger::trace_scope_time("Loading patches using device={}", stack_loader.compute_device());

        const auto patch_shape_2d = grid.patch_shape();
        const auto patches_shape = patch_shape_2d.push_front(Vec{grid.n_patches(), i64{1}});
        const auto fft_scale = 1 / static_cast<f32>(patch_shape_2d.n_elements());

        // The patches are loaded one slice at a time. So allocate enough for one slice.
        const auto options = ArrayOption{stack_loader.compute_device(), Allocator::ASYNC};
        const auto slice = Array<f32>(grid.slice_shape().push_front<2>(1), options);
        const auto patches_rfft = Array<c32>(patches_shape.rfft(), options);
        const auto patches = noa::fft::alias_to_real(patches_rfft.view(), patches_shape);

        // Fourier crop the patches to the target max frequency, and go back to real space to zero-pad,
        // effectively increasing the sampling and stretching the Thon rings as much as possible.
        const auto patches_cropped_shape = Shape{grid.n_patches(), i64{1}, fourier_cropped_size, fourier_cropped_size};
        const auto patches_cropped_rfft = Array<c32>(patches_cropped_shape.rfft(), options);
        const auto patches_cropped = noa::fft::alias_to_real(patches_cropped_rfft.view(), patches_cropped_shape);

        const auto patches_final_shape = Shape{grid.n_patches(), i64{1}, final_size, final_size};
        const auto zero_padding = (patches_final_shape - patches_cropped_shape).vec;

        // In case we zero-pad back to the original patch size, don't allocate and reuse the patches instead.
        Array<c32> buffer_padding;
        View<c32> patches_final_rfft;
        View<f32> patches_final;
        if (noa::any(zero_padding != 0)) {
            buffer_padding = Array<c32>(patches_final_shape.rfft(), options);
            patches_final_rfft = buffer_padding.view();
            patches_final = noa::fft::alias_to_real(patches_final_rfft, patches_final_shape);
        } else {
            patches_final_rfft = patches_rfft.view();
            patches_final = patches;
        }

        Logger::info(
            "Creating patches:\n"
            "  n_slices={}\n"
            "  n_patches={} (per_slice={})\n"
            "  size: original={}, cropped={}, final={}",
            metadata.ssize(), grid.n_patches() * metadata.ssize(), grid.n_patches(),
            grid.patch_size(), fourier_cropped_size, final_size
        );

        // Prepare the subregion origins, ready for extract_subregions.
        const auto patches_origins = grid.compute_subregion_origins().to(options);

        // Create the big array with all the patches.
        // Importantly, use managed memory in case it doesn't fit in the GPU discrete memory.
        auto all_patches = Patches(final_size, grid.n_patches(), metadata.ssize(), {
            .device = options.device,
            .allocator = Allocator::MANAGED,
        });

        auto metadata_sorted = metadata;
        metadata_sorted.reset_indices();
        for (const auto& slice_metadata: metadata_sorted) {
            // Extract the patches. Assume the slice is normalized and edges are tapered.
            stack_loader.read_slice(slice.view(), slice_metadata.index_file);
            noa::extract_subregions(slice.view(), patches, patches_origins.view());

            // Crop to the maximum frequency and oversample back to the original size
            // to nicely stretch the Thon rings, thus counteracting small pixel sizes and high defoci.
            nf::r2c(patches, patches_rfft);
            nf::resize<"h2h">(patches_rfft, patches.shape(), patches_cropped_rfft, patches_cropped.shape());
            nf::c2r(patches_cropped_rfft, patches_cropped);
            // TODO smooth edges?
            noa::normalize_per_batch(patches_cropped, patches_cropped);
            noa::resize(patches_cropped, patches_final, {}, zero_padding);

            // Compute the power-spectra of these tiles and save it into the main array.
            noa::fft::r2c(patches_final, patches_final_rfft, {.norm = nf::Norm::NONE});
            noa::ewise(patches_final_rfft, all_patches.rfft_ps(slice_metadata.index),
                       [=]NOA_HD(const c32& i, value_type& o) {
                           o = static_cast<value_type>(noa::abs_squared(i) * fft_scale);
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
        const size_t n_bytes = patch_shape.as<size_t>().n_elements() * sizeof(value_type);
        Logger::trace(
            "Patches(): allocating {:.2f}GB on {} ({}, shape={}, dtype={})",
            static_cast<f64>(n_bytes) * 1e-9, options.device, options.allocator, patch_shape, noa::string::stringify<value_type>()
        );

        // This is the big array with all the patches.
        m_rfft_ps = noa::Array<value_type>(patch_shape, options);
    }
}
