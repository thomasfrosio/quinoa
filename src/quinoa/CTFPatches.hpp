#pragma once

#include <noa/Array.hpp>

#include "quinoa/CTFGrid.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Types.hpp"

namespace qn::ctf {
    class Patches {
    public:
        // Use half-precision floating-points to store the patches.
        // Given that the array is multiple GBytes, this is quite worth it.
        // Since the power-spectra are min-max normalized and this is just for storage,
        // this essentially has no effect on the computed single-precision rotation-averages.
        using value_type = f16;

    public:
        static auto from_stack(
            StackLoader& stack_loader,
            const MetadataStack& metadata,
            const Grid& grid,
            i64 fourier_cropped_size,
            i64 final_size
        ) -> Patches;

    public:
        Patches() = default;
        Patches(i64 patch_size, i64 n_patch_per_slice, i64 n_slices, ArrayOption options);

    public:
        [[nodiscard]] auto rfft_ps() const noexcept -> View<value_type> { return m_rfft_ps.view(); }
        [[nodiscard]] auto rfft_ps(i64 chunk_index) const -> View<value_type> {
            return rfft_ps().subregion(chunk_slice(chunk_index));
        }

        [[nodiscard]] auto n_slices() const noexcept -> i64 { return m_n_slices; }
        [[nodiscard]] auto n_patches_per_slice() const noexcept -> i64 { return m_n_patches_per_slice; }
        [[nodiscard]] auto n_patches_per_stack() const noexcept -> i64 { return m_rfft_ps.shape()[0]; }

        [[nodiscard]] auto shape() const noexcept -> Shape<i64, 2> {
            const i64 logical_size = m_rfft_ps.shape()[2]; // patches are square
            return {logical_size, logical_size};
        }

        [[nodiscard]] auto chunk_shape() const noexcept -> Shape<i64, 4> {
            return shape().push_front(Vec{n_patches_per_slice(), i64{1}});
        }

        [[nodiscard]] auto chunk_slice(i64 chunk_index) const noexcept -> ni::Slice {
            const i64 start = chunk_index * n_patches_per_slice();
            return ni::Slice{start, start + n_patches_per_slice()};
        }

    private:
        Array<value_type> m_rfft_ps{}; // (n, 1, h, w/2+1)
        i64 m_n_slices{};
        i64 m_n_patches_per_slice{};
    };
}
