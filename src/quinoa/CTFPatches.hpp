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
            const Vec<f64, 2>& resolution_range,
            i64 patch_size,
            i64 patch_padded_size,
            i64 phi_size,
            noa::Interp polar_interp = noa::Interp::CUBIC_BSPLINE
        ) -> Patches;

    public:
        Patches() = default;
        [[nodiscard]] auto view() const noexcept { return m_polar.view(); }
        [[nodiscard]] auto patches(i64 index) const noexcept {
            return m_polar.view().subregion(index).permute({1, 0, 2, 3});
        }

        [[nodiscard]] auto view_batched() const noexcept {
            return m_polar.view().reshape({n_patches_total(), 1, height(), width()});
        }
        [[nodiscard]] auto chunk(i64 index) const noexcept {
            const i64 start = index * n_patches_per_image();
            return ni::Slice{start, start + n_patches_per_image()};
        }

        [[nodiscard]] auto phi() const noexcept { return m_phi_range; }
        [[nodiscard]] auto rho() const noexcept { return m_rho_range; }
        [[nodiscard]] auto rho_vec() const noexcept { return Vec{rho().start, rho().stop}; }

        [[nodiscard]] auto phi_step() const noexcept -> f64 { return phi().for_size(height()).step; }
        [[nodiscard]] auto rho_step() const noexcept -> f64 { return rho().for_size(width()).step; }

        [[nodiscard]] auto n_images() const noexcept -> i64 { return m_polar.shape().batch(); }
        [[nodiscard]] auto n_patches_per_image() const noexcept -> i64 { return m_polar.shape().depth(); }
        [[nodiscard]] auto n_patches_total() const noexcept -> i64 { return n_images() * n_patches_per_image(); }
        [[nodiscard]] auto height() const noexcept -> i64 { return m_polar.shape().height(); }
        [[nodiscard]] auto width() const noexcept -> i64 { return m_polar.shape().width(); }

        void exclude_views(SpanContiguous<const i64> indices);

    private:
        Array<value_type> m_polar{}; // (n,p,phi,rho)
        noa::Linspace<f64> m_phi_range{};
        noa::Linspace<f64> m_rho_range{};
    };
}
