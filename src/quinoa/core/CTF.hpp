#pragma once

#include <algorithm>

#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Memory.hpp>
#include <noa/Signal.hpp>
#include <noa/core/utils/Indexing.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Utilities.h"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/CubicGrid.hpp"
#include "quinoa/core/Stack.hpp"

namespace qn {
    class CTFFitter {
    public:
        // Grid of patches. Each slice is divided into a set of patches, called a grid.
        // This type creates and helps to refer to this grid, especially the origins and centers
        // of the patches within that grid.
        class Grid {
        public:
            Grid(const Shape2<i64>& slice_shape,
                 i64 patch_size,
                 i64 patch_step
            ) : m_slice_shape(slice_shape),
                m_patch_size(patch_size),
                m_patch_step(patch_step)
            {
                set_grid_();
            }

        public:
            [[nodiscard]] const Shape2<i64>& slice_shape() const noexcept { return m_slice_shape; }

            [[nodiscard]] constexpr i64 patch_size() const noexcept { return m_patch_size; }
            [[nodiscard]] constexpr Shape2<i64> patch_shape() const noexcept { return {patch_size(), patch_size()}; }
            [[nodiscard]] i64 n_patches() const noexcept { return static_cast<i64>(patches_centers().size()); }

            // Returns the center of each patch within the slice/grid.
            // These coordinates are 0 at the slice origin.
            [[nodiscard]] Span<const Vec2<f64>> patches_centers() const noexcept {
                return {m_centers.data(), m_centers.size()};
            }

            // Converts the patch origins to the subregion origins, used for extraction.
            template<typename Integer = i32>
            [[nodiscard]] auto compute_subregion_origins(i64 batch_index = 0) const {
                std::vector<Vec4<Integer>> subregion_origins;
                subregion_origins.reserve(m_origins.size());
                for (const auto& origin: m_origins)
                    subregion_origins.emplace_back(batch_index, 0, origin[0], origin[1]);
                return subregion_origins;
            }

            // Same as above, but removes the patches that are not within the specified z-range.
            template<typename Integer = i32>
            [[nodiscard]] auto compute_subregion_origins(
                    const MetadataSlice& metadata,
                    Vec2<f64> sampling_rate,
                    Vec2<f64> delta_z_range_nanometers,
                    i64 batch_index = 0
            ) const {
                std::vector<Vec4<Integer>> subregion_origins;
                subregion_origins.reserve(m_origins.size());

                const Vec2<f64>& slice_shifts = metadata.shifts;
                const Vec3<f64>& slice_angles = metadata.angles;
                const Vec2<f64> patch_center = MetadataSlice::center<f64>(patch_shape());

                for (const auto& origin: m_origins) {
                    // Get the 3d position of the patch.
                    const auto patch_coordinates = patch_transformed_coordinate(
                            slice_shifts, slice_angles, sampling_rate,
                            origin.as<f64>() + patch_center);

                    // Filter based on its z position.
                    // TODO Filter to remove patches at the corners?
                    const auto z_nanometers = patch_coordinates[0] * 1e3; // micro -> nano
                    if (z_nanometers < delta_z_range_nanometers[0] ||
                        z_nanometers > delta_z_range_nanometers[1])
                        continue;

                    subregion_origins.emplace_back(batch_index, 0, origin[0], origin[1]);
                }
                return subregion_origins;
            }

            [[nodiscard]] auto patch_transformed_coordinate(
                    Vec2<f64> slice_shifts,
                    Vec3<f64> slice_angles,
                    Vec2<f64> slice_spacing,
                    Vec2<f64> patch_center
            ) const -> Vec3<f64> {
                slice_angles = noa::math::deg2rad(slice_angles);

                // By convention, the rotation angle is the additional rotation of the image.
                // Subtracting it aligns the tilt-axis to the y-axis.
                slice_angles[0] *= -1;

                // Switch coordinates from pixels to micrometers.
                const auto scale = slice_spacing * 1e-4;
                const auto slice_center = MetadataSlice::center(slice_shape()).as<f64>();
                const auto slice_center_3d = (slice_center * scale).push_front(0);
                const auto slice_shifts_3d = (slice_shifts * scale).push_front(0);

                // Place the slice into a 3d volume, with the center of the slice at the origin of the volume.
                namespace ng = noa::geometry;
                const Double44 image2microscope_matrix =
                        ng::linear2affine(ng::euler2matrix(slice_angles, /*axes=*/ "zyx", /*intrinsic=*/ false)) *
                        ng::translate(-slice_center_3d - slice_shifts_3d);

                const auto patch_center_3d = (patch_center * scale).push_front(0).push_back(1);
                const Vec3<f64> patch_center_transformed = (image2microscope_matrix * patch_center_3d).pop_back();
                return patch_center_transformed;
            }

        private:
            void set_grid_() {
                m_origins = patch_grid_2d(slice_shape(), patch_shape(), Vec2<i64>(m_patch_step));

                m_centers.reserve(m_origins.size());
                const Vec2<i64> patch_center = MetadataSlice::center<i64>(patch_shape());
                for (const Vec2<i64>& patch_origin: m_origins)
                    m_centers.push_back((patch_origin + patch_center).as<f64>());
            }

        private:
            Shape2<i64> m_slice_shape;
            i64 m_patch_size;
            i64 m_patch_step;
            std::vector<Vec2<i64>> m_origins;
            std::vector<Vec2<f64>> m_centers;
        };

        // Frequency range used for the fitting.
        class FittingRange {
        public:
            i64 original_logical_size{}; // size of the real-space patch(es)
            f64 spacing{}; // spacing/pixel_size of the real-space patch(es), in Angstrom/pix.
            Vec2<f64> resolution; // fitting range, in Angstroms.
            Vec2<f64> fftfreq; // fitting range, in cycle/pix.
            i64 size{}; // logical_size/2+1
            i64 logical_size{}; // (size-1)*2

            // Slice operator, to extract, from the full rotational average, the subregion containing the fitting range.
            // The full rotational average has "original_logical_size/2+1" elements.
            // The fitting range has "size" elements.
            noa::indexing::Slice slice;

            // Store the background with the fitting range, since it precisely matches the fitting range.
            // CubicSplineGrid uses coordinates [0,1], corresponding to [fitting_range[0], fitting_range[1]] exactly.
            CubicSplineGrid<f64, 1> background;

        public:
            // Create a fitting range.
            constexpr FittingRange(
                    const Vec2<f64>& target_fitting_range, // angstrom
                    f64 spacing_, // angstrom/pixel
                    i64 logical_size_ // of the patches
            ) : original_logical_size(logical_size_),
                spacing(spacing_) {

                // Find the frequency cutoffs in the current spectrum.
                // Round to the nearest pixel (this is where we cut the spectrum)
                // and compute the actual fitting range at these rounded frequencies.
                const auto original_logical_size_f = static_cast<f64>(original_logical_size);
                const auto frequency_cutoff = noa::math::round(
                        spacing / target_fitting_range * original_logical_size_f);
                fftfreq = frequency_cutoff / original_logical_size_f;
                resolution = spacing / fftfreq;

                auto index_cutoff = frequency_cutoff.as<i64>();
                index_cutoff[1] += 1; // fitting range is inclusive, so add +1 at the end because slice is exclusive
                size = index_cutoff[1] - index_cutoff[0];
                logical_size = (size - 1) * 2; // simply go to the even logical size
                slice = noa::indexing::Slice{index_cutoff[0], index_cutoff[1]};
            }

            // Compute the Fourier cropped logical shape.
            // This corresponds to the range [0, fftfreq[1]], i.e. the high frequencies are removed,
            // but of course the low frequencies cannot be cropped by simple Fourier cropping.
            [[nodiscard]] constexpr Shape2<i64> fourier_cropped_shape() const noexcept {
                const i64 fourier_cropped_size = slice.end;
                return {fourier_cropped_size, fourier_cropped_size};
            }

            // The patches are already Fourier cropped to (exactly) the end of fitting range.
            // We still need to remove the low frequencies, but we cannot do this with a Fourier cropping.
            // Instead, we'll use the rotational average output frequency range to only output the frequencies
            // that are within the fitting range by excluding the low frequencies before the range.
            [[nodiscard]] constexpr Vec2<f64> fourier_cropped_fftfreq_range() const noexcept {
                constexpr f64 NYQUIST = 0.5; // logical size is even, so max fftfreq is 0.5
                const f64 rescale = NYQUIST / fftfreq[1]; // fftfreq[1] becomes NYQUIST after Fourier crop
                return {fftfreq[0] * rescale, NYQUIST}; // NYQUIST=NYQUIST*rescale
            }
        };

    public: // -- Average fitting --
        static auto fit_average_ps(
                StackLoader& stack_loader,
                const Grid& grid,
                const MetadataStack& metadata,
                const Path& debug_directory,
                Vec2<f64> delta_z_range_nanometers,
                f64 delta_z_shift_nanometers,
                f64 max_tilt_for_average,
                bool fit_phase_shift,
                bool fit_astigmatism,
                Device compute_device,

                // inout
                FittingRange& fitting_range,
                CTFAnisotropic64& ctf
        ) -> std::array<f64, 3>; // defocus ramp

    private:
        static Array<f32> compute_average_patch_rfft_ps_(
                Device compute_device,
                StackLoader& stack_loader,
                const MetadataStack& metadata,
                const CTFFitter::Grid& grid,
                Vec2<f64> delta_z_range_nanometers,
                f64 max_tilt_for_average,
                const Path& debug_directory
        );

        static void fit_ctf_to_patch_(
                Array<f32> patch_rfft_ps,
                FittingRange& fitting_range,
                CTFAnisotropic64& ctf_anisotropic, // contains the initial defocus and phase shift, return best
                bool fit_phase_shift,
                bool fit_astigmatism,
                const Path& debug_directory
        );

    public: // -- Global fitting --

        struct GlobalFit {
            bool rotation{};
            bool tilt{};
            bool elevation{};
            bool phase_shift{};
            bool astigmatism{};
        };

        struct GlobalFitOutput {
            Vec3<f64> angle_offsets{};
            f64 phase_shift{};
            f64 astigmatism_value{};
            f64 astigmatism_angle{};
            f64 average_defocus{};
            std::vector<f64> defoci{};
        };

        // The cropped power-spectra, of every patch, of every slice.
        //  - The patches are Fourier cropped to save memory.
        //  - This type contains the actual (Fourier cropped) power-spectrum of each patch, of every slice.
        //    The patches of a given slice are saved sequentially in "chunks" and these chunks are saved
        //    sequentially too, along the batch dimension of the main (contiguous) array. In practice,
        //    chunks are simple slice-operators: an index range such as [start,end) (the end is exclusive).
        class Patches {
        public:
            Patches(
                    const Grid& grid,
                    const FittingRange& fitting_range,
                    i64 n_slices,
                    ArrayOption option
            ) : m_n_slices(n_slices),
                m_n_patches_per_slice(grid.n_patches())
            {
                // Here, to save memory and computation, we store the Fourier crop spectra directly.
                const auto cropped_patch_shape = fitting_range
                        .fourier_cropped_shape()
                        .push_front<2>({n_patches_per_slice() * n_slices, 1});

                const size_t bytes = cropped_patch_shape.rfft().as<size_t>().elements() * sizeof(f32);
                qn::Logger::trace("Allocating for the patches (n_slices={}, n_patches={} ({}GB, device={})).",
                                  n_slices, n_patches_per_slice(), static_cast<f64>(bytes) * 1e-9, option.device());

                // This is the big array with all the patches.
                m_rfft_ps = noa::memory::empty<f32>(cropped_patch_shape.rfft(), option);
            }

            [[nodiscard]] View<f32> rfft_ps() const noexcept { return m_rfft_ps.view(); }

            [[nodiscard]] View<f32> rfft_ps(i64 chunk_index) const {
                return rfft_ps().subregion(chunk(chunk_index));
            }

        public:
            [[nodiscard]] i64 n_slices() const noexcept { return m_n_slices; }
            [[nodiscard]] i64 n_patches_per_slice() const noexcept { return m_n_patches_per_slice; }
            [[nodiscard]] i64 n_patches_per_stack() const noexcept { return m_rfft_ps.shape()[0]; }

            [[nodiscard]] Shape2<i64> shape() const noexcept {
                const i64 logical_size = m_rfft_ps.shape()[2]; // patches are square
                return {logical_size, logical_size};
            }

            [[nodiscard]] Shape4<i64> chunk_shape() const noexcept {
                return shape().push_front<2>({n_patches_per_slice(), 1});
            }

            [[nodiscard]] noa::indexing::Slice chunk(i64 chunk_index) const noexcept {
                const i64 start = chunk_index * n_patches_per_slice();
                return noa::indexing::Slice{start, start + n_patches_per_slice()};
            }

        private:
            Array<f32> m_rfft_ps; // (n, 1, h, w/2+1)
            i64 m_n_slices;
            i64 m_n_patches_per_slice;
        };

        static Patches compute_patches_rfft_ps(
                Device compute_device,
                StackLoader& stack_loader,
                const MetadataStack& metadata,
                const FittingRange& fitting_range,
                const Grid& grid,
                const Path& debug_directory
        );

        static auto fit_ctf_to_patches(
                const MetadataStack& metadata,
                const Patches& patches_rfft_ps,
                const FittingRange& fitting_range,
                const Grid& grid,
                const CTFAnisotropic64& ctf_anisotropic,
                GlobalFit fit,
                const Path& debug_directory
        ) -> CTFFitter::GlobalFitOutput;
    };
}
