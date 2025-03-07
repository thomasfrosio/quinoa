#pragma once

#include <algorithm>

#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/CubicGrid.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/SplineInterpolate.hpp"

namespace qn::ctf {
    /// Grid of patches. Each slice is divided into a set of patches, called a grid.
    /// This type creates and helps to manipulate this grid, especially the origins
    /// and centers of each path within that grid.
    class Grid {
    public:
        Grid(const Shape<i64, 2>& slice_shape, i64 patch_size, i64 patch_step) :
            m_slice_shape(slice_shape),
            m_patch_size(patch_size),
            m_patch_step(patch_step)
        {
            const std::vector origins_along_y = patch_grid_1d_(m_slice_shape[0], m_patch_size, m_patch_step);
            const std::vector origins_along_x = patch_grid_1d_(m_slice_shape[1], m_patch_size, m_patch_step);

            m_origins.reserve(origins_along_y.size() * origins_along_x.size());
            for (i64 y: origins_along_y)
                for (i64 x: origins_along_x)
                    m_origins.push_back({y, x});

            m_centers.reserve(m_origins.size());
            const auto patch_center = (patch_shape() / 2).vec;
            for (const auto& patch_origin: m_origins)
                m_centers.push_back((patch_origin + patch_center).as<f64>());
        }

    public:
        [[nodiscard]] auto slice_shape() const noexcept -> const Shape<i64, 2>& { return m_slice_shape; }
        [[nodiscard]] auto patch_size() const noexcept -> i64 { return m_patch_size; }
        [[nodiscard]] auto patch_shape() const noexcept -> Shape<i64, 2> { return Shape{patch_size(), patch_size()}; }
        [[nodiscard]] auto n_patches() const noexcept -> i64 { return static_cast<i64>(patches_centers().size()); }

        /// Returns the center of each patch within the slice/grid.
        /// These coordinates are 0 at the slice origin.
        [[nodiscard]] auto patches_centers() const noexcept -> SpanContiguous<const Vec2<f64>> {
            return {m_centers.data(), static_cast<i64>(m_centers.size())};
        }

        /// Converts the patch origins to the subregion origins, used for extraction.
        template<nt::sinteger I = i32>
        [[nodiscard]] auto compute_subregion_origins(i64 batch_index = 0) const {
            std::vector<Vec<I, 4>> subregion_origins{};
            subregion_origins.reserve(m_origins.size());
            for (const auto& origin: m_origins) {
                auto origin_4d = Vec<I, 4>::from_values(batch_index, 0, origin[0], origin[1]);
                subregion_origins.push_back(origin_4d);
            }
            return subregion_origins;
        }

        /// Same as above, but removes the patches that are not within the specified z-range.
        template<nt::sinteger I = i32>
        [[nodiscard]] auto compute_subregion_origins(
            const MetadataSlice& metadata,
            const Vec<f64, 2>& sampling_rate,
            const Vec<f64, 2>& delta_z_range_nanometers,
            i64 batch_index = 0
        ) const {
            std::vector<Vec<I, 4>> subregion_origins;
            subregion_origins.reserve(m_origins.size());

            const Vec<f64, 2> patch_center = (patch_shape() / 2).vec.as<f64>();
            for (const auto& origin: m_origins) {
                // Get the 3d position of the patch.
                const auto patch_z_offset_um = patch_z_offset(
                    metadata.shifts, noa::deg2rad(metadata.angles), sampling_rate,
                    origin.as<f64>() + patch_center);

                // Filter based on its z position.
                const auto patch_z_offset_nm = patch_z_offset_um * 1e3; // micro -> nano
                if (patch_z_offset_nm < delta_z_range_nanometers[0] or
                    patch_z_offset_nm > delta_z_range_nanometers[1])
                    continue;

                subregion_origins.push_back(Vec<I, 4>::from_values(batch_index, 0, origin[0], origin[1]));
            }
            return subregion_origins;
        }

        /// Applies the tilt and pitch to the patch to get its z-offset (in micrometers) from the tilt-axis.
        [[nodiscard]] auto patch_z_offset(
            const Vec<f64, 2>& slice_shifts,
            const Vec<f64, 3>& slice_angles, // radians
            const Vec<f64, 2>& slice_spacing, // angstrom
            const Vec<f64, 2>& patch_center
        ) const -> f64 {
            // Switch coordinates from pixels to micrometers.
            const auto scale = slice_spacing * 1e-4;
            const auto slice_center = (slice_shape() / 2).vec.as<f64>();
            const auto slice_center_3d = (slice_center * scale).push_front(0);
            const auto slice_shifts_3d = (slice_shifts * scale).push_front(0);

            // Apply the tilt and pitch.
            const Mat<f64, 3, 4> image_to_microscope = (
                ng::linear2affine(ng::rotate_x(-slice_angles[2])) * // 4. align pitch
                ng::linear2affine(ng::rotate_y(-slice_angles[1])) * // 3. align tilt
                ng::linear2affine(ng::rotate_z(-slice_angles[0])) * // 2. align tilt-axis
                ng::translate(-slice_center_3d - slice_shifts_3d)   // 1. align rotation center
            ).pop_back();

            const auto patch_center_3d = (patch_center * scale).push_front(0);
            const Vec<f64, 3> patch_center_transformed = image_to_microscope * patch_center_3d.push_back(1);
            return patch_center_transformed[0]; // z
        }

    private:
        static auto patch_grid_1d_(i64 grid_size, i64 patch_size, i64 patch_step) -> std::vector<i64> {
            // Arange:
            const auto max = grid_size - patch_size - 1;
            std::vector<i64> patch_origin;
            for (i64 i{}; i < max; i += patch_step)
                patch_origin.push_back(i);

            if (patch_origin.empty())
                return patch_origin;

            // Center:
            const i64 end = patch_origin.back() + patch_size;
            const i64 offset = (grid_size - end) / 2;
            for (auto& origin: patch_origin)
                origin += offset;

            return patch_origin;
        }

    private:
        Shape<i64, 2> m_slice_shape;
        i64 m_patch_size;
        i64 m_patch_step;
        std::vector<Vec<i64, 2>> m_origins;
        std::vector<Vec<f64, 2>> m_centers;
    };

    /// The cropped power-spectra, of every patch, of every slice.
    /// => The patches are Fourier cropped to the target resolution.
    /// => The patches of a given slice are saved sequentially in a "chunk" and these chunks are saved
    ///    sequentially, for each slice, and along the batch dimension of the main (contiguous) array.
    ///    Chunks can be accessed with a simple slice-operator: an index range such as [start,end).
    /// => Importantly, chunks have the same size, i.e. the number of patches is the same for every slice.
    class Patches {
    public:
        /// Loads the patches in ascending order of exposure.
        /// The metadata index is reset to match the chunk index in the returned patches.
        static auto from_stack(
            StackLoader& stack_loader,
            MetadataStack& metadata,
            const Grid& grid,
            i64 fourier_cropped_size
        ) -> Patches;

    public:
        Patches(i64 patch_size, i64 n_patch_per_slice, i64 n_slices, ArrayOption options);

        [[nodiscard]] auto rfft_ps() const noexcept -> View<f32> { return m_rfft_ps.view(); }

        /// Retrieves the patches of a given slice.
        [[nodiscard]] auto rfft_ps(i64 chunk_index) const -> View<f32> {
            return rfft_ps().subregion(chunk_slice(chunk_index));
        }

    public:
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
        Array<f32> m_rfft_ps; // (n, 1, h, w/2+1)
        i64 m_n_slices;
        i64 m_n_patches_per_slice;
    };

    struct Background {
    public:
        static auto fit_coarse_background_1d(
            const View<const f32>& spectrum,
            i64 spline_resolution = 3
        ) -> CubicSplineGrid<f64, 1>;

        static auto fit_coarse_background_2d(
            const View<const f32>& spectrum,
            const Vec<f64, 2>& fftfreq_range,
            i64 spline_resolution = 3
        ) -> CubicSplineGrid<f64, 1>;

    public:
        Spline spline;

        void fit_1d(
            const View<const f32>& spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFIsotropic<f64>& ctf
        );

        void fit_2d(
            const View<const f32>& spectrum,
            const Vec<f64, 2>& fftfreq_range,
            const ns::CTFAnisotropic<f64>& ctf
        );

        void sample(
            const View<f32>& spectrum,
            const Vec<f64, 2>& fftfreq_range
        ) const;

        void subtract(
            const View<const f32>& input,
            const View<f32>& output,
            const Vec<f64, 2>& fftfreq_range
        ) const;
    };
}

namespace qn::ctf {
    struct FitCoarseOptions {
        Vec<f64, 2> fftfreq_range;
        bool fit_phase_shift{};
        bool has_user_rotation;
        Path output_directory{};
    };

    /// Initial fitting.
    /// 1. The defocus of every slice is estimated.
    /// 2. A background of the average spectrum is computed.
    /// 3. The defocus ramp is checked.
    auto coarse_fit(
        const Grid& grid,
        const Patches& patches,
        const ns::CTFIsotropic<f64>& ctf,
        MetadataStack& metadata, // .defocus updated, angles[0] may be flipped
        const FitCoarseOptions& options
    ) -> Background;

    struct FitRefineOptions {
        bool fit_rotation{};
        bool fit_tilt{};
        bool fit_pitch{};
        bool fit_phase_shift{};
        bool fit_astigmatism{};
    };

    /// Fits the stage orientation by fitting the CTF of the patches.
    /// \param[in,out] metadata     Metadata of the stack, corresponding to the provided patches.
    ///                             The angles are updated, as well as the phase shift and defocus fields.
    /// \param grid                 Slice grid.
    /// \param patches_rfft_ps      Patches to fit. Should be within [0, fftfreq_range[1]].
    /// \param fftfreq_range        Frequency range.
    /// \param isotropic_ctf        Average CTF. Used to get the constant microscope settings.
    ///                             The defocus is ignored, the per-slice defocus of the metadata is used instead.
    /// \param[in,out] background   Background, and optional envelope, within the fftfreq range. This is updated.
    /// \param[in,out] phase_shift  Time-resolved phase shift. This is updated, up to resolution=3.
    /// \param[in,out] astigmatism  Time-resolved astigmatism (0=value, 1=angle). This is updated, up to resolution=3.
    /// \param fit                  Parameters to fit.
    void refine_fit_patches_ps(
        MetadataStack& metadata,
        const Grid& grid,
        const Patches& patches_rfft_ps,
        const Vec<f64, 2>& fftfreq_range,
        ns::CTFIsotropic<f64>& isotropic_ctf,
        const Background& background,
        CubicSplineGrid<f64, 1>& phase_shift,
        CubicSplineGrid<f64, 1>& astigmatism,
        const FitRefineOptions& fit,
        const View<f32>& rotational_average
    );
}
