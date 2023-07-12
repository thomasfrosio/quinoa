#pragma once

#include <algorithm>

#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Memory.hpp>
#include <noa/Signal.hpp>
#include <noa/core/fft/Frequency.hpp>
#include <noa/core/traits/Utilities.hpp>
#include <noa/core/utils/Indexing.hpp>

#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Utilities.h"
#include "quinoa/core/Optimizer.hpp"
#include "quinoa/core/CubicGrid.hpp"
#include "quinoa/core/Stack.hpp"

namespace qn {
    using CTFIsotropic64 = noa::signal::fft::CTFIsotropic<f64>;
    using CTFAnisotropic64 = noa::signal::fft::CTFAnisotropic<f64>;

    class CTF {
    public:
        // Frequency range used for the fitting.
        class FittingRange {
        public:
            i64 original_logical_size{};
            f64 spacing{};
            Vec2<f64> resolution;
            Vec2<f64> fftfreq;
            i64 size{}; // logical_size/2+1
            i64 logical_size{}; // (size-1)/2
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

    public:
        CTF(const Shape2<i64>& slice_shape,
            i64 patch_size,
            i64 patch_step,
            Device compute_device
        );

        void fit_average(
                StackLoader& stack_loader,
                MetadataStack& metadata,
                FittingRange& fitting_range,
                CTFIsotropic64& ctf,
                Vec2<f64> delta_z_range_nanometers,
                f64 delta_z_shift_nanometers,
                f64 max_tilt_for_average,
                bool fit_phase_shift,
                bool fit_astigmatism,
                f64& astigmatism_value,
                f64& astigmatism_angle,
                bool flip_rotation_to_match_defocus_ramp,
                const Path& debug_directory
        );

        void fit_global(
                StackLoader& stack_loader,
                MetadataStack& metadata,
                f64 max_tilt,
                Vec3<bool> fit_angles, // rotation, tilt, elevation
                bool fit_phase_shift,
                bool fit_astigmatism,
                const Path& debug_directory
        );

    public: // Utilities
        static auto patch_transformed_coordinate(
                Shape2<i64> slice_shape,
                Vec2<f64> slice_shifts,
                Vec3<f64> slice_angles,
                Vec2<f64> slice_spacing,
                Vec2<f64> patch_center
        ) -> Vec3<f64>;

        [[nodiscard]] static auto extract_patches_origins(
                const Shape2<i64>& slice_shape,
                const MetadataSlice& metadata,
                Vec2<f64> sampling_rate,
                Shape2<i64> patch_shape,
                Vec2<i64> patch_step,
                Vec2<f64> delta_z_range_nanometers =
                        {noa::math::Limits<f64>::lowest(),
                         noa::math::Limits<f64>::max()}
        ) -> std::vector<Vec4<i32>>;

        [[nodiscard]] static auto extract_patches_centers(
                const Shape2<i64>& slice_shape,
                Shape2<i64> patch_shape,
                Vec2<i64> patch_step
        ) -> std::vector<Vec2<f32>>;

        static void update_slice_patches_ctfs(
                Span<const Vec2<f32>> patches_centers,
                Span<CTFIsotropic64> patches_ctfs,
                const Shape2<i64>& slice_shape,
                const Vec2<f64>& slice_shifts,
                const Vec3<f64>& slice_angles,
                f64 slice_defocus,
                f64 additional_phase_shift
        );

    private: // Average fitting
        Array<f32> compute_average_patch_rfft_ps_(
                StackLoader& stack_loader,
                const MetadataStack& metadata,
                Vec2<f64> delta_z_range_nanometers,
                f64 max_tilt_for_average,
                const Path& debug_directory);

        static void fit_ctf_to_patch_(
                Array<f32> patch_rfft_ps,
                FittingRange& fitting_range,
                CTFIsotropic64& ctf, // contains the initial defocus and phase shift, return best
                bool fit_phase_shift,
                bool fit_astigmatism,
                f64& astigmatism_value,
                f64& astigmatism_angle,
                const Path& debug_directory
        );

    private: // Global fitting
        // The cropped power-spectra, of every patch, of every slice.
        //  - The patches are Fourier cropped to save memory. However, we want to keep track of the original
        //    spacing and logical size so that we can apply fitting range as if it was the full spectrum.
        //  - We need to slice through the main array to get the patches of a given slice.
        //    We do this by saving the slice indexing operators of every slice. The order in which the
        //    slice patches are saved is the one from the metadata used for the extraction.
        //    It is therefore important to not reorder the metadata after the extraction!
        struct Patches {
            Array<f32> rfft_ps; // (n, 1, h, w/2+1)
            std::vector<Vec2<f32>> center_coordinates; // coordinates of the patches within a slice.
            std::vector<noa::indexing::Slice> slices{};

            // The index is the index within the metadata of the MetadataSlice to extract.
            [[nodiscard]] const noa::indexing::Slice& slicing_operator(i64 slice_index) const {
                return slices[static_cast<size_t>(slice_index)];
            }

            // Extract the patches belonging to a given slice.
            View<f32> patches_from_slice(i64 index) {
                return rfft_ps.view().subregion(slicing_operator(index));
            }

            View<f32> patches_from_last_slice() {
                return rfft_ps.view().subregion(slices.back());
            }

            [[nodiscard]] i64 n_slices() const noexcept { return static_cast<i64>(slices.size()); }
            [[nodiscard]] i64 n_patches() const noexcept { return rfft_ps.shape()[0]; }

            [[nodiscard]] Vec2<i64> range(i64 slice_index) const {
                const auto& slice_indexing = slicing_operator(slice_index);
                return {slice_indexing.start, slice_indexing.end}; // [) range
            }

            [[nodiscard]] constexpr i64 logical_size() const noexcept {
                return rfft_ps.shape()[2]; // patches are square
            }

            [[nodiscard]] constexpr Shape2<i64> logical_shape() const noexcept {
                return {logical_size(), logical_size()};
            }
        };

        Patches compute_patches_rfft_ps_(
                StackLoader& stack_loader,
                const MetadataStack& metadata,
                const FittingRange& fitting_range,
                f64 max_tilt,
                const Path& debug_directory
        );

        static void fit_ctf_to_patches_(
                MetadataStack& metadata,
                const Shape2<i64>& slice_shape,
                const Patches& patches_rfft_ps,
                const FittingRange& fitting_range,
                CTFIsotropic64& average_ctf,
                Vec3<bool> fit_angles,
                bool fit_phase_shift,
                bool fit_astigmatism,
                f64 initial_astigmatism_value,
                f64 initial_astigmatism_angle,
                const Path& debug_directory
        );

    private:
        // For loading the stack into the patches (one slice at a time).
        Array<f32> m_slice;
        Array<c32> m_patches_rfft; // (p, 1, h, w/2+1), where p is the number of patches within a slice

        i64 m_patch_size;
        i64 m_patch_step;
    };
}

#include <quinoa/core/CTF.inl>
