#pragma once

#include "quinoa/Types.h"
#include "quinoa/Exception.h"
#include "quinoa/io/YAML.h"
#include "quinoa/io/Options.h"

namespace qn {
    /// Metadata of a 2D slice.
    /// \note For the transformation, the rotation center is fixed at n // 2, where n is the size of the axis.
    ///       Furthermore, the shifts should be applied before the rotation.
    struct MetadataSlice {
    public:
        float3_t angles{};      // Euler angles, in degrees, of the slice. ZYX extrinsic (yaw, tilt, pitch)
        float2_t shifts{};      // YX shifts, in pixels, of the slice.
        float exposure{};       // Cumulated exposure, in e-/A2.
        int32_t index{};        // Index [0, N) of the slice within the array.
        int32_t index_file{};   // Index [0, N) of the slice within the original file.

        static float2_t center(dim_t height, dim_t width) noexcept {
            // Use integral division to always have the center onto a pixel.
            // This is actually important for Fourier cropping and resizing,
            // so that we don't have to shift by 0.5 for even dimensions.
            return {height / 2, width / 2};
        }

        static float2_t center(const dim4_t& shape) noexcept {
            return center(shape[2], shape[3]);
        }
    };

    struct TiltScheme {
        float starting_angle{};         // Angle, in degrees, of the first image that was collected.
        int32_t starting_direction{};   // Direction (>0 or <0) after collecting the first image.
        float angle_increment{};        // Angle increment, in degrees, between images.
        int32_t group{};                // Number of images that are collected before switching to the opposite side.
        bool exclude_start{};           // Exclude the first image from the first group.
        float per_view_exposure{};      // Per view exposure, in e-/A^2.

        [[nodiscard]] std::vector<MetadataSlice> generate(int32_t count, float rotation_angle = 0.f) const;
    };

    /// Metadata of a stack of 2D slices.
    class MetadataStack {
    public:
        MetadataStack() = default;

        /// Creates the metadata.
        /// This also exclude the views if needed (but does not squeeze).
        explicit MetadataStack(const Options& options);

        /// Creates the metadata from a mdoc file.
        explicit MetadataStack(const path_t& mdoc_filename);

        /// Creates the metadata from a tlt and exposure file.
        MetadataStack(const path_t& tlt_filename,
                      const path_t& exposure_filename,
                      float rotation_angle = 0);

        /// Creates the metadata from the tilt scheme.
        MetadataStack(TiltScheme tilt_scheme,
                      int32_t order_count,
                      float rotation_angle = 0);

    public: // Stack manipulations
        /// Exclude the slice(s) according to their "index" field.
        /// The metadata is sorted in ascending order according to the "index" field
        /// and the "index" field is reset from [0, N), N being the new slices count.
        MetadataStack& exclude(const std::vector<int32_t>& indexes_to_exclude) noexcept;

        /// (Stable) sorts the slices based on a given key.
        /// Valid keys: "index", "index_file", "tilt", "absolute_tilt", "exposure".
        MetadataStack& sort(std::string_view key, bool ascending = true);

    public: // Getters
        [[nodiscard]] constexpr const std::vector<MetadataSlice>& slices() const noexcept { return m_slices; }
        [[nodiscard]] constexpr std::vector<MetadataSlice>& slices() noexcept { return m_slices; }
        [[nodiscard]] size_t size() const noexcept { return m_slices.size(); }

        /// Returns a view of the slice at \p idx, as currently sorted in this instance (see sort()).
        template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
        [[nodiscard]] constexpr MetadataSlice& operator[](T idx) noexcept {
            NOA_ASSERT(idx >= 0 && static_cast<size_t>(idx) < size());
            return m_slices[static_cast<size_t>(idx)];
        }

        /// Returns a view of the slice at \p idx, as currently sorted in this instance (see sort()).
        template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
        [[nodiscard]] constexpr const MetadataSlice& operator[](T idx) const noexcept {
            NOA_ASSERT(idx >= 0 && static_cast<size_t>(idx) < size());
            return m_slices[static_cast<size_t>(idx)];
        }

        /// Find the index (as currently sorted in this instance)
        /// of the slice with the lowest absolute tilt angle.
        [[nodiscard]] size_t lowestTilt() const;

    public:
        static void logUpdate(const MetadataStack& origin, const MetadataStack& current);

    private:
        void sortBasedOnIndexes_(bool ascending = true);
        void sortBasedOnIndexesFile_(bool ascending = true);
        void sortBasedOnTilt_(bool ascending = true);
        void sortBasedOnAbsoluteTilt_(bool ascending = true);
        void sortBasedOnExposure_(bool ascending = true);

    private:
        std::vector<MetadataSlice> m_slices;
    };
}
