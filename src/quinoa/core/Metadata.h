#pragma once

#include "quinoa/Types.h"
#include "quinoa/Exception.h"
#include "quinoa/io/YAML.h"
#include "quinoa/io/Options.h"

namespace qn {
    // Metadata of a 2d slice.
    // The rotation center is fixed at n // 2, where n is the size of the axis.
    // The shifts are applied before the rotations.
    struct MetadataSlice {
    public:
        Vec3<f32> angles{}; // Euler angles, in degrees, of the slice. zyx extrinsic (rotation, tilt, elevation)
        Vec2<f32> shifts{}; // yx shifts, in pixels, of the slice.
        f32 exposure{};     // Cumulated exposure, in e-/A2.
        i32 index{};        // Index [0, N) of the slice within the array.
        i32 index_file{};   // Index [0, N) of the slice within the original file.

        static Vec2<f32> center(i64 height, i64 width) noexcept {
            // Just make it a function to make it less ambiguous
            return {height / 2, width / 2};
        }

        static Vec2<f32> center(const Shape4<i64>& shape) noexcept {
            return center(shape[2], shape[3]);
        }

        static constexpr f32 UNSET_ROTATION_VALUE = std::numeric_limits<f32>::max();
    };

    struct TiltScheme {
        f32 starting_angle{};       // Angle, in degrees, of the first image that was collected.
        i32 starting_direction{};   // Direction (>0 or <0) after collecting the first image.
        f32 angle_increment{};      // Angle increment, in degrees, between images.
        i32 group{};                // Number of images that are collected before switching to the opposite side.
        bool exclude_start{};       // Exclude the first image from the first group.
        f32 per_view_exposure{};    // Per view exposure, in e-/A^2.

        [[nodiscard]] std::vector<MetadataSlice> generate(i32 count, f32 rotation_angle = 0.f) const;
    };

    /// Metadata of a stack of 2D slices.
    class MetadataStack {
    public:
        MetadataStack() = default;

        /// Creates the metadata.
        /// Excluded views are not included, obviously.
        explicit MetadataStack(const Options& options);

        /// Creates the metadata from a mdoc file.
        explicit MetadataStack(const Path& mdoc_filename);

        /// Creates the metadata from a tlt and exposure file.
        MetadataStack(const Path& tlt_filename,
                      const Path& exposure_filename,
                      f32 rotation_angle = 0);

        /// Creates the metadata from the tilt scheme.
        MetadataStack(TiltScheme tilt_scheme,
                      i32 order_count,
                      f32 rotation_angle = 0);

    public: // Stack manipulations
        /// Excludes the slice(s) according to their "index" field.
        /// The metadata is sorted in ascending order according to the "index" field
        /// and the "index" field is reset from [0, N), N being the new slices count.
        MetadataStack& exclude(const std::vector<i32>& indexes_to_exclude) noexcept;

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
        [[nodiscard]] i64 find_lowest_tilt_index() const;

    public:
        static void log_update(const MetadataStack& origin, const MetadataStack& current);

    private:
        void sort_on_indexes_(bool ascending = true);
        void sort_on_file_indexes_(bool ascending = true);
        void sort_on_tilt_(bool ascending = true);
        void sort_on_absolute_tilt_(bool ascending = true);
        void sort_on_exposure_(bool ascending = true);

    private:
        std::vector<MetadataSlice> m_slices;
    };
}
