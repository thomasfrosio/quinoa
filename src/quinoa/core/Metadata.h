#pragma once

#include "quinoa/Types.h"
#include "quinoa/Exception.h"
#include "quinoa/io/YAML.h"
#include "quinoa/io/Options.h"

namespace qn {
    // Metadata of a 2d slice.
    // * The shifts are applied before the rotations. These are "by how much the slice is shifted",
    //   so to align the slice, one must subtract the shifts.
    // * The rotation center is fixed at n // 2, where n is the size of the axis.
    // * The Euler angles are in degrees, ZYX extrinsic (). All angles are positive-CCW when looking
    //   at the origin from the positive side. The rotation is "by how much the slice is rotated",
    //   so to align the slice, one must subtract the rotation. However, for the tilt and elevation,
    //   these are simply the angle of the slices in 3d space, so to insert a slice in 3d space,
    //   one must simply add these angles.
    struct MetadataSlice {
        Vec3<f64> angles{}; // Euler angles, in degrees, of the slice. zyx extrinsic (rotation, tilt, elevation)
        Vec2<f64> shifts{}; // yx shifts, in pixels, of the slice.
        f32 exposure{};     // Cumulated exposure, in e-/A2.
        i32 index{};        // Index [0, N) of the slice within the array.
        i32 index_file{};   // Index [0, N) of the slice within the original file.

        template<typename Real = f32>
        static constexpr Vec2<Real> center(i64 height, i64 width) noexcept {
            // Center is defined at n // 2 (integer division). We rely on this during resizing (as opposed to n / 2).
            return {height / 2, width / 2};
        }

        template<typename Real = f32, size_t N>
        static constexpr Vec2<Real> center(const Shape<i64, N>& shape) noexcept {
            return center<Real>(shape[N - 2], shape[N - 1]);
        }

        static constexpr f64 UNSET_ROTATION_VALUE = std::numeric_limits<f64>::max();
    };

    struct TiltScheme {
        f32 starting_angle{};       // Angle, in degrees, of the first image that was collected.
        i32 starting_direction{};   // Direction (>0 or <0) after collecting the first image.
        f32 angle_increment{};      // Angle increment, in degrees, between images.
        i32 group{};                // Number of images that are collected before switching to the opposite side.
        bool exclude_start{};       // Exclude the first image from the first group.
        f32 per_view_exposure{};    // Per view exposure, in e-/A^2.

        [[nodiscard]] std::vector<MetadataSlice> generate(i32 n_slices) const;
    };

    // Metadata of a stack of 2D slices.
    class MetadataStack {
    public:
        MetadataStack() = default;

        // Initializes the slices:
        //  - The tilt angles and exposure are set using the tilt_scheme:order or the tilt/exposure file.
        //  - The rotation and elevation angles, as well as the shifts, are set to 0.
        //  - The known angle offsets are not added at this point: the alignment will deal with them.
        //  - The slice index and index_file are set, either from the tilt_scheme:order (in which case
        //    slices are assumed to be saved in tilt-ascending order), or from the tilt file.
        explicit MetadataStack(const Options& options);

    public: // Stack manipulations
        // Excludes slice(s) according to a predicate.
        // Predicate: The predicate is a function taking a MetadataSlice a retuning a boolean
        //            If the predicate returns true, the slice should be removed.
        // Reset indexes: The filtered metadata is sorted in ascending order according to the "index" field,
        //                and the "index" field is reset from [0, N), N being the new slices count.
        template<typename Predicate,
                 typename = std::enable_if_t<std::is_invocable_r_v<bool, Predicate, const MetadataSlice&>>>
        MetadataStack& exclude(
                Predicate&& predicate,
                bool reset_index_field = true
        ) {
            const auto end_of_new_range = std::remove_if(m_slices.begin(), m_slices.end(), predicate);
            m_slices.erase(end_of_new_range, m_slices.end());

            // Sort and reset index.
            if (reset_index_field) {
                sort_on_indexes_();
                i32 count{0};
                for (auto& slice: m_slices)
                    slice.index = count++;
            }

            return *this;
        }

        // Excludes the slice(s) according to their "index" field.
        MetadataStack& exclude(const std::vector<i64>& indexes_to_exclude, bool reset_index_field = true) {
            return exclude(
                    [&](const MetadataSlice& slice) {
                        const i64 slice_index = slice.index;
                        return std::any_of(indexes_to_exclude.begin(), indexes_to_exclude.end(),
                                           [=](i64 index_to_exclude) { return slice_index == index_to_exclude; });
                    },
                    reset_index_field);
        }

        // (Stable) sorts the slices based on a given key.
        // Valid keys: "index", "index_file", "tilt", "absolute_tilt", "exposure".
        MetadataStack& sort(std::string_view key, bool ascending = true);

    public: // Getters
        [[nodiscard]] constexpr const std::vector<MetadataSlice>& slices() const noexcept { return m_slices; }
        [[nodiscard]] constexpr std::vector<MetadataSlice>& slices() noexcept { return m_slices; }
        [[nodiscard]] size_t size() const noexcept { return m_slices.size(); }
        [[nodiscard]] i64 ssize() const noexcept { return static_cast<i64>(size()); }

        // Returns a view of the slice at "idx", as currently sorted in this instance (see sort()).
        template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
        [[nodiscard]] constexpr MetadataSlice& operator[](T idx) noexcept {
            NOA_ASSERT(idx >= 0 && static_cast<size_t>(idx) < size());
            return m_slices[static_cast<size_t>(idx)];
        }

        // Returns a view of the slice at "idx", as currently sorted in this instance (see sort()).
        template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
        [[nodiscard]] constexpr const MetadataSlice& operator[](T idx) const noexcept {
            NOA_ASSERT(idx >= 0 && static_cast<size_t>(idx) < size());
            return m_slices[static_cast<size_t>(idx)];
        }

        // Find the index (as currently sorted in this instance)
        // of the slice with the lowest absolute tilt angle.
        [[nodiscard]] i64 find_lowest_tilt_index() const;

    public:
        static void log_update(const MetadataStack& origin, const MetadataStack& current);

    private:
        void generate_(const Path& tlt_filename, const Path& exposure_filename);
        void generate_(TiltScheme tilt_scheme, i32 order_count);

        void sort_on_indexes_(bool ascending = true);
        void sort_on_file_indexes_(bool ascending = true);
        void sort_on_tilt_(bool ascending = true);
        void sort_on_absolute_tilt_(bool ascending = true);
        void sort_on_exposure_(bool ascending = true);

    private:
        std::vector<MetadataSlice> m_slices;
    };
}
