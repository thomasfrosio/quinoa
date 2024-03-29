#pragma once

#include <noa/Geometry.hpp>

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
    public:
        Vec3<f64> angles{};     // Euler angles, in degrees, of the slice. zyx extrinsic (rotation, tilt, elevation)
        Vec2<f64> shifts{};     // yx shifts, in pixels, of the slice.
        Vec2<f64> exposure{};   // Pre- and post-exposure, in e-/A2.
        f64 defocus{};          // Slice defocus, in micrometers.
        i32 index{};            // Index [0, N) of the slice within the array.
        i32 index_file{};       // Index [0, N) of the slice within the original file.

    public:
        template<typename Real = f32>
        [[nodiscard]] static constexpr auto center(i64 size) noexcept -> Real {
            // Center is defined at n // 2 (integer division).
            // We rely on this during resizing (as opposed to n / 2).
            return static_cast<Real>(size / 2);
        }

        template<typename Real = f32>
        [[nodiscard]] static constexpr auto center(i64 height, i64 width) noexcept -> Vec2<Real> {
            // Center is defined at n // 2 (integer division). We rely on this during resizing (as opposed to n / 2).
            return {center(height), center(width)};
        }

        template<typename Real = f32, size_t N>
        [[nodiscard]] static constexpr auto center(const Shape<i64, N>& shape) noexcept -> Vec2<Real> {
            return center<Real>(shape[N - 2], shape[N - 1]);
        }

        // Convert the angle (in degrees) to the [-180,180]degrees range.
        [[nodiscard]] static constexpr auto to_angle_range(f64 angle) noexcept -> f64 {
            if (angle < -180)
                angle += 360;
            else if (angle > 180)
                angle -= 360;
            return angle;
        }

        [[nodiscard]] static constexpr auto to_angle_range(Vec3<f64> angles) noexcept -> Vec3<f64> {
            return {to_angle_range(angles[0]),
                    to_angle_range(angles[1]),
                    to_angle_range(angles[2])};
        }
    };

    // Metadata of a stack of 2D slices.
    class MetadataStack {
    public:
        using container = std::vector<MetadataSlice>;
        using const_iterator = container::const_iterator;
        using iterator = container::iterator;
        using const_reference = container::const_reference;
        using reference = container::reference;

    public: // Static functions
        static MetadataStack load_csv(const Path& filename);

    public:
        MetadataStack() = default;
        explicit MetadataStack(container&& slices) : m_slices(std::move(slices)) {}

        // Initializes the slices:
        //  - The tilt angles and exposure are set using the tilt_scheme:order or the tilt/exposure file.
        //  - The rotation and elevation angles, as well as the shifts, are set to 0.
        //  - The known angle offsets are not added at this point: the alignment will deal with them.
        //  - The slice index and index_file are set, either from the tilt_scheme:order (in which case
        //    slices are assumed to be saved in tilt-ascending order), or from the tilt file.
        explicit MetadataStack(const Options& options);

        void save(
                const Path& filename,
                Shape2<i64> shape,
                Vec2<f64> spacing,
                f64 defocus_astigmatism_value = 0,
                f64 defocus_astigmatism_angle = 0,
                f64 phase_shift = 0
        ) const;

    public: // Stack manipulations
        // Excludes slice(s) according to a predicate.
        // Predicate: The predicate is a function taking a MetadataSlice a retuning a boolean
        //            If the predicate returns true, the slice should be removed.
        // Reset indexes: The filtered metadata is sorted in ascending order according to the "index" field,
        //                and the "index" field is reset from [0, N), N being the new slices count.
        template<typename Predicate,
                 typename = std::enable_if_t<std::is_invocable_r_v<bool, Predicate, const MetadataSlice&>>>
        auto exclude(
                Predicate&& predicate,
                bool reset_index_field = true
        ) -> MetadataStack& {
            const auto end_of_new_range = std::remove_if(
                    m_slices.begin(), m_slices.end(), std::forward<Predicate>(predicate));
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
        auto exclude(const std::vector<i64>& indexes_to_exclude, bool reset_index_field = true) -> MetadataStack& {
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
        auto sort(std::string_view key, bool ascending = true) -> MetadataStack&;

        // Update the metadata using the values of the input metadata.
        // The input and output (i.e. self) slices are matched using the .index field.
        // The angles and shifts can be updated. A scaling factor is applied to the input shifts first.
        auto update_from(
                const MetadataStack& input,
                bool update_angles,
                bool update_shifts,
                bool update_defocus,
                Vec2<f64> input_spacing = {1, 1},
                Vec2<f64> current_spacing = {1, 1}
        ) -> MetadataStack& {
            const auto scale = input_spacing / current_spacing;
            for (MetadataSlice& output_slice: slices()) {
                for (const MetadataSlice& input_slice: input.slices()) {
                    if (output_slice.index == input_slice.index) {
                        if (update_angles)
                            output_slice.angles = input_slice.angles;
                        if (update_shifts)
                            output_slice.shifts = input_slice.shifts * scale;
                        if (update_defocus)
                            output_slice.defocus = input_slice.defocus;
                    }
                }
            }
            return *this;
        }

        auto update_angles_from(const MetadataStack& input) -> MetadataStack& {
            return update_from(input, true, false, false);
        }
        auto update_shift_from(
                const MetadataStack& input,
                Vec2<f64> input_spacing,
                Vec2<f64> current_spacing
        ) -> MetadataStack& {
            return update_from(input, false, true, false, input_spacing, current_spacing);
        }
        auto update_defocus_from(const MetadataStack& input) -> MetadataStack& {
            return update_from(input, false, false, true);
        }

        // Shift the sample by a given amount.
        auto add_global_shift(Vec3<f64> global_shift) -> MetadataStack& {
            for (auto& slice: slices()) {
                const Vec3<f64> angles = noa::math::deg2rad(slice.angles);
                const Double33 rotm{
                        noa::geometry::rotate_z(angles[0]) *
                        noa::geometry::rotate_x(angles[2]) *
                        noa::geometry::rotate_y(angles[1]) *
                        noa::geometry::rotate_z(-angles[0])
                };
                const auto rotated_shift = rotm * global_shift;
                slice.shifts += rotated_shift.pop_front(); // project along z
            }
            return *this;
        }

        auto add_global_shift(Vec2<f64> global_shift) -> MetadataStack& {
            return add_global_shift(global_shift.push_front(0));
        }

        auto add_global_angles(Vec3<f64> global_angles) -> MetadataStack& {
            for (auto& slice: slices())
                slice.angles = MetadataSlice::to_angle_range(slice.angles + global_angles);
            return *this;
        }

        auto rescale_shifts(Vec2<f64> current_spacing, Vec2<f64> desired_spacing) -> MetadataStack& {
            const auto scale = current_spacing / desired_spacing;
            for (auto& slice: slices())
                slice.shifts *= scale;
            return *this;
        }

        // Move the average shift to 0.
        auto center_shifts() -> MetadataStack& {
            Vec2<f64> mean{0};
            auto mean_scale = 1 / static_cast<f64>(size());
            for (auto& slice: slices()) {
                const Vec3<f64> angles = noa::math::deg2rad(slice.angles);
                const Vec2<f64> elevation_tilt = angles.filter(2, 1);
                const Double22 stretch_to_0deg{
                        noa::geometry::rotate(angles[0]) *
                        noa::geometry::scale(1 / noa::math::cos(elevation_tilt)) * // 1 = cos(0deg)
                        noa::geometry::rotate(-angles[0])
                };
                const Vec2<f64> shift_at_0deg = stretch_to_0deg * slice.shifts;
                mean += shift_at_0deg * mean_scale;
            }
            return add_global_shift(-mean);
        }

    public: // Getters
        [[nodiscard]] constexpr auto slices() const noexcept -> const std::vector<MetadataSlice>& { return m_slices; }
        [[nodiscard]] constexpr auto slices() noexcept -> std::vector<MetadataSlice>& { return m_slices; }
        [[nodiscard]] auto size() const noexcept -> size_t { return m_slices.size(); }
        [[nodiscard]] auto ssize() const noexcept -> i64 { return static_cast<i64>(size()); }

        [[nodiscard]] auto begin() const noexcept -> const_iterator { return m_slices.cbegin(); }
        [[nodiscard]] auto begin() noexcept -> iterator { return m_slices.begin(); }
        [[nodiscard]] auto end() const noexcept -> const_iterator { return m_slices.cend(); }
        [[nodiscard]] auto end() noexcept -> iterator { return m_slices.end(); }

        [[nodiscard]] auto front() const noexcept -> const_reference { return m_slices.front(); }
        [[nodiscard]] auto front() noexcept -> reference { return m_slices.front(); }
        [[nodiscard]] auto back() const noexcept -> const_reference { return m_slices.back(); }
        [[nodiscard]] auto back() noexcept -> reference { return m_slices.back(); }

        // Returns a view of the slice at "idx", as currently sorted in this instance (see sort()).
        template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
        [[nodiscard]] constexpr auto operator[](T idx) noexcept -> MetadataSlice& {
            NOA_ASSERT(idx >= 0 && static_cast<size_t>(idx) < size());
            return m_slices[static_cast<size_t>(idx)];
        }

        // Returns a view of the slice at "idx", as currently sorted in this instance (see sort()).
        template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
        [[nodiscard]] constexpr auto operator[](T idx) const noexcept -> const MetadataSlice& {
            NOA_ASSERT(idx >= 0 && static_cast<size_t>(idx) < size());
            return m_slices[static_cast<size_t>(idx)];
        }

        // Find the index (as currently sorted in this instance)
        // of the slice with the lowest absolute tilt angle.
        [[nodiscard]] auto find_lowest_tilt_index() const -> i64;

        // Find the index (as currently sorted in this instance)
        // of the slice with the highest absolute tilt angle.
        [[nodiscard]] auto minmax_tilts() const -> std::pair<f64, f64>;

    private:
        void generate_(const Path& tlt_filename, const Path& exposure_filename);
        void generate_(f64 starting_angle, i64 starting_direction, f64 tilt_increment,
                       i64 group_of, bool exclude_start, f64 per_view_exposure, i32 n_slices);

        void sort_on_indexes_(bool ascending = true);
        void sort_on_file_indexes_(bool ascending = true);
        void sort_on_tilt_(bool ascending = true);
        void sort_on_absolute_tilt_(bool ascending = true);
        void sort_on_exposure_(bool ascending = true);

    private:
        std::vector<MetadataSlice> m_slices;
    };
}
