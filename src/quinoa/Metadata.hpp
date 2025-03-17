#pragma once

#include <noa/Core.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Options.hpp"

namespace qn {
    /// Metadata of a 2d slice.
    /// \note Shifts:
    ///     - The shifts are applied before the rotations.
    ///     - They are offsets ("by how much the slice is shifted"), in pixels.
    ///       As such, to align the slice (aka to go from image to volume space), we subtract the shifts.
    /// \note Angles:
    ///     - The rotation center is fixed at n // 2.
    ///     - The Euler angles are in degrees, ZYX extrinsic (rotation, tilt, pitch).
    ///     - All angles are positive-CCW when looking at the origin from the positive side.
    ///     - They are offsets ("by how much the slice is rotated"), in degrees.
    ///       As such, to align the slice (aka to go from image to volume space), we subtract the angles.
    struct MetadataSlice {
    public:
        i32 index{};            /// Index [0, N) of the slice in the array.
        i32 index_file{};       /// Index [0, N) of the slice in the original file.
        Vec<f64, 3> angles{};   /// Euler angles, in degrees, of the slice. ZYX extrinsic (rotation, tilt, pitch)
        Vec<f64, 2> shifts{};   /// YX shifts, in pixels, of the slice.
        Vec<f64, 2> exposure{}; /// Pre- and post-exposure, in e-/A2.
        f64 phase_shift{};      /// Phase shift, in radians.
        ns::DefocusAstigmatic<f64> defocus{};

    public:
        /// Convert the angle (in degrees) to the [-180,180] degrees range.
        [[nodiscard]] static constexpr auto to_angle_range(f64 angle) noexcept -> f64 {
            if (angle < -180)
                angle += 360;
            else if (angle > 180)
                angle -= 360;
            return angle;
        }

        [[nodiscard]] static constexpr auto to_angle_range(const Vec<f64, 3>& angles) noexcept -> Vec<f64, 3> {
            return {to_angle_range(angles[0]),
                    to_angle_range(angles[1]),
                    to_angle_range(angles[2])};
        }
    };

    /// Metadata of 2d slices.
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

        /// Initializes the slices:
        ///  - The tilt angles and exposure are set using the tilt_scheme:order or the tilt/exposure file.
        ///  - The known angle offsets are added.
        ///  - The slice index and index_file are set, either from the tilt_scheme:order (in which case
        ///    slices are assumed to be saved in tilt-ascending order), or from the tilt file.
        explicit MetadataStack(const Options& options);

        void save_csv(
            const Path& filename,
            Shape<i64, 2> shape,
            Vec<f64, 2> spacing
        ) const;

    public: // Stack manipulations
        /// Excludes slice(s) according to a predicate.
        /// \param predicate A function taking a MetadataSlice a retuning a boolean
        ///                  If the predicate returns true, the slice should be removed.
        template<typename Predicate> requires std::is_invocable_r_v<bool, Predicate, const MetadataSlice&>
        auto exclude(Predicate&& predicate) -> MetadataStack& { // TODO rename to exclude_if
            std::erase_if(m_slices, std::forward<Predicate>(predicate));
            return *this;
        }

        /// Resets the index field from [0, N), using the current order.
        auto reset_indices() -> MetadataStack& {
            i32 count{};
            for (auto& slice: m_slices)
                slice.index = count++;
            return *this;
        }

        /// (Stable) sorts the slices based on a given key.
        /// Valid keys: "index", "index_file", "tilt", "absolute_tilt", "exposure".
        auto sort(std::string_view key, bool ascending = true) -> MetadataStack&;

        struct UpdateOptions {
            bool update_angles{false};
            bool update_shifts{false};
            bool update_defocus{false};
            bool update_phase_shift{false};
        };
        /// Update the metadata using the values of the input metadata.
        /// The input and output (i.e. self) slices are matched using the .index field.
        /// The angles and shifts can be updated. A scaling factor is applied to the input shifts first.
        auto update_from(
            const MetadataStack& input,
            const UpdateOptions& options,
            Vec<f64, 2> input_spacing = {1, 1},
            Vec<f64, 2> current_spacing = {1, 1}
        ) -> MetadataStack& {
            const auto scale = input_spacing / current_spacing;
            for (MetadataSlice& output_slice: slices()) {
                for (const MetadataSlice& input_slice: input) {
                    if (output_slice.index == input_slice.index) {
                        check(output_slice.index_file == input_slice.index_file);
                        if (options.update_angles)
                            output_slice.angles = input_slice.angles;
                        if (options.update_shifts)
                            output_slice.shifts = input_slice.shifts * scale;
                        if (options.update_defocus)
                            output_slice.defocus = input_slice.defocus;
                        if (options.update_phase_shift)
                            output_slice.phase_shift = input_slice.phase_shift;
                    }
                }
            }
            return *this;
        }

        /// Shift the sample by a given amount.
        auto add_global_shift(const Vec<f64, 3>& global_shift) -> MetadataStack& {
            for (auto& slice: slices()) {
                // Go from volume->image space.
                const auto angles = noa::deg2rad(slice.angles);
                const auto volume2image = (
                    ng::rotate_z(+angles[0]) *
                    ng::rotate_y(+angles[1]) *
                    ng::rotate_x(+angles[2])
                ).pop_front(); // project along z
                slice.shifts += volume2image * global_shift;
            }
            return *this;
        }

        auto add_global_shift(const Vec<f64, 2>& global_shift) -> MetadataStack& {
            return add_global_shift(global_shift.push_front(0));
        }

        auto add_global_angles(const Vec<f64, 3>& global_angles) -> MetadataStack& {
            for (auto& slice: slices())
                slice.angles = MetadataSlice::to_angle_range(slice.angles + global_angles);
            return *this;
        }

        auto rescale_shifts(const Vec<f64, 2>& current_spacing, const Vec<f64, 2>& desired_spacing) -> MetadataStack& {
            const auto scale = current_spacing / desired_spacing;
            for (auto& slice: slices())
                slice.shifts *= scale;
            return *this;
        }

        /// Move the average shift to 0.
        auto center_shifts() -> MetadataStack& {
            Vec<f64, 2> mean{};
            // Compute the average shift in volume space...
            for (auto& slice: slices()) {
                const auto angles = noa::deg2rad(slice.angles);
                const auto image2volume = (
                    ng::rotate_x(-angles[2]) *
                    ng::rotate_y(-angles[1]) *
                    ng::rotate_z(-angles[0])
                );
                mean += image2volume * slice.shifts;
            }
            mean /= static_cast<f64>(size());

            // Then center the volume onto that point.
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

        /// Returns a view of the slice at "idx", as currently sorted in this instance (see sort()).
        [[nodiscard]] constexpr auto operator[](std::integral auto index) noexcept -> MetadataSlice& {
            ni::bounds_check<true>(ssize(), index);
            return m_slices[static_cast<size_t>(index)];
        }

        /// Returns a view of the slice at "idx", as currently sorted in this instance (see sort()).
        [[nodiscard]] constexpr auto operator[](std::integral auto index) const noexcept -> const MetadataSlice& {
            ni::bounds_check<true>(ssize(), index);
            return m_slices[static_cast<size_t>(index)];
        }

        /// Find the index (as currently sorted in this instance)
        /// of the slice with the lowest absolute tilt angle.
        [[nodiscard]] auto find_lowest_tilt_index() const -> i64;

        /// Find the index (as currently sorted in this instance)
        /// of the slice with the highest absolute tilt angle.
        [[nodiscard]] auto minmax_tilts() const -> std::pair<f64, f64>;

    private:
        void generate_(const Path& tilt_filename, const Path& exposure_filename);
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
