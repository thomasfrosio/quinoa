#pragma once

#include <noa/Core.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Settings.hpp"

namespace qn {
    struct Metadata {
        struct Image {
            using defocus_type = ns::DefocusAstigmatic<f64>;
            using time_type = stdc::time_point<stdc::system_clock, stdc::seconds>;

        public:
            /// Index [0, N) of the image in an array.
            i32 index{};

            /// Index [0, N) of the slice in the original file.
            i32 index_file{};

            /// Euler angles, in degrees, of the image. ZYX extrinsic (rotation, tilt, pitch).
            /// The rotation center is fixed at n // 2. Angles are positive-CCW when looking at the origin from the
            /// positive side. They are offsets ("by how much the central-slice is rotated"), in degrees. As such,
            /// to align the slice (aka to go from image to volume space), we subtract the angles.
            Vec<f64, 3> angles{};

            /// Shifts, in pixels, of the image.
            /// The shifts are applied before the rotations. They are offsets ("by how much the image is shifted").
            /// As such, to align the image (aka to go from image to volume space), we subtract the shifts.
            Vec<f64, 2> shifts{};

            /// Pre- and post-exposure, in e-/A2.
            Vec<f64, 2> exposure{};

            /// Phase shift, in degrees.
            f64 phase_shift{};

            /// Astigmatic defocus.
            defocus_type defocus{};

            /// Collection time-point.
            i64 time{};

            /// Path to the frame file.
            /// Frames are optional, so this pointer may be null.
            const Path* frames{};

        public:
            /// Convert the angle (in degrees) to the [-180,180] degrees range.
            [[nodiscard]] static constexpr auto to_angle_range(f64 angle) noexcept {
                if (angle < -180)
                    angle += 360;
                else if (angle > 180)
                    angle -= 360;
                return angle;
            }

            [[nodiscard]] static constexpr auto to_angle_range(const Vec<f64, 3>& angles) noexcept {
                return Vec{
                    to_angle_range(angles[0]),
                    to_angle_range(angles[1]),
                    to_angle_range(angles[2])
                };
            }
        };

        /// Sequence of images.
        struct Stack {
            std::vector<Image> images{};

            [[nodiscard]] auto clone() const { return Stack{images}; }

            /// Excludes images(s) according to a predicate.
            /// \param predicate A function taking a Metadata::Image and retuning a boolean.
            ///                  If the predicate returns true, the image is removed.
            template<typename Predicate> requires std::is_invocable_r_v<bool, Predicate, const Image&>
            auto exclude_if(Predicate&& predicate) -> Stack& {
                std::erase_if(images, std::forward<Predicate>(predicate));
                return *this;
            }

            /// Resets the .index field from [0, N), using the current image order.
            auto reset_indices() -> Stack& {
                i32 count{};
                for (auto& slice: images)
                    slice.index = count++;
                return *this;
            }

            /// (Stable) sorts the images based on a given key.
            /// Valid keys: "index", "index_file", "tilt", "absolute_tilt", "exposure", "time".
            auto sort(std::string_view key, bool ascending = true) -> Stack&;

            struct UpdateOptions {
                bool update_angles{false};
                bool update_shifts{false};
                bool update_defocus{false};
                bool update_phase_shift{false};
            };

            /// Update the metadata using the values of the input metadata.
            /// The input and output (i.e. self) slices are matched using the .index field.
            auto update_from(
                const Stack& input,
                const UpdateOptions& options
            ) -> Stack& {
                for (auto& output_slice: images) {
                    for (const auto& input_slice: input) {
                        if (output_slice.index == input_slice.index) {
                            check(output_slice.index_file == input_slice.index_file);
                            if (options.update_angles)
                                output_slice.angles = input_slice.angles;
                            if (options.update_shifts)
                                output_slice.shifts = input_slice.shifts;
                            if (options.update_defocus)
                                output_slice.defocus = input_slice.defocus;
                            if (options.update_phase_shift)
                                output_slice.phase_shift = input_slice.phase_shift;
                        }
                    }
                }
                return *this;
            }

            /// Shift the volume space (aka volume reference-frame) by the given amount.
            /// Importantly, this results in moving the field-of-view by the given amount,
            /// so for instance, to move the specimen up in Z by x, -x should be passed here.
            auto add_volume_shift(const Vec<f64, 3>& shift) -> Stack& {
                for (auto& image: images) {
                    // Go from volume->image space.
                    const auto angles = noa::deg2rad(image.angles);
                    const auto volume2image = (
                        nx::rotate_z(+angles[0]) *
                        nx::rotate_y(+angles[1]) *
                        nx::rotate_x(+angles[2])
                    ).pop_front(); // project along z
                    image.shifts += volume2image * shift;
                }
                return *this;
            }

            auto add_volume_shift(const Vec<f64, 2>& global_shift) -> Stack& {
                return add_volume_shift(global_shift.push_front(0));
            }

            auto rescale_shifts(const Vec<f64, 2>& current_spacing, const Vec<f64, 2>& desired_spacing) -> Stack& {
                const auto scale = current_spacing / desired_spacing;
                for (auto& slice: images)
                    slice.shifts *= scale;
                return *this;
            }

            /// Set the average shift in volume-space to 0.
            auto center_shifts() -> Stack& {
                // Compute the average shift in volume space.
                auto mean = Vec<f64, 2>{};
                for (auto& image: images) {
                    const auto angles = noa::deg2rad(image.angles);

                    const auto plane_rotation =
                            nx::rotate_z(angles[0]) *
                            nx::rotate_y(angles[1]) *
                            nx::rotate_x(angles[2]);
                    const auto [c, b, a] = plane_rotation * Vec{1., 0., 0.}; // plane normal
                    const auto z = -(a * image.shifts[1] + b * image.shifts[0]) / c;

                    const auto image2volume =
                            nx::rotate_x(-angles[2]) *
                            nx::rotate_y(-angles[1]) *
                            nx::rotate_z(-angles[0]);
                    const auto volume_shift = image2volume * image.shifts.push_front(z);

                    mean += volume_shift.pop_front(); // z should be zero
                }
                mean /= static_cast<f64>(size());

                // Center by subtracting the average shift.
                return add_volume_shift(-mean);
            }

            /// Add the given angles (in degrees) to all images.
            /// The backward projection creates an inverse relationship between volume and image space.
            /// To tilt/pitch the volume by an angle x, the angle -x should be added to the images.
            auto add_image_angles(const Vec<f64, 3>& angles) -> Stack& {
                for (auto& image: images)
                    image.angles = Image::to_angle_range(image.angles + angles);
                return *this;
            }

            [[nodiscard]] auto size() const noexcept -> usize { return images.size(); }
            [[nodiscard]] auto ssize() const noexcept -> isize { return static_cast<isize>(size()); }

            /// Returns a view of the image at "idx", as currently sorted in this instance (see sort()).
            [[nodiscard]] constexpr auto operator[](std::integral auto index) noexcept -> Image& {
                noa::bounds_check<true>(ssize(), index);
                return images[static_cast<usize>(index)];
            }

            /// Returns a view of the image at "idx", as currently sorted in this instance (see sort()).
            [[nodiscard]] constexpr auto operator[](std::integral auto index) const noexcept -> const Image& {
                noa::bounds_check<true>(ssize(), index);
                return images[static_cast<usize>(index)];
            }

            /// Find the index (as currently sorted in this instance)
            /// of the slice with the lowest absolute tilt angle.
            [[nodiscard]] auto find_lowest_tilt_index() const -> isize;

            [[nodiscard]] auto tilt_range() const -> Vec<f64, 2>;
            [[nodiscard]] auto time_range() const -> Vec<i64, 2>;
            [[nodiscard]] auto defocus_range() const -> Vec<f64, 2>;

        public: // Range support
            using container = std::vector<Image>;
            using const_iterator = container::const_iterator;
            using iterator = container::iterator;
            using const_reference = container::const_reference;
            using reference = container::reference;

            [[nodiscard]] auto begin() const noexcept -> const_iterator { return images.cbegin(); }
            [[nodiscard]] auto begin() noexcept -> iterator { return images.begin(); }
            [[nodiscard]] auto end() const noexcept -> const_iterator { return images.cend(); }
            [[nodiscard]] auto end() noexcept -> iterator { return images.end(); }

            [[nodiscard]] auto front() const noexcept -> const_reference { return images.front(); }
            [[nodiscard]] auto front() noexcept -> reference { return images.front(); }
            [[nodiscard]] auto back() const noexcept -> const_reference { return images.back(); }
            [[nodiscard]] auto back() noexcept -> reference { return images.back(); }
        };

        /// Microscope/general.
        struct Sample {
            f64 voltage{};
            f64 cs{};
            f64 amplitude{};
            f64 thickness{}; // nm
        };

    public:
        Stack stack{};
        Sample sample{};

        /// Spacing (aka pixel size) of the currently referred images, in angstrom.
        /// \warning Changing this value should be done by set_spacing
        ///          to correctly rescale the image shifts at the same time.
        Vec<f64, 2> spacing{};

    public: // Load
        static auto load_from_mdoc(const Path& mdoc) -> Metadata;
        static auto load_from_star(const Path& star) -> Metadata;
        static auto load_from_settings(const Settings& settings) -> Metadata;

    public: // Save
        void save_star(const Path& filename) const;
        // void save_relion(const Path& filename, Shape<i64, 2> shape, Vec<f64, 2> spacing) const;
        // void save_warp(const Path& filename, Shape<i64, 2> shape, Vec<f64, 2> spacing) const;
        // void save_imod(const Path& filename, Shape<i64, 2> shape, Vec<f64, 2> spacing) const;

    public:
        auto set_spacing(const Vec<f64, 2>& new_spacing) -> Metadata& {
            const auto scale = spacing / new_spacing;
            for (auto& slice: stack)
                slice.shifts *= scale;
            spacing = new_spacing;
            return *this;
        }
        auto set_spacing(f64 new_spacing) -> Metadata& {
            return set_spacing(Vec{new_spacing, new_spacing});
        }

        [[nodiscard]] auto empty_ctf() const {
            return CTFIsotropic64::Parameters{
                .pixel_size = mean(spacing),
                .defocus = 0., // unset
                .voltage = sample.voltage,
                .amplitude = sample.amplitude,
                .cs = sample.cs,
                .phase_shift = 0, // unset
                .bfactor = 0,
                .scale = 1.,
            }.to_ctf();
        }

        [[nodiscard]] auto average_ctf() const {
            f64 average_defocus{};
            f64 average_phase_shift{};
            for (auto& image: stack) {
                average_defocus += image.defocus.value;
                average_phase_shift += image.phase_shift;
            }
            return CTFIsotropic64::Parameters{
                .pixel_size = mean(spacing),
                .defocus = average_defocus / static_cast<f64>(stack.size()),
                .voltage = sample.voltage,
                .amplitude = sample.amplitude,
                .cs = sample.cs,
                .phase_shift = average_phase_shift / static_cast<f64>(stack.size()),
                .bfactor = 0,
                .scale = 1.,
            }.to_ctf();
        }
    };
}
