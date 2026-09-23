#include "quinoa/Logger.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/Plot.hpp"

#include "quinoa/align/Run.hpp"
#include "quinoa/align/Tilter.hpp"
#include "quinoa/align/Projection.hpp"
#include "quinoa/align/Thickness.hpp"
#include "quinoa/postprocessing/FilterStack.hpp"

namespace {
    auto allocate_shared_buffer(Device device, isize n_bytes) {
        // If there's enough memory use device-only as it seems more performant on some systems.
        const bool has_enough_space = [&] {
            const auto available = device.memory_capacity().free;
            const auto rough_estimate = static_cast<usize>(n_bytes) + static_cast<usize>(n_bytes / 10);
            return available >= rough_estimate;
        }();
        return Array<std::byte>(noa::next_multiple_of(n_bytes, 16), {
            .device = device,
            .allocator = has_enough_space ? Allocator::ASYNC : Allocator::MANAGED,
        });
    }
}

namespace qn {
    void coarse_alignment(
        const Path& stack_path,
        Metadata& metadata,
        Device device,
        const Settings::Alignment::Coarse& settings,
        const Path& output_directory
    ) {
        auto t0 = Logger::status_scope_time("Coarse alignment");

        // The stage leveling needs the stack sorted with the tilt in ascending order and will throw otherwise.
        metadata.stack.sort("tilt").reset_indices();

        const auto tilt_series = load_stack(stack_path, metadata, {
            .compute_device = device,
            .allocator = Allocator::DEFAULT_ASYNC,

            // Fourier cropping:
            // Keep everything at low resolution, high frequencies are useless here.
            .precise_cutoff = false,
            .rescale_target_resolution = settings.resolution,
            .rescale_min_size = settings.min_size_pix,
            .rescale_max_size = settings.max_size_pix,

            // Signal processing after cropping:
            .bandpass{
                .highpass_cutoff = settings.bandpass.highpass_cutoff,
                .highpass_width = settings.bandpass.highpass_width,
                .lowpass_cutoff = settings.bandpass.lowpass_cutoff,
                .lowpass_width = settings.bandpass.lowpass_width,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .fake_sirt_iterations = settings.fake_sirt_iterations,
            .exposure_filter_voltage = metadata.sample.voltage,// TODO 0?

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.05,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        // Clean up FFT state.
        if (device.is_gpu()) {
            nf::clear_cache(device);
            nf::set_cache_limit(12, device);
        }

        // Prepare the tilter.
        const auto b0 = Allocator::bytes_currently_allocated(device);
        auto tilter = Tilter(tilt_series.shape());
        const auto shared_buffer = allocate_shared_buffer(tilt_series.device(), tilter.shared_buffer_bytes());
        tilter.set_shared_buffer(shared_buffer);
        const auto allocated = Allocator::bytes_currently_allocated(device) - b0;
        Logger::trace("Allocated {:.2f}GB on {} ({})",
                      static_cast<f64>(allocated) * 1e-9, device, shared_buffer.allocator());

        auto angle_offsets = Vec{0., 0., 0.};

        // We require the rotation angle from the mdoc, but we can check that this rotation matches the images.
        // In this case, do a quick shift alignment and search for the rotation offset across the full angle range.
        // The resulting angle isn't the most accurate but should be close enough (+-5deg) to the provided rotation.
        if (settings.check_rotation) {
            auto t1 = Logger::info_scope_time("Rotation check");
            auto metadata_check = metadata.stack;

            for (auto i: noa::irange<usize>(2)) {
                tilter.find_image_shifts(tilt_series.view(), metadata_check, {
                    .cosine_stretch = i > 0,
                    .update_count = 1,
                    .fov_mask = false,
                    .smooth_edge_percent = 0.08,
                    .max_shift_percent = 0.5,
                });

                constexpr auto ROTATION_STEP = std::array{0.25, 0.1};
                tilter.find_image_rotation(tilt_series.view(), metadata_check, angle_offsets, {
                    .accurate_fov = false,
                    .angle_range = 90.,
                    .angle_step = ROTATION_STEP[i],
                    .output_directory = &output_directory,
                });
                angle_offsets = 0; // we don't care about the offsets here
            }

            const auto expected_rotation = Metadata::Image::to_angle_range(metadata.stack[0].angles[0]);
            const auto measured_rotation_1 = Metadata::Image::to_angle_range(metadata_check[0].angles[0]);
            const auto measured_rotation_2 = Metadata::Image::to_angle_range(metadata_check[0].angles[0] + 180);
            auto distance_from_measured = [&](f64 e) {
                return std::min(std::abs(e - measured_rotation_1), std::abs(e - measured_rotation_2));
            };

            if (distance_from_measured(expected_rotation) > 5.) {
                if (settings.is_tilt_axis_from_mdoc and settings.allow_90_and_flip_rotation_from_mdoc) {
                    // The mdoc doesn't always encode the tilt-axis correctly.
                    // A 90degree offset and/or image flip may be present.
                    // Check if these configurations result in a closer distance from the measured line.
                    const auto expected_rotations = Vec{
                        Metadata::Image::to_angle_range(metadata.stack[0].angles[0] + 90.),
                        Metadata::Image::to_angle_range(metadata.stack[0].angles[0] * -1.),
                        Metadata::Image::to_angle_range(metadata.stack[0].angles[0] * -1. + 90.),
                    };
                    const auto distances = Vec{
                        distance_from_measured(expected_rotations[0]),
                        distance_from_measured(expected_rotations[1]),
                        distance_from_measured(expected_rotations[2]),
                    };
                    const auto MESSAGES = std::array{
                        "Rotating 90 degrees",
                        "Flipping",
                        "Flipping and rotating 90 degrees",
                    };

                    const auto index = noa::argmin(distances);
                    check(expected_rotations[index] <= 5.,
                          "The tilt-axis from the mdoc file ({:.2f}) does not seem to match the tilt images (rotation_estimate={:.2f}deg, or equivalently {:.2f}deg) and does not seem to be flipped and/or rotated 90 degrees (which is allowed given alignment.coarse.allow_90_and_flip_rotation_from_mdoc=true). Since this check is fairly reliable, the program will stop now. If you are certain that the provided tilt-axis is correct, this check can be turned off using alignment.coarse.check_rotation=false.",
                          expected_rotation, measured_rotation_1, measured_rotation_2);

                    const auto new_rotations = Vec{
                        expected_rotations[index],
                        Metadata::Image::to_angle_range(expected_rotations[index] + 180),
                    };
                    const auto new_rotation = new_rotations[argmin(abs(new_rotations))];
                    Logger::info(
                        "{} ({:.3f} to {:.3f}) to match the tilt images (alignment.coarse.allow_90_and_flip_rotation_from_mdoc=true)",
                        MESSAGES[index], expected_rotation, new_rotation);
                    for (auto& e: metadata.stack)
                        e.angles[0] = new_rotation;
                } else {
                    panic(
                        "The tilt-axis from the {} ({:.2f}) does not seem to match the tilt images (rotation_estimate={:.2f}deg, or equivalently {:.2f}deg) and alignment.coarse.allow_90_and_flip_rotation_from_mdoc=false. Since this check is fairly reliable, the program will stop now. If you are certain that the provided tilt-axis is correct, this check can be turned off using alignment.coarse.check_rotation=false.",
                        settings.is_tilt_axis_from_mdoc ? "mdoc file" : "experiment.tilt_axis setting",
                        expected_rotation, measured_rotation_1, measured_rotation_2
                    );
                }
            } else {
                Logger::info(
                    "The tilt-axis from the {} ({:.2f}) seem to match the tilt images (rotation_estimate={:.2f}deg, or equivalently {:.2f}deg)",
                    settings.is_tilt_axis_from_mdoc ? "mdoc file" : "experiment.tilt_axis setting",
                    expected_rotation, measured_rotation_1, measured_rotation_2
                );
            }
        }

        // Coarse alignment.
        for (auto i: noa::irange<usize>(4)) {
            tilter.find_image_shifts(tilt_series.view(), metadata.stack, {
                .cosine_stretch = i != 0,
                .update_count = i >= 2 ? 5 : 10,
                .fov_mask = i >= 2,
                .smooth_edge_percent = i == 0 ? 0.08 : 0.3,
                .max_shift_percent = i == 0 ? 0.5 : 0.1,
            });

            if (settings.fit_rotation) {
                constexpr auto ROTATION_RANGE = std::array{10., 5., 2., 1.};
                tilter.find_image_rotation(tilt_series.view(), metadata.stack, angle_offsets, {
                    .accurate_fov = i >= 2,
                    .angle_range = ROTATION_RANGE[i],
                    .angle_step = 0.01,
                    .output_directory = &output_directory,
                });
            }

            if (settings.fit_tilt or settings.fit_pitch) {
                constexpr auto TILT_RANGE = std::array{20., 10., 2., 1.};
                constexpr auto PITCH_RANGE = std::array{10., 5., 2., 1.};
                tilter.find_specimen_level(tilt_series.view(), metadata.stack, angle_offsets, {
                    .tilt_search_range = not settings.fit_tilt ? 0. : TILT_RANGE[i],
                    .pitch_search_range = not settings.fit_pitch ? 0. : PITCH_RANGE[i],
                    .n_global_search_evaluations = 0, // initial global search doesn't seem necessary
                    .fov_mask = i >= 2,
                    .smooth_edge_percent = i == 0 ? 0.08 : 0.3,
                    .max_shift_percent = i == 0 ? 0.5 : 0.1,
                });
            }

            // TODO Detect for view with huge shifts and remove them?
            //      Maybe only for higher tilts, e.g. >20deg, since low tilts
            //      are unlikely to blame and are very valuable.
        }

        tilter.find_image_shifts(tilt_series.view(), metadata.stack, {
            .cosine_stretch = true,
            .update_count = 15,
            .fov_mask = true,
            .smooth_edge_percent = 0.1,
            .max_shift_percent = 0.1,
        });
    }

    void refine_alignment(
        const Path& stack_filename,
        Metadata& metadata,
        Device device,
        const Settings::Alignment::Refine& settings,
        const Path& output_directory
    ) {
        auto timer = Logger::status_scope_time("Refine alignment");

        auto loader = StackLoader(stack_filename, {
            .compute_device = device,
            .allocator = Allocator::MANAGED,

            // Fourier cropping:
            .precise_cutoff = true, // ensure isotropic spacing
            .rescale_target_resolution = settings.resolution,
            .rescale_min_size = settings.min_size_pix,
            .rescale_max_size = settings.max_size_pix,

            // Signal processing after cropping:
            .bandpass{
                .highpass_cutoff = settings.bandpass.highpass_cutoff,
                .highpass_width = settings.bandpass.highpass_width,
                .lowpass_cutoff = settings.bandpass.lowpass_cutoff,
                .lowpass_width = settings.bandpass.lowpass_width,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .fake_sirt_iterations = settings.fake_sirt_iterations,
            .exposure_filter_voltage = metadata.sample.voltage,

            // Image processing after cropping:
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.03,
            .zero_pad_to_fast_fft_shape = true,
            .zero_pad_to_square_shape = false,
        });

        const auto stack_spacing = mean(loader.stack_spacing());
        metadata.set_spacing(stack_spacing);
        metadata.stack.sort("tilt").reset_indices(); // load ascending tilt order

        // Load and filter the tilt-series.
        // If the CTF is off, this simply loads the tilt-series by pulling images from the loader.
        // If the CTF is on, it corrects for the CTF at the "center of the sample", where most of the signal comes from.
        // It is up to the thickness estimation and tomogram centering to place that "center of the sample"
        // at the center of the tomogram. On the other hand, this is quite low resolution and the CTF correction
        // has little to no effect; it's mostly about the B-factor filtering, and even that can be replaced using
        // filtering from the stack loader.
        const auto tilt_series = filter_stack(std::move(loader), metadata, {
            .correct_ctf = settings.correct_ctf,
            .ctf_phase_flip_strength = settings.ctf_phase_flip_strength,
            .ctf_defocus_step_nm = 15, // TODO probably not worth it, decrease it
            .ctf_bfactor = 0,
        });

        // Prepare for the specimen thickness.
        // Loads a low-resolution tilt-series and allocate the full tomogram.
        // Most of the memory requirement comes from this tomogram.
        auto specimen_thickness = SpecimenThickness{};
        if (settings.fit_thickness)
            specimen_thickness = SpecimenThickness(stack_filename, metadata, device);

        // Clean up FFT state.
        if (device.is_gpu()) {
            nf::clear_cache(device);
            nf::set_cache_limit(12, device);
        }

        // Prepare for the specimen leveling.
        auto tilter = Tilter{};
        if (settings.fit_tilt or settings.fit_pitch)
            tilter = Tilter(tilt_series.shape());

        // Prepare for the projection matching.
        const auto n_images = metadata.stack.ssize();
        const auto image_shape = tilt_series.shape().filter(2, 3);
        auto projection_matcher = ProjectionMatcher(n_images, image_shape, settings.max_tilt_difference);

        // Reduce memory usage by sharing a write buffer between the different steps.
        const auto specimen_thickness_bytes = specimen_thickness.shared_buffer_bytes();
        const auto tilter_bytes = tilter.shared_buffer_bytes();
        const auto projection_matcher_bytes = projection_matcher.shared_buffer_bytes();
        const auto n_bytes_shared_buffer = std::max({
            specimen_thickness.shared_buffer_bytes(),
            tilter.shared_buffer_bytes(),
            projection_matcher.shared_buffer_bytes()
        });

        const auto b0 = Allocator::bytes_currently_allocated(device);

        auto shared_buffer = allocate_shared_buffer(device, n_bytes_shared_buffer);
        specimen_thickness.set_shared_buffer(shared_buffer.view());
        tilter.set_shared_buffer(shared_buffer);
        projection_matcher.set_shared_buffer(shared_buffer.view());

        const auto allocated = Allocator::bytes_currently_allocated(device) - b0;
        Logger::trace(
            "\nSharing buffer: (device={}, {}):\n"
            "  specimen_thickness={:.3f}GB\n"
            "  specimen_leveling={:.3f}GB\n"
            "  projection_matching={:.3f}GB",
             device, shared_buffer.allocator(),
            static_cast<f64>(specimen_thickness_bytes) * 1e-9,
            static_cast<f64>(tilter_bytes) * 1e-9,
            static_cast<f64>(projection_matcher_bytes) * 1e-9
        );
        Logger::trace(
            "Allocated {:.3f}GB (device={}, {})\n",
            static_cast<f64>(allocated) * 1e-9, device, shared_buffer.allocator()
        );

        // Run.
        auto angle_offsets = Vec{0., 0., 0.};
        constexpr f64 SMOOTH_EDGE_PERCENT = 0.1;
        const usize n_iterations = settings.fit_rotation or settings.fit_tilt or settings.fit_pitch ?
            noa::clamp_cast<usize>(settings.nb_iterations) : 1;

        if (settings.fit_thickness)
            metadata.sample.thickness = specimen_thickness.estimate(metadata, output_directory);

        for (auto i: noa::irange(n_iterations)) {
            projection_matcher.update_shifts(tilt_series.view(), metadata, {
                .max_tilt_difference = settings.max_tilt_difference,
                .smooth_edge_percent = SMOOTH_EDGE_PERCENT,
            });

            if (settings.fit_rotation) {
                Tilter::find_accurate_image_rotation(tilt_series.view(), metadata.stack, angle_offsets, {
                    .angle_range = i == 0 ? 4. : 1.,
                    .output_directory = &output_directory,
                });
            }

            if (settings.fit_tilt or settings.fit_pitch) {
                constexpr auto TILT_RANGE = std::array{5., 1.};
                constexpr auto PITCH_RANGE = std::array{5., 1.};
                tilter.find_specimen_level(tilt_series.view(), metadata.stack, angle_offsets, {
                    .tilt_search_range = not settings.fit_tilt ? 0. : TILT_RANGE[i],
                    .pitch_search_range = not settings.fit_pitch ? 0. : PITCH_RANGE[i],
                    .n_global_search_evaluations = 0, // initial global search doesn't seem necessary
                    .fov_mask = true,
                    .smooth_edge_percent = 0.3,
                    .max_shift_percent = 0.1,
                });
            }

            if (settings.fit_thickness)
                metadata.sample.thickness = specimen_thickness.estimate(metadata, output_directory);

            // Note that we don't recompute the CTF correction despite the possible change of tilt-axis.
            // Usually the axis barely changes, and while we could check and recompute it if the change is significant,
            // the CTF correction has little effect and is already an approximation anyway.
        }

        projection_matcher.update_shifts(tilt_series.view(), metadata, {
            .max_tilt_difference = settings.max_tilt_difference,
            .smooth_edge_percent = SMOOTH_EDGE_PERCENT,
        });
    }
}
