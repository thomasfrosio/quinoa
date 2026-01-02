#include <noa/Core.hpp>
#include <noa/Utils.hpp>
#include <forward_list>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <flux.hpp>
#pragma GCC diagnostic pop

#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Logger.hpp"

namespace {
    using namespace qn;

    // Store the frame path to prevent unnecessary copies when copying Metadata::Image around.
    // Use the datetime to keep track of the frame.
    std::forward_list<Pair<i64, Path>> frame_paths{};

    auto set_frame_path(i64 time, Path&& path) -> const Path* {
        for (auto& [k, v]: frame_paths) {
            if (k == time) {
                v = std::move(path);
                return &v;
            }
        }
        const auto& front_path = frame_paths.emplace_front(time, std::move(path)).second;
        return &front_path;
    }

    void apply_settings_and_defaults(const Settings& settings, Metadata& metadata, bool is_initialized) {
        const auto has_setting = [&](f64 value) {
            constexpr auto MAX = std::numeric_limits<f64>::max();
            return not noa::allclose(MAX, value);
        };

        // Set/add the angles and phase-shifts.
        // If not in the settings, let them unchanged (from mdoc/star, or zero).
        for (auto& e: metadata.stack) {
            // Set the tilt-axis.
            if (has_setting(settings.experiment.tilt_axis))
                e.angles[0] = Metadata::Image::to_angle_range(settings.experiment.tilt_axis);

            // Add the specimen angles.
            if (has_setting(settings.experiment.add_specimen_tilt))
                e.angles[1] += settings.experiment.add_specimen_tilt;
            if (has_setting(settings.experiment.add_specimen_pitch))
                e.angles[2] += settings.experiment.add_specimen_pitch;

            // Set an initial value for the phase-shift.
            if (has_setting(e.phase_shift))
                e.phase_shift = settings.experiment.phase_shift;
        }

        // Overwrite the frame directory and reset the metadata.
        if (not settings.files.frames_directory.empty()) {
            for (auto& [time, path]: frame_paths)
                path = settings.files.frames_directory / path.filename();

            for (auto& image: metadata.stack)
                for (auto& [time, path]: frame_paths)
                    if (image.time == time)
                        image.frames = &path;
        }

        // Use setting value > mdoc/star file > default value.
        const auto set_value = [&](f64& metadata_value, f64 settings_value, f64 default_value) {
            if (has_setting(settings_value))
                metadata_value = settings_value;
            if (is_initialized)
                return;
            metadata_value = default_value;
        };

        set_value(metadata.sample.voltage, settings.experiment.voltage, 300.);
        set_value(metadata.sample.cs, settings.experiment.cs, 2.7);
        set_value(metadata.sample.amplitude, settings.experiment.amplitude, 0.07);
        set_value(metadata.sample.thickness, settings.experiment.thickness, 0.);
    }

    void check_metadata(const Metadata& metadata) {
        // Check that the tilts are within a reasonable range.
        for (auto& e: metadata.stack) {
            if (std::abs(e.angles[1]) > 75.)
                panic("Tilt angle is greater than -+75deg, this is likely a input error");
        }
        // TODO
    }
}

namespace qn {
    auto Metadata::load_from_mdoc(const Path& mdoc) -> Metadata {
        auto file = noa::io::InputTextFile(mdoc, {.read = true});
        std::string line;

        // TODO Use Voltage.
        // TODO Use PriorRecordDose.

        auto metadata = Metadata{};
        auto& images = metadata.stack.images;

        auto frame_path = Path{};
        bool has_rotation{}, has_tilt{}, has_exposure{}, has_datetime{};
        while (file.next_line_or_throw(line)) {
            std::string_view trimmed = noa::string::trim(line);

            // Create a new image.
            if (trimmed.starts_with("[ZValue =")) {
                if (not images.empty()) {
                    // Before switching to the next image, check that we collected the necessary fields.
                    check(has_tilt and has_exposure and has_datetime,
                          "An image in the mdoc is missing a key value:\n"
                          "has_rotation={}, has_tilt={}, has_exposure={} and has_datetime={}",
                          has_rotation, has_tilt, has_exposure, has_datetime);
                    has_rotation = false;
                    has_tilt = false;
                    has_exposure = false;
                    has_datetime = false;

                    // Register the frame path.
                    images.back().frames = set_frame_path(images.back().time, std::move(frame_path));
                }
                images.push_back({});
                continue;
            }

            // "key = value" -> "value"
            auto get_substring = [&trimmed] {
                return noa::string::trim_left(trimmed.substr(trimmed.find_first_of('=') + 1));
            };

            if (trimmed.starts_with("RotationAngle")) {
                auto substring = get_substring();
                auto result = noa::string::parse<f64>(substring);
                check(result, "Could not parse RotationAngle = {}", substring);
                images.back().angles[0] = *result;
                has_rotation = true;

            } else if (trimmed.starts_with("TiltAngle")) {
                auto substring = get_substring();
                auto result = noa::string::parse<f64>(substring);
                check(result, "Could not parse TiltAngle = {}", substring);
                images.back().angles[1] = *result;
                has_tilt = true;

            } else if (trimmed.starts_with("ExposureDose")) {
                auto substring = get_substring();
                auto result = noa::string::parse<f64>(substring);
                check(result, "Could not parse ExposureDose = {}", substring);
                images.back().exposure[1] = *result;
                has_exposure = true;

            } else if (trimmed.starts_with("SubFramePath")) {
                // Assume '\' are Windows separators. On POSIX, they are valid filename characters.
                auto substring = std::string(get_substring());
                stdr::replace(substring, '\\', '/');

                // Note that we only get the filename. The base path is from the user settings.
                frame_path = Path(std::move(substring)).filename();

            } else if (trimmed.starts_with("DateTime")) {
                auto substring = get_substring();
                std::tm tm{};
                check(::strptime(substring.data(), "%d-%b-%y  %H:%M:%S", &tm) != nullptr or
                      ::strptime(substring.data(), "%d-%b-%Y  %H:%M:%S", &tm) != nullptr,
                      "Could not parse DateTime = {}", substring);
                std::time_t time = std::mktime(&tm);
                check(time != -1);
                images.back().time =
                    stdc::time_point_cast<stdc::seconds>(
                        stdc::system_clock::from_time_t(time)
                    ).time_since_epoch().count();
                has_datetime = true;
            }
        }

        // Compute pre- and post-exposure.
        // TODO If we can use PriorRecordDose, remove this.
        stdr::stable_sort(images, [](const Image& lhs, const Image& rhs) { return lhs.time < rhs.time; });
        f64 accumulated_exposure{};
        for (auto& image : images) {
            image.exposure[0] = accumulated_exposure;
            image.exposure[1] += image.exposure[0];
            accumulated_exposure = image.exposure[1];
        }

        // Compute the stack file index.
        // TODO Deal with cases where the same tilt is collected twice.
        stdr::stable_sort(images, [](const Image& lhs, const Image& rhs) { return lhs.angles[1] < rhs.angles[1]; });
        for (i32 i{}; auto& image : images) {
            image.index = i;
            image.index_file = i++;
        }

        return metadata;
    }

    auto Metadata::load_from_star(const Path& filename) -> Metadata {
        auto parse_key_value = []<typename T>(std::string_view line, std::string_view name, T& value, bool& has_field) {
            if (line.starts_with(name)) {
                line.remove_prefix(name.size());
                line = noa::string::trim_left(line);
                auto result = noa::string::parse<T>(line);
                check(result.has_value(),
                      "Failed to get the value of {}. {} could not be parsed to type {}",
                      name, line, noa::string::stringify<T>());
                check(not has_field, "{} is specified more than once", name);
                value = result.value();
                has_field = true;
                return true;
            }
            return false;
        };

        auto parse_value = []<typename T>(std::string_view str, T& value) {
            auto result = noa::string::parse<T>(str);
            check(result.has_value());
            value = result.value();
        };

        auto file = noa::io::InputTextFile(filename, {.read = true});
        std::string buffer;
        std::string_view line;
        bool exit_block{};
        i64 line_number{};
        auto next_line = [&] {
            for (;;) {
                if (exit_block) {
                    exit_block = false;
                    return true;
                }
                auto success = file.next_line(buffer);
                if (file.eof()) {
                    return false;
                } else if (not success) {
                    panic("{}. Failed to read a line", filename);
                } else {
                    line_number += 1;
                    line = noa::string::trim(buffer);
                    if (line.starts_with("#") or line.empty())
                        continue;
                    return true;
                }
            }
        };

        // Shifts are in pix/A. In other words, they are rescaled to a sampling rate of 1 A/pix.
        auto metadata = Metadata{.spacing = {1, 1}};
        i32 version{};
        bool has_version{};

        bool has_sample{}, has_stack{};
        while (next_line()) {
            if (parse_key_value(line, "_qnVersion", version, has_version)) {
                check(version <= 1, "Unsupported version. Should be <= 1, but got {}", version);
                continue;
            }

            if (line.starts_with("data_sample")) {
                exit_block = false;

                bool has_voltage{}, has_amplitude{}, has_cs{}, has_thickness{};
                while (next_line()) {
                    if (parse_key_value(line, "_qnVoltage", metadata.sample.voltage, has_voltage) or
                        parse_key_value(line, "_qnAmplitude", metadata.sample.amplitude, has_amplitude) or
                        parse_key_value(line, "_qnCs", metadata.sample.cs, has_cs) or
                        parse_key_value(line, "_qnThickness", metadata.sample.thickness, has_thickness))
                        continue;

                    if (line.starts_with("data_")) {
                        exit_block = true;
                        break;
                    }

                    panic("invalid entry within data_sample at line {}: {}", line_number, line);
                }

                check(has_voltage, "Missing \"_qnVoltage\" from data_sample block");
                check(has_amplitude, "Missing \"_qnAmplitude\" from data_sample block");
                check(has_cs, "Missing \"_qnCs\" from data_sample block");
                check(has_thickness, "Missing \"_qnThickness\" from data_sample block");
                has_sample = true;
            }

            if (line.starts_with("data_stack")) {
                exit_block = false;

                // Read until the loop starts.
                while (next_line()) {
                    if (line.starts_with("loop_"))
                        break;
                    panic("data_images is a loop block, but got an entry at line {} before the loop_ key: \"{}\"",
                          line_number, line);
                }

                // Parse the column names and set their column index.
                constexpr auto FIELDS = std::array{
                    "_qnIndex", "_qnRotation", "_qnTilt", "_qnPitch", "_qnShiftX", "_qnShiftY",
                    "_qnDefocus", "_qnAstigmatismValue", "_qnAstigmatismAngle", "_qnPhaseShift",
                    "_qnPreExposure", "_qnPostExposure", "_qnTimepoint", "_qnFrames",
                };
                auto column_indices = std::array<size_t, FIELDS.size()>{};
                for (auto& e: column_indices)
                    e = std::numeric_limits<size_t>::max();

                size_t current_index{};
                while (next_line()) {
                    // Trim comments and whitespace so we can check for an exact match.
                    auto offset = line.find_first_of('#');
                    if (offset != std::string_view::npos) {
                        line = line.substr(0, offset);
                        line = noa::string::trim_right(line);
                    }

                    bool is_field{};
                    for (size_t i{}; i < std::size(FIELDS); ++i) {
                        if (line == FIELDS[i]) {
                            column_indices[i] = current_index++;
                            is_field = true;
                            break;
                        }
                    }
                    if (is_field)
                        continue;

                    // Check that every column name was specified.
                    for (size_t i{}; i < column_indices.size(); ++i) {
                        check(column_indices[i] != std::numeric_limits<size_t>::max(),
                              "Missing column \"{}\" from data_stack loop block",
                              FIELDS[i]);
                    }

                    exit_block = true;
                    break;
                }

                // Parse line values.
                std::string frame_path{};
                while (next_line()) {
                    if (line.starts_with("data_")) {
                        exit_block = true;
                        break;
                    }

                    Image image{};
                    size_t index{};
                    flux::ref(line)
                        .split([](char c) { return c == ' ' or c == '\t'; })
                        .filter([](flux::sequence auto&& r) { return not r.is_empty(); })
                        .map([](flux::sequence auto&& r) { return flux::to<std::string_view>(r); })
                        .for_each([&](std::string_view str) {
                            if      (index == column_indices[0])  parse_value(str, image.index_file);
                            else if (index == column_indices[1])  parse_value(str, image.angles[0]);
                            else if (index == column_indices[2])  parse_value(str, image.angles[1]);
                            else if (index == column_indices[3])  parse_value(str, image.angles[2]);
                            else if (index == column_indices[4])  parse_value(str, image.shifts[1]);
                            else if (index == column_indices[5])  parse_value(str, image.shifts[0]);
                            else if (index == column_indices[6])  parse_value(str, image.defocus.value);
                            else if (index == column_indices[7])  parse_value(str, image.defocus.astigmatism);
                            else if (index == column_indices[8])  parse_value(str, image.defocus.angle);
                            else if (index == column_indices[9])  parse_value(str, image.phase_shift);
                            else if (index == column_indices[10]) parse_value(str, image.exposure[0]);
                            else if (index == column_indices[11]) parse_value(str, image.exposure[1]);
                            else if (index == column_indices[12]) parse_value(str, image.time);
                            else if (index == column_indices[13]) parse_value(str, frame_path);
                            index++;
                        });
                    check(index == FIELDS.size(),
                          "Missing value in data_stack at line {}. {} values are expected per line, but got {}",
                          line_number, FIELDS.size(), index);

                    image.index = image.index_file;
                    image.frames = frame_path == "<NA>" ? nullptr : set_frame_path(image.time, frame_path);
                    metadata.stack.images.push_back(image);
                    has_stack = true; // one image is technically enough
                }
            }
        }

        check(has_sample, "Missing \"data_sample\" block");
        check(has_stack, "Missing \"data_stack\" block");
        return metadata;
    }

    auto Metadata::load_from_settings(const Settings& settings) -> Metadata {
        Metadata metadata;
        bool is_initialized{};

        // Load the mdoc.
        if (not settings.files.mdoc_file.empty()) {
            Logger::info("Loading metadata from mdoc file {}.", settings.files.mdoc_file);
            metadata = load_from_mdoc(settings.files.mdoc_file);
            is_initialized = true;
        }

        // Overwrite with the star file.
        if (not settings.files.star_file.empty()) {
            Logger::info("Loading metadata from star file {}.", settings.files.star_file);
            metadata = load_from_star(settings.files.star_file);
            is_initialized = true;
        }

        // Overwrite with the user settings.
        apply_settings_and_defaults(settings, metadata, is_initialized);
        check_metadata(metadata);

        return metadata;
    }

    void Metadata::save_star(const Path& filename) const {
        const auto now = round<stdc::minutes>(stdc::system_clock::now());
        std::string buffer = fmt::format(
            "# Created by quinoa at {:%R} on {:%d/%m/%y}\n\n"
            "_qnVersion {}\n\n"
            "data_sample\n"
            "_qnVoltage   {:>7.2f}  # kV\n"
            "_qnAmplitude {:>7.2f}\n"
            "_qnCs        {:>7.2f}  # mm\n"
            "_qnThickness {:>7.2f}  # nm\n\n"
            "data_stack\n"
            "loop_\n"
            "_qnIndex             # index within input stack, from 0\n"
            "_qnRotation          # deg\n"
            "_qnTilt              # deg\n"
            "_qnPitch             # deg\n"
            "_qnShiftX            # pix/A (normalized to 1 A/pix)\n"
            "_qnShiftY            # pix/A (normalized to 1 A/pix)\n"
            "_qnDefocus           # um\n"
            "_qnAstigmatismValue  # um\n"
            "_qnAstigmatismAngle  # deg\n"
            "_qnPhaseShift        # deg\n"
            "_qnPreExposure       # e-/A2\n"
            "_qnPostExposure      # e-/A2\n"
            "_qnTimepoint         # collection time\n"
            "_qnFrames            # filepath of the frames\n",
            now, now, 1, sample.voltage, sample.amplitude, sample.cs, sample.thickness);

        constexpr std::string_view FORMAT =
            "{:>3} {:>8.3f} {:>7.3f} {:>7.3f} {:>9.3f} {:>9.3f} "
            "{:>7.3f} {:>7.3f} {:>7.2f} {:>7.2f} {:>7.2f} {:>8.2f} {:>10} {}\n";
        buffer.reserve(8'000);

        // Save in the same order as in the input file and normalize the shifts.
        // If images are removed from the stack, their index_file would still match the original file.
        auto sorted_stack = stack.clone().sort("index_file");
        sorted_stack.rescale_shifts(spacing, Vec{1., 1.});

        for (const auto& image: sorted_stack) {
            buffer += fmt::format(FORMAT,
                image.index_file, image.angles[0], image.angles[1], image.angles[2], image.shifts[1], image.shifts[0],
                image.defocus.value, image.defocus.astigmatism, noa::rad2deg(image.defocus.angle),
                noa::rad2deg(image.phase_shift), image.exposure[0], image.exposure[1], image.time,
                image.frames ? *image.frames : "<NA>"
            );
        }

        noa::write_text(buffer, filename);
    }

    auto Metadata::Stack::sort(std::string_view key, bool ascending) -> Metadata::Stack& {
        std::string lower_key = noa::string::to_lower(key);
        if (lower_key == "index") {
            stdr::stable_sort(
                images, [ascending](const Image& lhs, const Image& rhs) {
                    return ascending ? lhs.index < rhs.index : lhs.index > rhs.index;
                });
        } else if (lower_key == "index_file") {
            stdr::stable_sort(
                images, [ascending](const Image& lhs, const Image& rhs) {
                    return ascending ? lhs.index_file < rhs.index_file : lhs.index_file > rhs.index_file;
                });
        } else if (lower_key == "tilt") {
            stdr::stable_sort(
                images, [ascending](const Image& lhs, const Image& rhs) {
                    return ascending ? lhs.angles[1] < rhs.angles[1] : lhs.angles[1] > rhs.angles[1];
                });
        } else if (lower_key == "absolute_tilt") {
            stdr::stable_sort(
                images, [ascending](const Image& lhs, const Image& rhs) {
                    return ascending ? std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]) :
                                       std::abs(lhs.angles[1]) > std::abs(rhs.angles[1]);
                });
        } else if (lower_key == "exposure") {
            stdr::stable_sort(
                images, [ascending](const Image& lhs, const Image& rhs) {
                    return ascending ? lhs.exposure[0] < rhs.exposure[0] : lhs.exposure[0] > rhs.exposure[0];
                });
        } else if (lower_key == "time") {
            stdr::stable_sort(
                images, [ascending](const Image& lhs, const Image& rhs) {
                    return ascending ? lhs.time < rhs.time : lhs.time > rhs.time;
                });
        } else {
            panic("Invalid sorting key: {}", lower_key);
        }
        return *this;
    }

    auto Metadata::Stack::find_lowest_tilt_index() const -> i64 {
        const auto iter = stdr::min_element(
            images, [](const auto& lhs, const auto& rhs) {
                return std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]);
            });
        return iter - images.begin();
    }

    auto Metadata::Stack::tilt_range() const -> Vec<f64, 2> {
        const auto [iter_min, iter_max] = stdr::minmax_element(
            images, [](const Image& lhs, const Image& rhs) {
                return lhs.angles[1] < rhs.angles[1];
            });
        return Vec{iter_min->angles[1], iter_max->angles[1]};
    }

    auto Metadata::Stack::time_range() const -> Vec<i64, 2> {
        const auto [iter_min, iter_max] = stdr::minmax_element(
            images, [](const Image& lhs, const Image& rhs) {
                return lhs.time < rhs.time;
            });
        return Vec{
            iter_min->time,
            iter_max->time
        };
    }

    auto Metadata::Stack::defocus_range() const -> Vec<f64, 2> {
        const auto [iter_min, iter_max] = stdr::minmax_element(
            images, [](const Image& lhs, const Image& rhs) {
                return lhs.defocus.value < rhs.defocus.value;
            });
        return Vec{
            iter_min->defocus.value,
            iter_max->defocus.value
        };
    }
}
