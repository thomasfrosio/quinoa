#include <noa/Core.hpp>
#include <noa/io/ImageFile.hpp>
#include <noa/io/TextFile.hpp>
#include <forward_list>

#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Logger.hpp"

namespace {
    using namespace qn;

    // Store the frame path to prevent unnecessary copies when copying Metadata::Image around.
    // Use the datetime to keep track of the frame.
    auto frame_paths = std::forward_list<Pair<i64, Path>>{};

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

    void apply_settings(const Settings& settings, const Series& series, Metadata& metadata) {
        const auto has_setting = [&](f64 value) {
            constexpr auto MAX = std::numeric_limits<f64>::max();
            return not noa::allclose(MAX, value);
        };

        // By this point, the metadata is entirely initialized with the MDOC or the STAR file.
        // This function is here to apply the user settings provided by the user via the CLI or the TOML file.
        // These settings have the priority over the MDOC and STAR files.

        // Sample.
        if (has_setting(settings.experiment.voltage))
            metadata.sample.voltage = settings.experiment.voltage;
        if (has_setting(settings.experiment.cs))
            metadata.sample.cs = settings.experiment.cs;
        if (has_setting(settings.experiment.amplitude))
            metadata.sample.amplitude = settings.experiment.amplitude;
        if (has_setting(settings.experiment.thickness))
            metadata.sample.thickness = settings.experiment.thickness;

        // Stack.
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
            if (has_setting(settings.experiment.phase_shift))
                e.phase_shift = settings.experiment.phase_shift;
        }

        // Overwrite the frame directory and reset the metadata.
        if (not series.frames_directory.empty()) {
            for (auto& [time, path]: frame_paths)
                path = series.frames_directory / path.filename();

            for (auto& image: metadata.stack)
                for (auto& [time, path]: frame_paths)
                    if (image.time == time)
                        image.frames = &path;
        }
    }

    void check_metadata(const Metadata& metadata) {
        // Check that the tilts are within a reasonable range.
        for (const auto& e: metadata.stack) {
            if (std::abs(e.angles[1]) > 75.)
                panic("Tilt angle is greater than -+75deg, this is likely a input error");
        }
        // TODO
    }
}

namespace qn {
    auto Metadata::load_from_mdoc(const Path& mdoc, const Path& stack, const Path& rawtlt) -> Metadata {
        // TODO Use PriorRecordDose.
        auto metadata = Metadata{};
        auto& images = metadata.stack.images;

        // Parse "key = value" to "value".
        std::string_view trimmed;
        auto get_substring = [] (std::string_view substring) {
            return noa::details::trim_left(substring.substr(substring.find_first_of('=') + 1));
        };

        // Add the image.
        auto z_values = std::vector<i32>{};
        auto frame_path = Path{};
        auto rotation = f64{};
        bool is_header{true}, has_voltage{}, has_rotation{};
        bool has_tilt{}, has_exposure{}, has_datetime{};
        auto validate_last_image = [&] {
            // Before switching to the next image, check that we collected the necessary fields.
            check(has_tilt, "Image entry {} in the mdoc is missing the TiltAngle", z_values.back());
            check(has_exposure, "Image entry {} in the mdoc is missing the ExposureDose", z_values.back());
            check(has_datetime, "Image entry {} in the mdoc is missing the DateTime", z_values.back());
            has_tilt = false;
            has_exposure = false;
            has_datetime = false;

            // Register the frame path.
            images.back().angles[0] = rotation;
            images.back().frames = set_frame_path(images.back().time, std::move(frame_path));
        };

        for (auto&& line : noa::read_lines(mdoc)) {
             trimmed = noa::details::trim(line);

            // Header.
            if (is_header) {
                if (trimmed.starts_with("Voltage")) {
                    auto substring = get_substring(trimmed);
                    auto result = noa::details::parse<f64>(substring);
                    check(result, "Could not parse Voltage = {}", substring);
                    metadata.sample.voltage = *result;
                    has_voltage = true;
                }

                auto offset = trimmed.find("TiltAxisAngle =");
                if (offset == std::string::npos)
                    offset = trimmed.find("Tilt axis angle =");
                if (offset != std::string::npos) {
                    auto substring = get_substring(trimmed.substr(offset));
                    auto result = noa::details::parse<f64>(substring);
                    check(result, "Could not parse the tilt axis angle from \"{}\"", trimmed);
                    rotation = result.value();
                    has_rotation = true;
                }
            }

            // Create a new image.
            if (trimmed.starts_with("[ZValue =")) {
                auto result = noa::details::parse<i32>(trimmed.substr(9));
                check(result, "Could not parse ZValue: {}", trimmed);
                for (auto e: z_values)
                    if (e == result.value())
                        panic("ZValue entry is duplicated");
                z_values.push_back(*result);

                if (is_header) {
                    check(has_voltage, "Missing Voltage in the mdoc header");
                    check(has_rotation, "Missing TiltAxisAngle in the mdoc header");
                    metadata.sample.cs = 2.7;
                    metadata.sample.amplitude = 0.07;
                    metadata.sample.thickness = 0.;
                    is_header = false;
                }
                if (not images.empty())
                    validate_last_image();
                images.push_back({.index = -1}); // mark as unset for the stack image assignment
                continue;
            }

            // Parse image fields.
            if (trimmed.starts_with("TiltAngle")) {
                auto substring = get_substring(trimmed);
                auto result = noa::details::parse<f64>(substring);
                check(result, "Could not parse TiltAngle = {}", substring);
                images.back().angles[1] = *result;
                has_tilt = true;

            } else if (trimmed.starts_with("ExposureDose")) {
                auto substring = get_substring(trimmed);
                auto result = noa::details::parse<f64>(substring);
                check(result, "Could not parse ExposureDose = {}", substring);
                images.back().exposure[1] = *result;
                has_exposure = true;

            } else if (trimmed.starts_with("SubFramePath")) {
                // Assume '\' are Windows separators. On POSIX, they are valid filename characters.
                auto substring = std::string(get_substring(trimmed));
                stdr::replace(substring, '\\', '/');

                // Note that we only get the filename. The base path is from the user settings.
                frame_path = Path(std::move(substring)).filename();

            } else if (trimmed.starts_with("DateTime")) {
                auto substring = get_substring(trimmed);
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
        validate_last_image();

        // Check that the mdoc doesn't have duplicated tilts.
        constexpr auto TILT_TOLERANCE = 0.2;
        for (usize i{}; i < images.size(); ++i) {
            for (usize j{}; j < images.size(); ++j) {
                if (i != j and noa::allclose(images[i].angles[1], images[j].angles[1], TILT_TOLERANCE)) {
                    panic("mdoc contains entries with the same tilt: entry:{}:tilt={:.1}, entry:{}:tilt={:.1}, tolerance={}",
                          i, images[i].angles[1], j, images[j].angles[1], TILT_TOLERANCE);
                }
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

        // TODO If stack isn't specified, use the frames.

        // Assign each image in the stack to an MDOC entry.
        const auto stack_shape = ni::ImageFile(stack, {.read = true}).shape();
        const auto n_images = stack_shape[0] == 1 and stack_shape[1] > 1 ? stack_shape[1] : stack_shape[0];

        if (rawtlt.empty()) {
            if (n_images > std::ssize(images))
                panic("The stack has more images ({}) than the number of mdoc entries ({}). "
                      "The mdoc parsing probably failed", n_images, std::ssize(images));
            check(n_images <= std::ssize(images),
                  "The stack has fewer images ({}) than the number of mdoc entries ({}). "
                  "Use a rawtlt file to assign each image to a tilt", n_images, std::ssize(images));

            // No rawtlt, but the number of images matches the mdoc, so we should be able to safely
            // assume that the images where saved in ascending tilt order.
            stdr::stable_sort(images, [](const Image& lhs, const Image& rhs) { return lhs.angles[1] < rhs.angles[1]; });
            for (i32 i{}; auto& image : images) {
                image.index = i;
                image.index_file = i++;
            }
        } else {
            for (i32 i{}; const auto& line: ni::read_lines(rawtlt)) {
                if (noa::details::trim(line).empty())
                    continue;
                auto result = noa::details::parse<f64>(line);
                check(result.has_value(), "Could not parse {} as a tilt angle", line);

                // Find the matching entry in the mdoc based on the tilt.
                // TODO Should the frame filename be used instead?
                bool found{};
                for (auto& image: images) {
                    if (noa::allclose(image.angles[1], result.value(), TILT_TOLERANCE)) {
                        check(image.index == -1,
                              "Tilt {:.1f} from rawtlt file matches a mdoc entry already assigned to a previous tilt",
                              result.value());
                        image.index = i;
                        image.index_file = i++;
                        found = true;
                    }
                }
                check(found, "Tilt {:.1f} from rawtlt did not match any entry in the mdoc", result.value());
            }

            // Remove MDOC entries not present in the rawtlt
            std::erase_if(images, [](const auto& image) { return image.index == -1; });
        }

        return metadata;
    }

    auto Metadata::load_from_star(const Path& filename) -> Metadata {
        auto parse_key_value = []<typename T>(std::string_view line, std::string_view name, T& value, bool& has_field) {
            if (line.starts_with(name)) {
                line.remove_prefix(name.size());
                line = noa::details::trim_left(line);
                auto result = noa::details::parse<T>(line);
                check(result.has_value(),
                      "Failed to get the value of {}. {} could not be parsed to type {}",
                      name, line, noa::details::stringify<T>());
                check(not has_field, "{} is specified more than once", name);
                value = result.value();
                has_field = true;
                return true;
            }
            return false;
        };

        auto parse_value = []<typename T>(std::string_view str, T& value) {
            auto result = noa::details::parse<T>(str);
            check(result.has_value());
            value = result.value();
        };

        auto file = ni::InputTextFile(filename, {.read = true});
        auto buffer = std::string{};
        auto line = std::string_view{};
        auto exit_block = bool{};
        auto line_number = i64{};
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
                    line = noa::details::trim(buffer);
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
                    "_qnDefocus", "_qnDefocusDelta", "_qnDefocusAngle", "_qnPhaseShift",
                    "_qnCTFResolution", "_qnCTFScore",
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
                        line = noa::details::trim_right(line);
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

                    constexpr auto DELIMITERS = std::string_view{" \t"};
                    Image image{};
                    usize index{};
                    usize start = line.find_first_not_of(DELIMITERS);
                    while (start != std::string_view::npos) {
                        // Find the end of the current token
                        usize end = line.find_first_of(DELIMITERS, start);
                        std::string_view token = line.substr(start, end - start);

                        // Map token to the correct struct member based on column index
                        if      (index == column_indices[0])  parse_value(token, image.index_file);
                        else if (index == column_indices[1])  parse_value(token, image.angles[0]);
                        else if (index == column_indices[2])  parse_value(token, image.angles[1]);
                        else if (index == column_indices[3])  parse_value(token, image.angles[2]);
                        else if (index == column_indices[4])  parse_value(token, image.shifts[1]);
                        else if (index == column_indices[5])  parse_value(token, image.shifts[0]);
                        else if (index == column_indices[6])  parse_value(token, image.defocus.value);
                        else if (index == column_indices[7])  parse_value(token, image.defocus.astigmatism);
                        else if (index == column_indices[8])  parse_value(token, image.defocus.angle);
                        else if (index == column_indices[9])  parse_value(token, image.phase_shift);
                        else if (index == column_indices[10]) parse_value(token, image.ctf_resolution);
                        else if (index == column_indices[11]) parse_value(token, image.ctf_score);
                        else if (index == column_indices[12]) parse_value(token, image.exposure[0]);
                        else if (index == column_indices[13]) parse_value(token, image.exposure[1]);
                        else if (index == column_indices[14]) parse_value(token, image.time);
                        else if (index == column_indices[15]) parse_value(token, frame_path);

                        // Move to the next token, skipping any consecutive delimiters (filter equivalent)
                        start = line.find_first_not_of(DELIMITERS, end);
                        index++;
                    }
                    check(index == FIELDS.size(),
                          "Missing value in data_stack at line {}. {} values are expected per line, but got {}",
                          line_number, FIELDS.size(), index);

                    // CTF-related changes.
                    image.defocus.astigmatism /= 2; // saved as (u-v), need (u-v)/2
                    image.defocus.angle = noa::deg2rad(image.defocus.angle);
                    image.phase_shift = noa::deg2rad(image.phase_shift);

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

    auto Metadata::load_from_settings(const Settings& settings, const Series& series) -> Metadata {
        Metadata metadata;

        if (not series.star_file.empty()) {
            if (not settings.compute.dry) {
                Logger::info("Loading metadata from star file {}.", series.star_file);
                Logger::warn("Loading metadata from star file is currently experimental and intended for debugging/testing only");
            }
            metadata = load_from_star(series.star_file);
            apply_settings(settings, series, metadata);

        } else if (not series.mdoc_file.empty()) {
            if (not settings.compute.dry) {
                Logger::info("Initializing the metadata from the mdoc file {}.", series.mdoc_file);
                if (not series.rawtlt_file.empty())
                    Logger::info("Initializing the metadata from the rawtlt file {}.", series.rawtlt_file);
                else
                    Logger::info("Initializing the metadata without rawtlt file; assuming images where saved in ascending tilt order");
            }
            metadata = load_from_mdoc(series.mdoc_file, series.stack_file, series.rawtlt_file);
            apply_settings(settings, series, metadata);

        } else {
            panic("Cannot initialize the metadata. No mdoc or star file have been provided");
        }

        check_metadata(metadata);
        return metadata;
    }

    void Metadata::save_star(const Path& filename) const {
        const auto now = round<stdc::minutes>(stdc::system_clock::now());
        std::string buffer = fmt::format(
            "# Created by quinoa at {:%R} on {:%d/%m/%Y}\n\n"
            "_qnVersion {}\n\n"
            "data_sample\n"
            "_qnVoltage   {:>7.2f}  # kV\n"
            "_qnAmplitude {:>7.2f}\n"
            "_qnCs        {:>7.2f}  # mm\n"
            "_qnThickness {:>7.2f}  # nm\n\n"
            "data_stack\n"
            "loop_\n"
            "_qnIndex             # 0-based index within the input stack\n"
            "_qnRotation          # deg\n"
            "_qnTilt              # deg\n"
            "_qnPitch             # deg\n"
            "_qnShiftX            # pix/A (normalized to 1 A/pix)\n"
            "_qnShiftY            # pix/A (normalized to 1 A/pix)\n"
            "_qnDefocus           # um, (u+v)/2\n"
            "_qnDefocusDelta      # um, (u-v)\n"
            "_qnDefocusAngle      # deg\n"
            "_qnPhaseShift        # deg\n"
            "_qnCTFResolution     # A\n"
            "_qnCTFScore          # ZNCC score\n"
            "_qnPreExposure       # e/A2\n"
            "_qnPostExposure      # e/A2\n"
            "_qnTimepoint         # collection time UID\n"
            "_qnFrames            # filepath of the frames\n",
            now, now, 1, sample.voltage, sample.amplitude, sample.cs, sample.thickness);

        constexpr std::string_view FORMAT =
            "{:>3} {:>8.3f} {:>7.3f} {:>7.3f} {:>9.3f} {:>9.3f} "
            "{:>9.5f} {:>9.5f} {:>7.2f} {:>7.2f} {:>6.2f} {:>5.2f} "
            "{:>7.2f} {:>8.2f} {:>10} {}\n";
        buffer.reserve(8'000);

        // Save in the same order as in the input file and normalize the shifts.
        // If images are removed from the stack, their index_file would still match the original file.
        auto sorted_stack = stack.clone().sort("index_file");
        sorted_stack.rescale_shifts(spacing, Vec{1., 1.});

        for (const auto& image: sorted_stack) {
            buffer += fmt::format(FORMAT,
                image.index_file, image.angles[0], image.angles[1], image.angles[2], image.shifts[1], image.shifts[0],
                image.defocus.value, image.defocus.astigmatism * 2, noa::rad2deg(image.defocus.angle), noa::rad2deg(image.phase_shift),
                image.ctf_resolution, image.ctf_score,
                image.exposure[0], image.exposure[1], image.time, image.frames ? *image.frames : Path{}
            );
        }

        noa::write_text(buffer, filename);
    }

    auto Metadata::Stack::sort(std::string_view key, bool ascending) & -> Stack& {
        std::string lower_key = noa::details::to_lower(key);
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

    auto Metadata::Stack::find_lowest_tilt_index() const -> isize {
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

    auto Metadata::Stack::defocus_range(bool with_astigmatism) const -> Vec<f64, 2> {
        const auto [iter_min, iter_max] = stdr::minmax_element(
            images, [&](const Image& lhs, const Image& rhs) {
                auto lhs_defocus = lhs.defocus.value;
                auto rhs_defocus = rhs.defocus.value;
                if (with_astigmatism) {
                    lhs_defocus += abs(lhs.defocus.astigmatism);
                    rhs_defocus += abs(rhs.defocus.astigmatism);
                }
                return lhs_defocus < rhs_defocus;
            });
        return Vec{
            iter_min->defocus.value,
            iter_max->defocus.value
        };
    }

    auto Metadata::Stack::has_astigmatism_changed(
        const Stack& other,
        f64 maximum_magnitude_difference,
        f64 maximum_angle_difference,
        f64 ignore_angle_below_magnitude
    ) const -> bool {
        for (const auto& [lhs, rhs]: noa::zip(images, other.images)) {
            check(lhs.index_file == rhs.index_file);
            const auto magnitude_difference = std::abs(lhs.defocus.astigmatism - rhs.defocus.astigmatism);
            const auto angle_difference = std::abs(lhs.defocus.angle - rhs.defocus.angle);
            const auto abs_max_magnitude = std::max(std::abs(rhs.defocus.astigmatism), std::abs(rhs.defocus.astigmatism));
            const auto has_significant_magnitude = ignore_angle_below_magnitude < abs_max_magnitude;
            if (magnitude_difference > maximum_magnitude_difference or
                (has_significant_magnitude and angle_difference > maximum_angle_difference))
                return true;
        }
        return false;
    }
}
