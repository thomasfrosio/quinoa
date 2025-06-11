#include <noa/Core.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Logger.hpp"

namespace {
    using namespace qn;

    auto load_mdoc_file(const Path& mdoc_filename) {
        auto file = noa::io::InputTextFile(mdoc_filename, {.read = true});
        std::string line;

        // Header.
        f64 rotation{};
        bool has_rotation{};
        while (file.next_line_or_throw(line)) {
            // Get the rotation offset.
            constexpr std::string_view PATTERN = "TiltAxisAngle";
            auto offset = line.find(PATTERN);
            if (offset != std::string::npos) {
                auto substring = std::string_view(line).substr(offset + PATTERN.size() + 2);
                auto result = noa::string::parse<f64>(substring);
                check(result.has_value(), "Could not parse the TiltAxisAngle value from the mdoc file");
                rotation = *result;
                has_rotation = true;
                break;
            }
        }
        check(has_rotation, "Could not find the TiltAxisAngle in the mdoc file");

        // Images.
        std::vector<MetadataSlice> images;
        std::vector<Pair<i64, Path>> frames;
        bool has_tilt{}, has_exposure{}, has_datetime{};
        while (file.next_line_or_throw(line)) {
            std::string_view trimmed = noa::string::trim(line);

            // Create a new image.
            if (trimmed.starts_with("[ZValue =")) {
                // Before switching to the next image, check that we collected the necessary fields.
                if (not images.empty()) {
                    check(has_tilt and has_exposure and has_datetime,
                          "An image in the mdoc is missing a key value: has_tilt={}, has_exposure={} and has_datetime={}",
                          has_tilt, has_exposure, has_datetime);
                    has_tilt = false;
                    has_exposure = false;
                    has_datetime = false;

                    // Use the datetime to keep track of the frame.
                    frames.back().first = images.back().time;
                }
                images.push_back({.angles = {rotation, 0., 0.}});
                frames.push_back({});
                continue;
            }

            // "key = value" -> "value"
            auto get_substring = [&trimmed] {
                return noa::string::trim_left(trimmed.substr(trimmed.find_first_of('=') + 1));
            };

            if (trimmed.starts_with("TiltAngle")) {
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
                // Assume '\' are Windows separators.
                // On POSIX, they are valid filename characters, but it's unlikely the case here.
                auto substring = std::string(get_substring());
                stdr::replace(substring, '\\', '/');
                frames.back().second = Path(std::move(substring)).filename();

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
        stdr::stable_sort(images, [](const auto& lhs, const auto& rhs) { return lhs.time < rhs.time; });
        f64 accumulated_exposure{};
        for (auto& image : images) {
            image.exposure[0] = accumulated_exposure;
            image.exposure[1] += image.exposure[0];
            accumulated_exposure = image.exposure[1];
        }

        // Compute the stack file index.
        // TODO Deal with cases where the same tilt is collected twice.
        stdr::stable_sort(images, [](const auto& lhs, const auto& rhs) { return lhs.angles[1] < rhs.angles[1]; });
        for (i32 i{}; auto& image : images) {
            image.index = i;
            image.index_file = i++;
        }

        return Pair{std::move(images), std::move(frames)};
    }
}

namespace qn {
    auto MetadataStack::load_from_settings(Settings& options) -> MetadataStack {
        MetadataStack metadata;

        // Load the mdoc.
        Logger::info("Loading metadata from mdoc file {}.", options.files.mdoc_file);
        auto [images, frames] = load_mdoc_file(options.files.mdoc_file);
        metadata.m_slices = std::move(images);
        options.files.frames = std::move(frames);

        // Overwrite the mdoc with the CSV file (a serialized version of the metadata) if it exists.
        // Note: The CSV file is mostly used for debugging, when we want to take a snapshot of the program
        //       and start again. Using the CSV file from a previous run with different experiment settings,
        //       like the voltage, may not give the expected results.
        if (not options.files.csv_file.empty()) {
            Logger::info("Loading metadata from CSV file {}.", options.files.csv_file);
            metadata = load_from_csv(options.files.csv_file);
        }

        // Overwrite with the experiment settings.
        constexpr auto MAX = std::numeric_limits<f64>::max();
        for (auto& e: metadata) {
            if (not noa::allclose(MAX, options.experiment.tilt_axis))
                e.angles[0] = MetadataSlice::to_angle_range(options.experiment.tilt_axis);
            if (not noa::allclose(MAX, options.experiment.specimen_tilt))
                e.angles[1] += options.experiment.specimen_tilt;
            if (not noa::allclose(MAX, options.experiment.specimen_pitch))
                e.angles[2] += options.experiment.specimen_pitch;
            if (not noa::allclose(MAX, options.experiment.phase_shift))
                e.phase_shift = options.experiment.phase_shift;
        }

        // Check that the tilts are within a reasonable range.
        for (auto& e: metadata) {
            if (std::abs(e.angles[1]) > 75.)
                panic("Tilt angle is greater than -+75deg, this is likely a input error");
        }

        return metadata;
    }

    auto MetadataStack::load_from_csv(const Path& filename) -> MetadataStack {
        auto csv_file = noa::io::InputTextFile(filename, {.read = true});
        std::string line;

        // Check the header.
        {
            using namespace std::string_view_literals;
            constexpr std::array HEADER = {
                "index"sv, "spacing_x"sv, "spacing_y"sv, "size_x"sv, "size_y"sv, "center_x"sv, "center_y"sv,
                "rotation"sv, "tilt"sv, "pitch"sv, "shift_x"sv, "shift_y"sv,
                "d_value"sv, "d_astig"sv, "d_angle"sv, "p_shift"sv, "pre_exp"sv, "post_exp"sv, "timepoint"sv,
            };
            csv_file.next_line_or_throw(line);
            size_t count{};
            for (auto header = line | stdv::split(',');
                const auto& [result, expected] : noa::zip(header, HEADER)) {
                auto trimmed = noa::string::trim(std::string_view(result.data(), result.size()));
                check(expected == trimmed, "Header column mismatched: {} |= {}", expected, trimmed);
                ++count;
                }
            check(count == HEADER.size(), "Missing header column, got count={}, expected {}", count, HEADER.size());
        }

        MetadataStack stack;
        while (csv_file.next_line_or_throw(line)) {
            if (line.empty())
                continue;

            std::array<f64, 19> row{};
            size_t count{};
            for (size_t i{}; auto value: line | stdv::split(',')) {
                auto result = noa::string::parse<f64>(std::string_view(value.data(), value.size()));
                check(result.has_value(), "Failed to parse value: {}", result);
                row[i++] = result.value();
                ++count;
            }
            check(count == row.size(), "Missing column, got count={}, expected {}", count, row.size());

            stack.slices().push_back({
                .index = 0,
                .index_file = static_cast<i32>(std::round(row[0])),
                .angles = {row[7], row[8], row[9]},
                .shifts = {row[11], row[10]}, // FIXME correct for different spacing and center!
                .exposure = {row[16], row[17]},
                .phase_shift = row[15],
                .defocus = {row[12], row[13], row[14]},
                .time = static_cast<i64>(std::round(row[18])),
            });
        }
        return stack;
    }

    void MetadataStack::save_csv(
        const Path& filename,
        Shape<i64, 2> shape,
        Vec<f64, 2> spacing
    ) const {
        // Save in the same order as in the input file.
        MetadataStack sorted = *this;
        sorted.sort("index_file");

        const auto center = (shape / 2).vec.as<f64>();
        constexpr std::string_view HEADER =
            "index, spacing_x, spacing_y, size_x, size_y, center_x, center_y, "
            "rotation,    tilt,   pitch,   shift_x,   shift_y, "
            "d_value, d_astig, d_angle, p_shift, pre_exp, post_exp,  timepoint\n";
        constexpr std::string_view FORMAT =
            "{:>5}, {:>9.3f}, {:>9.3f}, {:>6}, {:>6}, {:>8.2f}, {:>8.2f}, "
            "{:>8.3f}, {:>7.3f}, {:>7.3f}, {:>9.3f}, {:>9.3f}, "
            "{:>7.3f}, {:>7.3f}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>8.2f}, {:>10}\n";

        std::string str;
        str.reserve(HEADER.size() * (sorted.size() + 1));

        str += HEADER;
        for (const auto& slice: slices()) {
            str += fmt::format(
                FORMAT,
                slice.index_file, spacing[1], spacing[0], shape[1], shape[0], center[1], center[0],
                slice.angles[0], slice.angles[1], slice.angles[2], slice.shifts[1], slice.shifts[0],
                slice.defocus.value, slice.defocus.astigmatism, noa::rad2deg(slice.defocus.angle),
                noa::rad2deg(slice.phase_shift), slice.exposure[0], slice.exposure[1],
                slice.time);
        }
        noa::write_text(str, filename);
    }

    auto MetadataStack::sort(std::string_view key, bool ascending) -> MetadataStack& {
        std::string lower_key = noa::string::to_lower(key);
        if (lower_key == "index") {
            stdr::stable_sort(
                slices(), [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                    return ascending ? lhs.index < rhs.index : lhs.index > rhs.index;
                });
        } else if (lower_key == "index_file") {
            stdr::stable_sort(
                slices(), [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                    return ascending ? lhs.index_file < rhs.index_file : lhs.index_file > rhs.index_file;
                });
        } else if (lower_key == "tilt") {
            stdr::stable_sort(
                slices(), [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                    return ascending ? lhs.angles[1] < rhs.angles[1] : lhs.angles[1] > rhs.angles[1];
                });
        } else if (lower_key == "absolute_tilt") {
            stdr::stable_sort(
                slices(), [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                    return ascending ? std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]) :
                                       std::abs(lhs.angles[1]) > std::abs(rhs.angles[1]);
                });
        } else if (lower_key == "exposure") {
            stdr::stable_sort(
                slices(), [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                    return ascending ? lhs.exposure[0] < rhs.exposure[0] : lhs.exposure[0] > rhs.exposure[0];
                });
        } else if (lower_key == "time") {
            stdr::stable_sort(
                slices(), [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                    return ascending ? lhs.time < rhs.time : lhs.time > rhs.time;
                });
        } else {
            panic(R"(The key should be "index", "tilt",  "absolute_tilt", "exposure" or "time, but got "{}")", lower_key);
        }
        return *this;
    }

    auto MetadataStack::find_lowest_tilt_index() const -> i64 {
        const auto iter = stdr::min_element(
            m_slices, [](const auto& lhs, const auto& rhs) {
                return std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]);
            });
        return iter - m_slices.begin();
    }

    auto MetadataStack::tilt_range() const -> Vec<f64, 2> {
        const auto [iter_min, iter_max] = stdr::minmax_element(
            m_slices, [](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                return lhs.angles[1] < rhs.angles[1];
            });
        return Vec{iter_min->angles[1], iter_max->angles[1]};
    }
    auto MetadataStack::time_range() const -> Vec<i64, 2> {
        const auto [iter_min, iter_max] = stdr::minmax_element(
            m_slices, [](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                return lhs.time < rhs.time;
            });
        return Vec{
            iter_min->time,
            iter_max->time
        };
    }
}
