#include <noa/Core.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Options.hpp"
#include "quinoa/Logger.hpp"

namespace qn {
    MetadataStack::MetadataStack(const Options& options) {
        if (not options.files.input_csv.empty()) {
            Logger::info("Loading metadata from {}.", options.files.input_csv);
            *this = load_csv(options.files.input_csv);
            return;
        }

        // Load tilt and exposure.
        if (not options.files.input_tilt.empty()) {
            Logger::info("Loading tilt offset {}.", options.files.input_tilt);
            generate_(options.files.input_tilt, options.files.input_exposure);

        } else if (options.experiment.collection_order.has_value()) {
            Logger::info("Loading tilt and exposure from tilt-scheme.");
            const auto shape = noa::io::ImageFile(options.files.input_stack, {.read = true}).shape();
            const auto count = static_cast<i32>(shape[0] == 1 and shape[1] > 1 ? shape[1] : shape[0]);

            const auto order = options.experiment.collection_order.value();
            generate_(order.starting_angle,
                      order.starting_direction,
                      order.tilt_increment,
                      order.group_of,
                      order.exclude_start,
                      order.per_view_exposure,
                      count);
        } else {
            panic("Missing option(s). Could not find enough information regarding the tilt geometry");
        }

        // Load angle offsets.
        constexpr auto MAX = std::numeric_limits<f64>::max();
        const auto angle_offsets = Vec{
            noa::allclose(MAX, options.experiment.rotation_offset) ? 0. : options.experiment.rotation_offset,
            noa::allclose(MAX, options.experiment.tilt_offset) ? 0. : options.experiment.tilt_offset,
            noa::allclose(MAX, options.experiment.pitch_offset) ? 0. : options.experiment.pitch_offset,
        };
        add_global_angles(angle_offsets);
    }

    auto MetadataStack::sort(std::string_view key, bool ascending) -> MetadataStack& {
        std::string lower_key = noa::string::to_lower(key);
        if (lower_key == "index")
            sort_on_indexes_(ascending);
        else if (lower_key == "index_file")
            sort_on_file_indexes_(ascending);
        else if (lower_key == "tilt")
            sort_on_tilt_(ascending);
        else if (lower_key == "absolute_tilt")
            sort_on_absolute_tilt_(ascending);
        else if (lower_key == "exposure")
            sort_on_exposure_(ascending);
        else {
            panic(R"(The key should be "index", "tilt",  "absolute_tilt" or "exposure", but got "{}")", lower_key);
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

    auto MetadataStack::minmax_tilts() const -> std::pair<f64, f64> {
        const auto [iter_min, iter_max] = stdr::minmax_element(
            m_slices, [](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                return lhs.angles[1] < rhs.angles[1];
            });
        return std::pair{iter_min->angles[1], iter_max->angles[1]};
    }

    auto MetadataStack::load_csv(const Path& filename) -> MetadataStack {
        auto csv_file = noa::io::InputTextFile(filename, {.read = true});
        std::string line;

        // Check the header.
        {
            using namespace std::string_view_literals;
            constexpr std::array HEADER = {
                "index"sv, "spacing_x"sv, "spacing_y"sv, "size_x"sv, "size_y"sv, "center_x"sv, "center_y"sv,
                "rotation"sv, "tilt"sv, "pitch"sv, "shift_x"sv, "shift_y"sv,
                "d_value"sv, "d_astig"sv, "d_angle"sv, "phase_shift"sv, "pre_exposure"sv, "post_exposure"sv,
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

            std::array<f64, 18> row{};
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
                .shifts = {row[11], row[10]}, // FIXME correct for spacing and center!
                .exposure = {row[16], row[17]},
                .phase_shift = row[15],
                .defocus = {row[12], row[13], row[14]},
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
            "d_value, d_astig, d_angle, phase_shift, pre_exposure, post_exposure\n";
        constexpr std::string_view FORMAT =
            "{:>5}, {:>9.3f}, {:>9.3f}, {:>6}, {:>6}, {:>8.2f}, {:>8.2f}, "
            "{:>8.3f}, {:>7.3f}, {:>7.3f}, {:>9.3f}, {:>9.3f}, "
            "{:>7.3f}, {:>7.3f}, {:>7.3f}, {:>11.3f}, {:>12.2f}, {:>13.2f}\n";

        std::string str;
        str.reserve(HEADER.size() * (sorted.size() + 1));

        str += HEADER;
        for (const auto& slice: slices()) {
            str += fmt::format(
                FORMAT,
                slice.index_file, spacing[1], spacing[0], shape[1], shape[0], center[1], center[0],
                slice.angles[0], slice.angles[1], slice.angles[2], slice.shifts[1], slice.shifts[0],
                slice.defocus.value, slice.defocus.astigmatism, slice.defocus.angle, slice.phase_shift,
                slice.exposure[0], slice.exposure[1]);
        }
        noa::write_text(str, filename);
    }
}

namespace qn {
    void MetadataStack::generate_(const Path& tilt_filename, const Path& exposure_filename) {
        auto splitter = [&](const std::string& str) {
            return str
                   | stdv::split('\n')
                   | stdv::transform([](auto s) { return std::string_view(s.data(), s.size()); })
                   | stdv::filter([](std::string_view s) { return not s.empty(); });
        };
        auto parse_f64 = [](std::string_view str) {
            auto result = noa::string::parse<f64>(str);
            check(result.has_value(), "Cannot parse {} to f64", str);
            return result.value();
        };

        // Tilt file.
        std::vector<f64> tilt_values{};
        try {
            std::string tilt_file = noa::read_text(tilt_filename);
            for (std::string_view line: splitter(tilt_file)) {
                tilt_values.push_back(parse_f64(line));
            }
        } catch (...) {
            panic("Failed to parse tilt file {}", tilt_filename);
        }

        // Exposure file.
        std::vector<Vec<f64, 2>> exposure_values{};
        if (not fs::is_regular_file(exposure_filename)) {
            exposure_values.resize(tilt_values.size(), Vec<f64, 2>{});
        } else {
            try {
                std::string exposure_file = noa::read_text(exposure_filename);
                for (std::string_view line: splitter(exposure_file)) {
                    auto view = line
                                | stdv::split(',')
                                | stdv::transform([&](auto s) {
                                    return parse_f64(std::string_view(s.data(), s.size()));
                                });
                    Vec<f64, 2> exposure{};
                    for (i32 i{}; const auto& e: view) {
                        check(i < 2, "Cannot parse exposure value {} to (f64, f64)", view);
                        exposure[i++] = e;
                    }
                    exposure_values.push_back(exposure);
                }
            } catch (...) {
                panic("Failed to parse exposure file {}", tilt_filename);
            }
            check(tilt_values.size() == exposure_values.size(),
             "The tilt ({}) and exposure ({}) files do not have the same number of slices",
             tilt_values.size(), exposure_values.size());
        }

        // Create the slices.
        for (i32 i{}; auto&& [tilt, exposure]: noa::zip(tilt_values, exposure_values)) {
            m_slices.push_back({
                .index = i,
                .index_file = i,
                .angles = {0, tilt, 0},
                .exposure = exposure,
            });
            ++i;
        }
    }

    void MetadataStack::generate_(
        f64 starting_angle,
        i64 starting_direction,
        f64 tilt_increment,
        i64 group_of,
        bool exclude_start,
        f64 per_view_exposure,
        i32 n_slices
    ) {
        m_slices.clear();
        m_slices.reserve(static_cast<size_t>(n_slices));

        auto direction = static_cast<f64>(starting_direction);
        f64 angle0 = starting_angle;
        f64 angle1 = angle0;

        m_slices.push_back(MetadataSlice{
            .angles = {0, angle0, 0},
            .exposure = {0, per_view_exposure}
        });
        i64 group_count = not exclude_start;
        f64 pre_exposure = per_view_exposure;
        f64 post_exposure = pre_exposure + per_view_exposure;

        for (i64 i = 1; i < n_slices; ++i) {
            angle0 += direction * tilt_increment;
            m_slices.push_back(MetadataSlice{
                .angles = {0, angle0, 0},
                .exposure = {pre_exposure, post_exposure}
            });

            if (group_count == group_of - 1) {
                direction *= -1;
                group_count = 0;
                std::swap(angle0, angle1);
            } else {
                ++group_count;
            }
            pre_exposure += per_view_exposure;
            post_exposure += per_view_exposure;
        }

        // Assume slices are saved in the ascending tilt order.
        stdr::sort(m_slices, [](const auto& lhs, const auto& rhs) { return lhs.angles[1] < rhs.angles[1]; });
        for (size_t i = 0; i < m_slices.size(); ++i) {
            m_slices[i].index = static_cast<i32>(i);
            m_slices[i].index_file = static_cast<i32>(i);
        }
    }

    void MetadataStack::sort_on_indexes_(bool ascending) {
        stdr::stable_sort(m_slices,
                         [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                             return ascending ?
                                    lhs.index < rhs.index :
                                    lhs.index > rhs.index;
                         });
    }

    void MetadataStack::sort_on_file_indexes_(bool ascending) {
        stdr::stable_sort(m_slices,
                         [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                             return ascending ?
                                    lhs.index_file < rhs.index_file :
                                    lhs.index_file > rhs.index_file;
                         });
    }

    void MetadataStack::sort_on_tilt_(bool ascending) {
        stdr::stable_sort(m_slices,
                         [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                             return ascending ?
                                    lhs.angles[1] < rhs.angles[1] :
                                    lhs.angles[1] > rhs.angles[1];
                         });
    }

    void MetadataStack::sort_on_absolute_tilt_(bool ascending) {
        stdr::stable_sort(m_slices,
                         [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                             return ascending ?
                                    std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]) :
                                    std::abs(lhs.angles[1]) > std::abs(rhs.angles[1]);
                         });
    }

    void MetadataStack::sort_on_exposure_(bool ascending) {
        stdr::stable_sort(m_slices,
                         [ascending](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                             return ascending ?
                                    lhs.exposure[0] < rhs.exposure[0] :
                                    lhs.exposure[0] > rhs.exposure[0];
                         });
    }
}
