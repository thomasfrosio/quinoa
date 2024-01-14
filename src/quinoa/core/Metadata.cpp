#include <noa/IO.hpp>

#include "quinoa/core/Metadata.h"
#include "quinoa/io/Options.h"
#include "quinoa/io/Logging.h"
#include "quinoa/Exception.h"

namespace qn {
    MetadataStack::MetadataStack(const Options& options) {
        if (!options.files.input_csv.empty()) {
            *this = load_csv(options.files.input_csv);

        } else if (!options.files.input_tlt.empty() && !options.files.input_exposure.empty()) {
            generate_(options.files.input_tlt, options.files.input_exposure);

        } else if (options.experiment.collection_order.has_value()) {
            const auto shape = noa::io::ImageFile(options.files.input_stack, noa::io::READ).shape();
            const auto count = static_cast<i32>(shape[0] == 1 && shape[1] > 1 ? shape[1] : shape[0]);

            const auto order = options.experiment.collection_order.value();
            generate_(order.starting_angle,
                      order.starting_direction,
                      order.tilt_increment,
                      order.group_of,
                      order.exclude_start,
                      order.per_view_exposure,
                      count);
        } else {
            QN_THROW("Missing option(s). Could not find enough information regarding the tilt geometry");
        }
    }

    auto MetadataStack::sort(std::string_view key, bool ascending) -> MetadataStack& {
        std::string lower_key = noa::string::lower(key);
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
            QN_THROW("The key should be \"index\", \"tilt\",  \"absolute_tilt\" or \"exposure\", but got \"{}\"",
                     lower_key);
        }
        return *this;
    }

    auto MetadataStack::find_lowest_tilt_index() const -> i64 {
        const auto iter = std::min_element(
                m_slices.begin(), m_slices.end(),
                [](const auto& lhs, const auto& rhs) {
                    return std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]);
                });
        return iter - m_slices.begin();
    }

    auto MetadataStack::minmax_tilts() const -> std::pair<f64, f64> {
        const auto [iter_min, iter_max] = std::minmax_element(
                begin(), end(),
                [](const MetadataSlice& lhs, const MetadataSlice& rhs) {
                    return lhs.angles[1] < rhs.angles[1];
                });
        return std::pair{iter_min->angles[1], iter_max->angles[1]};
    }

    MetadataStack MetadataStack::load_csv(const Path& filename) {
        noa::io::TextFile<std::ifstream> csv_file(filename, noa::io::READ);
        std::string line;

        // Check the header.
        constexpr std::array<std::string_view, 18> HEADER = {
                "index", "spacing_x", "spacing_y", "size_x", "size_y", "center_x", "center_y",
                "rotation", "tilt", "elevation", "shift_x", "shift_y",
                "d_value", "d_astig", "d_angle", "phase_shift", "pre_exposure", "post_exposure"
        };
        csv_file.get_line(line);
        [[maybe_unused]] auto columns = noa::string::split<std::string, 18>(line, ','); // FIXME

        // FIXME correct for spacing and center!
        MetadataStack stack;
        while (csv_file.get_line_or_throw(line)) {
            const auto tokens = noa::string::split<f64, 18>(line, ',');
            MetadataSlice slice{};
            slice.angles = {tokens[7], tokens[8], tokens[9]};
            slice.shifts = {tokens[11], tokens[10]};
            slice.exposure = {tokens[16], tokens[17]};
            slice.defocus = {tokens[12]};
            slice.index = 0;
            slice.index_file = {static_cast<i32>(std::round(tokens[0]))};
            stack.slices().push_back(slice);
        }
        return stack;
    }

    void MetadataStack::save(
            const Path& filename,
            Shape2<i64> shape,
            Vec2<f64> spacing,
            f64 defocus_astigmatism_value,
            f64 defocus_astigmatism_angle,
            f64 phase_shift
    ) const {
        // Save in the same order as in the input file.
        MetadataStack sorted = *this;
        sorted.sort("index_file");

        const auto center = MetadataSlice::center<f64>(shape);

        constexpr std::string_view HEADER =
                "index, spacing_x, spacing_y, size_x, size_y, center_x, center_y, "
                "rotation,    tilt, elevation,   shift_x,   shift_y, "
                "d_value, d_astig, d_angle, phase_shift, pre_exposure, post_exposure\n";
        constexpr std::string_view FORMAT =
                "{:>5}, {:>9.3f}, {:>9.3f}, {:>6}, {:>6}, {:>8.2f}, {:>8.2f}, "
                "{:>8.3f}, {:>7.3f}, {:>9.3f}, {:>9.3f}, {:>9.3f}, "
                "{:>7.3f}, {:>7.3f}, {:>7.3f}, {:>11.3f}, {:>12.2f}, {:>13.2f}\n";

        std::string str;
        str.reserve(HEADER.size() * (sorted.size() + 1));

        str += HEADER;
        for (const auto& slice: slices()) {
            str += fmt::format(
                    FORMAT,
                    slice.index_file, spacing[1], spacing[0], shape[1], shape[0], center[1], center[0],
                    slice.angles[0], slice.angles[1], slice.angles[2], slice.shifts[1], slice.shifts[0],
                    slice.defocus, defocus_astigmatism_value, defocus_astigmatism_angle, phase_shift,
                    slice.exposure[0], slice.exposure[1]);
        }
        noa::io::save_text(str, filename);
    }
}

namespace qn {
    void MetadataStack::generate_(const Path& tlt_filename, const Path& exposure_filename) {
        const auto is_empty = [](const auto& str) { return str.empty(); };
        std::string file;
        std::vector<std::string> lines;

        file = noa::io::read_text(tlt_filename);
        lines = noa::string::split<std::string>(file, '\n'); // TODO add "keep_empty"? Also, default to std::string
        lines.erase(std::remove_if(lines.begin(), lines.end(), is_empty), lines.end());
        const std::vector<f64> tlt_file = noa::string::parse<f64>(lines);

        file = noa::io::read_text(exposure_filename);
        lines = noa::string::split<std::string>(file, '\n');
        lines.erase(std::remove_if(lines.begin(), lines.end(), is_empty), lines.end());
        std::vector<Vec2<f64>> exposure_file;
        for (const auto& line: lines) {
            std::array exposure = noa::string::split<f64, 2>(line, ',');
            exposure_file.emplace_back(exposure[0], exposure[1]);
        }

        QN_CHECK(tlt_file.size() == exposure_file.size(),
                 "The tilt ({}) and exposure ({}) files do not have the same number of lines",
                 tlt_file.size(), exposure_file.size());

        // Create the slices.
        for (size_t i = 0; i < tlt_file.size(); ++i) {
            m_slices.push_back({{0, tlt_file[i], 0},
                                {},
                                {exposure_file[i][0], exposure_file[i][1]},
                                0,
                                static_cast<i32>(i),
                                static_cast<i32>(i)});
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

        m_slices.push_back({{0, angle0, 0}, {}, {0, per_view_exposure}}); // TODO C++20 use emplace_back()
        i64 group_count = !exclude_start;
        f64 pre_exposure = per_view_exposure;
        f64 post_exposure = pre_exposure + per_view_exposure;

        for (i64 i = 1; i < n_slices; ++i) {
            angle0 += direction * tilt_increment;
            m_slices.push_back({{0, angle0, 0}, {}, {pre_exposure, post_exposure}});

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
        std::sort(m_slices.begin(), m_slices.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.angles[1] < rhs.angles[1]; });
        for (size_t i = 0; i < m_slices.size(); ++i) {
            m_slices[i].index = static_cast<i32>(i);
            m_slices[i].index_file = static_cast<i32>(i);
        }
    }

    void MetadataStack::sort_on_indexes_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    lhs.index < rhs.index :
                                    lhs.index > rhs.index;
                         });
    }

    void MetadataStack::sort_on_file_indexes_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    lhs.index_file < rhs.index_file :
                                    lhs.index_file > rhs.index_file;
                         });
    }

    void MetadataStack::sort_on_tilt_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    lhs.angles[1] < rhs.angles[1] :
                                    lhs.angles[1] > rhs.angles[1];
                         });
    }

    void MetadataStack::sort_on_absolute_tilt_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]) :
                                    std::abs(lhs.angles[1]) > std::abs(rhs.angles[1]);
                         });
    }

    void MetadataStack::sort_on_exposure_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    lhs.exposure[0] < rhs.exposure[0] :
                                    lhs.exposure[0] > rhs.exposure[0];
                         });
    }
}
