#include <noa/IO.hpp>

#include "quinoa/core/Metadata.h"
#include "quinoa/io/Options.h"
#include "quinoa/io/Logging.h"
#include "quinoa/Exception.h"

namespace qn {
    std::vector<MetadataSlice> TiltScheme::generate(i32 count, f32 rotation_angle) const {
        std::vector<MetadataSlice> slices;
        slices.reserve(static_cast<size_t>(count));

        auto direction = static_cast<f32>(starting_direction);
        f32 angle0 = starting_angle;
        f32 angle1 = angle0;
        f32 exposure = per_view_exposure;

        slices.push_back({{rotation_angle, angle0, 0}, {}, 0.f, 0, false});
        i32 group_count = !exclude_start;

        for (i32 i = 1; i < count; ++i) {
            angle0 += direction * angle_increment;
            slices.push_back({{rotation_angle, angle0, 0}, {}, exposure, 0, false});

            if (group_count == group - 1) {
                direction *= -1;
                group_count = 0;
                std::swap(angle0, angle1);
            } else {
                ++group_count;
            }
            exposure += per_view_exposure;
        }

        // Assume slices are saved in the ascending tilt order.
        std::sort(slices.begin(), slices.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.angles[1] < rhs.angles[1]; });
        for (size_t i = 0; i < slices.size(); ++i) {
            slices[i].index = static_cast<i32>(i);
            slices[i].index_file = static_cast<i32>(i);
        }

        return slices;
    }

    MetadataStack::MetadataStack(const Options& options) {
        if (!options["stack_mdoc"].IsNull()) {
            *this = MetadataStack(options["stack_mdoc"].as<Path>());

        } else if (!options["stack_tlt"].IsNull() &&
                   !options["stack_exposure"].IsNull()) {
            *this = MetadataStack(options["stack_tlt"].as<Path>(),
                                  options["stack_exposure"].as<Path>(),
                                  options["rotation_angle"].as<f32>(MetadataSlice::UNSET_YAW_VALUE));

        } else if (!options["order_starting_angle"].IsNull() &&
                   !options["order_starting_direction"].IsNull() &&
                   !options["order_angle_increment"].IsNull() &&
                   !options["order_group"].IsNull() &&
                   !options["order_exclude_start"].IsNull()) {
            const auto stack_file = options["stack_file"].as<Path>();
            const auto shape = noa::io::ImageFile(stack_file, noa::io::READ).shape();
            const auto count = static_cast<i32>(shape[0] == 1 && shape[1] > 1 ? shape[1] : shape[0]);

            const TiltScheme scheme{
                    options["order_starting_angle"].as<f32>(),
                    options["order_starting_direction"].as<i32>(),
                    options["order_angle_increment"].as<f32>(),
                    options["order_group"].as<i32>(),
                    options["order_exclude_start"].as<bool>(),
                    options["order_per_view_exposure"].as<f32>(MetadataSlice::UNSET_YAW_VALUE),
            };
            *this = MetadataStack(scheme, count, options["rotation_angle"].as<f32>(MetadataSlice::UNSET_YAW_VALUE));

        } else {
            QN_THROW("Missing option(s). Could not find enough information regarding the tilt geometry");
        }
    }

    MetadataStack::MetadataStack(const Path&) {
        QN_THROW("TODO Extracting the tilt-scheme from a .mdoc file is not implemented yet");
    }

    MetadataStack::MetadataStack(const Path& tlt_filename,
                                 const Path& exposure_filename,
                                 f32 rotation_angle) {
        auto is_empty = [](const auto& str) { return str.empty(); };
        std::string file;
        std::vector<std::string> lines;

        using TextFile = noa::io::TextFile<std::ifstream>;
        file = TextFile(tlt_filename, noa::io::READ).read_all();
        lines = noa::string::split<std::string>(file, '\n');
        lines.erase(std::remove_if(lines.begin(), lines.end(), is_empty), lines.end());
        const std::vector<f32> tlt_file = noa::string::parse<f32>(lines);

        file = TextFile(exposure_filename, noa::io::READ).read_all();
        lines = noa::string::split<std::string>(file, '\n');
        lines.erase(std::remove_if(lines.begin(), lines.end(), is_empty), lines.end());
        const std::vector<f32> exposure_file = noa::string::parse<f32>(lines);

        QN_CHECK(tlt_file.size() == exposure_file.size(),
                 "The .tlt ({}) and .order ({}) file do not have the same number of lines",
                 tlt_file.size(), exposure_file.size());

        // Create the slices.
        for (size_t i = 0; i < tlt_file.size(); ++i) {
            m_slices.push_back({{rotation_angle, tlt_file[i], 0},
                                {},
                                exposure_file[i],
                                static_cast<i32>(i),
                                static_cast<i32>(i)});
        }
    }

    MetadataStack::MetadataStack(TiltScheme scheme, i32 count, f32 rotation_angle) {
        m_slices = scheme.generate(count, rotation_angle);
    }

    MetadataStack& MetadataStack::exclude(const std::vector<i32>& indexes_to_exclude) noexcept {
        const auto end_of_new_range = std::remove_if(
                m_slices.begin(), m_slices.end(),
                [&](const MetadataSlice& slice) {
                    const i32 slice_index = slice.index;
                    return std::any_of(indexes_to_exclude.begin(), indexes_to_exclude.end(),
                                       [=](i32 index_to_exclude) { return slice_index == index_to_exclude; });
                });

        m_slices.erase(end_of_new_range, m_slices.end());

        // Sort and reset index.
        sort_on_indexes_();
        i32 count{0};
        for (auto& slice: m_slices)
            slice.index = count++;

        return *this;
    }

    MetadataStack& MetadataStack::sort(std::string_view key, bool ascending) {
        key = noa::string::lower(key);
        if (key == "index")
            sort_on_indexes_(ascending);
        else if (key == "index_file")
            sort_on_file_indexes_(ascending);
        else if (key == "tilt")
            sort_on_tilt_(ascending);
        else if (key == "absolute_tilt")
            sort_on_absolute_tilt_(ascending);
        else if (key == "exposure")
            sort_on_exposure_(ascending);
        else
            QN_THROW("The key should be \"index\", \"tilt\",  \"absolute_tilt\" or \"exposure\", but got \"{}\"", key);
        return *this;
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
                                    lhs.exposure < rhs.exposure :
                                    lhs.exposure > rhs.exposure;
                         });
    }

    i64 MetadataStack::find_lowest_tilt_index() const {
        const auto iter = std::min_element(
                m_slices.begin(), m_slices.end(),
                [](const auto& lhs, const auto& rhs) {
                    return std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]);
                });
        return iter - m_slices.begin();
    }

    void MetadataStack::log_update(const MetadataStack& origin,
                                   const MetadataStack& current) {
        const size_t size = current.size();
        QN_CHECK(origin.size() == size,
                 "The two metadata should have the same number of slices, but got {} and {}",
                 origin.size(), size);
        if (size == 0)
            return;

        // TODO Improve formatting.
        qn::Logger::info("index, yaw, tilt, pitch, y-shift, x-shift");

        for (size_t i = 0; i < size; ++i) {
            const MetadataSlice& current_slice = current[i];
            const MetadataSlice* origin_slice = &origin[i];

            if (current_slice.index != origin_slice->index) {
                // They are not sorted in the same order, so retrieve corresponding slice.
                size_t count{0};
                for (const auto& slice: origin.slices()) {
                    if (slice.index == current_slice.index) {
                        origin_slice = &slice;
                        break;
                    }
                    ++count;
                }
                if (count == size - 1)
                    QN_THROW("Missing slice in the original metadata");
            }

            // Log:
            const auto angle_difference = current_slice.angles - origin_slice->angles;
            const auto shift_difference = current_slice.shifts - origin_slice->shifts;
            qn::Logger::info("{}, {}, ({:+f}), {}, ({:+f}), {}, ({:+f}), {}, ({:+f}), {}, ({:+f})",
                             current_slice.index,
                             current_slice.angles[0], angle_difference[0],
                             current_slice.angles[1], angle_difference[1],
                             current_slice.angles[2], angle_difference[2],
                             current_slice.shifts[0], shift_difference[0],
                             current_slice.shifts[1], shift_difference[1]);
        }
    }
}
