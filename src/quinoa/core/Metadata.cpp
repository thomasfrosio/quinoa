#include <noa/IO.h>

#include "quinoa/core/Metadata.h"
#include "quinoa/io/Options.h"
#include "quinoa/io/Logging.h"
#include "quinoa/Exception.h"

namespace qn {
    std::vector<MetadataSlice> TiltScheme::generate(int32_t count, float rotation_angle) const {
        std::vector<MetadataSlice> slices;
        slices.reserve(static_cast<size_t>(count));

        auto direction = static_cast<float>(starting_direction);
        float angle0 = starting_angle;
        float angle1 = angle0;
        float exposure = per_view_exposure;

        slices.push_back({float3_t{rotation_angle, angle0, 0}, float2_t{}, 0.f, 0, false});
        int group_count = !exclude_start;

        for (int32_t i = 1; i < count; ++i) {
            angle0 += direction * angle_increment;
            slices.push_back({float3_t{rotation_angle, angle0, 0}, float2_t{}, exposure, 0, false});

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
        for (size_t i = 0; i < slices.size(); ++i)
            slices[i].index = static_cast<int32_t>(i);

        return slices;
    }

    MetadataStack::MetadataStack(const Options& options) {
        if (!options["stack_mdoc"].IsNull()) {
            *this = MetadataStack(options["stack_mdoc"].as<path_t>());

        } else if (!options["stack_tlt"].IsNull() &&
                   !options["stack_exposure"].IsNull()) {
            *this = MetadataStack(options["stack_tlt"].as<path_t>(),
                                  options["stack_exposure"].as<path_t>(),
                                  options["rotation_angle"].as<float>(0.f));

        } else if (!options["order_starting_angle"].IsNull() &&
                   !options["order_starting_direction"].IsNull() &&
                   !options["order_angle_increment"].IsNull() &&
                   !options["order_group"].IsNull() &&
                   !options["order_exclude_start"].IsNull()) {
            const auto stack_file = options["stack_file"].as<path_t>();
            const size4_t shape = noa::io::ImageFile(stack_file, noa::io::READ).shape();
            const auto count = static_cast<int32_t>(shape[0] == 1 && shape[1] > 1 ? shape[1] : shape[0]);

            const TiltScheme scheme{
                    options["order_starting_angle"].as<float>(),
                    options["order_starting_direction"].as<int32_t>(),
                    options["order_angle_increment"].as<float>(),
                    options["order_group"].as<int32_t>(),
                    options["order_exclude_start"].as<bool>(),
                    options["order_per_view_exposure"].as<float>(0.f),
            };
            *this = MetadataStack(scheme, count, options["rotation_angle"].as<float>(0.f));

        } else {
            QN_THROW("Missing option(s). Could not find enough information regarding the tilt geometry");
        }

        if (!options["exclude_views_idx"].IsNull()) {
            std::vector<int32_t> exclude_views_idx;
            if (options["exclude_views_idx"].IsSequence())
                exclude_views_idx = options["exclude_views_idx"].as<std::vector<int32_t>>();
            else if (options["exclude_views_idx"].IsScalar())
                exclude_views_idx.emplace_back(options["exclude_views_idx"].as<int32_t>());
            else
                QN_THROW("The value of \"exclude_views_idx\" is not recognized");
            exclude(exclude_views_idx);
        }
    }

    MetadataStack::MetadataStack(const path_t&) {
        QN_THROW("TODO Extracting the tilt-scheme from a .mdoc file is not implemented yet");
    }

    MetadataStack::MetadataStack(const qn::path_t& tlt_filename,
                                 const qn::path_t& exposure_filename,
                                 float rotation_angle) {
        auto is_empty = [](const auto& str) { return str.empty(); };
        std::string file;
        std::vector<std::string> lines;

        file = noa::io::TextFile<std::ifstream>(tlt_filename, noa::io::READ).readAll();
        lines = noa::string::split<std::string>(file, '\n');
        lines.erase(std::remove_if(lines.begin(), lines.end(), is_empty), lines.end());
        std::vector<float> tlt_file = noa::string::parse<float>(lines);

        file = noa::io::TextFile<std::ifstream>(exposure_filename, noa::io::READ).readAll();
        lines = noa::string::split<std::string>(file, '\n');
        lines.erase(std::remove_if(lines.begin(), lines.end(), is_empty), lines.end());
        std::vector<float> exposure_file = noa::string::parse<float>(lines);

        QN_CHECK(tlt_file.size() == exposure_file.size(),
                 "The .tlt ({}) and .order ({}) file do not have the same number of lines",
                 tlt_file.size(), exposure_file.size());

        // Create the slices.
        for (size_t i = 0; i < tlt_file.size(); ++i) {
            m_slices.push_back({float3_t{rotation_angle, tlt_file[i], 0},
                                float2_t{},
                                exposure_file[i],
                                static_cast<int32_t>(i)});
        }
    }

    MetadataStack::MetadataStack(TiltScheme scheme, int32_t count, float rotation_angle) {
        m_slices = scheme.generate(count, rotation_angle);
    }

    MetadataStack& MetadataStack::exclude(const std::vector<int32_t>& indexes_to_exclude) noexcept {
        for (int32_t index_to_exclude: indexes_to_exclude)
            for (auto& slice: m_slices)
                if (slice.index == index_to_exclude)
                    slice.excluded = true;
        return *this;
    }

    MetadataStack& MetadataStack::keep(const std::vector<int32_t>& indexes_to_keep) noexcept {
        for (auto& slice: m_slices)
            for (int32_t index_to_keep: indexes_to_keep)
                if (slice.index != index_to_keep)
                    slice.excluded = true;
        return *this;
    }

    MetadataStack& MetadataStack::squeeze() {
        m_slices.erase(
                std::remove_if(m_slices.begin(), m_slices.end(), [](const auto& slice) { return slice.excluded; }),
                m_slices.end());
        return *this;
    }

    MetadataStack& MetadataStack::sort(std::string_view key, bool ascending) {
        key = noa::string::lower(key);
        if (key == "index")
            sortBasedOnIndexes_(ascending);
        else if (key == "tilt")
            sortBasedOnTilt_(ascending);
        else if (key == "absolute_tilt")
            sortBasedOnAbsoluteTilt_(ascending);
        else if (key == "exposure")
            sortBasedOnExposure_(ascending);
        else
            QN_THROW("The key should be \"index\", \"tilt\",  \"absolute_tilt\" or \"exposure\", but got \"{}\"", key);
        return *this;
    }

    void MetadataStack::sortBasedOnIndexes_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    lhs.index < rhs.index :
                                    lhs.index > rhs.index;
                         });
    }

    void MetadataStack::sortBasedOnTilt_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    lhs.angles[1] < rhs.angles[1] :
                                    lhs.angles[1] > rhs.angles[1];
                         });
    }

    void MetadataStack::sortBasedOnAbsoluteTilt_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]) :
                                    std::abs(lhs.angles[1]) > std::abs(rhs.angles[1]);
                         });
    }

    void MetadataStack::sortBasedOnExposure_(bool ascending) {
        std::stable_sort(m_slices.begin(), m_slices.end(),
                         [ascending](const auto& lhs, const auto& rhs) {
                             return ascending ?
                                    lhs.exposure < rhs.exposure :
                                    lhs.exposure > rhs.exposure;
                         });
    }

    size_t MetadataStack::lowestTilt() const {
        const auto iter = std::min_element(
                m_slices.begin(), m_slices.end(),
                [](const auto& lhs, const auto& rhs) {
                    return std::abs(lhs.angles[1]) < std::abs(rhs.angles[1]);
                });
        return static_cast<size_t>(iter - m_slices.begin());
    }

    static void logUpdate(const MetadataStack& origin, const MetadataStack& current) {
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
            if (current_slice.excluded)
                continue;

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
            const float3_t angle_difference = current_slice.angles - origin_slice->angles;
            const float2_t shift_difference = current_slice.shifts - origin_slice->shifts;
            qn::Logger::info("{}, {}, {}, {}, {}, {}",
                             current_slice.index,
                             current_slice.angles[0], angle_difference[0],
                             current_slice.angles[1], angle_difference[1],
                             current_slice.angles[2], angle_difference[2],
                             current_slice.shifts[0], shift_difference[0],
                             current_slice.shifts[1], shift_difference[1]);
        }
    }
}
