#include "quinoa/Plot.hpp"

namespace qn::details {
    auto has_plot_file_uuid(const Path& path) -> bool {
        bool has_uuid{};
        auto path_expanded = path;
        noa::io::expand_user(path_expanded);
        if (fs::is_regular_file(path_expanded)) {
            std::string buffer;
            auto file = noa::io::InputTextFile(std::move(path_expanded), {.read = true});
            while (file.next_line_or_throw(buffer)) {
                const auto line = std::string_view(buffer);
                const size_t index = line.find('=');
                if (noa::string::trim(line.substr(0, index)) == "uuid") {
                    std::optional result = noa::string::parse<u64>(line.substr(index + 1));
                    check(result.has_value(), "Invalid UUID: {}", line);
                    has_uuid = result.value() == Logger::s_uuid;
                    break;
                }
            }
        }
        return has_uuid;
    }
}

namespace qn {
    void save_plot_shifts(
        const MetadataStack& metadata,
        const Path& path,
        const SavePlotShiftsOptions& options
    ) {
        const bool has_uuid = details::has_plot_file_uuid(path);
        const bool append = options.append and has_uuid;
        auto text_file = noa::io::OutputTextFile(path, noa::io::Open{
            .write = true,
            .append = append,
            .backup = not append,
        });

        if (not append) {
            text_file.write(fmt::format("uuid={}\ntitle={}\nxname=x-shifts (pixels)\nyname=y-shifts (pixels)\n\n",
                Logger::s_uuid, options.title));
        }

        text_file.write("type=scatter-shifts\n");
        text_file.write(fmt::format("label={}\n", options.label));
        text_file.write(fmt::format("indices={}\n", fmt::join(metadata | stdv::transform([](auto& slice){ return slice.index; }), ",")));
        text_file.write(fmt::format("tilts={:.2f}\n", fmt::join(metadata | stdv::transform([](auto& slice){ return slice.angles[1]; }), ",")));
        text_file.write(fmt::format("x={:.5f}\n", fmt::join(metadata | stdv::transform([](auto& slice){ return slice.shifts[0]; }), ",")));
        text_file.write(fmt::format("y={:.5f}\n\n", fmt::join(metadata | stdv::transform([](auto& slice){ return slice.shifts[1]; }), ",")));
        Logger::trace("{} {}", path, append ? "appended" : "saved");
    }
}
