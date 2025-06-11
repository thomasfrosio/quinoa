#include "quinoa/Utilities.hpp"

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

}
