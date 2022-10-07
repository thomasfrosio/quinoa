#pragma once

#include <string>
#include "quinoa/Types.h"

namespace qn {
    /// Options for the program.
    class Options {
    public:
        Options(int argc, char* argv[]);
        explicit Options(const YAML::Node& node) : m_options(node) {}

    public:
        YAML::Node operator[](const std::string& key) const { return m_options[key]; }

    private:
        YAML::Node m_options;
    };
}
