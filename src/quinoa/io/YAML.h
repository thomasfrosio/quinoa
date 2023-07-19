#pragma once

#include "yaml-cpp/node/type.h"
#include <istream>
#include <noa/Array.hpp>
#include <ostream>

#if defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include <yaml-cpp/yaml.h>

#if defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#endif

namespace YAML {
    template<typename T, size_t N>
    struct convert<noa::Shape<T, N>> {
        static Node encode(const noa::Shape<T, N>& rhs) {
            Node node;
            for (auto e: rhs)
                node.push_back(e);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::Shape<T, N>& rhs) {
            if (!node.IsSequence() || node.size() != N)
                return false;
            for (size_t i = 0; i < N; ++i)
                rhs[i] = node[i].as<T>();
            return true;
        }
    };

    template<typename T, size_t N>
    struct convert<noa::Strides<T, N>> {
        static Node encode(const noa::Strides<T, N>& rhs) {
            Node node;
            for (auto e: rhs)
                node.push_back(e);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::Strides<T, N>& rhs) {
            if (!node.IsSequence() || node.size() != N)
                return false;
            for (size_t i = 0; i < N; ++i)
                rhs[i] = node[i].as<T>();
            return true;
        }
    };

    template<typename T, size_t N>
    struct convert<noa::Vec<T, N>> {
        static Node encode(const noa::Vec<T, N>& rhs) {
            Node node;
            for (auto e: rhs)
                node.push_back(e);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::Vec<T, N>& rhs) {
            if (!node.IsSequence() || node.size() != N)
                return false;
            for (size_t i = 0; i < N; ++i)
                rhs[i] = node[i].as<T>();
            return true;
        }
    };

    template<>
    struct convert<noa::Path> {
        static Node encode(const noa::Path& rhs) {
            return convert<std::string>::encode(rhs.string());
        }

        static bool decode(const Node& node, noa::Path& rhs) {
            std::string str;
            bool status = convert<std::string>::decode(node, str);
            rhs = str;
            return status;
        }
    };

    inline std::ostream& operator<<(std::ostream& os, NodeType::value type) {
        switch (type) {
            case NodeType::Undefined:
                os << "YAML::NodeType::Undefined";
                break;
            case NodeType::Null:
                os << "YAML::NodeType::Null";
                break;
            case NodeType::Scalar:
                os << "YAML::NodeType::Scalar";
                break;
            case NodeType::Sequence:
                os << "YAML::NodeType::Sequence";
                break;
            case NodeType::Map:
                os << "YAML::NodeType::Map";
                break;
        }
        return os;
    }
}

namespace fmt {
    template<> struct formatter<YAML::NodeType::value> : ostream_formatter {};
}
