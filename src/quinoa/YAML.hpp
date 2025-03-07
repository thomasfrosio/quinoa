#pragma once

#include <istream>
#include <ostream>

#if defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include <yaml-cpp/yaml.h>
#include <noa/Core.hpp>

#if defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#endif

namespace YAML {
    template<noa::traits::vec_shape_or_strides T>
    struct convert<T> {
        static auto encode(const T& rhs) -> Node {
            Node node;
            for (auto e: rhs)
                node.push_back(e);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static auto decode(const Node& node, T& rhs) -> bool {
            constexpr size_t N = T::SIZE;
            if (not node.IsSequence() or node.size() != N)
                return false;
            for (size_t i{}; i < N; ++i)
                rhs[i] = node[i].as<T>();
            return true;
        }
    };

    template<>
    struct convert<std::filesystem::path> {
        static auto encode(const std::filesystem::path& rhs) -> Node {
            return convert<std::string>::encode(rhs.string());
        }

        static auto decode(const Node& node, std::filesystem::path& rhs) -> bool {
            std::string str;
            bool status = convert<std::string>::decode(node, str);
            rhs = str;
            return status;
        }
    };

    inline auto operator<<(std::ostream& os, NodeType::value type) -> std::ostream& {
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

    template<typename T, size_t N>
    auto operator<<(Emitter& out, const noa::Vec<T, N>& vec) -> Emitter& {
        out << Flow;
        out << BeginSeq;
        for (size_t i = 0; i < N; ++i)
            out << vec[i];
        out << EndSeq;
        return out;
    }
}

namespace fmt {
    template<> struct formatter<YAML::NodeType::value> : ostream_formatter {};
}
