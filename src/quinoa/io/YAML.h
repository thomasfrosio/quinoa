#pragma once

#include <noa/Types.h>
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
    template<>
    struct convert<noa::size4_t> {
        static Node encode(const noa::size4_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.push_back(rhs[3]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::size4_t& rhs) {
            if (!node.IsSequence() || node.size() != 4)
                return false;
            rhs[0] = node[0].as<size_t>();
            rhs[1] = node[1].as<size_t>();
            rhs[2] = node[2].as<size_t>();
            rhs[3] = node[3].as<size_t>();
            return true;
        }
    };

    template<>
    struct convert<noa::size3_t> {
        static Node encode(const noa::size3_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::size3_t& rhs) {
            if (!node.IsSequence() || node.size() != 3)
                return false;
            rhs[0] = node[0].as<size_t>();
            rhs[1] = node[1].as<size_t>();
            rhs[2] = node[2].as<size_t>();
            return true;
        }
    };

    template<>
    struct convert<noa::size2_t> {
        static Node encode(const noa::size2_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::size2_t& rhs) {
            if (!node.IsSequence() || node.size() != 2)
                return false;
            rhs[0] = node[0].as<size_t>();
            rhs[1] = node[1].as<size_t>();
            return true;
        }
    };

    template<>
    struct convert<noa::int4_t> {
        static Node encode(const noa::int4_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.push_back(rhs[3]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::int4_t& rhs) {
            if (!node.IsSequence() || node.size() != 4)
                return false;
            rhs[0] = node[0].as<int>();
            rhs[1] = node[1].as<int>();
            rhs[2] = node[2].as<int>();
            rhs[3] = node[3].as<int>();
            return true;
        }
    };

    template<>
    struct convert<noa::int3_t> {
        static Node encode(const noa::int3_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::int3_t& rhs) {
            if (!node.IsSequence() || node.size() != 3)
                return false;
            rhs[0] = node[0].as<int>();
            rhs[1] = node[1].as<int>();
            rhs[2] = node[2].as<int>();
            return true;
        }
    };

    template<>
    struct convert<noa::uint4_t> {
        static Node encode(const noa::uint4_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.push_back(rhs[3]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::uint4_t& rhs) {
            if (!node.IsSequence() || node.size() != 4)
                return false;
            rhs[0] = node[0].as<uint>();
            rhs[1] = node[1].as<uint>();
            rhs[2] = node[2].as<uint>();
            rhs[3] = node[3].as<uint>();
            return true;
        }
    };

    template<>
    struct convert<noa::uint3_t> {
        static Node encode(const noa::uint3_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::uint3_t& rhs) {
            if (!node.IsSequence() || node.size() != 3)
                return false;
            rhs[0] = node[0].as<uint>();
            rhs[1] = node[1].as<uint>();
            rhs[2] = node[2].as<uint>();
            return true;
        }
    };


    template<>
    struct convert<noa::float4_t> {
        static Node encode(const noa::float4_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.push_back(rhs[3]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::float4_t& rhs) {
            if (!node.IsSequence() || node.size() != 4)
                return false;
            rhs[0] = node[0].as<float>();
            rhs[1] = node[1].as<float>();
            rhs[2] = node[2].as<float>();
            rhs[3] = node[3].as<float>();
            return true;
        }
    };

    template<>
    struct convert<noa::float3_t> {
        static Node encode(const noa::float3_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::float3_t& rhs) {
            if (!node.IsSequence() || node.size() != 3)
                return false;
            rhs[0] = node[0].as<float>();
            rhs[1] = node[1].as<float>();
            rhs[2] = node[2].as<float>();
            return true;
        }
    };

    template<>
    struct convert<noa::float2_t> {
        static Node encode(const noa::float2_t& rhs) {
            Node node;
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::float2_t& rhs) {
            if (!node.IsSequence() || node.size() != 2)
                return false;
            rhs[0] = node[0].as<float>();
            rhs[1] = node[1].as<float>();
            return true;
        }
    };

    template<>
    struct convert<noa::path_t> {
        static Node encode(const noa::path_t& rhs) {
            return convert<std::string>::encode(rhs.string());
        }

        static bool decode(const Node& node, noa::path_t& rhs) {
            std::string str;
            bool status = convert<std::string>::decode(node, str);
            rhs = str;
            return status;
        }
    };
}
