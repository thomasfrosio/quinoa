#pragma once

#include <noa/Types.h>
#include <noa/Array.h>
#include <noa/Texture.h>

#include <yaml-cpp/yaml.h>

namespace qn {
    namespace fs = std::filesystem;
    using path_t = fs::path;
    using byte_t = std::byte;
}

namespace qn {
    using ::noa::bool2_t;
    using ::noa::bool3_t;
    using ::noa::bool4_t;

    using ::noa::int2_t;
    using ::noa::int3_t;
    using ::noa::int4_t;

    using ::noa::uint2_t;
    using ::noa::uint3_t;
    using ::noa::uint4_t;

    using ::noa::long2_t;
    using ::noa::long3_t;
    using ::noa::long4_t;

    using ::noa::ulong2_t;
    using ::noa::ulong3_t;
    using ::noa::ulong4_t;

    using ::noa::size2_t;
    using ::noa::size3_t;
    using ::noa::size4_t;

    using ::noa::float2_t;
    using ::noa::float3_t;
    using ::noa::float4_t;

    using ::noa::double2_t;
    using ::noa::double3_t;
    using ::noa::double4_t;

    using ::noa::float22_t;
    using ::noa::float23_t;
    using ::noa::float33_t;
    using ::noa::float34_t;
    using ::noa::float44_t;

    using ::noa::double22_t;
    using ::noa::double23_t;
    using ::noa::double33_t;
    using ::noa::double34_t;
    using ::noa::double44_t;

    using ::noa::half_t;
    using ::noa::chalf_t;
    using ::noa::cfloat_t;
    using ::noa::cdouble_t;

    template<typename T>
    using Array = ::noa::Array<T>;

    template<typename T>
    using Texture = ::noa::Texture<T>;

    using Stream = ::noa::Stream;
    using Device = ::noa::Device;
    using Allocator = ::noa::Allocator;
    using ArrayOption = ::noa::ArrayOption;
}
