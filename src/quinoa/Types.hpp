#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <ranges>
#include <string>
#include <type_traits>

#include <noa/Core.hpp>

namespace qn {
    namespace stdr = std::ranges;
    namespace stdv = std::views;
    namespace fs = std::filesystem;
    using Path = std::filesystem::path;

    using namespace noa::types;
    namespace nf = noa::fft;
    namespace ng = noa::geometry;
    namespace ni = noa::indexing;
    namespace ns = noa::signal;
    namespace nt = noa::traits;

    using noa::panic;
    using noa::panic_at_location;
    using noa::check;
    using noa::check_at_location;
}
