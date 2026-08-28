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
    namespace stdc = std::chrono;
    namespace stdr = std::ranges;
    namespace stdv = std::views;
    namespace fs = std::filesystem;
    using Path = std::filesystem::path;

    using namespace noa::types;
    namespace nf = noa::fft;
    namespace nx = noa::xform;
    namespace ns = noa::signal;
    namespace nt = noa::traits;
    namespace ni = noa::io;

    using noa::panic;
    using noa::panic_at_location;
    using noa::check;
    using noa::check_at_location;

    using CTFIsotropic64 = ns::CTFIsotropic<f64>;
    using CTFAnisotropic64 = ns::CTFAnisotropic<f64>;

    struct Bandpass {
        f64 highpass_cutoff;
        f64 highpass_width;
        f64 lowpass_cutoff;
        f64 lowpass_width;

        static constexpr auto from_vec(const Vec<f64, 4>& v) {
            return Bandpass{
                .highpass_cutoff = v[0],
                .highpass_width = v[1],
                .lowpass_cutoff = v[2],
                .lowpass_width = v[3]
            };
        }
    };
}
