#pragma once

#include <noa/Array.hpp>
#include <noa/Signal.hpp>
#include <yaml-cpp/yaml.h>

namespace qn {
    using namespace ::noa;
    using CTFIsotropic64 = noa::signal::fft::CTFIsotropic<f64>;
    using CTFAnisotropic64 = noa::signal::fft::CTFAnisotropic<f64>;
}
