#pragma once

#include <noa/IO.hpp>
#include <noa/FFT.hpp>

#include "quinoa/Types.hpp"

namespace qn {
    template<typename Real>
    void save_vector_to_text(View<Real> x, const Path& filename) {
        check(noa::indexing::is_contiguous_vector_batched_strided(x));

        // Make sure it is dereferenceable and ready to read.
        Array<std::remove_const_t<Real>> x_cpu;
        if (not x.is_dereferenceable()) {
            x_cpu = x.to_cpu();
            x = x_cpu.view();
        }
        x.eval();

        std::string format;
        for (auto i : noa::irange(x.shape().batch()))
            format += fmt::format("{}\n", fmt::join(x.subregion(i).span_1d(), ","));
        noa::write_text(format, filename);
    }

    template<typename T>
    concept vec_or_real = nt::vec_real<T> or nt::real<T>;

    template<vec_or_real T, vec_or_real U>
    constexpr auto resolution_to_fftfreq(const T& spacing, const U& resolution) {
        return spacing / resolution;
    }

    template<vec_or_real T, vec_or_real U>
    constexpr auto fftfreq_to_resolution(const T& spacing, const U& fftfreq) {
        return spacing / fftfreq;
    }

    inline auto fourier_crop_to_resolution(i64 size, f64 spacing, f64 resolution) {
        const f64 input_size = static_cast<f64>(size);
        const f64 target_spacing = resolution / 2;
        const f64 target_size = input_size * spacing / target_spacing;
        const i64 final_size = std::min(size, nf::next_fast_size(static_cast<i64>(std::round(target_size))));
        const f64 final_fftfreq = (static_cast<f64>(final_size) / input_size) / 2;
        return Pair{static_cast<i64>(final_size), final_fftfreq};
    }
}
