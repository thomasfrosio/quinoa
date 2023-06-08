#pragma once

#include <noa/Memory.hpp>
#include <noa/core/Definitions.hpp>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Woverloaded-virtual"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

#include <matplot/matplot.h>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif

#include "quinoa/Types.h"

namespace qn {
    template<typename T>
    void plot(View<T> x, const View<T>& y, const Path& path) {
        NOA_CHECK(noa::indexing::is_contiguous_vector_batched(x) &&
                  noa::indexing::is_contiguous_vector_batched(y),
                  "");

        x = noa::indexing::broadcast(x, y.shape());
        const i64 size = y.shape().pop_front().size();
        const i64 batches = y.shape()[0];

        for (i64 i = 0; i < batches; ++i) {
            noa::Span<T> span_x(x.subregion(i).data(), size);
            noa::Span<T> span_y(y.subregion(i).data(), size);
            matplot::plot(span_x, span_y, "-o");
        }

        if (path.empty())
            matplot::show();
        else
            matplot::save(path.string());
    }

    template<typename T>
    void plot_uniform(const View<T>& y, const Path& path) {
        const auto x = noa::memory::arange<T>(y.shape().pop_front().size());
        plot(x.view(), y, path);
    }
}
