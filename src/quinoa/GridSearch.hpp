#pragma once

#include "quinoa/Types.hpp"

namespace qn {
    template<nt::scalar... T>
    class GridSearch {
    public:
        static constexpr size_t SIZE = sizeof...(T);

        template<typename U>
        struct Range {
            using value_type = U;
            U start;
            U end;
            U step;
        };

    public:
        constexpr explicit GridSearch(const Range<T>&... ranges) noexcept : m_ranges(noa::make_tuple(ranges...)) {}

        template<typename Function>
        constexpr void for_each(Function&& function) const {
            const auto shape = this->shape();
            for (size_t i{}; i < shape.n_elements(); ++i) {
                const auto indices = ni::offset2index(i, shape);
                [&]<size_t... I>(std::index_sequence<I...>) {
                    function(eval_step<I>(indices[I])...);
                }(std::make_index_sequence<SIZE>{});
            }
        }

        [[nodiscard]] constexpr auto shape() const noexcept -> Shape<size_t, SIZE>{
            auto get_size = [this](auto i){
                const auto& range = m_ranges[i];
                auto count = range.end - range.start + range.step;
                return static_cast<size_t>(count / range.step);
            };
            return [&]<size_t... I>(std::index_sequence<I...>) {
                return Shape<size_t, SIZE>{get_size(Tag<I>{})...};
            }(std::make_index_sequence<SIZE>{});
        }

        [[nodiscard]] constexpr size_t size() const noexcept {
            return static_cast<size_t>(shape().n_elements());
        }

        template<size_t N>
        [[nodiscard]] constexpr auto eval_step(size_t i) const noexcept {
            const auto& range = m_ranges[Tag<N>{}];
            using value_t = nt::value_type_t<decltype(range)>;
            return range.start + range.step * static_cast<value_t>(i);
        }

    private:
        noa::Tuple<Range<T>...> m_ranges{};
    };
}
