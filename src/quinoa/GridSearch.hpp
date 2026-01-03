#pragma once

#include "quinoa/Types.hpp"

namespace qn {
    template<nt::scalar... T>
    class GridSearch {
    public:
        static constexpr usize SIZE = sizeof...(T);

        template<typename U>
        struct Range {
            using value_type = U;
            U start;
            U end;
            U step;

            static constexpr auto from_vec(const Vec<U, 3>& vec) -> Range {
                return {vec[0], vec[1], vec[2]};
            }
        };

    public:
        constexpr explicit GridSearch(const Range<T>&... ranges) noexcept : m_ranges(noa::make_tuple(ranges...)) {}
        constexpr explicit GridSearch(const Vec<T, 3>&... ranges) noexcept : m_ranges(noa::make_tuple(Range<T>::from_vec(ranges)...)) {}

        template<typename Function>
        constexpr void for_each(Function&& function) const {
            const auto shape = this->shape();
            for (usize i{}; i < shape.n_elements(); ++i) {
                const auto indices = noa::offset2index(i, shape);
                [&]<usize... I>(std::index_sequence<I...>) {
                    function(eval_step<I>(indices[I])...);
                }(std::make_index_sequence<SIZE>{});
            }
        }

        [[nodiscard]] constexpr auto shape() const noexcept -> Shape<usize, SIZE>{
            auto get_size = [this](auto i){
                const auto& range = m_ranges[i];
                auto count = range.end - range.start + range.step;
                return static_cast<usize>(std::round(count / range.step));
            };
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Shape<usize, SIZE>{get_size(Tag<I>{})...};
            }(std::make_index_sequence<SIZE>{});
        }

        [[nodiscard]] constexpr usize size() const noexcept {
            return static_cast<usize>(shape().n_elements());
        }

        template<usize N>
        [[nodiscard]] constexpr auto eval_step(usize i) const noexcept {
            const auto& range = m_ranges[Tag<N>{}];
            using value_t = nt::value_type_t<decltype(range)>;
            return range.start + range.step * static_cast<value_t>(i);
        }

    private:
        noa::Tuple<Range<T>...> m_ranges{};
    };
}
