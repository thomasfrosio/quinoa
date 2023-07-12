#pragma once

#include "quinoa/Types.h"

namespace qn {
    template<typename T = f64>
    class GridSearch1D {
    public:
        using value_type = T;

    public:
        GridSearch1D(value_type start, value_type end, value_type step)
                : m_start(start), m_end(end), m_step(step) {}

        template<typename Function>
        constexpr void for_each(Function&& function) const {
            for (size_t i = 0; i < size(); ++i) {
                function(eval_step(i));
            }
        }

        [[nodiscard]] constexpr size_t size() const noexcept {
            value_type current = m_start;
            size_t count{0};
            while (true) {
                if (current > m_end)
                    break;
                current += m_step;
                ++count;
            }
            return count;
        }

        [[nodiscard]] constexpr value_type eval_step(size_t i) const noexcept {
            return m_start + m_step * static_cast<value_type>(i);
        }

    private:
        value_type m_start;
        value_type m_end;
        value_type m_step;
    };

    template<typename T0 = f64, typename T1 = f64>
    class GridSearch2D {
    public:
        using value0_type = T0;
        using value1_type = T1;

    public:
        GridSearch2D(value0_type start_0, value0_type end_0, value0_type step_0,
                     value1_type start_1, value1_type end_1, value1_type step_1)
                : m_start_0(start_0), m_end_0(end_0), m_step_0(step_0),
                  m_start_1(start_1), m_end_1(end_1), m_step_1(step_1) {}

        template<typename Function>
        constexpr void for_each(Function&& function) const {
            for (size_t i = 0; i < size<0>(); ++i) {
                for (size_t j = 0; j < size<1>(); ++j) {
                    function(eval_step<0>(i), eval_step<1>(j));
                }
            }
        }

        template<size_t N>
        [[nodiscard]] constexpr size_t size() const noexcept {
            if constexpr (N == 0)
                return size_(m_start_0, m_step_0, m_end_0);
            else
                return size_(m_start_1, m_step_1, m_end_1);
        }

        template<size_t N>
        [[nodiscard]] constexpr auto eval_step(size_t i) const noexcept {
            if constexpr (N == 0)
                return eval_step_(i, m_start_0, m_step_0);
            else
                return eval_step_(i, m_start_1, m_step_1);
        }

    private:
        template<typename U>
        [[nodiscard]] constexpr size_t size_(U start, U step, U end) const noexcept {
            U current = start;
            size_t count{0};
            while (true) {
                if (current > end)
                    break;
                current += step;
                ++count;
            }
            return count;
        }

        template<typename U>
        [[nodiscard]] constexpr U eval_step_(size_t i, U start, U step) const noexcept {
            return start + step * static_cast<U>(i);
        }

    private:
        value0_type m_start_0;
        value0_type m_end_0;
        value0_type m_step_0;
        value1_type m_start_1;
        value1_type m_end_1;
        value1_type m_step_1;
    };
}
