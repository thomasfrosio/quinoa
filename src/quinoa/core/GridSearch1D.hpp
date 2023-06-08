#pragma once

#include "quinoa/Types.h"

namespace qn {
    class GridSearch1D {
    public:
        GridSearch1D(f64 bound, f64 step) : m_bound(bound), m_step(step) {}

        template<typename Function>
        void for_each(Function&& function) const {
            for (size_t i = 0; i < size(); ++i) {
                function(eval_step(i));
            }
        }

        [[nodiscard]] constexpr size_t size() const noexcept {
            f64 current = -m_bound;
            size_t count{0};
            while (true) {
                if (current > m_bound)
                    break;
                current += m_step;
                ++count;
            }
            return count;
        }

        [[nodiscard]] constexpr f64 eval_step(size_t i) const noexcept {
            return -m_bound + m_step * static_cast<f64>(i);
        }

    private:
        f64 m_bound;
        f64 m_step;
    };
}
