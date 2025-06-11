#pragma once

#include <cmath>
#include <vector>

#include "quinoa/Types.hpp"
#include <noa/Array.hpp>

namespace qn::guts {
    /// Band matrix solver
    template<typename T>
    class BandMatrix {
    public:
        using value_type = T;

    public:
        BandMatrix(i64 dim, i64 n_u, i64 n_l) {
            m_upper.resize(static_cast<size_t>(n_u + 1));
            m_lower.resize(static_cast<size_t>(n_l + 1));
            for (auto& i: m_upper)
                i.resize(static_cast<size_t>(dim));
            for (auto& i: m_lower)
                i.resize(static_cast<size_t>(dim));
        }

        [[nodiscard]] auto dim() const -> i64 { // matrix dimension
            if (not m_upper.empty())
                return std::ssize(m_upper[0]);
            return 0;
        }

        // defines the new operator (), so that we can access the elements
        // by A(i,j), index going from i=0,...,dim()-1
        auto operator()(i64 i, i64 j) -> value_type& {
            ni::bounds_check(dim(), i);
            ni::bounds_check(dim(), j);

            auto k = j - i; // what band is the entry
            check(-num_lower() <= k and k <= num_upper());

            // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
            if (k >= 0)
                return m_upper[static_cast<size_t>(k)][static_cast<size_t>(i)];
            return m_lower[static_cast<size_t>(-k)][static_cast<size_t>(i)];
        }
        auto operator()(i64 i, i64 j) const -> const value_type& {
            return const_cast<BandMatrix&>(*this)(i, j);
        }

        // we can store an additional diagonal (in m_lower)
        // second diag (used in LU decomposition), saved in m_lower
        auto saved_diag(i64 i) const -> const value_type& {
            ni::bounds_check(dim(), i);
            return m_lower[0][static_cast<size_t>(i)];
        }
        auto saved_diag(i64 i) -> value_type& {
            ni::bounds_check(dim(), i);
            return m_lower[0][static_cast<size_t>(i)];
        }

        [[nodiscard]] auto num_upper() const -> i64 {
            return std::ssize(m_upper) - 1;
        }
        [[nodiscard]] auto num_lower() const -> i64 {
            return std::ssize(m_lower) - 1;
        }

        // LR-Decomposition of a band matrix
        void lu_decompose() {
            auto& self = *this;

            // preconditioning
            // normalize column i so that a_ii=1
            for (i64 i{}; i < dim(); ++i) {
                check(self(i, i) != 0.0);
                saved_diag(i) = 1.0 / self(i, i);
                const i64 j_min = std::max(i64{}, i - num_lower());
                const i64 j_max = std::min(dim() - 1, i + num_upper());
                for (i64 j = j_min; j <= j_max; ++j)
                    self(i, j) *= saved_diag(i);
                self(i, i) = 1.0; // prevents rounding errors
            }

            // Gauss LR-Decomposition
            for (i64 k{}; k < dim(); ++k) {
                const i64 i_max = std::min(dim() - 1, k + num_lower()); // num_lower not a mistake!
                for (i64 i = k + 1; i <= i_max; ++i) {
                    check(self(k, k) != 0.0);
                    const value_type x = -self(i, k) / self(k, k);
                    self(i, k) = -x; // assembly part of L
                    const i64 j_max = std::min(dim() - 1, k + num_upper());
                    for (i64 j = k + 1; j <= j_max; ++j) {
                        // assembly part of R
                        self(i, j) = self(i, j) + x * self(k, j);
                    }
                }
            }
        }

        // solves Ly=b
        void l_solve(
            SpanContiguous<const value_type> b,
            SpanContiguous<value_type> x
        ) const {
            check(dim() == std::ssize(b) and dim() == std::ssize(x));
            for (i64 i{}; i < dim(); ++i) {
                value_type sum{};
                for (i64 j = std::max(i64{}, i - num_lower()); j < i; ++j)
                    sum += (*this)(i, j) * x[j];
                x[i] = b[i] * saved_diag(i) - sum;
            }
        }

        // solves Rx=y
        void r_solve(
            SpanContiguous<const value_type> b,
            SpanContiguous<value_type> x
        ) const {
            check(dim() == std::ssize(b) and dim() == std::ssize(x));
            for (i64 i = dim() - 1; i >= 0; --i) {
                value_type sum{};
                for (i64 j = i + 1; j <= std::min(dim() - 1, i + num_upper()); ++j)
                    sum += (*this)(i, j) * x[j];
                x[i] = (b[i] - sum) / (*this)(i, i);
            }
        }

        void lu_solve(
            SpanContiguous<const value_type> b,
            SpanContiguous<value_type> y,
            SpanContiguous<value_type> x,
            bool is_lu_decomposed = false
        ) {
            if (is_lu_decomposed == false)
                lu_decompose();
            l_solve(b, y);
            r_solve(y, x);
        }

    private:
        std::vector<std::vector<value_type>> m_upper; // upper band
        std::vector<std::vector<value_type>> m_lower; // lower band
    };
}

namespace qn {
    /// (non-uniform) Interpolating spline.
    /// \note Adapted from https://github.com/ttk592/spline
    ///       The implementation originally had a .solve(y) function, to solve for all x that satisfied: spline(x)=y.
    ///       I removed it since I don't plan to use it, and it was 300 lines of code...
    class Spline {
    public:
        using value_type = f64;

        /// Spline types
        enum class Type {
            LINEAR = 10, // linear interpolation
            CSPLINE = 30, // cubic splines (classical C^2)
            CSPLINE_HERMITE = 31 // cubic hermite splines (local, only C^1)
        };

        /// Boundary condition type for the spline end-points
        enum class Boundary {
            FIRST_DERIVATIVE = 1,
            SECOND_DERIVATIVE = 2,
            NOT_A_KNOT = 3
        };
        using enum Type;
        using enum Boundary;

        struct Parameters {
            Type type{CSPLINE};
            bool monotonic{false};
            Boundary left{SECOND_DERIVATIVE};
            Boundary right{SECOND_DERIVATIVE};
            value_type left_value{};
            value_type right_value{};
        };

    public:
        /// Creates an uninitialized spline, with zero curvatures at both ends, i.e. natural splines.
        Spline() = default;

        Spline(
            const SpanContiguous<value_type>& x,
            const SpanContiguous<value_type>& y,
            const Parameters& parameters
        ) {
            fit(x, y, parameters);
        }

    public:
        /// Allocates n elements now.
        /// Future calls to fit(...) or set_points(...) will not need to allocate
        /// if called with vectors of fewer or as many elements.
        void reserve(i64 n) {
            allocate_(n, true);
        }

        /// Creates a spline.
        void fit(
            const SpanContiguous<value_type>& x,
            const SpanContiguous<value_type>& y,
            const Parameters& parameters
        ) {
            set_points(x, y, parameters);
            if (parameters.monotonic)
                make_monotonic();
        }

        /// Set all data points.
        void set_points(
            const SpanContiguous<value_type>& x,
            const SpanContiguous<value_type>& y,
            const Parameters& parameters
        ) {
            // not-a-knot with 3 points has many solutions, so require a minimum of 4.
            const bool is_not_a_know =
                m_left == NOT_A_KNOT or
                m_right == NOT_A_KNOT;
            check(x.size() == y.size() and x.ssize() >= 3 + is_not_a_know,
                  "Invalid number of input points");

            m_type = parameters.type;
            m_left = parameters.left;
            m_right = parameters.right;
            m_left_value = parameters.left_value;
            m_right_value = parameters.right_value;
            m_made_monotonic = false;
            const auto n = std::ssize(x);

            // Check the strict monotonicity of input vector x.
            bool is_monotonic{true};
            for (i64 i{}; i < n - 1; i++) {
                if (x[i] >= x[i + 1]) {
                    is_monotonic = false;
                    break;
                }
            }
            check(is_monotonic);

            // Allocate and save inputs.
            allocate_(n);
            for (i64 i{}; i < n; ++i) {
                m_x[i] = x[i];
                m_y[i] = y[i];
            }

            if (m_type == LINEAR) {
                // linear interpolation
                for (i64 i{}; i < n - 1; i++)
                    m_b[i] = (m_y[i + 1] - m_y[i]) / (m_x[i + 1] - m_x[i]);
                m_b[n - 1] = m_b[n - 2]; // ignore boundary conditions, set slope equal to the last segment

            } else if (m_type == CSPLINE) {
                // Classical cubic splines which are C^2.
                // This requires solving an equation system.

                // Set up the matrix and right-hand side of the equation system for the parameters b
                const i64 n_upper = m_left == NOT_A_KNOT ? 2 : 1;
                const i64 n_lower = m_right == NOT_A_KNOT ? 2 : 1;
                auto A = guts::BandMatrix<value_type>(n, n_upper, n_lower);
                auto rhs = m_b; // use b_i as a temporary buffer
                for (i64 i = 1; i < n - 1; i++) {
                    A(i, i - 1) = 1.0 / 3.0 * (m_x[i] - m_x[i - 1]);
                    A(i, i + 0) = 2.0 / 3.0 * (m_x[i + 1] - m_x[i - 1]);
                    A(i, i + 1) = 1.0 / 3.0 * (m_x[i + 1] - m_x[i]);
                    rhs[i] = (m_y[i + 1] - m_y[i]) / (m_x[i + 1] - m_x[i]) -
                             (m_y[i] - m_y[i - 1]) / (m_x[i] - m_x[i - 1]);
                }

                // Boundary conditions.
                if (m_left == SECOND_DERIVATIVE) {
                    // 2*c[0] = f''
                    A(0, 0) = 2.0;
                    A(0, 1) = 0.0;
                    rhs[0] = m_left_value;
                } else if (m_left == FIRST_DERIVATIVE) {
                    // b[0] = f', needs to be re-expressed in terms of c:
                    // (2c[0]+c[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
                    A(0, 0) = 2.0 * (m_x[1] - m_x[0]);
                    A(0, 1) = 1.0 * (m_x[1] - m_x[0]);
                    rhs[0] = 3.0 * ((m_y[1] - m_y[0]) / (m_x[1] - m_x[0]) - m_left_value);
                } else if (m_left == NOT_A_KNOT) {
                    // f'''(x[1]) exists, i.e. d[0]=d[1], or re-expressed in c:
                    // -h1*c[0] + (h0+h1)*c[1] - h0*c[2] = 0
                    A(0, 0) = -(m_x[2] - m_x[1]);
                    A(0, 1) = m_x[2] - m_x[0];
                    A(0, 2) = -(m_x[1] - m_x[0]);
                    rhs[0] = 0.0;
                }
                if (m_right == SECOND_DERIVATIVE) {
                    // 2*c[n-1] = f''
                    A(n - 1, n - 1) = 2.0;
                    A(n - 1, n - 2) = 0.0;
                    rhs[n - 1] = m_right_value;
                } else if (m_right == FIRST_DERIVATIVE) {
                    // b[n-1] = f', needs to be re-expressed in terms of c:
                    // (c[n-2]+2c[n-1])(x[n-1]-x[n-2])
                    // = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
                    A(n - 1, n - 1) = 2.0 * (m_x[n - 1] - m_x[n - 2]);
                    A(n - 1, n - 2) = 1.0 * (m_x[n - 1] - m_x[n - 2]);
                    rhs[n - 1] = 3.0 * (m_right_value - (m_y[n - 1] - m_y[n - 2]) / (m_x[n - 1] - m_x[n - 2]));
                } else if (m_right == NOT_A_KNOT) {
                    // f'''(x[n-2]) exists, i.e. d[n-3]=d[n-2], or re-expressed in c:
                    // -h_{n-2}*c[n-3] + (h_{n-3}+h_{n-2})*c[n-2] - h_{n-3}*c[n-1] = 0
                    A(n - 1, n - 3) = -(m_x[n - 1] - m_x[n - 2]);
                    A(n - 1, n - 2) = m_x[n - 1] - m_x[n - 3];
                    A(n - 1, n - 1) = -(m_x[n - 2] - m_x[n - 3]);
                    rhs[0] = 0.0; // FIXME shouldn't this be rhs[n-1]?
                }

                // Solve the equation system to get the parameters c.
                A.lu_solve(rhs, m_d, m_c); // use d_i as a temporary buffer

                // Calculate parameters b and d based on c.
                for (i64 i{}; i < n - 1; i++) {
                    m_d[i] = 1.0 / 3.0 * (m_c[i + 1] - m_c[i]) / (m_x[i + 1] - m_x[i]);
                    m_b[i] = (m_y[i + 1] - m_y[i]) / (m_x[i + 1] - m_x[i]) -
                             1.0 / 3.0 * (2.0 * m_c[i] + m_c[i + 1]) * (m_x[i + 1] - m_x[i]);
                }

                // For the right extrapolation coefficients (zero cubic term)
                // f_{n-1}(x) = y_{n-1} + b*(x-x_{n-1}) + c*(x-x_{n-1})^2
                const auto h = m_x[n - 1] - m_x[n - 2];
                // m_c[n-1] is determined by the boundary condition
                m_d[n - 1] = 0.0;
                m_b[n - 1] = 3.0 * m_d[n - 2] * h * h + 2.0 * m_c[n - 2] * h + m_b[n - 2]; // = f'_{n-2}(x_{n-1})
                if (m_right == FIRST_DERIVATIVE)
                    m_c[n - 1] = 0.0; // force linear extrapolation

            } else if (m_type == CSPLINE_HERMITE) {
                // Hermite cubic splines which are C^1
                // and derivatives are specified on each grid point
                // (here we use 3-point finite differences)

                // Set b to match 1st order derivative finite difference.
                for (i64 i = 1; i < n - 1; i++) {
                    const auto h = m_x[i + 1] - m_x[i];
                    const auto hl = m_x[i] - m_x[i - 1];
                    m_b[i] = -h / (hl * (hl + h)) * m_y[i - 1] +
                             (h - hl) / (hl * h) * m_y[i] +
                             hl / (h * (hl + h)) * m_y[i + 1];
                }
                // Boundary conditions determine b[0] and b[n-1].
                if (m_left == FIRST_DERIVATIVE) {
                    m_b[0] = m_left_value;
                } else if (m_left == SECOND_DERIVATIVE) {
                    const auto h = m_x[1] - m_x[0];
                    m_b[0] = 0.5 * (-m_b[1] - 0.5 * m_left_value * h + 3.0 * (m_y[1] - m_y[0]) / h);
                } else if (m_left == NOT_A_KNOT) {
                    // f''' continuous at x[1]
                    const auto h0 = m_x[1] - m_x[0];
                    const auto h1 = m_x[2] - m_x[1];
                    m_b[0] = -m_b[1] + 2.0 * (m_y[1] - m_y[0]) / h0 +
                             h0 * h0 / (h1 * h1) * (m_b[1] + m_b[2] - 2.0 * (m_y[2] - m_y[1]) / h1);
                }
                if (m_right == FIRST_DERIVATIVE) {
                    m_b[n - 1] = m_right_value;
                    m_c[n - 1] = 0.0;
                } else if (m_right == SECOND_DERIVATIVE) {
                    const auto h = m_x[n - 1] - m_x[n - 2];
                    m_b[n - 1] = 0.5 * (-m_b[n - 2] + 0.5 * m_right_value * h + 3.0 * (m_y[n - 1] - m_y[n - 2]) / h);
                    m_c[n - 1] = 0.5 * m_right_value;
                } else if (m_right == NOT_A_KNOT) {
                    // f''' continuous at x[n-2]
                    const auto h0 = m_x[n - 2] - m_x[n - 3];
                    const auto h1 = m_x[n - 1] - m_x[n - 2];
                    m_b[n - 1] =
                        -m_b[n - 2] + 2.0 * (m_y[n - 1] - m_y[n - 2]) / h1 + h1 * h1 / (h0 * h0) *
                        (m_b[n - 3] + m_b[n - 2] - 2.0 * (m_y[n - 2] - m_y[n - 3]) / h0);
                    // f'' continuous at x[n-1]: c[n-1] = 3*d[n-2]*h[n-2] + c[n-1]
                    m_c[n - 1] = (m_b[n - 2] + 2.0 * m_b[n - 1]) / h1 - 3.0 * (m_y[n - 1] - m_y[n - 2]) / (h1 * h1);
                }
                m_d[n - 1] = 0.0;

                // Parameters c and d are determined by continuity and differentiability.
                set_coeffs_from_b_();
            }

            // For left extrapolation coefficients.
            m_c0 = m_left == FIRST_DERIVATIVE ? 0.0 : m_c[0];
        }

        /// Adjusts coefficients so that the spline becomes piecewise monotonic where possible.
        /// This is done by adjusting the slope at grid points by a non-negative factor.
        /// This breaks C^2, and if adjustments need to be made at the boundary points,
        /// this can also break boundary conditions.
        ///
        /// \returns false if no adjustments have been made, true otherwise
        bool make_monotonic() {
            bool modified = false;
            const i64 n = m_x.ssize();

            // Check:
            //  input data monotonic increasing --> b_i>=0
            //  input data monotonic decreasing --> b_i<=0
            for (i64 i{}; i < n; i++) {
                const i64 im1 = std::max(i - 1, i64{0});
                const i64 ip1 = std::min(i + 1, n - 1);
                if ((m_y[im1] <= m_y[i] and m_y[i] <= m_y[ip1] and m_b[i] < 0.0) or
                    (m_y[im1] >= m_y[i] and m_y[i] >= m_y[ip1] and m_b[i] > 0.0)) {
                    modified = true;
                    m_b[i] = 0.0;
                }
            }

            // If input data is monotonic (b[i], b[i+1], avg have all the same sign)
            // ensure a sufficient criteria for the monotonicity to be satisfied:
            //      sqrt(b[i]^2+b[i+1]^2) <= 3 |avg|, with avg=(y[i+1]-y[i])/h,
            for (i64 i{}; i < n - 1; i++) {
                const auto h = m_x[i + 1] - m_x[i];
                const auto avg = (m_y[i + 1] - m_y[i]) / h;

                if (avg == 0.0 and (m_b[i] != 0.0 or m_b[i + 1] != 0.0)) {
                    modified = true;
                    m_b[i] = 0.0;
                    m_b[i + 1] = 0.0;
                } else if ((m_b[i] >= 0.0 and m_b[i + 1] >= 0.0 and avg > 0.0) or
                           (m_b[i] <= 0.0 and m_b[i + 1] <= 0.0 and avg < 0.0)) {
                    // input data is monotonic
                    const auto r = std::sqrt(m_b[i] * m_b[i] + m_b[i + 1] * m_b[i + 1]) / std::fabs(avg);
                    if (r > 3.0) {
                        // sufficient criteria for monotonicity: r<=3
                        // adjust b[i] and b[i+1]
                        modified = true;
                        m_b[i] *= 3.0 / r;
                        m_b[i + 1] *= 3.0 / r;
                    }
                }
            }

            if (modified == true) {
                set_coeffs_from_b_();
                m_made_monotonic = true;
            }

            return modified;
        }

        /// Evaluates the spline at point x.
        [[nodiscard]] constexpr auto interpolate_at(value_type x) const -> value_type {
            // polynomial evaluation using Horner's scheme
            // TODO: consider more numerically accurate algorithms, e.g.:
            //   - Clenshaw
            //   - Even-Odd method by A.C.R. Newbery
            //   - Compensated Horner Scheme
            const i64 n = m_x.ssize();
            const i64 idx = find_closest_(x);
            const auto h = x - m_x[idx];

            if (x < m_x[0]) // extrapolation to the left
                return (m_c0 * h + m_b[0]) * h + m_y[0];

            if (x > m_x[n - 1]) // extrapolation to the right
                return (m_c[n - 1] * h + m_b[n - 1]) * h + m_y[n - 1];

            // interpolation
            return ((m_d[idx] * h + m_c[idx]) * h + m_b[idx]) * h + m_y[idx];
        }

        /// Evaluates the spline's derivative at point x.
        [[nodiscard]] constexpr auto derive_at(i32 order, value_type x) const -> value_type {
            const i64 n = m_x.ssize();
            const i64 idx = find_closest_(x);
            const auto h = x - m_x[idx];

            if (x < m_x[0]) { // extrapolation to the left
                switch (order) {
                    case 1: return 2 * m_c0 * h + m_b[0];
                    case 2: return 2 * m_c0;
                    default: return 0;
                }
            }

            if (x > m_x[n - 1]) { // extrapolation to the right
                switch (order) {
                    case 1: return 2 * m_c[n - 1] * h + m_b[n - 1];
                    case 2: return 2 * m_c[n - 1];
                    default: return 0;
                }
            }

            switch (order) { // interpolation
                case 1: return (3 * m_d[idx] * h + 2 * m_c[idx]) * h + m_b[idx];
                case 2: return 6 * m_d[idx] * h + 2 * m_c[idx];
                case 3: return 6 * m_d[idx];
                default: return 0.0;
            }
        }

        [[nodiscard]] constexpr auto x() const -> SpanContiguous<value_type> { return m_x; }
        [[nodiscard]] constexpr auto y() const -> SpanContiguous<value_type> { return m_y; }
        [[nodiscard]] auto is_empty() const -> bool { return m_buffer == nullptr; }

    private:
        void allocate_(i64 n, bool reserve_only = false) {
            // Reuse the buffer if you can.
            const i64 n_to_allocate = n * 5;
            if (m_buffer == nullptr or n_allocated < n_to_allocate) {
                n_allocated = n_to_allocate;
                m_buffer = std::make_unique<value_type[]>(static_cast<size_t>(n_allocated));
            }
            if (reserve_only)
                return;

            std::fill_n(m_buffer.get(), n_to_allocate, 0.);
            m_x = SpanContiguous(m_buffer.get() + n * 0, n);
            m_y = SpanContiguous(m_buffer.get() + n * 1, n);
            m_b = SpanContiguous(m_buffer.get() + n * 2, n);
            m_c = SpanContiguous(m_buffer.get() + n * 3, n);
            m_d = SpanContiguous(m_buffer.get() + n * 4, n);
        }

        // Calculate c_i and d_i from b_i.
        void set_coeffs_from_b_() {
            for (i64 i{}; i < m_x.ssize() - 1; i++) {
                const auto h = m_x[i + 1] - m_x[i];
                // from continuity and differentiability condition
                m_c[i] = (3.0 * (m_y[i + 1] - m_y[i]) / h - (2.0 * m_b[i] + m_b[i + 1])) / h;
                // from differentiability condition
                m_d[i] = ((m_b[i + 1] - m_b[i]) / (3.0 * h) - 2.0 / 3.0 * m_c[i]) / h;
            }

            // for left extrapolation coefficients
            m_c0 = m_left == FIRST_DERIVATIVE ? 0.0 : m_c[0];
        }

        // Closest idx so that m_x[idx] <= x.
        [[nodiscard]] constexpr auto find_closest_(value_type x) const -> i64 {
            // x is sorted, so stop when we passed x.
            for (i64 i{}; i < m_x.ssize(); ++i)
                if (m_x[i] > x)
                    return std::max(i - 1, i64{0});
            return m_x.ssize() - 1;
        }

    private:
        std::unique_ptr<value_type[]> m_buffer;
        i64 n_allocated{};

        // interpolation parameters
        // f(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
        // where a_i = y_i, or else it won't go through grid points
        SpanContiguous<value_type> m_x;
        SpanContiguous<value_type> m_y;
        SpanContiguous<value_type> m_b;
        SpanContiguous<value_type> m_c;
        SpanContiguous<value_type> m_d;

        value_type m_c0{}; // for left extrapolation
        Type m_type{};
        Boundary m_left{};
        Boundary m_right{};
        value_type m_left_value{};
        value_type m_right_value{};
        bool m_made_monotonic{};
    };

    /// Samples the spline from 0 to 1.
    template<nt::real T, typename Op = noa::Copy>
    void sample_spline_1d(
        const Spline& spline,
        const SpanContiguous<T, 1>& output,
        Op&& op = Op{}
    ) {
        const f64 norm = 1 / static_cast<f64>(output.ssize() - 1);
        for (i64 i{}; i < output.ssize(); ++i) {
            const f64 coordinate = static_cast<f64>(i) * norm; // [0,1]
            output[i] = static_cast<T>(op(spline.interpolate_at(coordinate)));
        }
    }

    /// Samples the spline from 0 to 1.
    template<nt::writable_varray_of_real Output, typename Op = noa::Copy>
    void sample_cubic_bspline_1d(
        const Spline& spline,
        const Output& output,
        Op&& op = Op{}
    ) {
        sample_spline_1d(spline, output.span_1d_contiguous(), std::forward<Op>(op));
    }
}
