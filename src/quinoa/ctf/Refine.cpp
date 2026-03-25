#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Optimizer.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/SplineGrid.hpp"
#include "quinoa/ctf/CTF.hpp"

namespace {
    using namespace ::qn;
    using namespace ::qn::ctf;

    // Loading CTFs on the GPU takes a long time,
    // so pack and load only the actual variables.
    struct alignas(8) CTFIsotropicPacked {
        f32 defocus;
        f32 phase_shift;
    };
    struct alignas(16) CTFAnisotropicPacked {
        f32 defocus;
        f32 astigmatism;
        f32 angle;
        f32 phase_shift;
    };

    template<typename T>
    struct ReduceHeight {
        using value_type = T;

        SpanContiguous<const Patches::value_type, 3> polar{}; // (n*p,h,w)
        SpanContiguous<const CTFAnisotropicPacked, 1> packed{}; // (c*n*p)

        ns::CTFIsotropic<value_type> isotropic_ctf;
        ns::CTFAnisotropic<value_type> anisotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};
        value_type rho_range{};

        NOA_HD void operator()(isize batch, isize row, isize col, value_type& r0, value_type& r1) {
            auto phi = static_cast<value_type>(row) * phi_step + phi_start; // radians
            auto rho = static_cast<value_type>(col) * rho_step + rho_start; // fftfreq

            // Get the target phase.
            const auto& patch = packed[batch];
            isotropic_ctf.set_defocus(patch.defocus);
            isotropic_ctf.set_phase_shift(patch.phase_shift);
            auto phase = isotropic_ctf.phase_at(rho);

            // Get the corresponding fftfreq within the astigmatic field.
            anisotropic_ctf.set_defocus({patch.defocus, patch.astigmatism, patch.angle});
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            auto fftfreq = isotropic_ctf.fftfreq_at(phase);
            if (not noa::is_finite(fftfreq))
                return;

            // Scale back to unnormalized frequency.
            const auto width = polar.shape().width();
            const auto frequency = static_cast<value_type>(width - 1) * (fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = static_cast<value_type>(frequency - floored);
            const auto index = static_cast<isize>(floored);

            value_type v0{}, w0{}, v1{}, w1{};
            if (index >= 0 and index < width) {
                v0 = static_cast<value_type>(polar(batch % polar.shape()[0], row, index));
                w0 = 1;
            }
            if (index + 1 >= 0 and index + 1 < width) {
                v1 = static_cast<value_type>(polar(batch % polar.shape()[0], row, index + 1));
                w1 = 1;
            }
            r0 += v0 * (1 - fraction) + v1 * fraction;
            r1 += w0 * (1 - fraction) + w1 * fraction;
        }

        static constexpr void join(value_type r0, value_type r1, value_type& j0, value_type& j1) {
            j0 += r0;
            j1 += r1;
        }

        using remove_default_post = bool;
        static constexpr void post(value_type j0, value_type j1, value_type& f) {
            f = j1 > 1 ? j0 / j1 : j0;
        }
    };

    template<typename T>
    struct ReduceIsotropicDepth {
        using value_type = T;

        SpanContiguous<const value_type, 3> image_spectra{}; // (cn,p,w)
        SpanContiguous<const CTFIsotropicPacked, 1> defocus_images{}; // (cn)
        SpanContiguous<const value_type, 1> defocus_patches{}; // (cnp)

        ns::CTFIsotropic<value_type> isotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};
        value_type rho_range{};
        isize n_images{};

        NOA_HD void operator()(isize i, isize p, isize c, value_type& r0, value_type& r1) {
            auto rho = static_cast<value_type>(c) * rho_step + rho_start; // fftfreq

            const auto [n_patches, width] = image_spectra.shape().pop_front();
            const auto& image = defocus_images[i];
            const auto& patch_defocus = defocus_patches[i * n_patches + p];

            // Get the target phase.
            isotropic_ctf.set_defocus(image.defocus);
            isotropic_ctf.set_phase_shift(image.phase_shift);
            const auto phase = isotropic_ctf.phase_at(rho);

            // Get the corresponding fftfreq within the patch.
            isotropic_ctf.set_defocus(patch_defocus);
            const auto fftfreq = isotropic_ctf.fftfreq_at(phase);
            if (not noa::is_finite(fftfreq))
                return;

            // Scale back to unnormalized frequency.
            const auto frequency = static_cast<value_type>(width - 1) * (fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = static_cast<value_type>(frequency - floored);
            const auto index = static_cast<isize>(floored);

            value_type v0{}, w0{}, v1{}, w1{};
            if (index >= 0 and index < width) {
                v0 = static_cast<value_type>(image_spectra(i % n_images, p, index));
                w0 = 1;
            }
            if (index + 1 >= 0 and index + 1 < width) {
                v1 = static_cast<value_type>(image_spectra(i % n_images, p, index + 1));
                w1 = 1;
            }
            r0 += v0 * (1 - fraction) + v1 * fraction;
            r1 += w0 * (1 - fraction) + w1 * fraction;
        }

        NOA_HD static void join(value_type r0, value_type r1, value_type& j0, value_type& j1) {
            j0 += r0;
            j1 += r1;
        }

        using remove_default_post = bool;
        NOA_HD static void post(value_type j0, value_type j1, value_type& f) {
            f = j1 > 1 ? j0 / j1 : j0;
        }
    };

    template<typename T>
    struct ReduceAnisotropicDepth {
        using value_type = T;

        SpanContiguous<const Patches::value_type, 4> polar{}; // (n,p,h,w)
        SpanContiguous<const CTFAnisotropicPacked, 1> ctf_images_packed{}; // (c*n)
        SpanContiguous<const value_type, 1> defocus_patches{}; // (c*n*p)

        ns::CTFIsotropic<value_type> isotropic_ctf;
        ns::CTFAnisotropic<value_type> anisotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};
        value_type rho_range{};

        NOA_HD void operator()(isize i, isize p, isize r, isize c, value_type& r0, value_type& r1) {
            auto phi = static_cast<value_type>(r) * phi_step + phi_start; // radians
            auto rho = static_cast<value_type>(c) * rho_step + rho_start; // fftfreq

            const auto& [n_images, n_patches, height, width] = polar.shape();
            const auto& image = ctf_images_packed[i];
            const auto& patch_defocus = defocus_patches[i * n_patches + p];

            // Get the target phase.
            anisotropic_ctf.set_defocus({image.defocus, image.astigmatism, image.angle});
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            isotropic_ctf.set_phase_shift(image.phase_shift);
            const auto phase = isotropic_ctf.phase_at(rho);

            // Get the corresponding fftfreq within the patch.
            anisotropic_ctf.set_defocus({patch_defocus, image.astigmatism, image.angle});
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            const auto fftfreq = isotropic_ctf.fftfreq_at(phase);
            if (not noa::is_finite(fftfreq))
                return;

            // Scale back to unnormalized frequency.
            const auto frequency = static_cast<value_type>(width - 1) * (fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = static_cast<value_type>(frequency - floored);
            const auto index = static_cast<isize>(floored);

            value_type v0{}, w0{}, v1{}, w1{};
            if (index >= 0 and index < width) {
                v0 = static_cast<value_type>(polar(i % n_images, p, r, index));
                w0 = 1;
            }
            if (index + 1 >= 0 and index + 1 < width) {
                v1 = static_cast<value_type>(polar(i % n_images, p, r, index + 1));
                w1 = 1;
            }
            r0 += v0 * (1 - fraction) + v1 * fraction;
            r1 += w0 * (1 - fraction) + w1 * fraction;
        }

        NOA_HD static void join(value_type r0, value_type r1, value_type& j0, value_type& j1) {
            j0 += r0;
            j1 += r1;
        }

        using remove_default_post = bool;
        NOA_HD static void post(value_type j0, value_type j1, value_type& f) {
            f = j1 > 1 ? j0 / j1 : j0;
        }
    };

    template<typename T>
    struct ReduceWidth {
        using value_type = T;

        struct Reduced {
            value_type sum_lhs{};
            value_type sum_rhs{};
            value_type sum_lhs_lhs{};
            value_type sum_rhs_rhs{};
            value_type sum_lhs_rhs{};
        };

        SpanContiguous<const value_type, 4> image_spectra{}; // (c,n,h,w)
        SpanContiguous<const value_type, 2> image_baseline{}; // (n,w)
        SpanContiguous<const value_type, 3> image_thickness_modulation{}; // (c,n,w)
        SpanContiguous<const CTFAnisotropicPacked, 2> image_defoci{}; // (c,n)

        ns::CTFIsotropic<value_type> isotropic_ctf;
        ns::CTFAnisotropic<value_type> anisotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};

        NOA_HD void operator()(isize c, isize n, isize h, isize w, Reduced& reduced) {
            auto phi = static_cast<value_type>(h) * phi_step + phi_start; // radians
            auto rho = static_cast<value_type>(w) * rho_step + rho_start; // fftfreq

            // Get the target phase.
            const auto& image = image_defoci(c, n);
            anisotropic_ctf.set_defocus({image.defocus, image.astigmatism, image.angle});
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            isotropic_ctf.set_phase_shift(image.phase_shift);

            auto& fftfreq = rho;
            auto lhs = isotropic_ctf.value_at(fftfreq);
            lhs *= lhs;
            auto envelope = isotropic_ctf.envelope_at(fftfreq);
            envelope *= envelope;
            lhs -= envelope / 2; // [0,1] -> [-0.5, 0.5]
            lhs *= static_cast<value_type>(image_thickness_modulation(c, n, w));

            // Get the baseline-subtracted (aka zero-centered) spectrum.
            auto rhs = image_spectra(c, n, h, w);
            rhs -= static_cast<value_type>(image_baseline(n, w)); // baseline is already sampled

            reduced.sum_lhs += lhs;
            reduced.sum_rhs += rhs;
            reduced.sum_lhs_lhs += lhs * lhs;
            reduced.sum_rhs_rhs += rhs * rhs;
            reduced.sum_lhs_rhs += lhs * rhs;
        }

        NOA_HD static void join(const Reduced& ireduced, Reduced& reduced) {
            reduced.sum_lhs += ireduced.sum_lhs;
            reduced.sum_rhs += ireduced.sum_rhs;
            reduced.sum_lhs_lhs += ireduced.sum_lhs_lhs;
            reduced.sum_rhs_rhs += ireduced.sum_rhs_rhs;
            reduced.sum_lhs_rhs += ireduced.sum_lhs_rhs;
        }

        using remove_default_post = bool;
        NOA_HD void post(const Reduced& reduced, value_type& zncc) {
            const auto count = static_cast<value_type>(image_baseline.shape().width());
            const auto denominator_lhs = reduced.sum_lhs_lhs - reduced.sum_lhs * reduced.sum_lhs / count;
            const auto denominator_rhs = reduced.sum_rhs_rhs - reduced.sum_rhs * reduced.sum_rhs / count;
            auto denominator = denominator_lhs * denominator_rhs;
            if (denominator <= 0) {
                zncc = 0;
                return;
            }
            const auto numerator = reduced.sum_lhs_rhs - reduced.sum_lhs * reduced.sum_rhs / count;
            zncc = numerator / noa::sqrt(denominator);
        }
    };

    struct SimulateCTF2 {
        SpanContiguous<f32, 3> output;
        SpanContiguous<const CTFAnisotropicPacked, 1> ctfs;
        ns::CTFAnisotropic<f32> ctf;

        f32 phi_start{};
        f32 phi_step{};
        f32 rho_start{};
        f32 rho_step{};

        NOA_HD void operator()(isize i, isize h, isize w) {
            const auto phi = static_cast<f32>(h) * phi_step + phi_start; // radians
            const auto rho = static_cast<f32>(w) * rho_step + rho_start; // fftfreq
            const auto fftfreq = rho * noa::sincos(phi);

            const auto& packed = ctfs[i];
            ctf.set_defocus({packed.defocus, packed.astigmatism, packed.angle});
            ctf.set_phase_shift(packed.phase_shift);
            const auto value = ctf.value_at(fftfreq);
            output(i, h, w) = value * value;
        }
    };

    class Parameters;

    class Parameter {
    private:
        f64* m_buffer{};
        f64 m_delta{};
        isize m_ssize{};
        u64 m_offset{};
        bool m_fit{};

        friend Parameters;

    public:
        [[nodiscard]] auto is_fitted() const noexcept { return m_fit; }
        [[nodiscard]] auto ssize() const noexcept { return m_ssize; }
        [[nodiscard]] auto size() const noexcept { return static_cast<usize>(m_ssize); }
        [[nodiscard]] auto offset() const noexcept { return m_offset; }
        [[nodiscard]] auto delta() const noexcept { return m_delta; }
        [[nodiscard]] auto span() const noexcept { return SpanContiguous(m_buffer + m_offset, m_ssize); }
    };

    class Parameters {
    public:
        enum Index : usize {
            ROTATION = 0,
            TILT,
            PITCH,
            THICKNESS,
            PHASE_SHIFT,
            DEFOCUS,
            ASTIGMATISM_VALUE,
            ASTIGMATISM_ANGLE,
        };
        static constexpr std::array<Index, 8> INDICES = {
            ROTATION, TILT, PITCH, THICKNESS, PHASE_SHIFT, DEFOCUS, ASTIGMATISM_VALUE, ASTIGMATISM_ANGLE
        };

    private:
        std::array<Parameter, sizeof(INDICES)> m_parameters{};

        // Keep track of the initial/default values in case we don't fit them.
        std::vector<f64> m_initial_defocus{};
        std::vector<f64> m_initial_phase_shift{};
        std::vector<f64> m_initial_astigmatism_value{};
        std::vector<f64> m_initial_astigmatism_angle{};

        // Contiguous buffers, where parameters for the optimizer are saved sequentially.
        std::vector<f64> m_buffer{};
        std::vector<f64> m_lower_bounds{};
        std::vector<f64> m_upper_bounds{};
        std::vector<f64> m_abs_tolerance{};

    public:
        [[nodiscard]] auto operator[](Index index) const noexcept -> const Parameter& {
            return m_parameters[index];
        }

    public:
        [[nodiscard]] auto data() noexcept -> f64* { return m_buffer.data(); }
        [[nodiscard]] constexpr auto ssize() const noexcept -> isize { return std::ssize(m_buffer); }
        [[nodiscard]] constexpr auto size() const noexcept -> usize { return std::size(m_buffer); }

        [[nodiscard]] constexpr auto n_fit() const noexcept -> isize {
            isize n{};
            for (auto& index: INDICES)
                n += m_parameters[index].is_fitted();
            return n;
        }

    public: // Special access
        [[nodiscard]] auto angles() const noexcept {
            return Vec{
                m_parameters[ROTATION].is_fitted() ? m_buffer[m_parameters[ROTATION].offset()] : 0,
                m_parameters[PITCH].is_fitted() ? m_buffer[m_parameters[TILT].offset()] : 0,
                m_parameters[TILT].is_fitted() ? m_buffer[m_parameters[PITCH].offset()] : 0
            };
        }

        [[nodiscard]] auto thickness() const noexcept {
            return m_parameters[THICKNESS].is_fitted() ? m_buffer[m_parameters[THICKNESS].offset()] : 0;
        }

        [[nodiscard]] auto defoci() noexcept {
            return m_parameters[DEFOCUS].is_fitted() ?
                m_parameters[DEFOCUS].span() : SpanContiguous(m_initial_defocus.data(), m_parameters[DEFOCUS].ssize());
        }

        [[nodiscard]] auto phase_shift() noexcept {
            auto pointer = m_parameters[PHASE_SHIFT].is_fitted() ?
                m_buffer.data() + m_parameters[PHASE_SHIFT].offset() :
                m_initial_phase_shift.data();
            return SplineGridCubic<f64, 1>(SpanContiguous(pointer, m_parameters[PHASE_SHIFT].ssize()));
        }

        [[nodiscard]] auto astigmatism_value() noexcept {
            auto pointer = m_parameters[ASTIGMATISM_VALUE].is_fitted() ?
                m_buffer.data() + m_parameters[ASTIGMATISM_VALUE].offset() :
                m_initial_astigmatism_value.data();
            return SplineGridCubic<f64, 1>(SpanContiguous(pointer, m_parameters[ASTIGMATISM_VALUE].ssize()));
        }

        [[nodiscard]] auto astigmatism_angle() noexcept {
            auto pointer = m_parameters[ASTIGMATISM_ANGLE].is_fitted() ?
                m_buffer.data() + m_parameters[ASTIGMATISM_ANGLE].offset() :
                m_initial_astigmatism_angle.data();
            return SplineGridCubic<f64, 1>(SpanContiguous(pointer, m_parameters[ASTIGMATISM_ANGLE].ssize()));
        }

        [[nodiscard]] auto lower_bounds() noexcept { return  SpanContiguous(m_lower_bounds.data(), ssize()); }
        [[nodiscard]] auto upper_bounds() noexcept { return  SpanContiguous(m_upper_bounds.data(), ssize()); }
        [[nodiscard]] auto abs_tolerance() noexcept { return  SpanContiguous(m_abs_tolerance.data(), ssize()); }

        template<typename T>
        struct SetOptions {
            T rotation{}; // radians
            T tilt{}; // radians
            T pitch{}; // radians
            T thickness{}; // um
            T phase_shift{}; // radians
            T defocus{}; // um
            T astigmatism_value{}; // um
            T astigmatism_angle{}; // radians
        };

    public:
        Parameters() = default;

        Parameters(
            const Metadata::Stack& metadata,
            const SplineGridCubic<f64, 1>& phase_shift,
            const SplineGridCubic<f64, 1>& astigmatism_value,
            const SplineGridCubic<f64, 1>& astigmatism_angle,
            const SetOptions<Vec<f64, 2>>& relative_bounds
        ) {
            // Set the parameter sizes.
            m_parameters[ROTATION].m_ssize = 1;
            m_parameters[TILT].m_ssize = 1;
            m_parameters[PITCH].m_ssize = 1;
            m_parameters[THICKNESS].m_ssize = 1;
            m_parameters[DEFOCUS].m_ssize = metadata.ssize();
            m_parameters[PHASE_SHIFT].m_ssize = phase_shift.ssize();
            m_parameters[ASTIGMATISM_VALUE].m_ssize = astigmatism_value.ssize();
            m_parameters[ASTIGMATISM_ANGLE].m_ssize = astigmatism_angle.ssize();

            // Set whether they are fitted.
            auto is_fitted = [](const auto& relative_bound) { return not noa::allclose(relative_bound, 0.); };
            m_parameters[ROTATION].m_fit = is_fitted(relative_bounds.rotation);
            m_parameters[TILT].m_fit = is_fitted(relative_bounds.tilt);
            m_parameters[PITCH].m_fit = is_fitted(relative_bounds.pitch);
            m_parameters[THICKNESS].m_fit = is_fitted(relative_bounds.thickness);
            m_parameters[DEFOCUS].m_fit = is_fitted(relative_bounds.defocus);
            m_parameters[PHASE_SHIFT].m_fit = is_fitted(relative_bounds.phase_shift);
            m_parameters[ASTIGMATISM_VALUE].m_fit = is_fitted(relative_bounds.astigmatism_value);
            m_parameters[ASTIGMATISM_ANGLE].m_fit = is_fitted(relative_bounds.astigmatism_angle);

            // Set the offset and allocate the contiguous buffer.
            usize offset{};
            for (auto& data: m_parameters) {
                if (data.m_fit) {
                    data.m_offset = offset;
                    offset += static_cast<usize>(data.m_ssize);
                }
            }
            m_buffer.resize(offset, 0.);
            for (auto& data: m_parameters)
                data.m_buffer = m_buffer.data();

            // Allocate for the default values.
            m_initial_defocus.resize(metadata.size());
            m_initial_phase_shift.resize(phase_shift.size());
            m_initial_astigmatism_value.resize(astigmatism_value.size());
            m_initial_astigmatism_angle.resize(astigmatism_angle.size());

            // Initialize the values, whether they're the default or fitted values.
            for (auto&& [defocus, image]: noa::zip(defoci(), metadata))
                defocus = image.defocus.value;
            for (auto&& [o, i]: noa::zip(this->astigmatism_value().span, astigmatism_value.span))
                o = i;
            for (auto&& [o, i]: noa::zip(this->astigmatism_angle().span, astigmatism_angle.span))
                o = i;

            set_relative_bounds(relative_bounds);
        }

        void set_relative_bounds(const SetOptions<Vec<f64, 2>>& relative_bounds) {
            m_lower_bounds.resize(size(), 0.);
            m_upper_bounds.resize(size(), 0.);

            const auto set_buffer = [&](
                const Parameter& parameter,
                const Vec<f64, 2>& low_and_high_bounds,
                f64 minimum = std::numeric_limits<f64>::lowest(),
                f64 maximum = std::numeric_limits<f64>::max()
            ) {
                if (not parameter.is_fitted())
                    return;
                for (usize i{}; i < parameter.size(); ++i) {
                    const auto index = parameter.offset() + i;
                    const auto value = m_buffer[index];
                    m_lower_bounds[index] = std::max(value + low_and_high_bounds[0], minimum);
                    m_upper_bounds[index] = std::min(value + low_and_high_bounds[1], maximum);
                }
            };

            set_buffer(m_parameters[ROTATION], relative_bounds.rotation);
            set_buffer(m_parameters[TILT], relative_bounds.tilt);
            set_buffer(m_parameters[PITCH], relative_bounds.pitch);
            set_buffer(m_parameters[THICKNESS], relative_bounds.thickness, 0.04, 0.45);
            set_buffer(m_parameters[PHASE_SHIFT], relative_bounds.phase_shift, 0., noa::deg2rad(120.));
            set_buffer(m_parameters[DEFOCUS], relative_bounds.defocus, 0.5);
            set_buffer(m_parameters[ASTIGMATISM_VALUE], relative_bounds.astigmatism_value);
            set_buffer(m_parameters[ASTIGMATISM_ANGLE], relative_bounds.astigmatism_angle);
        }

        void set_abs_tolerance(const SetOptions<f64>& abs_tolerance) {
            m_abs_tolerance.resize(size(), 0.);

            const auto set_buffer = [&](const Parameter& parameter, const f64& tolerance) {
                if (not parameter.is_fitted())
                    return;
                for (usize i{}; i < parameter.size(); ++i) {
                    const auto index = parameter.offset() + i;
                    m_abs_tolerance[index] = tolerance;
                }
            };

            set_buffer(m_parameters[ROTATION], abs_tolerance.rotation);
            set_buffer(m_parameters[TILT], abs_tolerance.tilt);
            set_buffer(m_parameters[PITCH], abs_tolerance.pitch);
            set_buffer(m_parameters[THICKNESS], abs_tolerance.pitch);
            set_buffer(m_parameters[PHASE_SHIFT], abs_tolerance.phase_shift);
            set_buffer(m_parameters[DEFOCUS], abs_tolerance.defocus);
            set_buffer(m_parameters[ASTIGMATISM_VALUE], abs_tolerance.astigmatism_value);
            set_buffer(m_parameters[ASTIGMATISM_ANGLE], abs_tolerance.astigmatism_angle);
        }

        void set_deltas(const SetOptions<f64>& deltas) {
            m_parameters[ROTATION].m_delta = deltas.rotation;
            m_parameters[TILT].m_delta = deltas.tilt;
            m_parameters[PITCH].m_delta = deltas.pitch;
            m_parameters[THICKNESS].m_delta = deltas.thickness;
            m_parameters[PHASE_SHIFT].m_delta = deltas.phase_shift;
            m_parameters[DEFOCUS].m_delta = deltas.defocus;
            m_parameters[ASTIGMATISM_VALUE].m_delta = deltas.astigmatism_value;
            m_parameters[ASTIGMATISM_ANGLE].m_delta = deltas.astigmatism_angle;
        }
    };

    class Fitter {
    private:
        using enum Parameters::Index;

        // Input data.
        const Metadata& m_metadata;
        const Grid& m_grid;
        const Patches& m_patches;

        // Optimizer.
        Parameters m_parameters{};
        Memoizer m_memoizer{};
        SpanContiguous<Vec<f64, 2>> m_fitting_ranges{};

        // Splines.
        Vec<f64, 2> m_time_range{};
        Vec<f64, 2> m_tilt_range{};
        Array<f64> m_phase_shift_weights{};
        Array<f64> m_astigmatism_value_weights{};
        Array<f64> m_astigmatism_angle_weights{};

        // CTFs.
        isize m_n_channels;
        CTFIsotropic64 m_ctf;
        Array<CTFAnisotropicPacked> m_anisotropic_ctf_patches;
        Array<CTFAnisotropicPacked> m_anisotropic_ctf_images;
        Array<CTFIsotropicPacked> m_isotropic_ctf_images;
        Array<f32> m_defocus_patches;

        // Reduction operators.
        ReduceHeight<f32> m_reduce_height;
        ReduceAnisotropicDepth<f32> m_reduce_anisotropic_depth;
        ReduceIsotropicDepth<f32> m_reduce_isotropic_depth;
        ReduceWidth<f32> m_reduce_width;
        Array<f32> m_reduced_cnpw;
        Array<f32> m_reduced_cn1w;
        Array<f32> m_reduced_cnhw;
        Array<f32> m_reduced_cnh1;
        bool m_is_reduce_height_done{};

        // Thickness-aware CTF.
        Array<f32> m_thickness_modulations; // (c,n,1,w)
        bool m_is_thickness_sampled{};

        // Spectrum baseline.
        Array<f32> m_baselines_sampled; // (n,1,1,w)

        std::vector<f64> m_parameters_buffer;
        Array<f64> m_znccs; // (c,1,1,n)

    public:
        Fitter(
            const Metadata& metadata,
            const Grid& grid,
            const Patches& patches,
            const SpanContiguous<Vec<f64, 2>>& fitting_ranges,
            const SplineGridCubic<f64, 1>& phase_shift,
            const SplineGridCubic<f64, 1>& astigmatism_value,
            const SplineGridCubic<f64, 1>& astigmatism_angle,
            const Parameters::SetOptions<Vec<f64, 2>>& relative_bounds
        ) :
            m_metadata(metadata),
            m_grid(grid),
            m_patches(patches),
            m_fitting_ranges(fitting_ranges)
        {
            // Initialize and configure the optimization parameters.
            m_parameters = Parameters(metadata.stack, phase_shift, astigmatism_value, astigmatism_angle, relative_bounds);
            m_parameters.set_abs_tolerance({
                .rotation = noa::deg2rad(0.01),
                .tilt = noa::deg2rad(0.01),
                .pitch = noa::deg2rad(0.01),
                .thickness = 0.005,
                .phase_shift = noa::deg2rad(0.05),
                .defocus = 0.001,
                .astigmatism_value = 0.001,
                .astigmatism_angle = noa::deg2rad(0.1),
            });
            m_parameters.set_deltas({
                .rotation = noa::deg2rad(0.1),
                .tilt = noa::deg2rad(0.1),
                .pitch = noa::deg2rad(0.1),
                .thickness = 0.01,
                .phase_shift = noa::deg2rad(0.5),
                .defocus = 0.005,
                .astigmatism_value = 0.005,
                .astigmatism_angle = noa::deg2rad(0.1),
            });
            m_memoizer = Memoizer(m_parameters.ssize(), 5); // simple memoization if the linear optimizer gets stuck

            // Quick access of the dimensions.
            const auto [n, p, h, w] = m_patches.view().shape();

            // To compute the gradients efficiently, batch the calls for the finite-difference.
            m_n_channels = m_parameters.n_fit() * 2 + 1; // central finite-difference needs 2n+1 evaluations
            m_znccs = noa::Array<f64>({m_n_channels, 1, 1, n});

            // Allocate the spectra buffers. Most things need to be dereferenceable.
            // Since accesses are per row, use a pitched layout for better performance on the GPU.
            const auto device = patches.view().options().device;
            const auto options_pitched = ArrayOption{.device = device, .allocator = Allocator::PITCHED};
            const auto options_managed = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
            const auto options_pitched_managed = ArrayOption{.device = device, .allocator = Allocator::PITCHED_MANAGED};

            // Allocate for the CTFs. Everything needs to be dereferenceable.
            m_anisotropic_ctf_patches = Array<CTFAnisotropicPacked>({m_n_channels, 1, 1, n * p}, options_managed);
            m_anisotropic_ctf_images = Array<CTFAnisotropicPacked>({m_n_channels, 1, 1, n}, options_managed);
            m_isotropic_ctf_images = Array<CTFIsotropicPacked>({m_n_channels, 1, 1, n}, options_managed);
            m_defocus_patches = Array<f32>({m_n_channels, 1, 1, n * p}, options_managed);

            // Baseline and thickness-aware CTF-model.
            m_baselines_sampled = Array<f32>({n, 1, 1, w}, options_pitched_managed);
            m_thickness_modulations = Array<f32>({m_n_channels, n, 1, w}, options_pitched_managed);

            // Precompute the spline range and weights.
            // These are for the time-resolved phase-shift and tilt-resolved astigmatism.
            m_tilt_range = metadata.stack.tilt_range();
            m_time_range = metadata.stack.time_range().as<f64>();
            m_phase_shift_weights = Array<f64>({1, 1, phase_shift.ssize(), n});
            m_astigmatism_value_weights = Array<f64>({1, 1, astigmatism_value.ssize(), n});
            m_astigmatism_angle_weights = Array<f64>({1, 1, astigmatism_angle.ssize(), n});

            auto set_weights = [&](auto&& to_norm_coordinate, const auto& range, const auto& array) {
                auto span = array.template span<f64, 2>();
                for (isize i{}; i < span.shape()[0]; ++i) { // per node
                    for (isize j{}; j < span.shape()[1]; ++j) { // per image
                        const f64 nc = (to_norm_coordinate(metadata.stack[j]) - range[0]) / (range[1] - range[0]);
                        span(i, j) = SplineGridCubic<f64, 1>::weight_at(Vec{nc}, Vec{i}, span.shape().filter(0));
                    }
                }
            };
            set_weights([](auto& s) { return static_cast<f64>(s.time); }, m_time_range, m_phase_shift_weights);
            set_weights([](auto& s) { return s.angles[1]; }, m_tilt_range, m_astigmatism_value_weights);
            set_weights([](auto& s) { return s.angles[1]; }, m_tilt_range, m_astigmatism_angle_weights);

            // Set up the base settings for the CTF.
            m_ctf = CTFIsotropic64({
                .pixel_size = mean(metadata.spacing),
                .defocus = 0., // overwritten
                .voltage = metadata.sample.voltage,
                .amplitude = metadata.sample.amplitude,
                .cs = metadata.sample.cs,
                .phase_shift = 0, // overwritten
                .bfactor = 0,
                .scale = 1.,
            });

            // Operators and buffers for the non-astigmatic case.
            m_reduced_cnpw = Array<f32>({m_n_channels, n, p, w}, options_pitched);
            m_reduced_cn1w = Array<f32>({m_n_channels, n, 1, w}, options_pitched_managed);
            m_reduce_height = ReduceHeight{
                .polar = m_patches.view_batched().span().filter(0, 2, 3).as_contiguous(), // (np,h,w)
                .packed = m_anisotropic_ctf_patches.span_1d(), // (cnp)
                .isotropic_ctf = m_ctf.as<f32>(),
                .anisotropic_ctf = ns::CTFAnisotropic(m_ctf).as<f32>(),
                .phi_start = static_cast<f32>(m_patches.phi().start),
                .phi_step = static_cast<f32>(m_patches.phi_step()),
                .rho_start = static_cast<f32>(m_patches.rho().start),
                .rho_step = static_cast<f32>(m_patches.rho_step()),
                .rho_range = static_cast<f32>(m_patches.rho().stop - m_patches.rho().start), // assumes endpoint=true
            };
            m_reduce_isotropic_depth = ReduceIsotropicDepth<f32>{
                .image_spectra = m_reduced_cnpw.span().reshape({1, -1, p, w}).filter(1, 2, 3).as_contiguous(), // (cn,p,w)
                .defocus_images = m_isotropic_ctf_images.span_1d(), // (cn)
                .defocus_patches = m_defocus_patches.span_1d(), // (cnp)
                .isotropic_ctf = m_reduce_height.isotropic_ctf,
                .phi_start = m_reduce_height.phi_start,
                .phi_step = m_reduce_height.phi_step,
                .rho_start = m_reduce_height.rho_start,
                .rho_step = m_reduce_height.rho_step,
                .rho_range = m_reduce_height.rho_range,
                .n_images = n,
            };

            // Operators and buffers for the astigmatic case.
            m_reduced_cnhw = Array<f32>({m_n_channels, n, h, w}, options_pitched_managed);
            m_reduced_cnh1 = Array<f32>({m_n_channels, n, h, 1}, options_pitched_managed);
            m_reduce_anisotropic_depth = ReduceAnisotropicDepth<f32>{
                .polar = m_patches.view().span_contiguous(), // (n,p,h,w)
                .ctf_images_packed = m_anisotropic_ctf_images.span_1d(), // (c*n)
                .defocus_patches = m_defocus_patches.span_1d(), // (c*n*p)
                .isotropic_ctf = m_reduce_height.isotropic_ctf,
                .anisotropic_ctf = m_reduce_height.anisotropic_ctf,
                .phi_start = m_reduce_height.phi_start,
                .phi_step = m_reduce_height.phi_step,
                .rho_start = m_reduce_height.rho_start,
                .rho_step = m_reduce_height.rho_step,
                .rho_range = m_reduce_height.rho_range,
            };
            m_reduce_width = ReduceWidth<f32>{
                .image_spectra = m_reduced_cnhw.span_contiguous(), // (c,n,h,w)
                .image_baseline = m_baselines_sampled.span().filter(0, 3).as_contiguous(), // (n,w)
                .image_thickness_modulation = m_thickness_modulations.span().filter(0, 1, 3).as_contiguous(), // (c,n,w)
                .image_defoci = m_anisotropic_ctf_images.span().filter(0, 3).as_contiguous(), // (c,n)
                .isotropic_ctf = m_reduce_height.isotropic_ctf,
                .anisotropic_ctf = m_reduce_height.anisotropic_ctf,
                .phi_start = m_reduce_height.phi_start,
                .phi_step = m_reduce_height.phi_step,
                .rho_start = m_reduce_height.rho_start,
                .rho_step = m_reduce_height.rho_step,
            };
        }

        // Read the current parameters and update the CTF buffers for the given channel accordingly.
        void update_ctfs(isize channel) {
            const Vec<f64, 3> angle_offsets = m_parameters.angles();
            const SplineGridCubic<f64, 1> time_resolved_phase_shift = m_parameters.phase_shift();
            const SplineGridCubic<f64, 1> tilt_resolved_astigmatism_value = m_parameters.astigmatism_value();
            const SplineGridCubic<f64, 1> tilt_resolved_astigmatism_angle = m_parameters.astigmatism_angle();
            const SpanContiguous<f64> defoci = m_parameters.defoci();
            const f64 sample_thickness_um = m_metadata.sample.thickness * 1e-3 + m_parameters.thickness();

            const auto ctf_anisotropic_images = m_anisotropic_ctf_images.subregion(channel).span_1d();
            const auto ctf_isotropic_images = m_isotropic_ctf_images.subregion(channel).span_1d();
            for (isize i{}; i < m_patches.n_images(); ++i) {
                // Time-resolved phase-shift.
                const f64 itime = normalized_time(m_metadata.stack[i]);
                const f64 phase_shift = time_resolved_phase_shift.interpolate_at(itime);

                // Tilt-resolved astigmatism.
                const f64 itilt = normalized_tilt(m_metadata.stack[i]);
                const f64 slice_astigmatism_value = tilt_resolved_astigmatism_value.interpolate_at(itilt);
                const f64 slice_astigmatism_angle = tilt_resolved_astigmatism_angle.interpolate_at(itilt);

                // Set the defocus and phase-shift of the image CTF.
                ctf_anisotropic_images[i].defocus = static_cast<f32>(defoci[i]);
                ctf_anisotropic_images[i].astigmatism = static_cast<f32>(slice_astigmatism_value);
                ctf_anisotropic_images[i].angle = static_cast<f32>(slice_astigmatism_angle);
                ctf_anisotropic_images[i].phase_shift = static_cast<f32>(phase_shift);
                ctf_isotropic_images[i].defocus = ctf_anisotropic_images[i].defocus;
                ctf_isotropic_images[i].phase_shift = ctf_anisotropic_images[i].phase_shift;

                const auto chunk = m_patches.chunk(i);
                const auto ctf_patches = m_anisotropic_ctf_patches.subregion(channel).span_1d().subregion(chunk);
                const auto defocus_patches = m_defocus_patches.subregion(channel).span_1d().subregion(chunk);

                const auto image_spacing = Vec<f64, 2>::from_value(m_ctf.pixel_size());
                const auto image_angles = noa::deg2rad(m_metadata.stack[i].angles) + angle_offsets;
                const auto patch_centers = m_grid.patches_centers();

                // Sample the thickness modulation for this image. If the thickness isn't
                // modeled, the modulation is a row of ones and does nothing.
                if (not m_is_thickness_sampled) {
                    ThicknessModulation{
                        .wavelength = m_ctf.wavelength(),
                        .spacing = m_ctf.pixel_size(),
                        .thickness = effective_thickness(sample_thickness_um, image_angles) * 1e4,
                    }.sample(
                        m_thickness_modulations.span().subregion(channel, i).as_1d(),m_patches.rho_vec()
                    );
                }

                // Update the CTFs of the patches belonging to the current image.
                for (isize j{}; j < m_patches.n_patches_per_image(); ++j) {
                    const auto patch_z_offset_um = m_grid.patch_z_offset(image_angles, image_spacing, patch_centers[j]);
                    const auto patch_defocus = defoci[i] - patch_z_offset_um;
                    ctf_patches[j].defocus = static_cast<f32>(patch_defocus);
                    ctf_patches[j].astigmatism = ctf_anisotropic_images[i].astigmatism;
                    ctf_patches[j].angle = ctf_anisotropic_images[i].angle;
                    ctf_patches[j].phase_shift = ctf_anisotropic_images[i].phase_shift;
                    defocus_patches[j] = ctf_patches[j].defocus;
                }
            }
        }

        void update_channels(isize& channel, const Parameter& parameter) {
            if (not parameter.is_fitted())
                return;

            // Save original parameters.
            auto span = parameter.span();
            m_parameters_buffer.clear();
            for (usize i{}; i < span.size(); ++i)
                m_parameters_buffer.push_back(span[i]);

            // Save the CTFs, with +/- delta.
            for (usize i{}; i < span.size(); ++i)
                span[i] = m_parameters_buffer[i] - parameter.delta();
            update_ctfs(channel++);
            for (usize i{}; i < span.size(); ++i)
                span[i] = m_parameters_buffer[i] + parameter.delta();
            update_ctfs(channel++);

            // Restore to original parameters.
            for (usize i{}; i < span.size(); ++i)
                span[i] = m_parameters_buffer[i];
        }

        // Reduce the polar height of every patch, (c,n,p,h,w)->(c,n,p,1,w).
        void reduce_height(bool first_channel_only = false) {
            const auto h = m_patches.view().shape().height();
            const auto& [c, n, p, w] = m_reduced_cnpw.shape();
            const auto actual_c = first_channel_only ? 1 : c;
            const auto output = m_reduced_cnpw.view().subregion(Slice{0, actual_c});
            if (h == 1) {
                // The polar spectra are already reduced to 1d (astigmatism is ignored).
                // No reduction necessary, simply copy to the output buffer.
                auto broadcast = noa::broadcast(m_patches.view().reshape({1, n, p, w}), output.shape());
                noa::ewise(broadcast, output, noa::Copy{});
            } else {
                noa::reduce_axes_iwise( // (cnp,h,w)->(cnp,1,w)
                    Shape{actual_c * n * p, h, w}, m_patches.view().device(), noa::wrap(f32{0}, f32{0}),
                    output.reshape({1, actual_c * n * p, 1, w}), m_reduce_height
                );
            }
        }

        // Fuse the 1d spectrum of each patch into one 1d spectrum per image, (c,n,p,1,w)->(c,n,1,1,w).
        auto reduce_isotropic_depth(bool first_channel_only = false) {
            const auto [n, p, h, w] = m_patches.view().shape();
            const auto actual_c = first_channel_only ? 1 : m_n_channels;
            const auto output = m_reduced_cn1w.view().subregion(Slice{0, actual_c});
            noa::reduce_axes_iwise( // (cn,p,w)->(cn,1,w)
                Shape{actual_c * n, p, w}, m_patches.view().device(), noa::wrap(f32{0}, f32{0}),
                output.reshape({1, actual_c * n, 1, w}), m_reduce_isotropic_depth
            );
            return output;
        }

        void zncc_no_astigmatism() {
            if (not m_is_reduce_height_done)
                reduce_height();
            auto reduced_cn1w = reduce_isotropic_depth().eval();

            for (isize c{}; c < m_n_channels; ++c) {
                const auto spectra_nw = reduced_cn1w.span().subregion(c).filter(1, 3).as_contiguous(); // (c,n,1,w)->(n,w)
                const auto baselines = m_baselines_sampled.span().filter(0, 3).as_contiguous(); // (n,1,1,w)->(n,w)
                const auto thickness_modulations = m_thickness_modulations.span().subregion(c).filter(1, 3).as_contiguous(); // (n,w)
                const auto ctf_images = m_isotropic_ctf_images.span().subregion(c).as_1d(); // (c,1,1,w)->(n)
                const auto znccs = m_znccs.span().subregion(c).as_1d(); // (c,1,1,w)->(n)

                for (isize i{}; i < m_patches.n_images(); ++i) {
                    m_ctf.set_defocus(static_cast<f64>(ctf_images[i].defocus));
                    m_ctf.set_phase_shift(static_cast<f64>(ctf_images[i].phase_shift));
                    znccs[i] = zero_normalized_cross_correlation(
                        spectra_nw[i], m_ctf, m_patches.rho_vec(), m_fitting_ranges[i],
                        baselines[i], thickness_modulations[i]
                    );
                }
            }
        }

        // Fuse the patches together, accounting for their (astigmatic-)defocus difference.
        auto reduce_anisotropic_depth(bool first_channel_only = false) {
            const auto [n, p, h, w] = m_patches.view().shape();
            const auto actual_c = first_channel_only ? 1 : m_n_channels;
            const auto output = m_reduced_cnhw.view().subregion(Slice{0, actual_c});
            noa::reduce_axes_iwise( // (cn,p,h,w)->(cn,1,h,w)
                Shape{actual_c * n, p, h, w}, m_patches.view().device(), noa::wrap(f32{0}, f32{0}),
                output.reshape({actual_c * n, 1, h, w}), m_reduce_anisotropic_depth
            );
            return output;
        }

        // Compute the ZNCC for each line of the spectrum.
        auto reduce_width() {
            noa::reduce_axes_iwise( // (c,n,h,w)->(c,n,h,1)
                m_reduced_cnhw.shape(), m_reduced_cnhw.device(), ReduceWidth<f32>::Reduced{},
                m_reduced_cnh1.view(), m_reduce_width
            );
            return m_reduced_cnh1.view();
        }

        void zncc_astigmatism() {
            reduce_anisotropic_depth();
            auto reduced_cnh1 = reduce_width().eval();

            const auto znccs_cnh = reduced_cnh1.span().filter(0, 1, 2).as_contiguous(); // (c,n,h)
            const auto znccs = m_znccs.span().filter(0, 3).as_contiguous(); // (c,w)
            for (isize c{}; c < znccs_cnh.shape()[0]; ++c) {
                for (isize n{}; n < znccs_cnh.shape()[1]; ++n) {
                    f64 zncc{};
                    for (isize h{}; h < znccs_cnh.shape()[2]; ++h)
                        zncc += static_cast<f64>(znccs_cnh(c, n, h));
                    znccs(c, n) = zncc / static_cast<f64>(znccs_cnh.shape()[2]);
                }
            }
        }

        auto zncc() -> f64 {
            // When the astigmatism isn't fitted, reduce the height of each patch, then fuse the 1d patches.
            // This only needs to be computed once. However, when the astigmatism is fitted, fuse the polar patches
            // to one spectrum per image (keep the polar height). This should be more sensitive to astigmatism,
            // but it needs to be recomputed at every iteration.
            m_parameters[ASTIGMATISM_VALUE].is_fitted() ? zncc_astigmatism() : zncc_no_astigmatism();
            return simple_average(m_znccs.span().subregion(0).as_1d());
        }

        template<nt::any_of<SpanContiguous<f64, 2>, Empty> T = Empty>
        void gradient(
            isize& channel,
            const Parameter& parameter,
            f64* gradients,
            const T& weights = {}
        ) {
            if (not parameter.is_fitted())
                return;

            // Prepare for direct access.
            const auto nccs = this->m_znccs.span();
            const auto fx = nccs.subregion(0).as_1d();
            const auto fx_minus_delta = nccs.subregion(channel++).as_1d();
            const auto fx_plus_delta = nccs.subregion(channel++).as_1d();

            const auto span = parameter.span();
            gradients += parameter.offset();

            // Compute the gradient for each variable by reducing the per-image scores.
            const isize n = m_patches.n_images();
            for (isize i{}; i < span.ssize(); ++i) {
                f64 score_minus_delta{0};
                f64 score_plus_delta{0};
                for (isize j{}; j < n; ++j) {
                    f64 weight{};
                    if (span.ssize() == 1) {
                        // If there's a single variable, it affects every image.
                        weight = 1;
                    } else if (span.ssize() == n) {
                        // Each variable only affects its corresponding image, so recompose the total score based on that.
                        // The resulting score is equivalent to the single-variable case above but allows computing
                        // the score only twice, as opposed to twice per variable.
                        weight = static_cast<f64>(i == j);
                    } else {
                        // The weights tell us how much the image j is affected by the current variable i.
                        // We use this information to get an estimated score. This score is not exactly what
                        // we would have gotten with the single-variable case above, but still gives us
                        // good enough derivatives to guide the optimizer. This is equivalent to Warp's wiggle weights.
                        if constexpr (not nt::empty<T>)
                            weight = weights(i, j);
                        else
                            panic();
                    }
                    score_minus_delta += fx[j] * (1 - weight) + fx_minus_delta[j] * weight;
                    score_plus_delta += fx[j] * (1 - weight) + fx_plus_delta[j] * weight;
                }
                score_minus_delta /= static_cast<f64>(n);
                score_plus_delta /= static_cast<f64>(n);
                gradients[i] = CentralFiniteDifference::get(score_minus_delta, score_plus_delta, parameter.delta());
            }
        }

        static auto function_to_maximise(u32, const f64* parameters, f64* gradients, void* buffer) -> f64 {
            check(gradients);
            auto& self = *static_cast<Fitter*>(buffer);

            // The optimizer may pass its own array, so update our parameters.
            auto& params = self.parameters();
            if (parameters != params.data())
                std::copy_n(parameters, params.size(), params.data());

            // Memoization. Sometimes the linear search within L-BFGS is stuck,
            // so detect for these cases to not have to recompute the gradients each time.
            std::optional<f64> memoized_score = self.memoizer().find(params.data(), gradients, 1e-8);
            if (memoized_score.has_value()) {
                f64 score = memoized_score.value();
                Logger::trace("score={:.4f}, memoized=true", score);
                return score;
            }

            // 1. Update the CTFs for every channel.
            self.update_ctfs(0);
            for (isize channel{1}; auto& index: Parameters::INDICES)
                self.update_channels(channel, params[index]);

            // 2. Compute the scores.
            const f64 score = self.zncc();

            // 3. Compute the gradients.
            auto get_spline = [&self](Parameters::Index index) {
                switch (index) {
                    case PHASE_SHIFT:       return self.m_phase_shift_weights.span<f64, 2>().as_contiguous();
                    case ASTIGMATISM_VALUE: return self.m_astigmatism_value_weights.span<f64, 2>().as_contiguous();
                    case ASTIGMATISM_ANGLE: return self.m_astigmatism_angle_weights.span<f64, 2>().as_contiguous();
                    default:                return SpanContiguous<f64, 2>{};
                }
            };
            for (isize channel{1}; auto& index: Parameters::INDICES)
                self.gradient(channel, params[index], gradients, get_spline(index));

            // No need to compute certain buffers every time.
            self.m_is_thickness_sampled = not params[THICKNESS].is_fitted();
            self.m_is_reduce_height_done = not params[ASTIGMATISM_VALUE].is_fitted();

            //
            // self.memoizer().record(parameters, score, gradients);
            // Logger::trace("score={:.4f}, angles={::+.3f}, defoci={::.2f}",
            //     score, noa::rad2deg(params.angles()), params[DEFOCUS].span());
            //
            if (params[ASTIGMATISM_VALUE].is_fitted()) {
                Logger::trace(
                    "score={}, astig={::.2f}, astig_grad={::.4f}",
                    score, params[ASTIGMATISM_VALUE].span(),
                    SpanContiguous(gradients + params[ASTIGMATISM_VALUE].offset(), params[ASTIGMATISM_VALUE].size())
                );
            }
            return score;
        }

        auto setup_fitting_ranges_and_baselines_(i32 extra_peaks_to_append = 2) {
            // Reset caches (not necessary, but in case this is called twice).
            m_memoizer.reset_cache();
            m_is_reduce_height_done = false;

            // Compute the per-image spectra.
            update_ctfs(0);
            reduce_height(true);
            const auto spectrum_cn1w = reduce_isotropic_depth(true).eval();
            const auto spectrum_n = spectrum_cn1w.span().subregion(0).filter(1, 3).as_contiguous(); // (n,w)

            // Compute the per-image CTFs.
            const auto ctf_images_buffer = Array<CTFIsotropic64>(m_patches.n_images());
            const auto ctf_images = ctf_images_buffer.span_1d();
            for (auto&& [ctf, packed]: noa::zip(ctf_images, m_isotropic_ctf_images.span_1d())) {
                ctf = m_ctf;
                ctf.set_defocus(static_cast<f64>(packed.defocus));
                ctf.set_phase_shift(static_cast<f64>(packed.phase_shift));
            }

            // Prepare the baselines and fitting range for the fitting.
            auto baseline = Baseline{};
            for (isize i{}; i < m_patches.n_images(); ++i) {
                m_fitting_ranges[i] = baseline.fit_and_tune_fitting_range(
                    spectrum_n[i], m_patches.rho_vec(), ctf_images[i], {
                        .threshold = 1.2,
                        .keep_first_nth_peaks = 2,

                        // In the case of strong astigmatism, the initial spectrum may have only a few Thon-rings.
                        // By looking ahead, we give the optimizer more opportunities to improve the spectrum.
                        .n_extra_peaks_to_append = extra_peaks_to_append,
                        .n_recoveries_allowed = 1,
                        .maximum_n_consecutive_bad_peaks = 1,

                        .thickness_um = effective_thickness(m_metadata.sample.thickness * 1e-3, m_metadata.stack[i].angles), // FIXME?
                    });
                baseline.sample(m_baselines_sampled.subregion(i).span_1d(), m_patches.rho_vec());
            }

            return Pair{spectrum_n, ctf_images_buffer};
        }

        auto fit(nlopt_algorithm algorithm, i32 max_number_of_evaluations) -> f64 {
            setup_fitting_ranges_and_baselines_();

            // Solve.
            auto optimizer = Optimizer(algorithm, m_parameters.ssize());
            optimizer.set_max_objective(function_to_maximise, this);
            optimizer.set_bounds(
                m_parameters.lower_bounds().data(),
                m_parameters.upper_bounds().data()
            );
            optimizer.set_x_tolerance_abs(m_parameters.abs_tolerance().data());
            if (max_number_of_evaluations > 1)
                optimizer.set_max_number_of_evaluations(max_number_of_evaluations);
            return optimizer.optimize(m_parameters.data());
        }

        void plot_diagnostics(const Path& output_directory) {
            auto [spectrum_n, ctf_n] = setup_fitting_ranges_and_baselines_(0);

            // Per-image polar spectra with CTFs.
            auto m_reduced_n1hw = reduce_anisotropic_depth(true).permute({1, 0, 2, 3}); // not baseline-subtracted.
            auto filename = output_directory / "per_image_spectra.mrc";
            noa::write_image(m_reduced_n1hw, filename, {.dtype = "f16"});
            Logger::trace("{} saved", filename);

            noa::iwise(m_reduced_n1hw.shape().filter(0, 2, 3), m_reduced_n1hw.device(), SimulateCTF2{
                .output = m_reduced_n1hw.span().filter(0, 2, 3).as_contiguous(),
                .ctfs = m_anisotropic_ctf_images.span_1d(),
                .ctf = CTFAnisotropic64(m_ctf).as<f32>(),
                .phi_start = m_reduce_height.phi_start,
                .phi_step = m_reduce_height.phi_step,
                .rho_start = m_reduce_height.rho_start,
                .rho_step = m_reduce_height.rho_step,
            });
            filename = output_directory / "per_image_ctfs.mrc";
            noa::write_image(m_reduced_n1hw, filename, {.dtype = "f16"});
            Logger::trace("{} saved", filename);

            // Per-image EPA with CTFs.
            save_plot_ctf_fit(
                m_patches.rho(), spectrum_n, m_baselines_sampled.span_contiguous<const f32, 2>(),
                ctf_n.span_1d(),
                output_directory / "refined_fitting.txt", {
                    .title = "Per-image refined spectra",
                });

            // Per-image fitting ranges.
            save_plot_xy(
                m_metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
                m_fitting_ranges | stdv::transform([&](const auto& v) {
                    return fftfreq_to_resolution(m_ctf.pixel_size(), v[1]);
                }),
                output_directory / "fitting_ranges.txt", {
                    .title = "Resolution cutoff for CTF fitting",
                    .x_name = "Image index (as saved in the file)",
                    .y_name = "Resolution (A)",
                    .label = "Refine fitting",
                });

            // Prepare the EPA.
            const auto buffer = noa::zeros<f32>({3, 1, 1, spectrum_n.shape().width()});
            const auto spectrum_rescaled = buffer.view().subregion(0);
            const auto spectrum_average = buffer.view().subregion(1);
            const auto spectrum_weights = buffer.view().subregion(2);

            // Compute the target CTF for the equiphase-average.
            const auto average_ctf = [&] {
                f64 average_defocus{};
                f64 min_phase_shift{};
                for (auto& ctf_image: ctf_n.span_1d()) {
                    average_defocus += ctf_image.defocus();
                    min_phase_shift = std::min(min_phase_shift, ctf_image.phase_shift());
                }
                auto ctf = m_ctf;
                ctf.set_defocus(average_defocus / static_cast<f64>(m_patches.n_images()));
                ctf.set_phase_shift(min_phase_shift);
                return ctf;
            }();

            const auto fftfreq_step = m_patches.rho_step();
            for (isize i{}; i < m_patches.n_images(); ++i) {
                // Subtract the baseline.
                const auto baseline = m_baselines_sampled.subregion(i).span_1d();
                for (auto&& [s, b]: noa::zip(spectrum_n[i], baseline))
                    s -= b;

                // Rescale to the target CTF.
                const auto ctf_i = ctf_n.span_1d()[i];
                nx::phase_spectra(
                    View(spectrum_n[i]), m_patches.rho(), ctf_i,
                    spectrum_rescaled, m_patches.rho(), average_ctf
                );
                auto fitting_range = m_fitting_ranges[i];
                for (auto j: noa::irange(2)) {
                    auto phase = ctf_i.phase_at(fitting_range[j]);
                    fitting_range[j] = average_ctf.fftfreq_at(phase);
                }

                // To fuse spectra with different thicknesses, we need to correct for the thickness modulation.
                // We could multiply the spectrum with the thickness modulation curve, but that would
                // downweight regions near and at the node (and create visible artifacts from the flipping
                // if the baseline isn't perfectly centered on zero). Instead, skip these regions entirely
                // and flip the zero-centered spectrum (oscillations) when the curve goes negative.
                const auto thickness_modulation = ThicknessModulation{
                    .wavelength = ctf_i.wavelength(),
                    .spacing = ctf_i.pixel_size(),
                    .thickness = m_metadata.sample.thickness * 10, // nm->angstrom
                };

                // Before adding this spectrum to the average, get the L2-norm within the fitting range.
                f32 l2_norm{};
                for (isize j{}; const auto& e: spectrum_rescaled.span_1d()) {
                    const f64 fftfreq = static_cast<f64>(j++) * fftfreq_step + m_patches.rho().start;
                    if (fitting_range[0] <= fftfreq and fftfreq <= fitting_range[1] and
                        std::abs(thickness_modulation.sample_at(fftfreq)) >= 0.9)
                        l2_norm += e * e;
                }
                l2_norm = std::sqrt(l2_norm);

                // Exclude regions after the fitting range.
                for (isize j{}; auto&& [rescaled, weight, average]: noa::zip(
                    spectrum_rescaled.span_1d(),
                    spectrum_weights.span_1d(),
                    spectrum_average.span_1d())
                ) {
                    const f64 fftfreq = static_cast<f64>(j++) * fftfreq_step + m_patches.rho().start;
                    if (fftfreq <= fitting_range[1]) {
                        const auto modulation = static_cast<f32>(thickness_modulation.sample_at(fftfreq));
                        if (std::abs(modulation) < 0.9f)
                            continue;
                        weight += 1;
                        average += (rescaled / l2_norm) * std::copysign(1.f, modulation);
                    } else {
                        break;
                    }
                }
            }
            for (auto&& [s, w]: noa::zip(spectrum_average.span_1d(), spectrum_weights.span_1d()))
                if (w > 1e-6f)
                    s /= w;

            save_plot_xy(m_patches.rho(), spectrum_average, output_directory / "epa.txt", {
                .title = "Equiphase-average",
                .x_name = "fftfreq",
                .label =  fmt::format("defocus={:.3f}", average_ctf.defocus()),
            });
        }

        void update_metadata_and_state(
            Metadata& metadata,
            SplineGridCubic<f64, 1> phase_shift,
            SplineGridCubic<f64, 1> astigmatism_value,
            SplineGridCubic<f64, 1> astigmatism_angle,
            Vec<f64, 3>& final_angle_offsets
        ) {
            phase_shift.update_from_span(m_parameters.phase_shift().span);
            astigmatism_value.update_from_span(m_parameters.astigmatism_value().span);
            astigmatism_angle.update_from_span(m_parameters.astigmatism_angle().span);

            // Update metadata.
            // Note that the optimizer ignores the astigmatism and
            // phase-shift from the metadata, and uses the splines instead.
            const auto defoci = m_parameters.defoci();
            const auto angle_offsets = noa::rad2deg(m_parameters.angles());
            for (isize i{}; i < metadata.stack.ssize(); ++i) {
                auto& image = metadata.stack[i];
                const auto time = normalized_time(image);
                const auto tilt = normalized_tilt(image); // must be before updating the tilt angles

                image.angles = Metadata::Image::to_angle_range(image.angles + angle_offsets);
                image.phase_shift = phase_shift.interpolate_at(time);
                image.defocus = {
                    .value = defoci[i],
                    .astigmatism = astigmatism_value.interpolate_at(tilt),
                    .angle = astigmatism_angle.interpolate_at(tilt),
                };
            }
            metadata.sample.thickness = m_metadata.sample.thickness + m_parameters.thickness() * 1e3; // um->nm
            final_angle_offsets += angle_offsets;
        }

        [[nodiscard]] auto parameters() noexcept -> Parameters& { return m_parameters; }
        [[nodiscard]] auto memoizer() noexcept -> Memoizer& { return m_memoizer; }

        [[nodiscard]] auto normalized_time(const Metadata::Image& slice) noexcept -> f64 {
            return (static_cast<f64>(slice.time) - m_time_range[0]) / (m_time_range[1] - m_time_range[0]);
        }
        [[nodiscard]] auto normalized_tilt(const Metadata::Image& slice) noexcept -> f64 {
            return (slice.angles[1] - m_tilt_range[0]) / (m_tilt_range[1] - m_tilt_range[0]);
        }
    };

    void increase_tilt_spline_resolution(
        i64 new_resolution,
        const Metadata::Stack& metadata,
        const Vec<f64, 2>& range,
        Array<f64>& values
    ) {
        const auto spline = SplineGridCubic<f64, 1>(values.span_1d());
        const auto new_values = noa::zeros<f64>(new_resolution);
        const auto new_spline = SplineGridCubic<f64, 1>(new_values.span_1d());
        const auto metadata_sorted = metadata.clone().sort("tilt");
        for (auto&& [image, node]: noa::zip(metadata_sorted, new_spline.span)) {
            const auto tilt = (image.angles[1] - range[0]) / (range[1] - range[0]);
            node = spline.interpolate_at(tilt);
        }
        values = new_values;
    }

    struct Runner {
        Metadata& metadata;
        FitRefineState& state;
        const Grid& grid;
        const Patches& patches;
        const FitRefineOptions& options;

        void operator()(
            nlopt_algorithm algorithm,
            i32 max_number_of_evaluations,
            bool plot_diagnostics,
            const Parameters::SetOptions<Vec<f64, 2>>& relative_bounds
        ) const {
            auto t = Logger::trace_scope_time("Running optimizer");

            const auto phase_shift_spline = SplineGridCubic<f64, 1>(state.phase_shift.span_1d());
            const auto astigmatism_value_spline = SplineGridCubic<f64, 1>(state.astigmatism_value.span_1d());
            const auto astigmatism_angle_spline = SplineGridCubic<f64, 1>(state.astigmatism_angle.span_1d());

            auto fitter = Fitter(
                metadata, grid, patches, state.fitting_ranges.span_1d(),
                phase_shift_spline, astigmatism_value_spline, astigmatism_angle_spline,
                relative_bounds
            );

            Logger::trace(
                "Optimization:\n"
                "{}{}{}{}{}{}{}{}"
                "  n_parameters={}\n"
                "  max_number_of_evaluations={}\n"
                "  optimizer={}",
                noa::allclose(relative_bounds.rotation, 0.) ? "" : fmt::format("  rotation={::.2f}deg bound\n", noa::rad2deg(relative_bounds.rotation)),
                noa::allclose(relative_bounds.tilt, 0.) ? "" : fmt::format("  tilt={::.2f}deg bound\n", noa::rad2deg(relative_bounds.tilt)),
                noa::allclose(relative_bounds.pitch, 0.) ? "" : fmt::format("  pitch={::.2f}deg bound\n", noa::rad2deg(relative_bounds.pitch)),
                noa::allclose(relative_bounds.thickness, 0.) ? "" : fmt::format("  thickness={::.3f}um bound\n", relative_bounds.thickness),
                noa::allclose(relative_bounds.phase_shift, 0.) ? "" : fmt::format("  phase_shift={::.2f}deg bound\n", noa::rad2deg(relative_bounds.phase_shift)),
                noa::allclose(relative_bounds.defocus, 0.) ? "" : fmt::format("  defocus={::.2f}um bound\n", relative_bounds.defocus),
                noa::allclose(relative_bounds.astigmatism_value, 0.) ? "" : fmt::format("  astigmatism_value={::.2f}um bound\n", relative_bounds.astigmatism_value),
                noa::allclose(relative_bounds.astigmatism_angle, 0.) ? "" : fmt::format("  astigmatism_angle={::.2f}deg bound\n", noa::rad2deg(relative_bounds.astigmatism_angle)),
                fitter.parameters().ssize(),
                max_number_of_evaluations,
                algorithm == NLOPT_LD_LBFGS ? "L-BFGS (local, gradient-based)" : "StoGO (global, gradient-based)"
            );

            fitter.fit(algorithm, max_number_of_evaluations);
            fitter.update_metadata_and_state(
                metadata, phase_shift_spline, astigmatism_value_spline, astigmatism_angle_spline, state.angle_offsets);

            Logger::trace(
                "stage_angles=[rotation={:.2f}, tilt={:.2f}, pitch={:.2f}]",
                state.angle_offsets[0], state.angle_offsets[1], state.angle_offsets[2]
            );

            Logger::trace("defocus={::.2f}", metadata.stack | stdv::transform([](auto& s) { return s.defocus.value; }));
            Logger::trace("astig={::.2f}", metadata.stack | stdv::transform([](auto& s) { return s.defocus.astigmatism; }));

            if (not plot_diagnostics)
                return;

            // Diagnostics.
            fitter.plot_diagnostics(options.output_directory);
            save_plot_xy(
                metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
                metadata.stack | stdv::transform([](auto& s) { return s.defocus.value; }),
                options.output_directory / "defocus_fit.txt", {
                    .title = "Per-tilt defocus",
                    .x_name = "Image index (as saved in the stack)",
                    .y_name = "Defocus (μm)",
                    .label = "Refine fit",
                });
            save_plot_xy(
                metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
                metadata.stack | stdv::transform([](auto& s) { return noa::rad2deg(s.phase_shift); }),
                options.output_directory / "phase_shift_fit.txt", {
                    .title = "Time-resolved phase_shift",
                    .x_name = "Image index (as saved in the stack)",
                    .y_name = "Phase-shift (degrees)",
                    .label = "Refine fit",
                });
            save_plot_xy(
                metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
                metadata.stack | stdv::transform([](auto& s) { return s.defocus.astigmatism; }),
                options.output_directory / "astigmatism_value_fit.txt", {
                    .title = "Tilt-resolved astigmatism",
                    .x_name = "Image index (as saved in the stack)",
                    .y_name = "Astigmatism (μm)",
                    .label = "Refine fit",
                });
            save_plot_xy(
                metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
                metadata.stack | stdv::transform([](auto& s) { return noa::rad2deg(s.defocus.angle); }),
                options.output_directory / "astigmatism_angle_fit.txt", {
                    .title = "Tilt-resolved astigmatism",
                    .x_name = "Image index (as saved in the stack)",
                    .y_name = "Astigmatism angle (degrees)",
                    .label = "Refine fit",
                });
        }
    };
}

namespace qn::ctf {
    void refine_fit(
        Metadata& metadata,
        FitRefineState& state,
        const Grid& grid,
        const Patches& patches,
        const FitRefineOptions& options
    ) {
        auto timer = Logger::info_scope_time("Refine CTF fitting");

        const auto runner = Runner{
            .metadata = metadata,
            .state = state,
            .grid = grid,
            .patches = patches,
            .options = options,
        };

        if (options.full_fit) {
            // Reset the splines.
            for (auto& s: state.phase_shift.span_1d())       s = metadata.stack[0].phase_shift; // radians
            for (auto& s: state.astigmatism_value.span_1d()) s = 0.;
            for (auto& s: state.astigmatism_angle.span_1d()) s = noa::deg2rad(45.);

            runner(NLOPT_LD_LBFGS, 30, true, {
                .phase_shift = options.fit_phase_shift ? Vec{0., noa::deg2rad(120.)} : Vec{0., 0.},
                .defocus = Vec{-1.5, 1.5},
            });
            runner(NLOPT_LD_LBFGS, 30, not options.fit_astigmatism, {
                .rotation = options.fit_rotation ? deg2rad(Vec{-10., 10.}) : Vec{0., 0.},
                .tilt = options.fit_tilt ? deg2rad(Vec{-30., 30.}) : Vec{0., 0.},
                .pitch = options.fit_pitch ? deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
                .phase_shift = options.fit_phase_shift ? Vec{0., noa::deg2rad(120.)} : Vec{0., 0.},
                .defocus = Vec{-1., 1.},
            });

            if (not options.fit_astigmatism)
                return;

            // Fit the astigmatism.
            runner(NLOPT_GD_STOGO, 75, false, {
                .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
                .defocus = Vec{-1., 1.},
                .astigmatism_value = {-0.6, 0.6},
                .astigmatism_angle = {-noa::deg2rad(45.), noa::deg2rad(45.)},
            });
            runner(NLOPT_LD_LBFGS, 50, true, {
                .rotation = options.fit_rotation ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
                .tilt = options.fit_tilt ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
                .pitch = options.fit_pitch ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
                .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
                .defocus = Vec{-2.5, 2.5},
                .astigmatism_value = {-0.4, 0.4},
                .astigmatism_angle = {-noa::deg2rad(45.), noa::deg2rad(45.)},
            });

            // Return early if no astigmatism.
            if (not metadata.stack.has_astigmatism(0.05)) {
                Logger::info("No significant astigmatism detected, stopping the refinement here.");
                return;
            }

            // Increase the tilt-resolution of the astigmatism.
            auto new_resolution = patches.n_images() / 4;
            if (noa::is_even(new_resolution))
                new_resolution += 1;
            Logger::info(
                "Astigmatism detected. Increasing the astigmatism tilt-resolution to {} points and continuing the refinement.",
                new_resolution
            );
            const auto tilt_range = metadata.stack.tilt_range();
            increase_tilt_spline_resolution(new_resolution, metadata.stack, tilt_range, state.astigmatism_value);
            increase_tilt_spline_resolution(new_resolution, metadata.stack, tilt_range, state.astigmatism_angle);
        } else {
            runner(NLOPT_GD_STOGO, 100, false, {
                .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
                .defocus = Vec{-2., 2.},
                .astigmatism_value = {-0.6, 0.6},
                .astigmatism_angle = {-noa::deg2rad(45.), noa::deg2rad(45.)},
            });

            runner(NLOPT_LD_LBFGS, 30, true, {
                .phase_shift = options.fit_phase_shift ? Vec{0., noa::deg2rad(120.)} : Vec{0., 0.},
                .defocus = Vec{-2., 2.},
            });
            runner(NLOPT_LD_LBFGS, 30, not options.fit_astigmatism, {
                .rotation = options.fit_rotation ? deg2rad(Vec{-10., 10.}) : Vec{0., 0.},
                .tilt = options.fit_tilt ? deg2rad(Vec{-30., 30.}) : Vec{0., 0.},
                .pitch = options.fit_pitch ? deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
                .phase_shift = options.fit_phase_shift ? Vec{0., noa::deg2rad(120.)} : Vec{0., 0.},
                .defocus = Vec{-2., 2.},
            });
        }

        // Refine.
        runner(NLOPT_LD_LBFGS, 50, false, {
            .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .defocus = Vec{-2., 2.},
            .astigmatism_value = {-0.3, 0.3},
            .astigmatism_angle = {-noa::deg2rad(25.), noa::deg2rad(25.)},
        });

        runner(NLOPT_LD_LBFGS, 50, true, {
            .rotation =  options.fit_rotation ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .tilt = options.fit_tilt ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .pitch = options.fit_pitch ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .defocus = Vec{-1., 1.},
            .astigmatism_value = {-0.2, 0.2},
            .astigmatism_angle = {-noa::deg2rad(25.), noa::deg2rad(25.)},
        });
    }
}
