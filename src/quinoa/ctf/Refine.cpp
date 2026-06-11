#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Optimizer.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/SplineGrid.hpp"
#include "quinoa/ctf/CTF.hpp"
#include "quinoa/ctf/Refine.hpp"

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

    template<typename T, typename I = isize>
    struct ReduceHeight {
        using value_type = T;
        using index_type = I;

        SpanContiguous<const Patches::value_type, 4, index_type> polar{}; // (n,p,h,w)
        SpanContiguous<const CTFAnisotropicPacked, 2, index_type> packed{}; // (cn,p)

        ns::CTFIsotropic<value_type> isotropic_ctf;
        ns::CTFAnisotropic<value_type> anisotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};
        value_type rho_range{};

        NOA_HD void operator()(index_type cn, index_type p, index_type h, index_type w, value_type& r0, value_type& r1) {
            const auto phi = static_cast<value_type>(h) * phi_step + phi_start; // radians
            const auto rho = static_cast<value_type>(w) * rho_step + rho_start; // fftfreq

            // Get the target phase.
            const auto& [defocus, astigmatism, angle, phase_shift] = packed(cn, p);
            isotropic_ctf.set_defocus(defocus);
            isotropic_ctf.set_phase_shift(phase_shift);
            const auto phase = isotropic_ctf.phase_at(rho);

            // Get the corresponding fftfreq within the astigmatic field.
            anisotropic_ctf.set_defocus({defocus, astigmatism, angle});
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            const auto fftfreq = isotropic_ctf.fftfreq_at(phase);
            if (not fftfreq)
                return;

            // Scale back to unnormalized frequency.
            const auto width = polar.shape().width();
            const auto frequency = static_cast<value_type>(width - 1) * (*fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = static_cast<value_type>(frequency - floored);
            const auto index = static_cast<index_type>(floored);

            value_type v0{}, w0{}, v1{}, w1{};
            if (index >= 0 and index < width) {
                v0 = static_cast<value_type>(polar(cn % polar.shape()[0], p, h, index));
                w0 = 1;
            }
            if (index + 1 >= 0 and index + 1 < width) {
                v1 = static_cast<value_type>(polar(cn % polar.shape()[0], p, h, index + 1));
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
    struct ScorePatch {
        using value_type = T;
        struct reduce_type {
            value_type sum_lhs{};
            value_type sum_rhs{};
            value_type sum_lhs_lhs{};
            value_type sum_rhs_rhs{};
            value_type sum_lhs_rhs{};

            NOA_HD void add(value_type lhs, value_type rhs) {
                sum_lhs += lhs;
                sum_rhs += rhs;
                sum_lhs_lhs += lhs * lhs;
                sum_rhs_rhs += rhs * rhs;
                sum_lhs_rhs += lhs * rhs;
            }
            NOA_HD void join(const reduce_type& reduced) {
                sum_lhs += reduced.sum_lhs;
                sum_rhs += reduced.sum_rhs;
                sum_lhs_lhs += reduced.sum_lhs_lhs;
                sum_rhs_rhs += reduced.sum_rhs_rhs;
                sum_lhs_rhs += reduced.sum_lhs_rhs;
            }
            [[nodiscard]] NOA_HD auto zncc(nt::integer auto n) const -> value_type {
                const auto count = static_cast<value_type>(n);
                const auto denominator_lhs = sum_lhs_lhs - sum_lhs * sum_lhs / count;
                const auto denominator_rhs = sum_rhs_rhs - sum_rhs * sum_rhs / count;
                auto denominator = denominator_lhs * denominator_rhs;
                if (denominator <= 0)
                    return 0;
                const auto numerator = sum_lhs_rhs - sum_lhs * sum_rhs / count;
                return numerator / noa::sqrt(denominator);
            }
        };
    };

    template<typename T, typename I = isize>
    struct ScorePatch1D {
        using value_type = ScorePatch<T>::value_type;
        using reduce_type = ScorePatch<T>::reduce_type;
        using index_type = I;

        SpanContiguous<const value_type, 4, index_type> patch_spectra{}; // (c,n,p,w)
        SpanContiguous<const CTFIsotropicPacked, 3, index_type> patch_ctfs{}; // (c,n,p)
        SpanContiguous<const value_type, 2, index_type> image_baseline{}; // (n,w)
        SpanContiguous<const value_type, 3, index_type> image_thickness_modulation{}; // (c,n,w)

        ns::CTFIsotropic<value_type> isotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};

        NOA_HD void operator()(index_type c, index_type n, index_type p, index_type w, reduce_type& reduced) {
            auto rho = static_cast<value_type>(w) * rho_step + rho_start; // fftfreq

            // Set up the CTF for the current patch.
            const auto& [defocus, phase_shift] = patch_ctfs(c, n, p);
            isotropic_ctf.set_defocus(defocus);
            isotropic_ctf.set_phase_shift(phase_shift);

            // Get the CTF at the current frequency.
            auto lhs = isotropic_ctf.value_at(rho);
            lhs *= lhs;
            auto envelope = isotropic_ctf.envelope_at(rho);
            envelope *= envelope;
            lhs -= envelope / 2; // [0,1] -> [-0.5, 0.5]
            lhs *= static_cast<value_type>(image_thickness_modulation(c, n, w));

            // Get the baseline-subtracted (aka zero-centered) spectrum.
            auto rhs = patch_spectra(c, n, p, w);
            rhs -= static_cast<value_type>(image_baseline(n, w)); // baseline is already sampled

            reduced.add(lhs, rhs);
        }

        NOA_HD static void join(const reduce_type& reduced, reduce_type& joined) {
            joined.join(reduced);
        }

        using remove_default_post = bool;
        NOA_HD void post(const reduce_type& joined, value_type& zncc) {
            zncc = joined.zncc(image_baseline.shape().width());
        }
    };

    template<typename T, typename I = isize>
    struct ScorePatch2D {
        using value_type = ScorePatch<T>::value_type;
        using reduce_type = ScorePatch<T>::reduce_type;
        using index_type = I;

        SpanContiguous<const Patches::value_type, 4, index_type> patch_spectra{}; // (n,p,h,w)
        SpanContiguous<const CTFAnisotropicPacked, 3, index_type> patch_ctfs{}; // (c,n,p)
        SpanContiguous<const value_type, 2, index_type> image_baseline{}; // (n,w)
        SpanContiguous<const value_type, 3, index_type> image_thickness_modulation{}; // (c,n,w)

        ns::CTFAnisotropic<value_type> anisotropic_ctf;
        ns::CTFIsotropic<value_type> isotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};

        NOA_HD void operator()(index_type c, index_type n, index_type p, index_type hw, reduce_type& reduced) {
            const auto [h, w] = noa::offset2index(hw, patch_spectra.shape().width());
            auto phi = static_cast<value_type>(h) * phi_step + phi_start; // radians
            auto rho = static_cast<value_type>(w) * rho_step + rho_start; // fftfreq

            // Set up the CTF for the current patch.
            const auto& [defocus, astigmatism, angle, phase_shift] = patch_ctfs(c, n, p);
            anisotropic_ctf.set_defocus({defocus, astigmatism, angle});
            auto defocus_at_phi = anisotropic_ctf.defocus_at(phi);
            isotropic_ctf.set_defocus(defocus_at_phi);
            isotropic_ctf.set_phase_shift(phase_shift);

            // Get the CTF at the current frequency.
            auto lhs = isotropic_ctf.value_at(rho);
            lhs *= lhs;
            auto envelope = isotropic_ctf.envelope_at(rho);
            envelope *= envelope;
            lhs -= envelope / 2; // [0,1] -> [-0.5, 0.5]
            lhs *= static_cast<value_type>(image_thickness_modulation(c, n, w));

            // Get the baseline-subtracted (aka zero-centered) spectrum.
            auto rhs = static_cast<value_type>(patch_spectra(n, p, h, w));
            rhs -= static_cast<value_type>(image_baseline(n, w)); // baseline is already sampled

            reduced.add(lhs, rhs);
        }

        NOA_HD static void join(const reduce_type& reduced, reduce_type& joined) {
            joined.join(reduced);
        }

        using remove_default_post = bool;
        NOA_HD void post(const reduce_type& joined, value_type& zncc) {
            zncc = joined.zncc(patch_spectra.shape().height() * patch_spectra.shape().width());
        }
    };

    template<typename T, typename I = isize>
    struct ReducePatchToImage {
        using value_type = T;
        using index_type = I;

        SpanContiguous<const value_type, 3, index_type> patch_spectra{}; // (n,p,w)
        SpanContiguous<const f32, 2, index_type> patch_defoci{}; // (n,p)
        SpanContiguous<const CTFIsotropicPacked, 1, index_type> image_defoci{}; // (n)

        ns::CTFIsotropic<value_type> isotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};
        value_type rho_range{};

        NOA_HD void operator()(index_type n, index_type p, index_type w, value_type& r0, value_type& r1) {
            const auto rho = static_cast<value_type>(w) * rho_step + rho_start; // fftfreq

            const auto& [image_defocus, phase_shift] = image_defoci[n];
            const auto patch_defocus = patch_defoci(n, p);

            // Get the target phase.
            isotropic_ctf.set_defocus(image_defocus);
            isotropic_ctf.set_phase_shift(phase_shift);
            const auto phase = isotropic_ctf.phase_at(rho);

            // Get the corresponding fftfreq within the patch.
            isotropic_ctf.set_defocus(patch_defocus);
            const auto fftfreq = isotropic_ctf.fftfreq_at(phase);
            if (not fftfreq)
                return;

            // Scale back to unnormalized frequency.
            const auto width = patch_spectra.shape().width();
            const auto frequency = static_cast<value_type>(width - 1) * (*fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = static_cast<value_type>(frequency - floored);
            const auto index = static_cast<index_type>(floored);

            value_type v0{}, w0{}, v1{}, w1{};
            if (index >= 0 and index < width) {
                v0 = static_cast<value_type>(patch_spectra(n, p, index));
                w0 = 1;
            }
            if (index + 1 >= 0 and index + 1 < width) {
                v1 = static_cast<value_type>(patch_spectra(n, p, index + 1));
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

    template<typename T, typename I = isize>
    struct ReducePolarPatchToImage {
        using value_type = T;
        using index_type = I;

        SpanContiguous<const Patches::value_type, 4, index_type> polar{}; // (n,p,h,w)
        SpanContiguous<const CTFAnisotropicPacked, 1, index_type> ctf_images_packed{}; // (c*n)
        SpanContiguous<const value_type, 1, index_type> defocus_patches{}; // (c*n*p)

        ns::CTFIsotropic<value_type> isotropic_ctf;
        ns::CTFAnisotropic<value_type> anisotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};
        value_type rho_range{};

        NOA_HD void operator()(index_type cn, index_type p, index_type r, index_type c, value_type& r0, value_type& r1) {
            auto phi = static_cast<value_type>(r) * phi_step + phi_start; // radians
            auto rho = static_cast<value_type>(c) * rho_step + rho_start; // fftfreq

            const auto& [n_images, n_patches, height, width] = polar.shape();
            const auto& image = ctf_images_packed[cn];
            const auto& patch_defocus = defocus_patches[cn * n_patches + p];

            // Get the target phase.
            anisotropic_ctf.set_defocus({image.defocus, image.astigmatism, image.angle});
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            isotropic_ctf.set_phase_shift(image.phase_shift);
            const auto phase = isotropic_ctf.phase_at(rho);

            // Get the corresponding fftfreq within the patch.
            anisotropic_ctf.set_defocus({patch_defocus, image.astigmatism, image.angle});
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            const auto fftfreq = isotropic_ctf.fftfreq_at(phase);
            if (not fftfreq)
                return;

            // Scale back to unnormalized frequency.
            const auto frequency = static_cast<value_type>(width - 1) * (*fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = static_cast<value_type>(frequency - floored);
            const auto index = static_cast<index_type>(floored);

            value_type v0{}, w0{}, v1{}, w1{};
            if (index >= 0 and index < width) {
                v0 = static_cast<value_type>(polar(cn % n_images, p, r, index));
                w0 = 1;
            }
            if (index + 1 >= 0 and index + 1 < width) {
                v1 = static_cast<value_type>(polar(cn % n_images, p, r, index + 1));
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

    struct SimulateCTF2 {
        SpanContiguous<f32, 3, i32> output; // (n,h,w)
        SpanContiguous<const CTFAnisotropicPacked, 1, i32> ctfs; // (n)
        SpanContiguous<const f32, 2, i32> thickness_modulation; // (n,w)
        ns::CTFAnisotropic<f32> ctf;

        f32 phi_start{};
        f32 phi_step{};
        f32 rho_start{};
        f32 rho_step{};

        NOA_HD void operator()(i32 i, i32 h, i32 w) {
            const auto phi = static_cast<f32>(h) * phi_step + phi_start; // radians
            const auto rho = static_cast<f32>(w) * rho_step + rho_start; // fftfreq
            const auto fftfreq = rho * noa::sincos(phi);

            const auto& [defocus, astigmatism, angle, phase_shift] = ctfs[i];
            ctf.set_defocus({defocus, astigmatism, angle});
            ctf.set_phase_shift(phase_shift);
            auto value = ctf.value_at(fftfreq);
            value *= value;
            auto envelope = ctf.envelope_at(fftfreq);
            envelope *= envelope;
            value -= envelope / 2;
            output(i, h, w) = value * thickness_modulation(i, w);
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
        f64 m_initial_thickness{};
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
        [[nodiscard]] auto angle_offsets() const noexcept {
            return Vec{
                m_parameters[ROTATION].is_fitted() ? m_buffer[m_parameters[ROTATION].offset()] : 0,
                m_parameters[TILT].is_fitted() ? m_buffer[m_parameters[TILT].offset()] : 0,
                m_parameters[PITCH].is_fitted() ? m_buffer[m_parameters[PITCH].offset()] : 0
            };
        }

        [[nodiscard]] auto thickness() const noexcept {
            return m_parameters[THICKNESS].is_fitted() ? m_buffer[m_parameters[THICKNESS].offset()] : m_initial_thickness;
        }

        [[nodiscard]] auto set_thickness(f64 thickness) noexcept {
            if (m_parameters[THICKNESS].is_fitted())
                m_buffer[m_parameters[THICKNESS].offset()] = thickness;
            else
                m_initial_thickness = thickness;
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

    public:
        Parameters() = default;

        Parameters(
            const Metadata& metadata,
            const SplineGridCubic<f64, 1>& phase_shift,
            const SplineGridCubic<f64, 1>& astigmatism_value,
            const SplineGridCubic<f64, 1>& astigmatism_angle,
            const RefineFittingParameters<Vec<f64, 2>>& relative_bounds
        ) {
            // Set the parameter sizes.
            m_parameters[ROTATION].m_ssize = 1;
            m_parameters[TILT].m_ssize = 1;
            m_parameters[PITCH].m_ssize = 1;
            m_parameters[THICKNESS].m_ssize = 1;
            m_parameters[DEFOCUS].m_ssize = metadata.stack.ssize();
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
            m_initial_defocus.resize(metadata.stack.size());
            m_initial_phase_shift.resize(phase_shift.size());
            m_initial_astigmatism_value.resize(astigmatism_value.size());
            m_initial_astigmatism_angle.resize(astigmatism_angle.size());

            // Initialize the values, whether they're the default or fitted values.
            set_thickness(metadata.sample.thickness * 1e-3); // nm->um
            for (auto&& [defocus, image]: noa::zip(defoci(), metadata.stack)) defocus = image.defocus.value;
            for (auto&& [o, i]: noa::zip(this->phase_shift().span, phase_shift.span)) o = i;
            for (auto&& [o, i]: noa::zip(this->astigmatism_value().span, astigmatism_value.span)) o = i;
            for (auto&& [o, i]: noa::zip(this->astigmatism_angle().span, astigmatism_angle.span)) o = i;

            set_relative_bounds(relative_bounds);
        }

        void set_relative_bounds(const RefineFittingParameters<Vec<f64, 2>>& relative_bounds) {
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
            set_buffer(m_parameters[PHASE_SHIFT], relative_bounds.phase_shift, 0., noa::deg2rad(130.));
            set_buffer(m_parameters[DEFOCUS], relative_bounds.defocus, 0.5);
            set_buffer(m_parameters[ASTIGMATISM_VALUE], relative_bounds.astigmatism_value);
            set_buffer(m_parameters[ASTIGMATISM_ANGLE], relative_bounds.astigmatism_angle);
        }

        void set_abs_tolerance(const RefineFittingParameters<f64>& abs_tolerance) {
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

        void set_deltas(const RefineFittingParameters<f64>& deltas) {
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
        const Metadata::Stack& m_metadata;
        const Grid& m_grid;
        const Patches& m_patches;

        // Optimizer.
        Parameters m_parameters{};
        Memoizer m_memoizer{};
        SpanContiguous<Vec<f64, 2>> m_fitting_ranges{};
        isize m_n_channels;
        std::vector<f64> m_parameters_buffer;
        Array<f64> m_znccs; // (c,1,1,n)

        // Splines.
        Vec<f64, 2> m_time_range{};
        Vec<f64, 2> m_tilt_range{};
        Array<f64> m_phase_shift_weights{};
        Array<f64> m_astigmatism_value_weights{};
        Array<f64> m_astigmatism_angle_weights{};

        // CTFs.
        CTFIsotropic64 m_ctf;
        Array<CTFAnisotropicPacked> m_anisotropic_ctf_patches;
        Array<CTFAnisotropicPacked> m_anisotropic_ctf_images;
        Array<CTFIsotropicPacked> m_isotropic_ctf_patches;
        Array<CTFIsotropicPacked> m_isotropic_ctf_images;
        Array<f32> m_defocus_patches;

        // Reduction operators.
        ReduceHeight<f32, i32> m_reduce_height;
        ScorePatch1D<f32, i32> m_score_patch_1d;
        ScorePatch2D<f32, i32> m_score_patch_2d;
        Array<f32> m_reduced_cnpw;
        Array<f32> m_reduced_cnp1;
        bool m_is_reduce_height_done{};

        // Thickness-aware CTF.
        Array<f32> m_thickness_modulations; // (c,n,1,w)
        bool m_is_thickness_sampled{};

        // Spectrum baseline.
        Array<f32> m_baselines_sampled; // (n,1,1,w)

    public:
        Fitter(
            const Metadata& metadata,
            const Grid& grid,
            const Patches& patches,
            const SpanContiguous<Vec<f64, 2>>& fitting_ranges,
            const SplineGridCubic<f64, 1>& phase_shift,
            const SplineGridCubic<f64, 1>& astigmatism_value,
            const SplineGridCubic<f64, 1>& astigmatism_angle,
            const RefineFittingParameters<Vec<f64, 2>>& relative_bounds = {}
        ) :
            m_metadata(metadata.stack),
            m_grid(grid),
            m_patches(patches),
            m_fitting_ranges(fitting_ranges)
        {
            // Initialize and configure the optimization parameters.
            m_parameters = Parameters(metadata, phase_shift, astigmatism_value, astigmatism_angle, relative_bounds);
            m_parameters.set_abs_tolerance({
                .rotation = noa::deg2rad(0.01),
                .tilt = noa::deg2rad(0.01),
                .pitch = noa::deg2rad(0.01),
                .thickness = 0.002,
                .phase_shift = noa::deg2rad(0.05),
                .defocus = 0.001,
                .astigmatism_value = 0.001,
                .astigmatism_angle = noa::deg2rad(0.1),
            });
            m_parameters.set_deltas({
                .rotation = noa::deg2rad(0.1),
                .tilt = noa::deg2rad(0.1),
                .pitch = noa::deg2rad(0.1),
                .thickness = 0.005,
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
            const auto device = m_patches.view().options().device;
            const auto options_pitched = ArrayOption{.device = device, .allocator = Allocator::PITCHED};
            const auto options_managed = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
            const auto options_pitched_managed = ArrayOption{.device = device, .allocator = Allocator::PITCHED_MANAGED};

            // Allocate for the CTFs. Everything needs to be dereferenceable.
            m_anisotropic_ctf_patches = Array<CTFAnisotropicPacked>({m_n_channels, n, p, 1}, options_managed);
            m_isotropic_ctf_patches =  Array<CTFIsotropicPacked>({m_n_channels, n, p, 1}, options_managed);
            m_anisotropic_ctf_images = Array<CTFAnisotropicPacked>({m_n_channels, n, 1, 1}, options_managed);
            m_isotropic_ctf_images = Array<CTFIsotropicPacked>({m_n_channels, n, 1, 1}, options_managed);
            m_defocus_patches = Array<f32>({1, n, p, 1}, options_managed);

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

            // Reduction operators and buffers for the scoring functions.
            m_reduced_cnpw = Array<f32>({m_n_channels, n, p, w}, options_pitched);
            m_reduced_cnp1 = Array<f32>({m_n_channels, n, p, 1}, options_managed);
            m_reduce_height = ReduceHeight<f32, i32>{
                .polar = m_patches.view().span_contiguous().as_index<i32>(), // (n,p,h,w)
                .packed = m_anisotropic_ctf_patches.span().reshape({-1, p, 1, 1}).filter(0, 1).as_contiguous().as_index<i32>(), // (cn,p)
                .isotropic_ctf = m_ctf.as<f32>(),
                .anisotropic_ctf = ns::CTFAnisotropic(m_ctf).as<f32>(),
                .phi_start = static_cast<f32>(m_patches.phi().start),
                .phi_step = static_cast<f32>(m_patches.phi_step()),
                .rho_start = static_cast<f32>(m_patches.rho().start),
                .rho_step = static_cast<f32>(m_patches.rho_step()),
                .rho_range = static_cast<f32>(m_patches.rho().stop - m_patches.rho().start), // assumes endpoint=true
            };
            m_score_patch_1d = ScorePatch1D<f32, i32>{
                .patch_spectra = m_reduced_cnpw.span_contiguous().as_index<i32>(), // (c,n,p,w)
                .patch_ctfs = m_isotropic_ctf_patches.span().filter(0, 1, 2).as_contiguous().as_index<i32>(), // (c,n,p)
                .image_baseline = m_baselines_sampled.span().filter(0, 3).as_contiguous().as_index<i32>(), // (n,w)
                .image_thickness_modulation = m_thickness_modulations.span().filter(0, 1, 3).as_contiguous().as_index<i32>(), // (c,n,w)
                .isotropic_ctf = m_reduce_height.isotropic_ctf,
                .phi_start = m_reduce_height.phi_start,
                .phi_step = m_reduce_height.phi_step,
                .rho_start = m_reduce_height.rho_start,
                .rho_step = m_reduce_height.rho_step,
            };
            m_score_patch_2d = ScorePatch2D<f32, i32>{
                .patch_spectra = m_patches.view().span_contiguous().as_index<i32>(), // (n,p,h,w)
                .patch_ctfs = m_anisotropic_ctf_patches.span().filter(0, 1, 2).as_contiguous().as_index<i32>(), // (c,n,p)
                .image_baseline = m_baselines_sampled.span().filter(0, 3).as_contiguous().as_index<i32>(), // (n,w)
                .image_thickness_modulation = m_thickness_modulations.span().filter(0, 1, 3).as_contiguous().as_index<i32>(), // (c,n,w)
                .anisotropic_ctf = m_reduce_height.anisotropic_ctf,
                .isotropic_ctf = m_reduce_height.isotropic_ctf,
                .phi_start = m_reduce_height.phi_start,
                .phi_step = m_reduce_height.phi_step,
                .rho_start = m_reduce_height.rho_start,
                .rho_step = m_reduce_height.rho_step,
            };
        }

        // Read the current parameters and update the CTF buffers for the given channel accordingly.
        void update_ctfs(isize channel) {
            const Vec<f64, 3> angle_offsets = m_parameters.angle_offsets();
            const SplineGridCubic<f64, 1> time_resolved_phase_shift = m_parameters.phase_shift();
            const SplineGridCubic<f64, 1> tilt_resolved_astigmatism_value = m_parameters.astigmatism_value();
            const SplineGridCubic<f64, 1> tilt_resolved_astigmatism_angle = m_parameters.astigmatism_angle();
            const SpanContiguous<f64> defoci = m_parameters.defoci();
            const f64 sample_thickness_um = m_parameters.thickness();

            const auto anisotropic_ctf_images = m_anisotropic_ctf_images.subregion(channel).span_1d();
            const auto isotropic_ctf_images = m_isotropic_ctf_images.subregion(channel).span_1d();
            for (isize i{}; i < m_patches.n_images(); ++i) {
                // Time-resolved phase-shift.
                const f64 itime = normalized_time(m_metadata[i]);
                const f64 phase_shift = time_resolved_phase_shift.interpolate_at(itime);

                // Tilt-resolved astigmatism.
                const f64 itilt = normalized_tilt(m_metadata[i]);
                const f64 slice_astigmatism_value = tilt_resolved_astigmatism_value.interpolate_at(itilt);
                const f64 slice_astigmatism_angle = tilt_resolved_astigmatism_angle.interpolate_at(itilt);

                // Set the defocus and phase-shift of the image CTF.
                anisotropic_ctf_images[i].defocus = static_cast<f32>(defoci[i]);
                anisotropic_ctf_images[i].astigmatism = static_cast<f32>(slice_astigmatism_value);
                anisotropic_ctf_images[i].angle = static_cast<f32>(slice_astigmatism_angle);
                anisotropic_ctf_images[i].phase_shift = static_cast<f32>(phase_shift);
                isotropic_ctf_images[i].defocus = anisotropic_ctf_images[i].defocus;
                isotropic_ctf_images[i].phase_shift = anisotropic_ctf_images[i].phase_shift;

                const auto anisotropic_ctf_patches = m_anisotropic_ctf_patches.subregion(channel, i).span_1d();
                const auto isotropic_ctf_patches = m_isotropic_ctf_patches.subregion(channel, i).span_1d();
                const auto defocus_patches = m_defocus_patches.subregion(0, i).span_1d();

                const auto image_spacing = Vec<f64, 2>::from_value(m_ctf.pixel_size());
                const auto image_angles = noa::deg2rad(m_metadata[i].angles) + angle_offsets;
                const auto patch_centers = m_grid.patches_centers();

                // Sample the thickness modulation for this image. If the thickness isn't
                // modeled, the modulation is a row of ones and does nothing.
                if (not m_is_thickness_sampled) {
                    ThicknessModulation<false>{
                        .wavelength = m_ctf.wavelength(),
                        .spacing = m_ctf.pixel_size(),
                        .thickness = effective_thickness(sample_thickness_um, noa::rad2deg(image_angles)) * 1e4, // um->ang
                    }.sample(
                        m_thickness_modulations.span().subregion(channel, i).as_1d(), m_patches.rho_vec()
                    );
                }

                // Update the CTFs of the patches belonging to the current image.
                for (isize j{}; j < m_patches.n_patches_per_image(); ++j) {
                    const auto patch_z_offset_um = m_grid.patch_z_offset(image_angles, image_spacing, patch_centers[j]);
                    const auto patch_defocus = defoci[i] - patch_z_offset_um;
                    anisotropic_ctf_patches[j].defocus = static_cast<f32>(patch_defocus);
                    anisotropic_ctf_patches[j].astigmatism = anisotropic_ctf_images[i].astigmatism;
                    anisotropic_ctf_patches[j].angle = anisotropic_ctf_images[i].angle;
                    anisotropic_ctf_patches[j].phase_shift = anisotropic_ctf_images[i].phase_shift;
                    isotropic_ctf_patches[j].defocus = static_cast<f32>(patch_defocus);
                    isotropic_ctf_patches[j].phase_shift = isotropic_ctf_images[i].phase_shift;
                    if (channel == 0)
                        defocus_patches[j] = isotropic_ctf_patches[j].defocus;
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

        void reduce_patch_height(bool first_channel_only = false) {
            const auto h = m_patches.view().shape().height();
            const auto& [c, n, p, w] = m_reduced_cnpw.shape();
            const auto actual_c = first_channel_only ? 1 : c;
            const auto reduced_cnpw = m_reduced_cnpw.view().subregion(Slice{0, actual_c});
            if (h == 1) {
                // The polar spectra are already reduced to 1d (astigmatism is ignored).
                // No reduction necessary, simply copy to the output buffer.
                auto broadcast = noa::broadcast(m_patches.view().reshape({1, n, p, w}), reduced_cnpw.shape());
                noa::ewise(broadcast, reduced_cnpw, noa::Copy{});
            } else {
                noa::reduce_axes_iwise( // (cn,p,h,w)->(cn,p,1,w)
                    Shape{actual_c * n, p, h, w}.as<i32>(), m_patches.view().device(), noa::wrap(f32{0}, f32{0}),
                    reduced_cnpw.reshape({actual_c * n, p, 1, w}), m_reduce_height
                );
            }
        }

        void zncc_no_astigmatism() {
            if (not m_is_reduce_height_done)
                reduce_patch_height();

            noa::reduce_axes_iwise( // (c,n,p,w)->(c,n,p,1)
                m_reduced_cnpw.shape().as<i32>(), m_reduced_cnpw.device(), ScorePatch1D<f32>::reduce_type{},
                m_reduced_cnp1.view(), m_score_patch_1d
            );

            const auto znccs_cnp = m_reduced_cnp1.view().eval().span().filter(0, 1, 2).as_contiguous(); // (c,n,p)
            const auto znccs_cn = m_znccs.span().filter(0, 3).as_contiguous(); // (c,w)
            for (isize c{}; c < znccs_cnp.shape()[0]; ++c) {
                for (isize n{}; n < znccs_cnp.shape()[1]; ++n) {
                    f64 zncc{};
                    for (isize p{}; p < znccs_cnp.shape()[2]; ++p)
                        zncc += static_cast<f64>(znccs_cnp(c, n, p));
                    znccs_cn(c, n) = zncc / static_cast<f64>(znccs_cnp.shape()[2]);
                }
            }
        }

        void zncc_astigmatism() {
            // Compute the per-patch ZNCCs.
            const auto reduced_cnp1 = m_reduced_cnp1.view();
            const auto [c, n, p] = reduced_cnp1.shape().pop_back();
            const auto [h, w] = m_patches.view().shape().filter(2, 3);
            noa::reduce_axes_iwise( // (c,n,p,hw)->(c,n,p,1)
                Shape{c, n, p, h * w}.as<i32>(), reduced_cnp1.device(), ScorePatch2D<f32>::reduce_type{},
                reduced_cnp1, m_score_patch_2d
            );

            // Recompose the per-image ZNCCs.
            const auto znccs_cnp = reduced_cnp1.eval().span().filter(0, 1, 2).as_contiguous(); // (c,n,p)
            const auto znccs_cn = m_znccs.span().filter(0, 3).as_contiguous(); // (c,w)
            for (isize i{}; i < c; ++i) {
                for (isize j{}; j < n; ++j) {
                    f64 zncc{};
                    for (isize k{}; k < p; ++k)
                        zncc += static_cast<f64>(znccs_cnp(i, j, k));
                    znccs_cn(i, j) = zncc / static_cast<f64>(p);
                }
            }
        }

        auto zncc() -> f64 {
            m_parameters[ASTIGMATISM_VALUE].is_fitted() ? zncc_astigmatism() : zncc_no_astigmatism();
            return simple_average(m_znccs.span().subregion(0).as_1d()); // TODO estimate Z-score instead?
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
                // Logger::trace("score={:.4f}, memoized=true", score);
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

            self.memoizer().record(parameters, score, gradients);


            // Logger::trace("score={:.4f}, angles={::+.3f}, defoci={::.2f}, g={::.4f}",
            //     score, noa::rad2deg(params.angle_offsets()), params[DEFOCUS].span(),
            //     SpanContiguous(gradients + params[DEFOCUS].offset(), params[DEFOCUS].size()));
            //
            // if (params[TILT].is_fitted()) {
            //     Logger::trace("score={:.4f}, angles={::+.3f}, g={::.4f}",
            //         score, noa::rad2deg(params.angle_offsets()), SpanContiguous(gradients, 4));
            // }
            // if (params[PHASE_SHIFT].is_fitted()) {
            //     Logger::trace(
            //         "phase_shift={::.6f}, phase_shift_grad={::.6f}",
            //         params[PHASE_SHIFT].span(),
            //         SpanContiguous(gradients + params[PHASE_SHIFT].offset(), params[PHASE_SHIFT].size())
            //     );
            // }
            // if (params[THICKNESS].is_fitted()) {
            //     Logger::trace(
            //         "score={:.6f}, thickness={::.6f}, thickness_grad={::.6f}",
            //         score, params[THICKNESS].span(),
            //         SpanContiguous(gradients + params[THICKNESS].offset(), params[THICKNESS].size())
            //     );
            // }
            // if (params[ASTIGMATISM_VALUE].is_fitted()) {
            //     Logger::trace(
            //         "score={}, astig={::.2f}, astig_grad={::.4f}",
            //         score, params[ASTIGMATISM_VALUE].span(),
            //         SpanContiguous(gradients + params[ASTIGMATISM_VALUE].offset(), params[ASTIGMATISM_VALUE].size())
            //     );
            // }
            return score;
        }

        auto compute_image_spectra() {
            // NOTE: update_ctfs(0) should have been called by this point.

            // Equiphase average the polar height of each patch.
            reduce_patch_height(true); // (n,p,h,w)->(n,p,1,w)
            const auto reduced_1npw = m_reduced_cnpw.view().subregion(0);

            const auto device = m_reduced_cnpw.device();
            const auto patches_shape = reduced_1npw.shape();
            auto reduced_1n1w = Array<f32>(patches_shape.set<2>(1), {
                .device = device, .allocator = Allocator::PITCHED_MANAGED
            });

            // Equiphase average per-patches to per-image.
            auto reduce_isotropic_depth = ReducePatchToImage<f32, i32>{
                .patch_spectra = reduced_1npw.span().filter(1, 2, 3).as_contiguous().as_index<i32>(), // (n,p,w)
                .patch_defoci = m_defocus_patches.span().filter(1, 2).as_contiguous().as_index<i32>(), // (n,p)
                .image_defoci =  m_isotropic_ctf_images.subregion(0).span_1d().as_index<i32>(), // (n)
                .isotropic_ctf = m_reduce_height.isotropic_ctf,
                .phi_start = m_reduce_height.phi_start,
                .phi_step = m_reduce_height.phi_step,
                .rho_start = m_reduce_height.rho_start,
                .rho_step = m_reduce_height.rho_step,
                .rho_range = m_reduce_height.rho_range,
            };
            noa::reduce_axes_iwise( // (n,p,w)->(n,1,w)
                patches_shape.pop_front().as<i32>(), device, noa::wrap(f32{0}, f32{0}),
                reduced_1n1w, reduce_isotropic_depth
            );
            return reduced_1n1w;
        }

        auto setup_fitting_ranges_and_baselines_(i32 extra_peaks_to_append = 2) {
            // Reset caches (not necessary, but in case this is called twice).
            m_memoizer.reset_cache();
            m_is_reduce_height_done = false;
            m_is_thickness_sampled = false;

            // Compute the per-image spectra.
            update_ctfs(0);
            const auto spectrum_1n1w = compute_image_spectra().eval();
            const auto spectrum_n = spectrum_1n1w.span().filter(1, 3).as_contiguous(); // (n,w)

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
                auto fitting_range = baseline.fit_and_tune_fitting_range(
                    spectrum_n[i], m_patches.rho_vec(), ctf_images[i], {
                        .threshold = 1.2,
                        .keep_first_nth_peaks = 2,

                        // In the case of strong astigmatism, the initial spectrum may have only a few Thon-rings.
                        // By looking ahead, we give the optimizer more opportunities to improve the spectrum.
                        .n_extra_peaks_to_append = extra_peaks_to_append,
                        .n_recoveries_allowed = 1,
                        .maximum_n_consecutive_bad_peaks = 1,

                        // The thickness can have a huge impact on the fitting range. A wrong estimate will
                        // remove everything after the first node (+extra peaks to append). Of course, if the
                        // resolution range of the spectra is too low, the thickness may have no effect.
                        .thickness_um = effective_thickness(m_parameters.thickness(), m_metadata[i].angles),
                    });

                // When fitting the thickness, include the entire right side of the spectrum
                // to not be affected by an initial wrong thickness estimate.
                if (m_parameters[THICKNESS].is_fitted())
                    fitting_range[1] = m_patches.rho_vec()[1];

                m_fitting_ranges[i] = fitting_range;
                baseline.sample(m_baselines_sampled.subregion(i).span_1d(), m_patches.rho_vec());
            }

            return Pair{spectrum_1n1w.permute({1, 0, 2, 3}), ctf_images_buffer};
        }

        auto fit(nlopt_algorithm algorithm, i32 max_number_of_evaluations) -> f64 {
            setup_fitting_ranges_and_baselines_();

            if (m_parameters.ssize() == 0)
                return zncc();

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

        void save_synthetic_image_polar_spectra(const Path& output_directory) const {
            // NOTE: setup_fitting_ranges_and_baselines_ should have been called by this point.

            const auto device = m_patches.view().device();
            const auto [n, p, h, w] = m_patches.view().shape(); // (n,p,h,w)

            // The polar range is 180 degrees, so expand it to 360 degrees with one quadrant being the simulated CTF.
            auto synthetic_polar_spectra = Array<f32>({n, 1, h * 2, w}, {
                .device = device, .allocator = Allocator::PITCHED_MANAGED
            });

            // Compute the per-image spectra.
            const auto first_two_quadrants = synthetic_polar_spectra.view().subregion(Ellipsis{}, Slice{0, h}, Full{});
            noa::reduce_axes_iwise( // (n,p,h,w)->(n,1,h,w)
                Shape4{n, p, h, w}.as<i32>(), device, noa::wrap(f32{0}, f32{0}),
                first_two_quadrants, ReducePolarPatchToImage<f32, i32>{
                    .polar = m_patches.view().span_contiguous().as_index<i32>(), // (n,p,h,w)
                    .ctf_images_packed = m_anisotropic_ctf_images.span_1d().as_index<i32>(), // (c*n)
                    .defocus_patches = m_defocus_patches.span_1d().as_index<i32>(), // (c*n*p)
                    .isotropic_ctf = m_reduce_height.isotropic_ctf,
                    .anisotropic_ctf = m_reduce_height.anisotropic_ctf,
                    .phi_start = m_reduce_height.phi_start,
                    .phi_step = m_reduce_height.phi_step,
                    .rho_start = m_reduce_height.rho_start,
                    .rho_step = m_reduce_height.rho_step,
                    .rho_range = m_reduce_height.rho_range,
                });

            // Simulate the CTF2 in the third quadrant.
            const auto third_quadrant = synthetic_polar_spectra.view().subregion(Ellipsis{}, Offset{h}, Full{});
            noa::iwise(Shape3{n, h / 2, w}.as<i32>(), device, SimulateCTF2{
                .output = third_quadrant.span().filter(0, 2, 3).as_contiguous().as_index<i32>(),
                .ctfs = m_anisotropic_ctf_images.span_1d().as_index<i32>(),
                .thickness_modulation = m_thickness_modulations.span_contiguous().subregion(0).as<const f32, 2, i32>(),
                .ctf = CTFAnisotropic64(m_ctf).as<f32>(),
                .phi_start = m_reduce_height.phi_start + m_reduce_height.phi_step * static_cast<f32>(h),
                .phi_step = m_reduce_height.phi_step,
                .rho_start = m_reduce_height.rho_start,
                .rho_step = m_reduce_height.rho_step,
            });

            // Fill the last quadrant.
            noa::ewise(
                synthetic_polar_spectra.view().subregion(Ellipsis{}, Slice{h / 2, h}, Full{}),
                synthetic_polar_spectra.view().subregion(Ellipsis{}, Offset{h + h / 2}, Full{}),
                noa::Copy{}
            );

            noa::normalize(
                synthetic_polar_spectra, synthetic_polar_spectra,
                ReduceAxes{.width = true}, {.mode = noa::Norm::MEAN_L2}
            );

            auto filename = output_directory / "synthetic_image_polar_spectra.mrc";
            noa::write_image(synthetic_polar_spectra, filename, {.dtype = "f16"});
            Logger::trace("{} saved", filename);
        }

        void save_stack_spectrum(
            SpanContiguous<const f32, 2> spectrum_bs_n,
            SpanContiguous<const CTFIsotropic64, 1> ctf_n,
            const Path& output_directory
        ) const {
            // Prepare the stack EPA.
            const auto buffer = noa::zeros<f32>({3, 1, 1, spectrum_bs_n.shape().width()});
            const auto spectrum_rescaled = buffer.view().subregion(0);
            const auto spectrum_average = buffer.view().subregion(1);
            const auto spectrum_weights = buffer.view().subregion(2);

            // Compute the target CTF for the equiphase-average.
            const auto average_ctf = [&] {
                f64 average_defocus{};
                f64 min_phase_shift{};
                for (auto& ctf_image: ctf_n) {
                    average_defocus += ctf_image.defocus();
                    min_phase_shift = std::min(min_phase_shift, ctf_image.phase_shift());
                }
                auto ctf = m_ctf;
                ctf.set_defocus(average_defocus / static_cast<f64>(m_patches.n_images()));
                ctf.set_phase_shift(min_phase_shift);
                return ctf;
            }();

            auto baseline = Baseline{};
            const auto fftfreq_step = m_patches.rho_step();
            for (isize i{}; i < m_patches.n_images(); ++i) {
                // Rescale to the target CTF.
                const auto ctf_i = ctf_n[i];
                nx::phase_spectra(
                    View(spectrum_bs_n[i]), m_patches.rho(), ctf_i,
                    spectrum_rescaled, m_patches.rho(), average_ctf
                );
                auto fitting_range = m_fitting_ranges[i];
                for (auto j: noa::irange(2)) {
                    auto phase = ctf_i.phase_at(fitting_range[j]);
                    auto fftfreq = average_ctf.fftfreq_at(phase);
                    if (not fftfreq)
                        fftfreq = fitting_range[1];
                    fitting_range[j] = *fftfreq;
                }

                // Sometimes the background is not perfectly subtracted, and this shows in the average spectrum.
                // This is mostly for cases with a wide fftfreq range, or including the 3.7A bump from amorphous ice.
                // It shouldn't affect the fitting significantly, so I don't think it's worth tweaking the background
                // fitting for this, but to make the EPA look great, fit and subtract the background again.
                baseline.fit(spectrum_rescaled.span_1d(), m_patches.rho_vec(), average_ctf);
                baseline.subtract(spectrum_rescaled, spectrum_rescaled, m_patches.rho_vec());

                // To fuse spectra with different thicknesses, we need to correct for the thickness modulation.
                // We could multiply the spectrum with the thickness modulation curve, but that would
                // downweight regions near and at the node (and create visible artifacts from the flipping
                // if the baseline isn't perfectly centered on zero). Instead, skip these regions entirely
                // and flip the zero-centered spectrum (oscillations) when the curve goes negative.
                const auto thickness_modulation = ThicknessModulation<true>{
                    .wavelength = ctf_i.wavelength(),
                    .spacing = ctf_i.pixel_size(),
                    .thickness = effective_thickness(m_parameters.thickness() * 1e4, m_metadata[i].angles), // um->angstrom
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

        void plot_diagnostics(const Path& output_directory) {
            auto [spectrum_n11w, ctf_n] = setup_fitting_ranges_and_baselines_(0);
            auto spectrum_nw = spectrum_n11w.span_contiguous<const f32, 2>();

            save_plot_xy(m_patches.rho(), spectrum_nw, output_directory / "refined_spectra.txt", {.title = "Per-image spectra", .label = "spectrum"});
            save_plot_xy(m_patches.rho(), m_baselines_sampled, output_directory / "refined_spectra.txt", {.label = "background"});

            // Subtract the baseline.
            for (isize i{}; i < spectrum_nw.shape()[0]; ++i) {
                for (auto&& [s, b]: noa::zip(
                    spectrum_n11w.span().subregion(i).as_1d(),
                    m_baselines_sampled.span().subregion(i).as_1d())
                ) {
                    s -= b;
                }
            }

            // Save per-image EPA with CTFs.
            save_plot_ctf_fit(
                m_patches.rho(), spectrum_nw, ctf_n.span_1d(),
                m_thickness_modulations.span_contiguous().subregion(0).as<const f32, 2>(),
                output_directory / "refined_fitting.txt", {
                    .title = "Per-image refined spectra",
                });

            save_synthetic_image_polar_spectra(output_directory);
            save_stack_spectrum(spectrum_nw, ctf_n.span_1d(), output_directory);

            // Per-image fitting ranges.
            save_plot_xy(
                m_metadata | stdv::transform([](auto& s) { return s.index_file; }),
                m_fitting_ranges | stdv::transform([&](const auto& v) {
                    return fftfreq_to_resolution(m_ctf.pixel_size(), v[1]);
                }),
                output_directory / "fitting_ranges.txt", {
                    .title = "Resolution cutoff for CTF fitting",
                    .x_name = "Image index (as saved in the file)",
                    .y_name = "Resolution (A)",
                    .label = "Refine fitting",
                });
        }

        void update_metadata_and_state(
            Metadata& metadata,
            SplineGridCubic<f64, 1> phase_shift,
            SplineGridCubic<f64, 1> astigmatism_value,
            SplineGridCubic<f64, 1> astigmatism_angle,
            Vec<f64, 3>& final_angle_offsets
        ) {
            const auto values = m_parameters.astigmatism_value().span;
            const auto angles = m_parameters.astigmatism_angle().span;
            constexpr auto RAD_180 = noa::deg2rad(180.);
            constexpr auto RAD_90 = noa::deg2rad(90.);

            // For simplicity, only deal with positive astigmatism value.
            for (auto&& [value, angle]: noa::zip(values, angles)) {
                if (value < 0) {
                    // The astigmatic has 180-degree symmetry.
                    // Negating the value applies 90deg, so another +/- 90 is needed.
                    value = -value;
                    angle += RAD_90;
                }
            }

            // Normalize the astigmatism angle spline by selecting
            // the symmetry half minimizing the delta between control points.
            angles[0] = std::fmod(angles[0], RAD_180);
            for (isize i{1}; i < angles.ssize(); ++i) {
                const auto previous = angles[i - 1];
                const auto current = angles[i];
                const auto delta = current - previous;
                const auto wrapped_delta =  std::fmod(std::fmod(delta + RAD_90, RAD_180) + RAD_180, RAD_180) - RAD_90;
                angles[i] = previous + wrapped_delta;
            }

            phase_shift.update_from_span(m_parameters.phase_shift().span);
            astigmatism_value.update_from_span(values);
            astigmatism_angle.update_from_span(angles);

            // Update metadata.
            // Note that the optimizer ignores the astigmatism and
            // phase-shift from the metadata, and uses the splines instead.
            const auto defoci = m_parameters.defoci();
            const auto scores = znccs();
            const auto angle_offsets = noa::rad2deg(m_parameters.angle_offsets());
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

                // Note that we don't save the ctf_resolution here because for the fit
                // the fitting ranges are slightly extended beyond the last good peak.
                // The ctf_resolution will be saved during plot_diagnostics.
                image.ctf_score = scores[i];
            }
            metadata.sample.thickness = m_parameters.thickness() * 1e3; // um->nm
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

        auto znccs() noexcept -> SpanContiguous<f64> {
            return m_znccs.subregion(0).span_1d();
        }
    };

    template<typename T, typename F>
    void increase_spline_resolution_(
        isize new_resolution,
        const Metadata::Stack& metadata_sorted,
        const Vec<T, 2>& range,
        Array<f64>& values,
        F&& projection
    ) {
        const auto spline = SplineGridCubic<f64, 1>(values.span_1d());
        const auto new_values = noa::zeros<f64>(new_resolution);
        const auto new_spline = SplineGridCubic<f64, 1>(new_values.span_1d());
        for (auto&& [image, node]: noa::zip(metadata_sorted, new_spline.span)) {
            const auto tilt = static_cast<f64>((projection(image) - range[0]) / (range[1] - range[0]));
            node = spline.interpolate_at(tilt);
        }
        values = new_values;
    }

    void increase_tilt_spline_resolution_(
        isize new_resolution,
        const Metadata::Stack& metadata,
        const Vec<f64, 2>& range,
        Array<f64>& values
    ) {
        increase_spline_resolution_(
            new_resolution, metadata.clone().sort("tilt"), range, values,
            [](const auto& image) { return image.angles[1]; }
        );
    }

    void increase_time_spline_resolution_(
        isize new_resolution,
        const Metadata::Stack& metadata,
        const Vec<i64, 2>& range,
        Array<f64>& values
    ) {
        increase_spline_resolution_(
            new_resolution, metadata.clone().sort("time"), range, values,
            [](const auto& image) { return image.time; }
        );
    }
}

namespace qn::ctf {
    void RefineFitting::run(
        nlopt_algorithm algorithm,
        i32 max_number_of_evaluations,
        const RefineFittingParameters<Vec<f64, 2>>& relative_bounds
    ) {
        auto t = Logger::trace_scope_time<true>("Running optimizer");

        const auto phase_shift_spline = SplineGridCubic<f64, 1>(m_phase_shift.span_1d());
        const auto astigmatism_value_spline = SplineGridCubic<f64, 1>(m_astigmatism_value.span_1d());
        const auto astigmatism_angle_spline = SplineGridCubic<f64, 1>(m_astigmatism_angle.span_1d());

        auto fitter = Fitter(
            m_metadata, m_grid, m_patches, m_fitting_ranges.span_1d(),
            phase_shift_spline, astigmatism_value_spline, astigmatism_angle_spline,
            relative_bounds
        );

        const auto n = m_fitting_ranges.span_1d().size();
        auto fitting_range_mean = Vec{0., 0.};
        auto fitting_range_min = 0.5;
        auto fitting_range_max = 0.;
        for (auto e: m_fitting_ranges.span_1d()) {
            fitting_range_mean += e;
            fitting_range_min = std::min(fitting_range_min, e[0]);
            fitting_range_max = std::max(fitting_range_max, e[1]);
        }
        fitting_range_mean /= static_cast<f64>(n);

        Logger::trace(
            "Optimization:\n"
            "  fitting_ranges=[mean={::.2f}, min={:.3f}, max={:.2f}]fftfreq\n"
            "{}{}{}{}{}{}{}{}"
            "  n_parameters={}\n"
            "  max_number_of_evaluations={}\n"
            "  optimizer={}",
            fitting_range_mean, fitting_range_min, fitting_range_max,
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
            m_metadata, phase_shift_spline, astigmatism_value_spline, astigmatism_angle_spline, m_angle_offsets);

        auto stats = [c = static_cast<f64>(n)](auto r) {
            auto o = Vec<f64, 4>{0, 0, 1000, -1000};
            for (auto e: r) {
                o[0] += e;
                o[1] += e * e;
                o[2] = std::min(o[2], e);
                o[3] = std::max(o[3], e);
            }
            const auto tmp = (o[0] * o[0]) / c;
            o[1] = sqrt((o[1] - tmp) / c);
            o[0] /= c;
            return o;
        };
        auto stats_znccs = stats(fitter.znccs());
        auto stats_defoc = stats(m_metadata.stack | stdv::transform([](auto& s) { return s.defocus.value; }));
        auto stats_astig = stats(m_metadata.stack | stdv::transform([](auto& s) { return s.defocus.astigmatism; }));
        auto stats_angle = stats(m_metadata.stack | stdv::transform([](auto& s) { return noa::rad2deg(s.defocus.angle); }));
        auto stats_phase = stats(m_metadata.stack | stdv::transform([](auto& s) { return s.phase_shift; }));

        Logger::trace(
            "Optimization results:\n"
            "  specimen=[[rotation={:.2f}, tilt={:.2f}, pitch={:.2f}]deg, thickness={:.2f}nm]\n"
            "  zncc:    [mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}]\n"
            "  defocus: [mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}]um\n"
            "  phase:   [mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}]deg\n"
            "  astig:   [mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}]um\n"
            "  angle:   [mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}]um",
            m_angle_offsets[0], m_angle_offsets[1], m_angle_offsets[2], m_metadata.sample.thickness,
            stats_znccs[0], stats_znccs[1], stats_znccs[2], stats_znccs[3],
            stats_defoc[0], stats_defoc[1], stats_defoc[2], stats_defoc[3],
            stats_phase[0], stats_phase[1], stats_phase[2], stats_phase[3],
            stats_astig[0], stats_astig[1], stats_astig[2], stats_astig[3],
            stats_angle[0], stats_angle[1], stats_angle[2], stats_angle[3]
        );
    }

    void RefineFitting::plot_diagnostics(const Path& diagnostics_directory) const {
        const auto phase_shift_spline = SplineGridCubic<f64, 1>(m_phase_shift.span_1d());
        const auto astigmatism_value_spline = SplineGridCubic<f64, 1>(m_astigmatism_value.span_1d());
        const auto astigmatism_angle_spline = SplineGridCubic<f64, 1>(m_astigmatism_angle.span_1d());

        auto fitter = Fitter(
            m_metadata, m_grid, m_patches, m_fitting_ranges.span_1d(),
            phase_shift_spline, astigmatism_value_spline, astigmatism_angle_spline
        );

        fitter.plot_diagnostics(diagnostics_directory);

        // Save the estimated resolution.
        const auto spacing = mean(m_metadata.spacing);
        for (auto&& [image, fitting_range]: noa::zip(m_metadata.stack, m_fitting_ranges.span_1d()))
            image.ctf_resolution = fftfreq_to_resolution(spacing, fitting_range[1]);

        save_plot_xy(
            m_metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
            m_metadata.stack | stdv::transform([](auto& s) { return s.defocus.value; }),
            diagnostics_directory / "defocus_fit.txt", {
                .title = "Per-tilt defocus",
                .x_name = "Image index (as saved in the stack)",
                .y_name = "Defocus (μm)",
                .label = "Refine fit",
            });
        save_plot_xy(
            m_metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
            m_metadata.stack | stdv::transform([](auto& s) { return noa::rad2deg(s.phase_shift); }),
            diagnostics_directory / "phase_shift_fit.txt", {
                .title = "Time-resolved phase_shift",
                .x_name = "Image index (as saved in the stack)",
                .y_name = "Phase-shift (degrees)",
                .label = "Refine fit",
            });
        save_plot_xy(
            m_metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
            m_metadata.stack | stdv::transform([](auto& s) { return s.defocus.astigmatism; }),
            diagnostics_directory / "astigmatism_value_fit.txt", {
                .title = "Tilt-resolved astigmatism",
                .x_name = "Image index (as saved in the stack)",
                .y_name = "Astigmatism (μm)",
                .label = "Refine fit",
            });
        save_plot_xy(
            m_metadata.stack | stdv::transform([](auto& s) { return s.index_file; }),
            m_metadata.stack | stdv::transform([](auto& s) { return noa::rad2deg(s.defocus.angle); }),
            diagnostics_directory / "astigmatism_angle_fit.txt", {
                .title = "Tilt-resolved astigmatism",
                .x_name = "Image index (as saved in the stack)",
                .y_name = "Astigmatism angle (degrees)",
                .label = "Refine fit",
            });
    }

    auto RefineFitting::increase_phase_shift_resolution(isize new_resolution) -> bool {
        if (m_phase_shift.ssize() >= new_resolution)
            return false;

        const auto time_range = m_metadata.stack.time_range();
        increase_time_spline_resolution_(new_resolution, m_metadata.stack, time_range, m_phase_shift);
        return true;
    }

    auto RefineFitting::increase_astigmatism_resolution(isize new_resolution) -> bool {
        if (m_astigmatism_angle.ssize() >= new_resolution)
            return false;

        const auto tilt_range = m_metadata.stack.tilt_range();
        increase_tilt_spline_resolution_(new_resolution, m_metadata.stack, tilt_range, m_astigmatism_value);
        increase_tilt_spline_resolution_(new_resolution, m_metadata.stack, tilt_range, m_astigmatism_angle);
        return true;
    }
}
