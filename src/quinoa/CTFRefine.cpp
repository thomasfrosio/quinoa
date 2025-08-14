#include <noa/Array.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Optimizer.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/CTF.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/SplineGrid.hpp"

namespace {
    using namespace ::qn;
    using namespace ::qn::ctf;

    // Loading the CTFs on the GPU takes a long time, so pack and load only the actual variables.
    struct alignas(16) CTFPacked {
        f32 defocus;
        f32 astigmatism;
        f32 angle;
        f32 phase_shift;
    };

    template<typename T>
    struct ReduceAnisotropic {
        using value_type = T;

        SpanContiguous<const Patches::value_type, 3> polar{}; // (n*p,h,w)
        SpanContiguous<const CTFPacked, 1> packed{}; // (c*n*p)

        ns::CTFIsotropic<value_type> isotropic_ctf;
        ns::CTFAnisotropic<value_type> anisotropic_ctf;

        value_type phi_start{};
        value_type phi_step{};
        value_type rho_start{};
        value_type rho_step{};
        value_type rho_range{};

        NOA_HD void init(i64 batch, i64 row, i64 col, value_type& r0, value_type& r1) {
            auto phi = static_cast<value_type>(row) * phi_step + phi_start; // radians
            auto rho = static_cast<value_type>(col) * rho_step + rho_start; // fftfreq

            const auto& [defocus, astigmatism, angle, phase_shift] = packed[batch];

            // Get the target phase.
            isotropic_ctf.set_defocus(defocus);
            isotropic_ctf.set_phase_shift(phase_shift);
            auto phase = isotropic_ctf.phase_at(rho);

            // Get the corresponding fftfreq within the astigmatic field.
            anisotropic_ctf.set_defocus({defocus, astigmatism, angle});
            isotropic_ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            auto fftfreq = isotropic_ctf.fftfreq_at(phase);

            // Scale back to unnormalized frequency.
            const auto width = polar.shape().width();
            const auto frequency = static_cast<value_type>(width - 1) * (fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = static_cast<value_type>(frequency - floored);
            const auto index = static_cast<i64>(floored);

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

        using remove_default_final = bool;
        static constexpr void final(value_type j0, value_type j1, value_type& f) {
            f = j1 > 1 ? j0 / j1 : 0;
        }
    };

    class Parameters;

    class Parameter {
    private:
        f64* m_buffer{};
        f64 m_delta{};
        i64 m_ssize{};
        u64 m_offset{};
        bool m_fit{};

        friend Parameters;

    public:
        [[nodiscard]] auto is_fitted() const noexcept { return m_fit; }
        [[nodiscard]] auto ssize() const noexcept { return m_ssize; }
        [[nodiscard]] auto size() const noexcept { return static_cast<size_t>(m_ssize); }
        [[nodiscard]] auto offset() const noexcept { return m_offset; }
        [[nodiscard]] auto delta() const noexcept { return m_delta; }
        [[nodiscard]] auto span() const noexcept { return SpanContiguous(m_buffer + m_offset, m_ssize); }
    };

    class Parameters {
    public:
        enum Index : size_t {
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
        [[nodiscard]] constexpr auto ssize() const noexcept -> ssize_t { return std::ssize(m_buffer); }
        [[nodiscard]] constexpr auto size() const noexcept -> size_t { return std::size(m_buffer); }

        [[nodiscard]] constexpr auto n_fit() const noexcept -> i64 {
            i64 n{};
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
            const MetadataStack& metadata,
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
            auto is_fitted = [](const auto& relative_bound) { return not noa::all(noa::allclose(relative_bound, 0.)); };
            m_parameters[ROTATION].m_fit = is_fitted(relative_bounds.rotation);
            m_parameters[TILT].m_fit = is_fitted(relative_bounds.tilt);
            m_parameters[PITCH].m_fit = is_fitted(relative_bounds.pitch);
            m_parameters[THICKNESS].m_fit = is_fitted(relative_bounds.thickness);
            m_parameters[DEFOCUS].m_fit = is_fitted(relative_bounds.defocus);
            m_parameters[PHASE_SHIFT].m_fit = is_fitted(relative_bounds.phase_shift);
            m_parameters[ASTIGMATISM_VALUE].m_fit = is_fitted(relative_bounds.astigmatism_value);
            m_parameters[ASTIGMATISM_ANGLE].m_fit = is_fitted(relative_bounds.astigmatism_angle);

            // Set the offset and allocate the contiguous buffer.
            size_t offset{};
            for (auto& data: m_parameters) {
                if (data.m_fit) {
                    data.m_offset = offset;
                    offset += static_cast<size_t>(data.m_ssize);
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
            for (auto&& [defocus, slice]: noa::zip(defoci(), metadata))
                defocus = slice.defocus.value;
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
                for (size_t i{}; i < parameter.size(); ++i) {
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
                for (size_t i{}; i < parameter.size(); ++i) {
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
        const MetadataStack& m_metadata;
        const Grid& m_grid;
        const Patches& m_patches;

        // Optimizer.
        Parameters m_parameters{};
        Memoizer m_memoizer{};

        // Patches and their ctfs.
        SpanContiguous<Vec<f64, 2>> m_fitting_ranges{}; // (n)

        Vec<f64, 2> m_time_range{};
        Vec<f64, 2> m_tilt_range{};
        Array<f64> m_phase_shift_weights{};
        Array<f64> m_astig_value_weights{};
        Array<f64> m_astig_angle_weights{};

        // CTFs.
        i64 m_c;
        Array<CTFIsotropic64> m_ctfs;               // (c,1,1,n)
        Array<CTFIsotropic64> m_ctfs_isotropic;     // (c,1,1,t)
        Array<CTFPacked> m_ctfs_packed;
        ReduceAnisotropic<f32> m_astig_reduce;

        // 1d spectra.
        Array<f32> m_spectra;                // (n*p,1,1,w)
        Array<f32> m_spectra_average;        // (n,1,1,w)
        Array<f32> m_spectra_average_smooth; // (n,1,1,w)

        // Thickness-aware CTF.
        f64 m_thickness;
        std::vector<f64> m_thickness_c;
        Array<f64> m_thickness_modulation;

        Array<f64> m_nccs; // (c,1,1,n)
        bool m_are_rotational_averages_ready{false};
        std::vector<Baseline> m_baseline;

        std::vector<f64> m_parameters_buffer;

    public:
        Fitter(
            const MetadataStack& metadata,
            const Grid& grid,
            const Patches& patches,
            const ns::CTFIsotropic<f64>& average_ctf,
            const SpanContiguous<Vec<f64, 2>>& fitting_ranges,
            const SplineGridCubic<f64, 1>& phase_shift,
            const SplineGridCubic<f64, 1>& astigmatism_value,
            const SplineGridCubic<f64, 1>& astigmatism_angle,
            const Parameters::SetOptions<Vec<f64, 2>>& relative_bounds,
            const f64 thickness_estimate_um
        ) :
            m_metadata(metadata),
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

            // The optimizer may ask for the same cost multiple times, so memoize it.
            m_memoizer = Memoizer(m_parameters.ssize(), 5);

            // Quick access of the dimensions.
            const auto [n, p, h, w] = m_patches.view().shape();

            // Per-image baselines.
            m_baseline.resize(static_cast<size_t>(n));

            // To compute the gradients efficiently, batch the calls for the finite-difference.
            m_c = m_parameters.n_fit() * 2 + 1; // central finite-difference needs 2n+1 evaluations
            m_nccs = noa::Array<f64>({m_c, 1, 1, n});

            // Allocate the spectra. Everything needs to be dereferenceable.
            // Since accesses are per row, use a pitched layout for better performance on the GPU.
            const auto options = patches.view().options();
            const auto options_managed = ArrayOption{options}.set_allocator(Allocator::MANAGED);
            const auto options_pitched = ArrayOption{options}.set_allocator(Allocator::PITCHED_MANAGED);
            m_spectra = Array<f32>({m_c, n * p, 1, w}, options_pitched);
            m_spectra_average = Array<f32>({m_c, n, 1, w}, options_pitched);
            m_spectra_average_smooth = Array<f32>({m_c, n, 1, w}, options_pitched);

            // Allocate for the CTFs. Everything needs to be dereferenceable.
            // Initialize CTFs with the microscope parameters.
            // The defocus and phase-shift are going to be overwritten.
            m_ctfs = Array<CTFIsotropic64>({m_c, 1, 1, n}, options_managed);
            m_ctfs_isotropic = Array<CTFIsotropic64>({m_c, 1, 1, n * p}, options_managed);
            m_ctfs_packed = Array<CTFPacked>({m_c, 1, 1, n * p}, options_managed);
            for (auto& ictf: m_ctfs.span_1d())
                ictf = average_ctf;
            for (auto& pctf: m_ctfs_isotropic.span_1d())
                pctf = average_ctf;

            // Thickness aware CTF-model.
            m_thickness = thickness_estimate_um;
            m_thickness_c.resize(static_cast<size_t>(m_c));
            m_thickness_modulation = Array<f64>({m_c, 1, 1, w}, options_pitched);

            // Precompute the spline range and weights.
            // These are for the time-resolved phase-shift and tilt-resolved astigmatism.
            m_tilt_range = metadata.tilt_range();
            m_time_range = metadata.time_range().as<f64>();
            m_phase_shift_weights = Array<f64>({1, 1, phase_shift.ssize(), n});
            m_astig_value_weights = Array<f64>({1, 1, astigmatism_value.ssize(), n});
            m_astig_angle_weights = Array<f64>({1, 1, astigmatism_angle.ssize(), n});

            auto set_weights = [&](auto&& to_norm_coordinate, const auto& range, const auto& array) {
                auto span = array.template span<f64, 2>();
                for (i64 i{}; i < span.shape()[0]; ++i) { // per node
                    for (i64 j{}; j < span.shape()[1]; ++j) { // per image
                        const f64 nc = (to_norm_coordinate(metadata[j]) - range[0]) / (range[1] - range[0]);
                        span(i, j) = SplineGridCubic<f64, 1>::weight_at(Vec{nc}, Vec{i}, span.shape().filter(0));
                    }
                }
            };
            set_weights([](auto& s) { return static_cast<f64>(s.time); }, m_time_range, m_phase_shift_weights);
            set_weights([](auto& s) { return s.angles[1]; }, m_tilt_range, m_astig_value_weights);
            set_weights([](auto& s) { return s.angles[1]; }, m_tilt_range, m_astig_angle_weights);

            // Initialize the reduction operator.
            // Reduce the height of the polar patches: (c*n*p,1,h,w) -> (c*n*p,1,1,w)
            const auto ctf = m_ctfs_isotropic.first().as<f32>();
            m_astig_reduce = ReduceAnisotropic{
                .polar = m_patches.view_batched().span().filter(0, 2, 3).as_contiguous(), // (n*p,h,w)
                .packed = m_ctfs_packed.span_1d(), // (c*n*p)
                .isotropic_ctf = ctf,
                .anisotropic_ctf = ns::CTFAnisotropic(ctf),
                .phi_start = static_cast<f32>(m_patches.phi().start),
                .phi_step = static_cast<f32>(m_patches.phi_step()),
                .rho_start = static_cast<f32>(m_patches.rho().start),
                .rho_step = static_cast<f32>(m_patches.rho_step()),
                .rho_range = static_cast<f32>(m_patches.rho().stop - m_patches.rho().start), // assumes endpoint=true
            };
        }

        // Read the current parameters and update the CTF of each patch accordingly.
        // Only the defocus, astigmatism and phase shift of the given channel are updated.
        void update_ctfs(i64 channel) {
            const Vec<f64, 3> angle_offsets = m_parameters.angles();
            const SplineGridCubic<f64, 1> time_resolved_phase_shift = m_parameters.phase_shift();
            const SplineGridCubic<f64, 1> tilt_resolved_astigmatism_value = m_parameters.astigmatism_value();
            const SplineGridCubic<f64, 1> tilt_resolved_astigmatism_angle = m_parameters.astigmatism_angle();
            const SpanContiguous<f64> defoci = m_parameters.defoci();

            // Save the thickness for this channel.
            // This is the thickness of the sample. The per-tilt thickness is computed when simulating the CTF.
            const f64 thickness_offset = m_parameters.thickness();
            m_thickness_c[static_cast<size_t>(channel)] = m_thickness + thickness_offset;

            const auto ictfs = m_ctfs.subregion(channel).span_1d();
            for (i64 i{}; i < m_patches.n_images(); ++i) {
                // Time-resolved phase-shift.
                const auto itime = normalized_time(m_metadata[i]);
                const f64 phase_shift = time_resolved_phase_shift.interpolate_at(itime);

                // Tilt-resolved astigmatism.
                const auto itilt = normalized_tilt(m_metadata[i]);
                const auto slice_astigmatism_value = tilt_resolved_astigmatism_value.interpolate_at(itilt);
                const auto slice_astigmatism_angle = tilt_resolved_astigmatism_angle.interpolate_at(itilt);

                // Set the defocus and phase-shift of the image CTF.
                ictfs[i].set_defocus(defoci[i]);
                ictfs[i].set_phase_shift(phase_shift);

                // Update the CTFs of the patches belonging to the current image.
                const auto chunk = m_patches.chunk(i);
                const auto ctfs_isotropic = m_ctfs_isotropic.subregion(channel).span_1d().subregion(chunk);
                const auto ctf_packed = m_ctfs_packed.subregion(channel).span_1d().subregion(chunk);

                const auto slice_spacing = Vec<f64, 2>::from_value(ictfs[i].pixel_size());
                const auto slice_angles = noa::deg2rad(m_metadata[i].angles) + angle_offsets;
                const auto patch_centers = m_grid.patches_centers();

                for (i64 j{}; j < m_patches.n_patches_per_image(); ++j) {
                    const auto patch_z_offset_um = m_grid.patch_z_offset(slice_angles, slice_spacing, patch_centers[j]);
                    const auto patch_defocus = defoci[i] - patch_z_offset_um;
                    ctfs_isotropic[j].set_phase_shift(phase_shift);
                    ctfs_isotropic[j].set_defocus(patch_defocus);
                    ctf_packed[j].defocus = static_cast<f32>(patch_defocus);
                    ctf_packed[j].astigmatism = static_cast<f32>(slice_astigmatism_value);
                    ctf_packed[j].angle = static_cast<f32>(slice_astigmatism_angle);
                    ctf_packed[j].phase_shift = static_cast<f32>(phase_shift);
                }
            }
        }

        void update_channels(i64& channel, const Parameter& parameter) {
            if (not parameter.is_fitted())
                return;

            // Save original parameters.
            auto span = parameter.span();
            m_parameters_buffer.clear();
            for (size_t i{}; i < span.size(); ++i)
                m_parameters_buffer.push_back(span[i]);

            // Save the CTFs, with +/- delta.
            for (size_t i{}; i < span.size(); ++i)
                span[i] = m_parameters_buffer[i] - parameter.delta();
            update_ctfs(channel++);
            for (size_t i{}; i < span.size(); ++i)
                span[i] = m_parameters_buffer[i] + parameter.delta();
            update_ctfs(channel++);

            // Restore to original parameters.
            for (size_t i{}; i < span.size(); ++i)
                span[i] = m_parameters_buffer[i];
        }

        auto cost() -> f64 {
            // Batch the arrays for noa.
            auto width = m_spectra.shape().width();
            auto spectra_batched = m_spectra.view().reshape({-1, 1, 1, width}); // (c,n*p,1,w) -> (c*n*p,1,1,w)
            auto spectra_average_batched = m_spectra_average.view().reshape({-1, 1, 1, width}); // (c,n,1,w) -> (c*n,1,1,w)
            auto spectra_average_smooth_batched = m_spectra_average_smooth.view().reshape({-1, 1, 1, width}); // (c,n,1,w) -> (c*n,1,1,w)

            // Compute the rotational average of the power-spectra, accounting for the astigmatism.
            if (not m_are_rotational_averages_ready) {
                auto polar_shape = m_patches.view_batched().shape(); // (n*p,1,h,w)
                polar_shape[0] = spectra_batched.shape()[0]; // (n*p,1,1,w) -> (c*n*p,1,h,w)
                noa::reduce_axes_iwise( // (c*n*p,1,h,w) -> (c*n*p,1,1,w)
                    polar_shape.filter(0, 2, 3), m_patches.view().device(),
                    noa::wrap(f32{0}, f32{0}), spectra_batched.permute({1, 0, 2, 3}), m_astig_reduce
                );
                if (not m_parameters[ASTIGMATISM_VALUE].is_fitted())
                    m_are_rotational_averages_ready = true;
            }

            // We can fit based on the NCC of each patch, or of each image.
            // I'm still not sure which is best. I'm leaning towards per image just because it's simpler.
            // Both seem to give similar results on the datasets I've tried.
            constexpr bool FIT_PER_PATCH = false;
            if constexpr (FIT_PER_PATCH) {
                // Wait for the compute device. Everything below is done on the CPU.
                m_spectra.eval();

                const auto n_threads = static_cast<i32>((m_c + 1) / 2);
                parallel_for(n_threads, m_c, [this](i32, i64 c) {
                    // Get the channel.
                    const auto spectrum_np = m_spectra.span().subregion(c).filter(1, 3).as_contiguous(); // (c,n*p,1,w) -> (n*p,w)
                    const auto ctf_np = m_ctfs_isotropic.span().subregion(c).as_1d(); // (c,1,1,n*p) -> (n*p)
                    const auto ncc_n = m_nccs.span().subregion(c).as_1d(); // (c,1,1,n) -> (n)
                    const auto n_patches = m_patches.n_patches_per_image();

                    for (i64 i = 0; i < m_patches.n_images(); ++i) {
                        const auto chunk = m_patches.chunk(i);
                        const auto spectrum_p = spectrum_np.subregion(chunk);
                        const auto ctf_p = ctf_np.subregion(chunk);

                        f64 ncc{};
                        if (m_thickness > 1e-6) {
                            // Use a thickness-aware CTF model.
                            const auto thickness = m_thickness_c[static_cast<size_t>(c)];
                            const auto thickness_modulation = ThicknessModulation{
                                .wavelength = ctf_p[0].wavelength(),
                                .spacing = ctf_p[0].pixel_size(),
                                .thickness = effective_thickness(thickness, m_metadata[i].angles) * 1e4,
                            };

                            // Since we loop through the patches, sample the thickness modulation curve once beforehand.
                            auto thickness_modulation_curve = m_thickness_modulation.subregion(c).span_1d();
                            auto fftfreq_step = m_patches.rho_step();
                            for (i64 ii{}; ii < thickness_modulation_curve.ssize(); ++ii) {
                                auto fftfreq = static_cast<f64>(ii) * fftfreq_step + m_patches.rho().start;
                                thickness_modulation_curve[ii] = thickness_modulation.sample_at(fftfreq);
                            }
                            for (i64 j{}; j < n_patches; ++j) {
                                ncc += zero_normalized_cross_correlation(
                                    spectrum_p[j], ctf_p[j], m_patches.rho_vec(), m_fitting_ranges[i],
                                    m_baseline[i], thickness_modulation_curve
                                );
                            }
                        } else {
                            // Use the classic CTF model.
                            for (i64 j{}; j < n_patches; ++j) {
                                ncc += zero_normalized_cross_correlation(
                                    spectrum_p[j], ctf_p[j], m_patches.rho_vec(), m_fitting_ranges[i], m_baseline[i]
                                );
                            }
                        }
                        ncc /= static_cast<f64>(n_patches);
                        ncc_n[i] = ncc;
                    }
                });
            } else {
                // Compute the average spectrum of each image.
                ng::fuse_spectra( // (c*n*p,1,1,s) -> (c*n,1,1,s)
                    spectra_batched, m_patches.rho(), m_ctfs_isotropic.view().flat(),
                    spectra_average_batched, m_patches.rho(), m_ctfs.view().flat(),
                    spectra_average_smooth_batched
                );

                // Wait for the compute device. Everything below is done on the CPU.
                m_spectra.eval();

                for (i64 c{}; c < m_c; ++c) {
                    // Get the channel.
                    const auto spectrum_n = m_spectra_average.span().subregion(c).filter(1, 3).as_contiguous(); // (c,n,1,w) -> (n,w)
                    const auto ctf_n = m_ctfs.span().subregion(c).as_1d(); // (c,1,1,w) -> (n)
                    const auto ncc_n = m_nccs.span().subregion(c).as_1d(); // (c,1,1,w) -> (n)

                    for (i64 i{}; i < m_patches.n_images(); ++i) {
                        // Use the thickness-aware CTF model.
                        // If the thickness is 0, the thickness modulation is a constant 1.
                        const auto thickness = m_thickness_c[static_cast<size_t>(c)];
                        const auto thickness_modulation = ThicknessModulation{
                            .wavelength = ctf_n[i].wavelength(),
                            .spacing = ctf_n[i].pixel_size(),
                            .thickness = effective_thickness(thickness, m_metadata[i].angles) * 1e4,
                        };
                        ncc_n[i] = zero_normalized_cross_correlation(
                            spectrum_n[i], ctf_n[i], m_patches.rho_vec(), m_fitting_ranges[i],
                            m_baseline[i], thickness_modulation
                        );
                    }
                }
            }

            // The first channel is the cost.
            f64 sum{};
            for (f64& ncc: m_nccs.span().subregion(0).as_1d())
                sum += ncc;
            return sum / static_cast<f64>(m_patches.n_images());
        }

        template<nt::any_of<SpanContiguous<f64, 2>, Empty> T = Empty>
        void gradient(
            i64& channel,
            const Parameter& parameter,
            f64* gradients,
            const T& weights = {}
        ) {
            if (not parameter.is_fitted())
                return;

            // Prepare for direct access.
            const auto nccs = this->m_nccs.span();
            const auto fx = nccs.subregion(0).as_1d();
            const auto fx_minus_delta = nccs.subregion(channel++).as_1d();
            const auto fx_plus_delta = nccs.subregion(channel++).as_1d();

            auto span = parameter.span();
            gradients += parameter.offset();

            // Compute the gradient for each variable by reducing the per-image costs.
            const i64 n = m_patches.n_images();
            for (i64 i{}; i < span.ssize(); ++i) {
                f64 cost_minus_delta{0};
                f64 cost_plus_delta{0};
                for (i64 j{}; j < n; ++j) {
                    f64 weight{};
                    if (span.ssize() == 1) {
                        // If there's a single variable, it affects every image.
                        weight = 1;
                    } else if (span.ssize() == n) {
                        // Each variable only affects its corresponding image, so recompose the total cost based on that.
                        // The resulting cost is equivalent to the single-variable case above, but allows to compute
                        // the cost only twice, as opposed to twice per variable.
                        weight = static_cast<f64>(i == j);
                    } else {
                        // The weights tell us how much the image j is affected by the current variable i.
                        // We use this information to get an estimated cost. This cost is not exactly what
                        // we would have gotten with the single-variable case above, but still gives us
                        // good enough derivatives to guide the optimizer. This is equivalent to Warp's wiggle weights.
                        if constexpr (not nt::empty<T>)
                            weight = weights(i, j);
                        else
                            panic();
                    }
                    cost_minus_delta += fx[j] * (1 - weight) + fx_minus_delta[j] * weight;
                    cost_plus_delta += fx[j] * (1 - weight) + fx_plus_delta[j] * weight;
                }
                cost_minus_delta /= static_cast<f64>(n);
                cost_plus_delta /= static_cast<f64>(n);
                gradients[i] = CentralFiniteDifference::get(cost_minus_delta, cost_plus_delta, parameter.delta());
            }
        }

        static auto function_to_maximise(u32, const f64* parameters, f64* gradients, void* buffer) -> f64 {
            // auto t = Logger::trace_scope_time("maximise"); // FIXME

            auto& self = *static_cast<Fitter*>(buffer);

            // The optimizer may pass its own array, so update our parameters.
            auto& params = self.parameters();
            if (parameters != params.data())
                std::copy_n(parameters, params.size(), params.data());

            // In case the optimizer only needs the cost.
            if (not gradients) {
                // panic(); // FIXME
                self.update_ctfs(0);
                auto cost = self.cost();
                Logger::trace("cost={:.4f}, thickness={:.4f}", cost, params.thickness());
                return cost;
            }

            // Memoization. Sometimes the linear search within L-BFGS is stuck,
            // so detect for these cases to not have to recompute the gradients each time.
            std::optional<f64> memoized_cost = self.memoizer().find(params.data(), gradients, 1e-8);
            if (memoized_cost.has_value()) {
                f64 cost = memoized_cost.value();
                Logger::trace("cost={:.4f}, memoized=true", cost);
                return cost;
            }

            // 1. Update the CTFs for every channel.
            self.update_ctfs(0);
            for (i64 channel{1}; auto& index: Parameters::INDICES)
                self.update_channels(channel, params[index]);

            // 2. Compute the scores.
            const f64 cost = self.cost();

            // 3. Compute the gradients.
            auto get_spline = [&self](Parameters::Index index) {
                switch (index) {
                    case PHASE_SHIFT:       return self.m_phase_shift_weights.span<f64, 2>().as_contiguous();
                    case ASTIGMATISM_VALUE: return self.m_astig_value_weights.span<f64, 2>().as_contiguous();
                    case ASTIGMATISM_ANGLE: return self.m_astig_angle_weights.span<f64, 2>().as_contiguous();
                    default:                return SpanContiguous<f64, 2>{};
                }
            };
            for (i64 channel{1}; auto& index: Parameters::INDICES)
                self.gradient(channel, params[index], gradients, get_spline(index));

            //
            self.memoizer().record(parameters, cost, gradients);
            Logger::trace("cost={:.4f}, angles={::+.3f}, thick={:.4f}, {::+.3f}, astig={::.4f}, {::.4f}",
                cost, noa::rad2deg(params.angles()), params.thickness(), SpanContiguous(gradients, 2),
                params[ASTIGMATISM_VALUE].span(), params[ASTIGMATISM_ANGLE].span());
            return cost;
        }

        auto fit(nlopt_algorithm algorithm, i32 max_number_of_evaluations) -> f64 {
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

        void update_fitting_range(const Path& output_directory, std::string_view label, std::string_view prefix) {
            update_ctfs(0);

            // Compute the average spectrum of each image.
            auto width = m_spectra.shape().width();
            auto spectra_batched = m_spectra.view().subregion(0).reshape({-1, 1, 1, width}); // (c,n*p,1,w) -> (n*p,1,1,w)
            auto spectra_average_batched = m_spectra_average.view().subregion(0).reshape({-1, 1, 1, width}); // (1,n*p,1,w) -> (n*p,1,1,w)
            auto spectra_average_weights_batched = m_spectra_average_smooth.view().subregion(0).reshape({-1, 1, 1, width}); // (1,n*p,1,w) -> (n*p,1,1,w)

            noa::reduce_axes_iwise( // (n*p,1,h,w) -> (n*p,1,1,w)
                m_patches.view_batched().shape().filter(0, 2, 3), m_patches.view().device(),
                noa::wrap(f32{0}, f32{0}), spectra_batched.permute({1, 0, 2, 3}), m_astig_reduce
            );
            ng::fuse_spectra( // (n*p,1,1,w) -> (n,1,1,w)
                spectra_batched, m_patches.rho(), m_ctfs_isotropic.view().subregion(0).flat(),
                spectra_average_batched, m_patches.rho(), m_ctfs.view().subregion(0).flat(),
                spectra_average_weights_batched
            );

            // Wait for the compute device and prepare for direct access.
            const auto spectrum_n = spectra_average_batched.reinterpret_as_cpu().span().filter(0, 3).as_contiguous();
            const auto ctf_n = m_ctfs.span().subregion(0).as_1d();

            // Fit the background of each image and tune based on the local NCC between the background-subtracted
            // spectrum and the simulated CTF. Note that these baselines will be needed for the optimization.
            for (i64 i{}; i < m_patches.n_images(); ++i) {
                m_fitting_ranges[i] = m_baseline[i].fit_and_tune_fitting_range(
                    spectrum_n[i], m_patches.rho_vec(), ctf_n[i], {
                        .threshold = 1.2,
                        .keep_first_nth_peaks = 2,

                        // In case of strong astigmatism, the initial spectrum may have only a few Thon-rings.
                        // By looking ahead, we give the optimizer more opportunities to improve the spectrum.
                        .n_extra_peaks_to_append = 2,
                        .n_recoveries_allowed = 1,
                        .maximum_n_consecutive_bad_peaks = 1,

                        .thickness_um = effective_thickness(m_thickness_c[0], m_metadata[i].angles),
                    });
            }
            save_plot_xy(
                m_metadata | stdv::transform([](auto& s) { return s.index_file; }),
                m_fitting_ranges | stdv::transform([&](const auto& v) {
                    return fftfreq_to_resolution(ctf_n[0].pixel_size(), v[1]);
                }),
                output_directory / "fitting_ranges.txt", {
                    .title = "Resolution cutoff for CTF fitting",
                    .x_name = "Image index (as saved in the file)",
                    .y_name = "Resolution (A)",
                    .label = "Refine fitting",
                });

            // That's technically all we need, however, for diagnostics, reconstruct the spectrum of the stack.
            auto buffer = spectra_average_weights_batched.view().reinterpret_as_cpu();
            auto phased = buffer.subregion(0);
            auto spectrum = buffer.subregion(1);
            auto weights = buffer.subregion(2);
            noa::fill(spectrum, 0);
            noa::fill(weights, 0);

            // Target CTF. The spectra will be phased to this CTF.
            f64 average_defocus{};
            f64 min_phase_shift{};
            for (auto& ictf: m_ctfs.span().subregion(0).as_1d()) {
                average_defocus += ictf.defocus();
                min_phase_shift = std::min(min_phase_shift, ictf.phase_shift());
            }
            ns::CTFIsotropic<f64> target_ctf = m_ctfs.first();
            target_ctf.set_defocus(average_defocus / static_cast<f64>(m_patches.n_images()));
            target_ctf.set_phase_shift(min_phase_shift);

            const auto fftfreq_step = m_patches.rho_step();
            auto baseline = Baseline{};
            for (i64 i{}; i < m_patches.n_images(); ++i) {
                // Recompute the fitting range, but without extra peaks or recovery.
                const auto thickness = effective_thickness(m_thickness_c[0], m_metadata[i].angles);
                auto fitting_range = baseline.fit_and_tune_fitting_range(
                    spectrum_n[i], m_patches.rho_vec(), ctf_n[i], {
                        .threshold = 1.5,
                        .keep_first_nth_peaks = 2,
                        .n_extra_peaks_to_append = 0,
                        .n_recoveries_allowed = 0,
                        .thickness_um = thickness,
                    });

                // Scale to the target CTF.
                baseline.subtract(spectrum_n[i], spectrum_n[i], m_patches.rho_vec());
                ng::phase_spectra(
                    View(spectrum_n[i]), m_patches.rho(), ctf_n[i],
                    phased, m_patches.rho(), target_ctf
                );
                for (auto j: noa::irange(2)) {
                    auto phase = ctf_n[i].phase_at(fitting_range[j]);
                    fitting_range[j] = target_ctf.fftfreq_at(phase);
                }

                // To fuse spectra with different thicknesses, we need to correct for the thickness modulation.
                // We could multiply the spectrum with the thickness modulation curve, but that would
                // downweight regions near and at the node (and create visible artifacts from the flipping
                // if the baseline isn't perfectly centered on zero). Instead, skip these regions entirely
                // and flip the zero-centered spectrum (oscillations) when the curve goes negative.
                const auto thickness_modulation = ThicknessModulation{
                    .wavelength = ctf_n[i].wavelength(),
                    .spacing = ctf_n[i].pixel_size(),
                    .thickness = thickness * 1e4,
                };

                // Before adding this spectrum to the average, get the L2-norm within the fitting range.
                f32 l2_norm{};
                for (i64 j{}; const auto& e: phased.span_1d()) {
                    const f64 fftfreq = static_cast<f64>(j++) * fftfreq_step + m_patches.rho().start;
                    if (fitting_range[0] <= fftfreq and fftfreq <= fitting_range[1] and
                        std::abs(thickness_modulation.sample_at(fftfreq)) >= 0.9)
                        l2_norm += e * e;
                }
                l2_norm = std::sqrt(l2_norm);

                // Exclude regions after the fitting range from the average.
                for (i64 j{}; auto&& [p, w, s]: noa::zip(phased.span_1d(), weights.span_1d(), spectrum.span_1d())) {
                    const f64 fftfreq = static_cast<f64>(j++) * fftfreq_step + m_patches.rho().start;
                    if (fftfreq <= fitting_range[1]) {
                        const auto modulation = static_cast<f32>(thickness_modulation.sample_at(fftfreq));
                        if (std::abs(modulation) < 0.9f)
                            continue;

                        w += 1;
                        s += (p / l2_norm) * std::copysign(1.f, modulation);
                    } else {
                        break;
                    }
                }
            }
            for (auto&& [s, w]: noa::zip(spectrum.span_1d(), weights.span_1d()))
                if (w > 1e-6f)
                    s /= w;

            save_plot_xy(m_patches.rho(), spectrum_n, output_directory / "refined_spectra.txt", {
                .title = "Per-image refined spectra (sorted by collection order)",
                .x_name = "fftfreq",
                .label = fmt::format("{}-{}", label, prefix),
            });
            save_plot_xy(m_patches.rho(), spectrum, output_directory / "reconstructed_spectrum.txt", {
                .title = "Reconstructed spectrum",
                .x_name = "fftfreq",
                .label =  fmt::format("defocus={:.3f}", target_ctf.defocus()),
            });

            // The spectra have changed, so reset for the next run.
            m_are_rotational_averages_ready = false;
            m_memoizer.reset_cache();
        }

        void update_metadata(
            MetadataStack& metadata,
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
            for (i64 i{}; i < metadata.ssize(); ++i) {
                auto& slice = metadata[i];
                const auto time = normalized_time(slice);
                const auto tilt = normalized_tilt(slice); // must be before updating the tilt angles

                slice.angles = MetadataSlice::to_angle_range(slice.angles + angle_offsets);
                slice.phase_shift = phase_shift.interpolate_at(time);
                slice.defocus = {
                    .value = defoci[i],
                    .astigmatism = astigmatism_value.interpolate_at(tilt),
                    .angle = astigmatism_angle.interpolate_at(tilt),
                };
            }

            final_angle_offsets += angle_offsets;
        }

        [[nodiscard]] auto parameters() noexcept -> Parameters& { return m_parameters; }
        [[nodiscard]] auto memoizer() noexcept -> Memoizer& { return m_memoizer; }

        [[nodiscard]] auto normalized_time(const MetadataSlice& slice) noexcept -> f64 {
            return (static_cast<f64>(slice.time) - m_time_range[0]) / (m_time_range[1] - m_time_range[0]);
        }
        [[nodiscard]] auto normalized_tilt(const MetadataSlice& slice) noexcept -> f64 {
            return (slice.angles[1] - m_tilt_range[0]) / (m_tilt_range[1] - m_tilt_range[0]);
        }
    };
}

namespace qn::ctf {
    void refine_fit(
        MetadataStack& metadata,
        const Grid& grid,
        const Patches& patches,
        ns::CTFIsotropic<f64>& isotropic_ctf,
        const FitRefineOptions& options
    ) {
        auto timer = Logger::info_scope_time("Refine CTF fitting");

        // Time-resolved phase-shift.
        auto phase_shift_buffer = Array<f64>(5);
        auto phase_shift = SplineGridCubic<f64, 1>(phase_shift_buffer.span_1d());
        for (auto& s: phase_shift.span)
            s = metadata[0].phase_shift;

        // Tilt-resolved astigmatism value.
        auto astigmatism_value_buffer = Array<f64>(9);
        auto astigmatism_value = SplineGridCubic<f64, 1>(astigmatism_value_buffer.span_1d());
        for (auto& s: astigmatism_value.span)
            s = 0.;

        // Tilt-resolved astigmatism angle.
        auto astigmatism_angle_buffer = Array<f64>(9);
        auto astigmatism_angle = SplineGridCubic<f64, 1>(astigmatism_angle_buffer.span_1d());
        for (auto& s: astigmatism_angle.span)
            s = noa::deg2rad(45.);

        // Set up the optimization pass.
        auto run_optimization = [
            &, final_angle_offsets = Vec<f64, 3>{},
            fitting_ranges = Array<Vec<f64, 2>>(patches.n_images())
        ](
            nlopt_algorithm algorithm, i32 max_number_of_evaluations, std::string_view label,
            const Parameters::SetOptions<Vec<f64, 2>>& relative_bounds
        ) mutable {
            auto t = Logger::trace_scope_time("Running optimizer");

            auto fitter = Fitter(
                metadata, grid, patches, isotropic_ctf, fitting_ranges.span_1d(),
                phase_shift, astigmatism_value, astigmatism_angle, relative_bounds, options.thickness
            );

            Logger::trace(
                "Optimization:\n"
                "{}{}{}{}{}{}{}{}"
                "  n_parameters={}\n"
                "  max_number_of_evaluations={}\n"
                "  optimizer={}",
                all(noa::allclose(relative_bounds.rotation, 0.)) ? "" : fmt::format("  rotation={::.2f}deg bound\n", noa::rad2deg(relative_bounds.rotation)),
                all(noa::allclose(relative_bounds.tilt, 0.)) ? "" : fmt::format("  tilt={::.2f}deg bound\n", noa::rad2deg(relative_bounds.tilt)),
                all(noa::allclose(relative_bounds.pitch, 0.)) ? "" : fmt::format("  pitch={::.2f}deg bound\n", noa::rad2deg(relative_bounds.pitch)),
                all(noa::allclose(relative_bounds.thickness, 0.)) ? "" : fmt::format("  thickness={::.3f}um bound\n", relative_bounds.thickness),
                all(noa::allclose(relative_bounds.phase_shift, 0.)) ? "" : fmt::format("  phase_shift={::.2f}deg bound\n", noa::rad2deg(relative_bounds.phase_shift)),
                all(noa::allclose(relative_bounds.defocus, 0.)) ? "" : fmt::format("  defocus={::.2f}um bound\n", relative_bounds.defocus),
                all(noa::allclose(relative_bounds.astigmatism_value, 0.)) ? "" : fmt::format("  astigmatism_value={::.2f}um bound\n", relative_bounds.astigmatism_value),
                all(noa::allclose(relative_bounds.astigmatism_angle, 0.)) ? "" : fmt::format("  astigmatism_angle={::.2f}deg bound\n", noa::rad2deg(relative_bounds.astigmatism_angle)),
                fitter.parameters().ssize(),
                max_number_of_evaluations,
                algorithm == NLOPT_LD_LBFGS ? "L-BFGS (local, gradient-based)" : "StoGO (global, gradient-based)"
            );

            fitter.update_fitting_range(options.output_directory, label, "pre");
            fitter.fit(algorithm, max_number_of_evaluations);
            fitter.update_fitting_range(options.output_directory, label, "post");
            fitter.update_metadata(metadata, phase_shift, astigmatism_value, astigmatism_angle, final_angle_offsets);

            // Diagnostics.
            save_plot_xy(
                metadata | stdv::transform([](auto& s) { return s.index_file; }),
                metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
                options.output_directory / "defocus_fit.txt", {
                    .title = "Per-tilt defocus",
                    .x_name = "Image index (as saved in the stack)",
                    .y_name = "Defocus (μm)",
                    .label = "Refine fit",
                });
            save_plot_xy(
                metadata | stdv::transform([](auto& s) { return s.index_file; }),
                metadata | stdv::transform([](auto& s) { return noa::rad2deg(s.phase_shift); }),
                options.output_directory / "phase_shift_fit.txt", {
                    .title = "Time-resolved phase_shift",
                    .x_name = "Image index (as saved in the stack)",
                    .y_name = "Phase-shift (degrees)",
                    .label = "Refine fit",
                });
            save_plot_xy(
                metadata | stdv::transform([](auto& s) { return s.index_file; }),
                metadata | stdv::transform([](auto& s) { return s.defocus.astigmatism; }),
                options.output_directory / "astigmatism_value_fit.txt", {
                    .title = "Tilt-resolved astigmatism",
                    .x_name = "Image index (as saved in the stack)",
                    .y_name = "Astigmatism (μm)",
                    .label = "Refine fit",
                });
            save_plot_xy(
                metadata | stdv::transform([](auto& s) { return s.index_file; }),
                metadata | stdv::transform([](auto& s) { return noa::rad2deg(s.defocus.angle); }),
                options.output_directory / "astigmatism_angle_fit.txt", {
                    .title = "Tilt-resolved astigmatism",
                    .x_name = "Image index (as saved in the stack)",
                    .y_name = "Astigmatism angle (degrees)",
                    .label = "Refine fit",
                });

            Logger::trace(
                "stage_angles=[rotation={:.2f}, tilt={:.2f}, pitch={:.2f}]",
                final_angle_offsets[0], final_angle_offsets[1], final_angle_offsets[2]
            );
        };

        // Refine the stage-angles, the per-image defocus, and optionally fit the time-resolved phase-shift.
        run_optimization(NLOPT_LD_LBFGS, 30, "1", {
            .rotation = options.fit_rotation ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .tilt =  options.fit_tilt ? deg2rad(Vec{-30., 30.}) : Vec{0., 0.},
            .pitch = options.fit_pitch ? deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .phase_shift = options.fit_phase_shift ? Vec{0., noa::deg2rad(120.)} : Vec{0., 0.},
            .defocus = Vec{-1., 1.},
        });

        if (not options.fit_astigmatism)
            return;

        // Fit the astigmatism.
        Logger::trace("Enable astigmatism with tilt-resolution={}", astigmatism_value.size());
        run_optimization(NLOPT_GD_STOGO, 50, "2", {
            .rotation =  options.fit_rotation ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .tilt =  options.fit_tilt ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .pitch = options.fit_pitch ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .defocus = Vec{-0.3, 0.3},
            .astigmatism_value = {-0.7, 0.7},
            .astigmatism_angle = {-noa::deg2rad(45.), noa::deg2rad(45.)},
        });
        run_optimization(NLOPT_LD_LBFGS, 50, "3",{
            .rotation =  options.fit_rotation ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .tilt =  options.fit_tilt ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .pitch = options.fit_pitch ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .defocus = Vec{-0.3, 0.3},
            .astigmatism_value = {-0.4, 0.4},
            .astigmatism_angle = {-noa::deg2rad(45.), noa::deg2rad(45.)},
        });

        // Increase the tilt-resolution of the astigmatism.
        // auto astigmatism_buffer2 = noa::zeros<f64>({2, 1, 1, patches.n_images() / 2 + 1});
        // auto astigmatism_value2 = SplineGridCubic<f64, 1>(astigmatism_buffer2.span().subregion(0).as_1d());
        // auto astigmatism_angle2 = SplineGridCubic<f64, 1>(astigmatism_buffer2.span().subregion(1).as_1d());
        // auto tilt_range = metadata.tilt_range();
        // auto metadata_sorted = metadata;
        // metadata_sorted.sort("tilt");
        // for (auto&& [s, v, a]: noa::zip(metadata_sorted, astigmatism_value2.span, astigmatism_angle2.span)) {
        //     auto tilt = (s.angles[1] - tilt_range[0]) / (tilt_range[1] - tilt_range[0]);
        //     v = astigmatism_value.interpolate_at(tilt);
        //     a = astigmatism_angle.interpolate_at(tilt);
        // }
        // astigmatism_value = astigmatism_value2;
        // astigmatism_angle = astigmatism_angle2;
        //
        // // Fit an average astigmatism.
        // run_optimization(NLOPT_LD_LBFGS, 30, "4", {
        //     .tilt =  options.fit_tilt ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
        //     .pitch = options.fit_pitch ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
        //     .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
        //     .defocus = Vec{-0.1, 0.1},
        //     .astigmatism_value = {-0.1, 0.1},
        //     .astigmatism_angle = {-noa::deg2rad(25.), noa::deg2rad(25.)},
        // });
        // panic();
    }
}
