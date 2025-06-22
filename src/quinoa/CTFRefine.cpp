#include <noa/Array.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Optimizer.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/CTF.hpp"

namespace {
    using namespace ::qn;
    using namespace ::qn::ctf;

    struct ReduceAnisotropic {
        SpanContiguous<const Patches::value_type, 3> polar{}; // (n*p,h,w)
        SpanContiguous<const ns::CTFIsotropic<f64>, 1> isotropic_ctfs{}; // (n*p)
        SpanContiguous<const ns::CTFAnisotropic<f64>, 1> anisotropic_ctfs{}; // (n*p)

        f64 phi_start{};
        f64 phi_step{};
        f64 rho_start{};
        f64 rho_step{};
        f64 rho_range{};

        NOA_HD void init(i64 batch, i64 row, i64 col, f32& r0, f32& r1) const {
            auto phi = static_cast<f64>(row) * phi_step + phi_start; // radians
            auto rho = static_cast<f64>(col) * rho_step + rho_start; // fftfreq

            // Get the target phase.
            auto phase = isotropic_ctfs[batch].phase_at(rho);

            // Get the corresponding fftfreq within the astigmatic field.
            const auto& anisotropic_ctf = anisotropic_ctfs[batch];
            auto ctf = ns::CTFIsotropic(anisotropic_ctf);
            ctf.set_defocus(anisotropic_ctf.defocus_at(phi));
            auto fftfreq = ctf.fftfreq_at(phase);

            // Scale back to unnormalized frequency.
            const auto width = polar.shape().width();
            const auto frequency = static_cast<f64>(width - 1) * (fftfreq - rho_start) / rho_range;

            // Lerp the polar array at this frequency.
            const auto floored = noa::floor(frequency);
            const auto fraction = static_cast<f32>(frequency - floored);
            const auto index = static_cast<i64>(floored);

            f32 v0{}, w0{}, v1{}, w1{};
            if (index >= 0 and index < width) {
                v0 = static_cast<f32>(polar(batch, row, index));
                w0 = 1;
            }
            if (index + 1 >= 0 and index + 1 < width) {
                v1 = static_cast<f32>(polar(batch, row, index + 1));
                w1 = 1;
            }
            r0 += v0 * (1 - fraction) + v1 * fraction;
            r1 += w0 * (1 - fraction) + w1 * fraction;
        }

        static constexpr void join(f32 r0, f32 r1, f32& j0, f32& j1) {
            j0 += r0;
            j1 += r1;
        }

        using remove_default_final = bool;
        static constexpr void final(f32 j0, f32 j1, f32& f) {
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
    private:
        std::array<Parameter, 7> m_parameters{};

        // Keep track of the initial/default values in case we don't fit them.
        std::vector<f64> m_initial_phase_shift{};
        std::vector<f64> m_initial_astigmatism_value{};
        std::vector<f64> m_initial_astigmatism_angle{};

        // Contiguous buffers, where parameters for the optimizer are saved sequentially.
        std::vector<f64> m_buffer{};
        std::vector<f64> m_lower_bounds{};
        std::vector<f64> m_upper_bounds{};
        std::vector<f64> m_abs_tolerance{};

    public:
        enum Index : size_t {
            ROTATION = 0,
            TILT,
            PITCH,
            PHASE_SHIFT,
            DEFOCUS,
            ASTIGMATISM_VALUE,
            ASTIGMATISM_ANGLE,
        };

        [[nodiscard]] auto operator[](Index index) const noexcept -> const Parameter& {
            return m_parameters[index];
        }

    public:
        [[nodiscard]] auto data() noexcept -> f64* { return m_buffer.data(); }
        [[nodiscard]] constexpr auto ssize() const noexcept -> ssize_t { return std::ssize(m_buffer); }
        [[nodiscard]] constexpr auto size() const noexcept -> size_t { return std::size(m_buffer); }

    public: // Special access
        [[nodiscard]] auto angles() const noexcept {
            return Vec{
                m_parameters[ROTATION].is_fitted() ? m_buffer[m_parameters[ROTATION].offset()] : 0,
                m_parameters[PITCH].is_fitted() ? m_buffer[m_parameters[TILT].offset()] : 0,
                m_parameters[TILT].is_fitted() ? m_buffer[m_parameters[PITCH].offset()] : 0
            };
        }

        [[nodiscard]] auto defoci() noexcept {
            return m_parameters[DEFOCUS].span();
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
            const SetOptions<Vec<f64, 2>>& relative_bounds,
            f64 smallest_defocus_allowed
        ) {
            // Set the parameter sizes.
            m_parameters[ROTATION].m_ssize = 1;
            m_parameters[TILT].m_ssize = 1;
            m_parameters[PITCH].m_ssize = 1;
            m_parameters[DEFOCUS].m_ssize = metadata.ssize();
            m_parameters[PHASE_SHIFT].m_ssize = phase_shift.ssize();
            m_parameters[ASTIGMATISM_VALUE].m_ssize = astigmatism_value.ssize();
            m_parameters[ASTIGMATISM_ANGLE].m_ssize = astigmatism_angle.ssize();

            // Set whether they are fitted.
            auto is_fitted = [](const auto& relative_bound) { return not noa::all(noa::allclose(relative_bound, 0.)); };
            m_parameters[ROTATION].m_fit = is_fitted(relative_bounds.rotation);
            m_parameters[TILT].m_fit = is_fitted(relative_bounds.tilt);
            m_parameters[PITCH].m_fit = is_fitted(relative_bounds.pitch);
            m_parameters[DEFOCUS].m_fit = true;
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

            set_relative_bounds(relative_bounds, smallest_defocus_allowed);
        }

        void set_relative_bounds(const SetOptions<Vec<f64, 2>>& relative_bounds, f64 smallest_defocus_allowed) {
            m_lower_bounds.resize(size(), 0.);
            m_upper_bounds.resize(size(), 0.);

            const auto set_buffer = [&](
                const Parameter& parameter,
                const Vec<f64, 2>& low_and_high_bounds,
                f64 minimum = std::numeric_limits<f64>::lowest()
            ) {
                if (not parameter.is_fitted())
                    return;
                for (size_t i{}; i < parameter.size(); ++i) {
                    const auto index = parameter.offset() + i;
                    const auto value = m_buffer[index];
                    m_lower_bounds[index] = std::max(value + low_and_high_bounds[0], minimum);
                    m_upper_bounds[index] = value + low_and_high_bounds[1];
                }
            };

            set_buffer(m_parameters[ROTATION], relative_bounds.rotation);
            set_buffer(m_parameters[TILT], relative_bounds.tilt);
            set_buffer(m_parameters[PITCH], relative_bounds.pitch);
            set_buffer(m_parameters[PHASE_SHIFT], relative_bounds.phase_shift, 0.);
            set_buffer(m_parameters[DEFOCUS], relative_bounds.defocus, smallest_defocus_allowed);
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
            set_buffer(m_parameters[PHASE_SHIFT], abs_tolerance.phase_shift);
            set_buffer(m_parameters[DEFOCUS], abs_tolerance.defocus);
            set_buffer(m_parameters[ASTIGMATISM_VALUE], abs_tolerance.astigmatism_value);
            set_buffer(m_parameters[ASTIGMATISM_ANGLE], abs_tolerance.astigmatism_angle);
        }

        void set_deltas(const SetOptions<f64>& deltas) {
            m_parameters[ROTATION].m_delta = deltas.rotation;
            m_parameters[TILT].m_delta = deltas.tilt;
            m_parameters[PITCH].m_delta = deltas.pitch;
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

        ReduceAnisotropic m_astig_reduce;

        // Patches and their ctfs.
        SpanContiguous<Vec<f64, 2>> m_fitting_ranges{}; // (n)

        Vec<f64, 2> m_time_range{};
        Vec<f64, 2> m_tilt_range{};
        Array<f64> m_phase_shift_weights{};
        Array<f64> m_astig_value_weights{};
        Array<f64> m_astig_angle_weights{};

        //
        Array<CTFIsotropic64> m_ctfs;               // (1,1,1,n)
        Array<CTFIsotropic64> m_ctfs_isotropic;     // (1,1,1,t)
        Array<CTFAnisotropic64> m_ctfs_anisotropic; // (1,1,1,t)

        // 1d spectra.
        Array<f32> m_buffer;
        View<f32> m_spectra;                // (n*p,1,1,w)
        View<f32> m_spectra_average;        // (n,1,1,w)
        View<f32> m_spectra_average_smooth; // (n,1,1,w)

        Array<f32> m_gaussian_filter;

        Array<f64> m_nccs; // (3,1,1,n)
        bool m_are_rotational_averages_ready{false};
        Background m_background;
        std::array<Background, 4> m_background2;

        std::vector<f64> m_gradient_buffer;

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
            const Parameters::SetOptions<Vec<f64, 2>>& relative_bounds
        ) :
            m_metadata(metadata),
            m_grid(grid),
            m_patches(patches),
            m_fitting_ranges(fitting_ranges)
        {
            // Initialize and configure the optimization parameters.
            m_parameters = Parameters(
                metadata, phase_shift, astigmatism_value, astigmatism_angle,
                relative_bounds, Background::smallest_defocus_for_fitting(m_patches.rho_vec(), average_ctf)
            );
            m_parameters.set_abs_tolerance({
                .rotation = noa::deg2rad(0.01),
                .tilt = noa::deg2rad(0.01),
                .pitch = noa::deg2rad(0.01),
                .phase_shift = noa::deg2rad(0.05),
                .defocus = 0.001,
                .astigmatism_value = 0.001,
                .astigmatism_angle = noa::deg2rad(0.1),
            });
            m_parameters.set_deltas({
                .rotation = noa::deg2rad(0.1),
                .tilt = noa::deg2rad(0.1),
                .pitch = noa::deg2rad(0.1),
                .phase_shift = noa::deg2rad(0.5),
                .defocus = 0.005,
                .astigmatism_value = 0.005,
                .astigmatism_angle = noa::deg2rad(0.1),
            });

            // The optimizer may ask for the same cost multiple times, so memoize it.
            m_memoizer = Memoizer(m_parameters.ssize(), 5);

            // Quick access of the dimensions.
            const auto [n, p, h, w] = m_patches.view().shape();

            // Prepare for the rotational averages.
            // Since accesses are per row, use a pitched layout for better performance on the GPU.
            const auto options = patches.view().options();
            const auto options_managed = ArrayOption{options}.set_allocator(Allocator::MANAGED);

            m_buffer = Array<f32>({n * p + n + n, 1, 1, w}, options_managed); // FIXME MANAGED_PITCHED
            m_spectra = m_buffer.view().subregion(ni::Slice{0, n * p}); // (n*p,1,1,s)
            m_spectra_average = m_buffer.view().subregion(ni::Slice{n * p, n * p + n}); // (n,1,1,s)
            m_spectra_average_smooth = m_buffer.view().subregion(ni::Slice{n * p + n}); // (n,1,1,s)

            // Allocate for the CTFs. Everything needs to be dereferenceable.
            // Initialize CTFs with the microscope parameters.
            // The defocus and phase-shift are going to be overwritten.
            m_ctfs = Array<CTFIsotropic64>(n, options_managed);
            m_ctfs_isotropic = Array<CTFIsotropic64>(n * p, options_managed);
            m_ctfs_anisotropic = Array<CTFAnisotropic64>(n * p, options_managed);
            for (auto& ictf: m_ctfs.span_1d())
                ictf = average_ctf;
            for (auto& pctf: m_ctfs_isotropic.span_1d())
                pctf = average_ctf;
            for (auto& pctf: m_ctfs_anisotropic.span_1d())
                pctf = ns::CTFAnisotropic(average_ctf);

            //
            m_gaussian_filter = ns::window_gaussian<f32>(11, 2.5, {.normalize = true}).to(options_managed);

            // Allocate buffers for the cross-correlation. In total, to compute the gradients efficiently,
            // we need a set of 3 NCCs per patch, and these are only accessed on the CPU.
            m_nccs = noa::Array<f64>({3, 1, 1, n});

            // Precompute the spline range and weights.
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

            // Initialize the reduction operator. Reduce: (n*p,1,h,w) -> (n*p,1,1,w)
            m_astig_reduce = ReduceAnisotropic{
                .polar = m_patches.view_batched().span().filter(0, 2, 3).as_contiguous(), // (n*p,h,w)
                .isotropic_ctfs = m_ctfs_isotropic.span_1d(), // (n*p)
                .anisotropic_ctfs = m_ctfs_anisotropic.span_1d(), // (n*p)
                .phi_start = m_patches.phi().start,
                .phi_step = m_patches.phi_step(),
                .rho_start = m_patches.rho().start,
                .rho_step = m_patches.rho_step(),
                .rho_range = m_patches.rho().stop - m_patches.rho().start, // assumes endpoint=true
            };
        }

        // Read the current parameters and update the CTF of each patch accordingly.
        // Only the defocus, and optionally the astigmatism and phase shift, are updated.
        void update_ctfs() {
            const Vec<f64, 3> angle_offsets = m_parameters.angles();
            const SplineGridCubic<f64, 1> time_resolved_phase_shift = m_parameters.phase_shift();
            const SplineGridCubic<f64, 1> tilt_resolved_astigmatism_value = m_parameters.astigmatism_value();
            const SplineGridCubic<f64, 1> tilt_resolved_astigmatism_angle = m_parameters.astigmatism_angle();
            const SpanContiguous<f64> defoci = m_parameters.defoci();

            const auto ictfs = m_ctfs.span_1d();
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
                const auto ctfs_isotropic = m_ctfs_isotropic.span_1d().subregion(chunk);
                const auto ctfs_anisotropic = m_ctfs_anisotropic.span_1d().subregion(chunk);
                const auto slice_spacing = Vec<f64, 2>::from_value(ictfs[i].pixel_size());
                const auto slice_angles = noa::deg2rad(m_metadata[i].angles) + angle_offsets;
                const auto patch_centers = m_grid.patches_centers();

                for (i64 j{}; j < m_patches.n_patches_per_image(); ++j) {
                    const auto patch_z_offset_um = m_grid.patch_z_offset(slice_angles, slice_spacing, patch_centers[j]);
                    const auto patch_defocus = defoci[i] - patch_z_offset_um;
                    ctfs_isotropic[j].set_phase_shift(phase_shift);
                    ctfs_isotropic[j].set_defocus(patch_defocus);
                    ctfs_anisotropic[j].set_phase_shift(phase_shift);
                    ctfs_anisotropic[j].set_defocus({
                        .value = patch_defocus,
                        .astigmatism = slice_astigmatism_value,
                        .angle = slice_astigmatism_angle,
                    });
                }
            }
        }

        auto cost(i64 nccs_index) -> f64 {
            // auto t = Logger::trace_scope_time("cost");
            update_ctfs();

            // Compute the 1d spectra, if needed.
            // noa::Event start{}, stop{};
            // noa::Stream stream = noa::Stream::current(m_patches.view().device());
            // start.record(stream);
            if (not m_are_rotational_averages_ready) {
                if (m_parameters[ASTIGMATISM_VALUE].is_fitted()) {
                    noa::reduce_axes_iwise(
                        m_patches.view_batched().shape().filter(0, 2, 3), m_patches.view().device(),
                        noa::wrap(f32{0}, f32{0}), m_spectra.permute({1, 0, 2, 3}), m_astig_reduce
                    );
                } else {
                    noa::reduce_axes_ewise(m_patches.view_batched(), f32{0}, m_spectra, noa::ReduceMean{m_patches.height()});
                    m_are_rotational_averages_ready = true;
                }
            }

            // Compute the average spectrum of each image to compute the per-image backgrounds.
            ng::fuse_spectra( // (n*p,1,1,s) -> (n,1,1,s)
                m_spectra, m_patches.rho(), m_ctfs_isotropic.view(),
                m_spectra_average, m_patches.rho(), m_ctfs.view(),
                m_spectra_average_smooth
            );

            ns::convolve(
                m_spectra_average, m_spectra_average_smooth,
                m_gaussian_filter.view(), {.border = noa::Border::REFLECT}
            );

            // stop.record(stream);
            // stop.synchronize();
            // fmt::println("gpu={}", noa::Event::elapsed(start, stop));

            // Prepare direct access.
            const auto spectrum_np = m_spectra.span().filter(0, 3).as_contiguous();
            const auto ctf_np = m_ctfs_isotropic.span_1d();
            const auto spectrum_n = m_spectra_average_smooth.span().filter(0, 3).as_contiguous();
            const auto ctf_n = m_ctfs.span_1d();
            const auto ncc_n = m_nccs.subregion(nccs_index).span_1d();
            // const auto baseline = m_spectra_average.subregion(0).span_1d();

            // Wait for the compute device. Everything below is done on the CPU.
            m_spectra.eval();

            // auto timer = Logger::trace_scope_time("cpu");

            // Compute the background based on the average spectrum of the image.
            // Then for every patch of the image, compute the NCC between the
            // background-subtracted spectrum and the simulated CTF.
            f64 sum{};
            #pragma omp parallel for reduction(+:sum) num_threads(2)
            for (i64 i=0; i < m_patches.n_images(); ++i) {
                auto& b = m_background2[omp_get_thread_num()];
                const auto baseline = m_spectra_average.subregion(omp_get_thread_num()).span_1d();

                // As opposed to the fitting range, the background is reevaluated every time.
                // This is mostly useful for the astigmatism search; fitting the background helps to identify
                // cases where the Thon rings are averaged out due to a wrong astigmatism.
                b.fit(spectrum_n[i], m_patches.rho_vec(), ctf_n[i]);
                b.sample(baseline, m_patches.rho_vec());

                // Get the image spectra and CTFs.
                const auto chunk = m_patches.chunk(i);
                const auto spectrum_p = spectrum_np.subregion(chunk);
                const auto ctf_p = ctf_np.subregion(chunk);

                f64 ncc{};
                for (i64 j{}; j < m_patches.n_patches_per_image(); ++j) {
                    ncc += normalized_cross_correlation(
                        spectrum_p[j], ctf_p[j], m_patches.rho_vec(), m_fitting_ranges[i], baseline);
                }
                ncc /= static_cast<f64>(m_patches.n_patches_per_image());

                ncc_n[i] = ncc;
                sum += ncc;
            }
            return sum / static_cast<f64>(m_patches.n_images());
        }

        template<nt::any_of<SpanContiguous<f64, 2>, Empty> T = Empty>
        void gradient(
            const Parameter& parameter,
            f64* gradients,
            const T& weights = {}
        ) {
            if (not parameter.is_fitted())
                return;

            // Save original parameters.
            auto span = parameter.span();
            m_gradient_buffer.clear();
            for (size_t i{}; i < span.size(); ++i)
                m_gradient_buffer.push_back(span[i]);

            // Compute the per-image cost, with +/- delta.
            for (size_t i{}; i < span.size(); ++i)
                span[i] = m_gradient_buffer[i] - parameter.delta();
            cost(1);
            for (size_t i{}; i < span.size(); ++i)
                span[i] = m_gradient_buffer[i] + parameter.delta();
            cost(2);

            // Restore to original parameters.
            for (size_t i{}; i < span.size(); ++i)
                span[i] = m_gradient_buffer[i];

            // Prepare for direct access.
            const auto nccs = this->m_nccs.span();
            const auto fx = nccs.subregion(0).as_1d(); // assume cost(0) was called before calling gradient()
            const auto fx_minus_delta = nccs.subregion(1).as_1d();
            const auto fx_plus_delta = nccs.subregion(2).as_1d();

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
                        // good enough derivatives to guide the optimizer.
                        // Note: This is equivalent to Warp's wiggle weights.
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
            auto t = Logger::trace_scope_time("maximise"); // FIXME

            auto& self = *static_cast<Fitter*>(buffer);

            // The optimizer may pass its own array, so update our parameters.
            auto& params = self.parameters();
            if (parameters != params.data())
                std::copy_n(parameters, params.size(), params.data());

            // In case the optimizer only needs the cost.
            if (not gradients)
                return self.cost(0);

            // Memoization. Sometimes the linear search within L-BFGS is stuck,
            // so detect for these cases to not have to recompute the gradients each time.
            std::optional<f64> memoized_cost = self.memoizer().find(params.data(), gradients, 1e-8);
            if (memoized_cost.has_value()) {
                f64 cost = memoized_cost.value();
                Logger::trace("cost={:.4f}, memoized=true", cost);
                return cost;
            }

            // Compute the cost and the gradient of each parameter.
            // If the parameter isn't fitted, gradient() simply returns.
            const f64 cost = self.cost(0);
            self.gradient(params[ROTATION], gradients);
            self.gradient(params[TILT], gradients);
            self.gradient(params[PITCH], gradients);
            self.gradient(params[DEFOCUS], gradients);
            self.gradient(params[PHASE_SHIFT], gradients, self.m_phase_shift_weights.span<f64, 2>().as_contiguous());
            self.gradient(params[ASTIGMATISM_VALUE], gradients, self.m_astig_value_weights.span<f64, 2>().as_contiguous());
            self.gradient(params[ASTIGMATISM_ANGLE], gradients, self.m_astig_angle_weights.span<f64, 2>().as_contiguous());

            self.memoizer().record(parameters, cost, gradients);
            Logger::trace("cost={:.4f}, angles={::+.3f}, {::+.3f}, astig={::.4f}, {::.4f}",
                cost, noa::rad2deg(params.angles()), SpanContiguous(gradients, 2),
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

        void update_fitting_range(const Path& output_directory) {
            update_ctfs();

            // Target CTF. The spectra will be phased to this CTF.
            f64 average_defocus{};
            f64 min_phase_shift{};
            for (auto& ictf: m_ctfs.span_1d()) {
                average_defocus += ictf.defocus();
                min_phase_shift = std::min(min_phase_shift, ictf.phase_shift());
            }
            ns::CTFIsotropic<f64> target_ctf = m_ctfs.first();
            target_ctf.set_defocus(average_defocus / static_cast<f64>(m_patches.n_images()));
            target_ctf.set_phase_shift(min_phase_shift);

            // Compute the average spectrum of each image.
            noa::reduce_axes_iwise( // (n*p,1,h,w) -> (n*p,1,1,w)
                m_patches.view_batched().shape().filter(0, 2, 3), m_patches.view().device(),
                noa::wrap(f32{0}, f32{0}), m_spectra.permute({1, 0, 2, 3}), m_astig_reduce
            );
            ng::fuse_spectra( // (n*p,1,1,w) -> (n,1,1,w)
                m_spectra, m_patches.rho(), m_ctfs_isotropic.view(),
                m_spectra_average, m_patches.rho(), m_ctfs.view(),
                m_spectra_average_smooth
            );
            m_spectra_average.to(m_spectra_average_smooth); // FIXME
            // ns::convolve(
                // m_spectra_average, m_spectra_average_smooth, m_gaussian_filter.view(),
                // {.border = noa::Border::REFLECT}
            // );

            // Wait for the compute device and prepare for direct access.
            auto buffer = m_spectra_average.view().reinterpret_as_cpu();
            auto phased = buffer.subregion(0);
            auto spectrum = buffer.subregion(1);
            auto weights = buffer.subregion(2);
            noa::fill(spectrum, 0);
            noa::fill(weights, 0);

            const auto fftfreq_step = m_patches.rho_step();
            const auto spectrum_n = m_spectra_average_smooth.span().filter(0, 3).as_contiguous();
            const auto ctf_n = m_ctfs.span_1d();

            for (i64 i{}; i < m_patches.n_images(); ++i) {
                // Fit the background of each image and tune based on the local NCC
                // between the background-subtracted spectrum and the simulated CTF.
                auto fitting_range = m_background.fit_and_tune_fitting_range(
                    spectrum_n[i], m_patches.rho_vec(), ctf_n[i], 1.5, 3); // FIXME
                m_fitting_ranges[i] = fitting_range;

                // That's technically all we need, however, for diagnostics, reconstruct the spectrum of the stack:

                // Scale to the target CTF.
                m_background.subtract(spectrum_n[i], spectrum_n[i], m_patches.rho_vec());
                ng::phase_spectra(
                    View(spectrum_n[i]), m_patches.rho(), ctf_n[i],
                    phased, m_patches.rho(), target_ctf, {.interp = noa::Interp::CUBIC}
                );
                for (auto j: noa::irange(2)) {
                    auto phase = ctf_n[i].phase_at(fitting_range[j]);
                    fitting_range[j] = target_ctf.fftfreq_at(phase);
                }

                // Before adding this spectrum to the average, get the L2-norm within the fitting range.
                f32 l2_norm{};
                for (i64 j{}; const auto& e: phased.span_1d()) {
                    const f64 fftfreq = static_cast<f64>(j++) * fftfreq_step + m_patches.rho().start;
                    if (fitting_range[0] <= fftfreq and fftfreq <= fitting_range[1])
                        l2_norm += e * e;
                }
                l2_norm = std::sqrt(l2_norm);

                // Exclude regions outside the fitting range from the average.
                for (i64 j{}; auto&& [p, w, s]: noa::zip(phased.span_1d(), weights.span_1d(), spectrum.span_1d())) {
                    const f64 fftfreq = static_cast<f64>(j++) * fftfreq_step + m_patches.rho().start;
                    if (fftfreq < fitting_range[1]) {
                        w += 1;
                        s += p / l2_norm;
                    } else {
                        break;
                    }
                }
            }
            for (auto&& [s, w]: noa::zip(spectrum.span_1d(), weights.span_1d()))
                if (w > 1e-6f)
                    s /= w;

            save_plot_xy(m_patches.rho(), spectrum, output_directory / "fitting_range.txt", {
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
        auto astigmatism_value_buffer = Array<f64>(11);
        auto astigmatism_value = SplineGridCubic<f64, 1>(astigmatism_value_buffer.span_1d());
        for (auto& s: astigmatism_value.span)
            s = 0.;

        // Tilt-resolved astigmatism angle.
        auto astigmatism_angle_buffer = Array<f64>(11);
        auto astigmatism_angle = SplineGridCubic<f64, 1>(astigmatism_angle_buffer.span_1d());
        for (auto& s: astigmatism_angle.span)
            s = noa::deg2rad(45.);

        // Set up the optimization pass.
        auto fitting_ranges = Array<Vec<f64, 2>>(patches.n_images());
        auto final_angle_offsets = Vec<f64, 3>{};
        auto run_optimization = [&](
            nlopt_algorithm algorithm, i32 max_number_of_evaluations,
            const Parameters::SetOptions<Vec<f64, 2>>& relative_bounds
        ) {
            auto fitter = Fitter(
                metadata, grid, patches, isotropic_ctf, fitting_ranges.span_1d(),
                phase_shift, astigmatism_value, astigmatism_angle, relative_bounds
            );
            fitter.update_fitting_range(options.output_directory);
            fitter.fit(algorithm, max_number_of_evaluations);
            fitter.update_fitting_range(options.output_directory);
            fitter.update_metadata(metadata, phase_shift, astigmatism_value, astigmatism_angle, final_angle_offsets);
        };

        // 1. Refine the stage-angles, the per-image defocus, and optionally fit the time-resolved phase-shift.
        run_optimization(NLOPT_LD_LBFGS, 30, {
            .rotation = options.fit_rotation ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .tilt =  options.fit_tilt ? deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .pitch = options.fit_pitch ? deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .phase_shift = options.fit_phase_shift ? Vec{0., noa::deg2rad(120.)} : Vec{0., 0.},
            .defocus = Vec{-1., 1.},
        });

        save_plot_xy(
            metadata | stdv::transform([](auto& s) { return s.angles[1]; }),
            metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
            options.output_directory / "defocus_fit.txt", {
                .title = "Per-tilt defocus",
                .x_name = "Tilts (degrees)",
                .y_name = "Defocus (μm)",
                .label = "Refine fit",
            });
        save_plot_xy(
            metadata | stdv::transform([](auto& s) { return s.angles[1]; }),
            metadata | stdv::transform([](auto& s) { return noa::rad2deg(s.phase_shift); }),
            options.output_directory / "phase_shift_fit.txt", {
                .title = "Time-resolved phase_shift",
                .x_name = "Tilts (degrees)",
                .y_name = "Phase-shift (degrees)",
                .label = "Refine + Astigmatism fit",
            });

        // 2. Fit the astigmatism.
        Logger::trace("Enable astigmatism");
        run_optimization(NLOPT_LD_LBFGS, 50, {
            .tilt =  options.fit_tilt ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .pitch = options.fit_pitch ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .defocus = Vec{-0.3, 0.3},
            .astigmatism_value = {-0.5, 0.5},
            .astigmatism_angle = {-noa::deg2rad(45.), noa::deg2rad(45.)},
        });

        save_plot_xy(
            metadata | stdv::transform([](auto& s) { return s.angles[1]; }),
            metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
            options.output_directory / "defocus_fit.txt", {
                .title = "Per-tilt defocus",
                .x_name = "Tilts (degrees)",
                .y_name = "Defocus (μm)",
                .label = "Refine + Astigmatism fit",
            });

        auto metadata_sorted = metadata;
        metadata_sorted.sort("tilt");
        metadata_sorted.reset_indices();
        save_plot_xy({}, //metadata_sorted | stdv::transform([](auto& s) { return s.index; }),
             metadata_sorted | stdv::transform([](auto& s) { return s.defocus.astigmatism; }),
             options.output_directory / "astig.txt");
        save_plot_xy({}, //metadata_sorted | stdv::transform([](auto& s) { return s.index; }),
             metadata_sorted | stdv::transform([](auto& s) { return noa::rad2deg(s.defocus.angle); }),
             options.output_directory / "angles.txt");

        Logger::trace("angles={}", final_angle_offsets);

        panic();

        // auto increase_spline_resolution = [&](SplineGridCubic<f64, 1>& spline, i64 new_size, auto&& norm_coord) -> Array<f64> {
        //     auto buffer = Array<f64>(new_size);
        //     for (auto& s: metadata) {
        //         auto tilt = norm_coord(s);
        //         v = astigmatism_value.interpolate_at(tilt);
        //         a = astigmatism_angle.interpolate_at(tilt);
        //     }
        // };

        // 3. Fit an average astigmatism.
        // auto astigmatism_buffer2 = noa::zeros<f64>({2, 1, 1, patches.n_images()});
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
        // save_plot_xy(metadata_sorted.slices() | stdv::transform([](auto& s) { return s.angles[1]; }),
        //              astigmatism_value2.span,
        //              "/dls/ebic/data/staff-scratch/thomas2/tmp/10164/TS_03/003/astig.txt");
        // save_plot_xy(metadata_sorted.slices() | stdv::transform([](auto& s) { return s.angles[1]; }),
        //      astigmatism_angle2.span,
        //      "/dls/ebic/data/staff-scratch/thomas2/tmp/10164/TS_03/003/angles.txt");
        //
        // run_optimization(NLOPT_GD_STOGO, 40, {
        //     .tilt =  options.fit_tilt ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
        //     .pitch = options.fit_pitch ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
        //     .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
        //     .defocus = Vec{-0.3, 0.3},
        //     .astigmatism_value = {-0.15, 0.15},
        //     .astigmatism_angle = {-noa::deg2rad(45.), noa::deg2rad(45.)},
        // });
        //
        // save_plot_xy(
        //     metadata | stdv::transform([](auto& s) { return s.angles[1]; }),
        //     metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
        //     options.output_directory / "defocus_fit.txt", {
        //         .title = "Per-tilt defocus",
        //         .x_name = "Tilts (degrees)",
        //         .y_name = "Defocus (μm)",
        //         .label = "Defocus - Astig fit",
        //     });
        //
        // save_plot_xy(metadata_sorted.slices() | stdv::transform([](auto& s) { return s.angles[1]; }),
        //              astigmatism_value2.span,
        //              options.output_directory / "astigmatism_values.txt");
        // save_plot_xy(metadata_sorted.slices() | stdv::transform([](auto& s) { return s.angles[1]; }),
        //              astigmatism_angle2.span,
        //              options.output_directory / "astigmatism_angles.txt");
        //
        // Logger::info("stage_angles={::.2f}", final_angle_offsets);

        return;

        // // 2. Same as 1, but fit phase-shift and astigmatism. This is the first time the astigmatism is fitted!
        // if (not options.fit_phase_shift and not options.fit_astigmatism)
        //     return;
        // run_optimization(options);
        //
        // // 3. Same as 2, but fit a time-resolved (3 control points) phase-shift and astigmatism.
        // if ((not options.fit_phase_shift or phase_shift.resolution()[0] == 3) and
        //     (not options.fit_astigmatism or astigmatism.resolution()[0] == 3))
        //     return;
        // if (options.fit_phase_shift and phase_shift.resolution()[0] == 1) { // increase phase-shift resolution
        //     auto input_phase_shift = phase_shift.span()[0][0];
        //     phase_shift = SplineGrid<f64, 1>(3);
        //     for (auto& e: phase_shift.span()[0])
        //         e = input_phase_shift;
        // }
        // if (options.fit_astigmatism and astigmatism.resolution()[0] == 1) { // increase astigmatism resolution
        //     auto input_astigmatism_value = astigmatism.span()[0][0];
        //     auto input_astigmatism_angle = astigmatism.span()[1][0];
        //     astigmatism = SplineGrid<f64, 1>(3, 2);
        //     for (auto& e: astigmatism.span()[0])
        //         e = input_astigmatism_value;
        //     for (auto& e: astigmatism.span()[1])
        //         e = input_astigmatism_angle;
        // }
        // run_optimization(options);
        //
        // // Plot average spectrum...
        // auto average_spectrum_cpu = average_spectrum.view().reinterpret_as_cpu();
        // background.subtract(average_spectrum_cpu, average_spectrum_cpu, fftfreq_range);
        // noa::normalize(average_spectrum_cpu, average_spectrum_cpu, {.mode = noa::Norm::L2});
        // auto title = fmt::format(
        //     "Background-subtracted average spectrum (defocus={:.3f}um, phase_shift={:.2f})",
        //     isotropic_ctf.defocus(), noa::rad2deg(isotropic_ctf.phase_shift()));
        // save_plot_xy(
        //     noa::Linspace{fftfreq_range[0], fftfreq_range[1]}, average_spectrum_cpu,
        //     output_directory / "average_spectrum_bs.txt", {
        //         .title = std::move(title),
        //         .x_name = "fftfreq",
        //         .label = "average spectrum",
        //     });
        //
        // // ... overlapped with the simulated CTF.
        // ns::ctf_isotropic<"h2h">(
        //     average_spectrum_cpu, Shape<i64, 4>{1, 1, 1, patches.shape().height()}, isotropic_ctf, {
        //         .fftfreq_range = noa::Linspace{.start = fftfreq_range[0], .stop = fftfreq_range[1], .endpoint = true},
        //         .ctf_abs = true,
        //     });
        // noa::normalize(average_spectrum_cpu, average_spectrum_cpu, {.mode = noa::Norm::L2});
        // save_plot_xy(
        //     noa::Linspace{fftfreq_range[0], fftfreq_range[1]}, average_spectrum_cpu,
        //     output_directory / "average_spectrum_bs.txt", {.label = "simulated CTF"}
        // );
        //
        // // Save a few other diagnostic plots.
        // auto metadata_plot = metadata;
        // metadata_plot.sort("tilt");
        // save_plot_xy(
        //     metadata | stdv::transform([](auto& s) { return s.angles[1]; }),
        //     metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
        //     output_directory / "defocus_fit.txt", {.label = "Defocus - Refine fit"}
        // );
    }
}
