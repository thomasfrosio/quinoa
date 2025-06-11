#include <span>
#include <noa/Array.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Optimizer.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/CTF.hpp"

#include <noa/gpu/cuda/Block.cuh>
#include <noa/core/utils/Atomic.hpp>

namespace {
    using namespace noa::types;
    namespace nt = noa::traits;

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_3d_static(Op op, Vec2<Index> end_hw) {
        auto dhw = Vec3<Index>::from_values(
            blockIdx.z,
            Config::block_work_size_y * blockIdx.y + threadIdx.y,
            Config::block_work_size_x * blockIdx.x + threadIdx.x
        );
        const Index ih = dhw[1];
        const Index iw = dhw[2];
        const Index tid = blockDim.x * threadIdx.y + threadIdx.x;

        // Initialize the shells for this block.
        __shared__ float shells[1024];
        __shared__ float shells_weights[1024];

        for (Index i = tid; i < op.m_max_shell_index + 1; i += Config::block_size) {
            shells[i] = 0;
            shells_weights[i] = 0;
        }
        noa::cuda::guts::block_synchronize();

        using coord_type = float;
        using coord_nd_type = Vec<coord_type, 2>;

        if (ih < end_hw[0] and iw < end_hw[1]) {
            // Input indices to fftfreq.
            const auto frequency = noa::fft::index2frequency<false, true>(noa::Vec{ih, iw}, op.m_shape);
            const auto fftfreq_nd = coord_nd_type::from_vec(frequency) * op.m_input_fftfreq_step;
            const auto fftfreq = static_cast<coord_type>(op.m_ctf[dhw[0]].isotropic_fftfreq(fftfreq_nd));

            // Remove most out-of-bounds asap.
            if (fftfreq >= op.m_fftfreq_cutoff[0] and fftfreq <= op.m_fftfreq_cutoff[1]) {
                const auto value = cast_or_abs_squared<float>(op.m_input(dhw[0], dhw[1], dhw[2]));

                const coord_type scaled_fftfreq = (fftfreq - op.m_output_fftfreq_start) / op.m_output_fftfreq_span;
                const coord_type radius = scaled_fftfreq * static_cast<coord_type>(op.m_max_shell_index);

                const auto shell = static_cast<Index>(round(radius));
                if (shell >= 0 and shell <= op.m_max_shell_index) {
                    atomicAdd(shells + shell, value);
                    atomicAdd(shells_weights + shell, 1.f);
                    // noa::guts::atomic_add(op.m_output, value, dhw[0], shell);
                    // noa::guts::atomic_add(op.m_weight, 1.f, dhw[0], shell);
                }
            }
        }

        noa::cuda::guts::block_synchronize();
        for (Index i = tid; i < op.m_max_shell_index + 1; i += Config::block_size) {
            noa::guts::atomic_add(op.m_output, shells[i], dhw[0], i);
            noa::guts::atomic_add(op.m_weight, shells_weights[i], dhw[0], i);
        }
    }

    template<typename Config, typename Index>
    auto iwise_3d_static_config(
        const Shape3<Index>& shape,
        size_t n_bytes_of_shared_memory
    ) -> noa::cuda::LaunchConfig {
        const auto iwise_shape = shape.template as_safe<u32>();
        const u32 n_blocks_x = noa::divide_up(iwise_shape[2], Config::block_work_size_x);
        const u32 n_blocks_y = noa::divide_up(iwise_shape[1], Config::block_work_size_y);
        return {
            .n_blocks = dim3(n_blocks_x, n_blocks_y, iwise_shape[0]),
            .n_threads = dim3(Config::block_size_x, Config::block_size_y),
            .n_bytes_of_shared_memory = n_bytes_of_shared_memory
        };
    }

    template<size_t N, typename Config = noa::cuda::IwiseConfig<32, 32, 1, 1>, typename Index, typename Op>
    void iwise_3d(
        const Shape<Index, N>& shape,
        Op&& op,
        noa::cuda::Stream& stream,
        size_t n_bytes_of_shared_memory = 0
    ) {
        const auto launch_config = iwise_3d_static_config<Config>(shape, n_bytes_of_shared_memory);
        const auto end_2d = shape.pop_front().vec;
        stream.enqueue(iwise_3d_static<Config, std::decay_t<Op>, Index>,
                       launch_config, std::forward<Op>(op), end_2d);
    }

    template<noa::Remap REMAP,
             size_t N,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<N + 1> Input,
             nt::atomic_addable_nd<2> Output,
             nt::atomic_addable_nd_optional<2> Weight,
             nt::batched_parameter Ctf>
    class RotationalAverage {
    public:
        static_assert((N == 2 or N == 3) and REMAP.is_xx2h());
        static constexpr bool IS_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2xx();

        using index_type = Index;
        using coord_type = Coord;
        using shape_type = noa::Shape<index_type, N - IS_RFFT>;
        using coord_nd_type = noa::Vec<coord_type, N>;
        using coord2_type = noa::Vec2<coord_type>;
        using shape_nd_type = noa::Shape<index_type, N>;

        using input_type = Input;
        using output_type = Output;
        using weight_type = Weight;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using weight_value_type = nt::value_type_t<weight_type>;
        static_assert(nt::spectrum_types<nt::value_type_t<input_type>, output_value_type>);
        static_assert(nt::same_as<weight_value_type, output_real_type>);

        using batched_ctf_type = Ctf;
        using ctf_type = nt::mutable_value_type_t<batched_ctf_type>;
        static_assert(nt::empty<ctf_type> or (N == 2 and nt::ctf_anisotropic<ctf_type>));

    public:
        constexpr RotationalAverage(
            const input_type& input,
            const shape_nd_type& input_shape,
            const batched_ctf_type& input_ctf,
            const output_type& output,
            const weight_type& weight,
            index_type n_shells,
            noa::Linspace<coord_type> input_fftfreq,
            noa::Linspace<coord_type> output_fftfreq
        ) :
            m_input(input),
            m_output(output),
            m_weight(weight),
            m_ctf(input_ctf),
            m_shape(input_shape.template pop_back<IS_RFFT>())
        {
            // If input_fftfreq.stop is negative, defaults to the highest frequency.
            // In this case, and if the frequency.start is 0, this results in the full frequency range.
            // The input is N-d, so we have to handle each axis separately.
            coord_type max_input_fftfreq{-1};
            for (size_t i{}; i < N; ++i) {
                const auto max_sample_size = input_shape[i] / 2 + 1;
                const auto fftfreq_end =
                    input_fftfreq.stop <= 0 ?
                    noa::fft::highest_fftfreq<coord_type>(input_shape[i]) :
                    input_fftfreq.stop;
                max_input_fftfreq = max(max_input_fftfreq, fftfreq_end);
                m_input_fftfreq_step[i] = noa::Linspace<coord_type>{
                    .start = 0,
                    .stop = fftfreq_end,
                    .endpoint = input_fftfreq.endpoint
                }.for_size(max_sample_size).step;
            }

            // The output defaults to the input range. Of course, it is a reduction to 1d, so take the max fftfreq.
            if (output_fftfreq.start < 0)
                output_fftfreq.start = 0;
            if (output_fftfreq.stop <= 0)
                output_fftfreq.stop = max_input_fftfreq;

            // Transform to inclusive range so that we only have to deal with one case.
            if (not output_fftfreq.endpoint) {
                output_fftfreq.stop -= output_fftfreq.for_size(n_shells).step;
                output_fftfreq.endpoint = true;
            }
            m_output_fftfreq_start = output_fftfreq.start;
            m_output_fftfreq_span = output_fftfreq.stop - output_fftfreq.start;
            m_max_shell_index = n_shells - 1;

            // To shortcut early, compute the fftfreq cutoffs where we know the output isn't affected.
            auto output_fftfreq_step = output_fftfreq.for_size(n_shells).step;
            m_fftfreq_cutoff[0] = output_fftfreq.start + -1 * output_fftfreq_step;
            m_fftfreq_cutoff[1] = output_fftfreq.stop + static_cast<coord_type>(m_max_shell_index) * output_fftfreq_step;
        }

        // 2d or 3d rotational average, with an optional anisotropic field correction.
        NOA_HD void operator()(index_type batch, index_type y, index_type x, __shared__ float* shells) const noexcept {
            // Input indices to fftfreq.
            const auto frequency = noa::fft::index2frequency<IS_CENTERED, IS_RFFT>(noa::Vec{y, x}, m_shape);
            const auto fftfreq_nd = coord_nd_type::from_vec(frequency) * m_input_fftfreq_step;

            coord_type fftfreq;
            if constexpr (nt::empty<ctf_type>) {
                fftfreq = sqrt(dot(fftfreq_nd, fftfreq_nd));
            } else {
                // Correct for anisotropic field (pixel size and defocus).
                fftfreq = static_cast<coord_type>(m_ctf[batch].isotropic_fftfreq(fftfreq_nd));
            }

            // Remove most out-of-bounds asap.
            if (fftfreq < m_fftfreq_cutoff[0] or fftfreq > m_fftfreq_cutoff[1])
                return;

            const auto value = cast_or_abs_squared<output_value_type>(m_input(batch, y, x));

            const coord_type scaled_fftfreq = (fftfreq - m_output_fftfreq_start) / m_output_fftfreq_span;
            const coord_type radius = scaled_fftfreq * static_cast<coord_type>(m_max_shell_index);

            const auto shell = static_cast<index_type>(round(radius));
            if (shell >= 0 and shell <= m_max_shell_index) {
                noa::guts::atomic_add(m_output, value, batch, shell);
                if (m_weight)
                    noa::guts::atomic_add(m_weight, static_cast<weight_value_type>(1), batch, shell);
            }
        }

    public:
        input_type m_input;
        output_type m_output;
        weight_type m_weight;
        NOA_NO_UNIQUE_ADDRESS batched_ctf_type m_ctf;

        shape_type m_shape;
        coord_nd_type m_input_fftfreq_step;
        coord2_type m_fftfreq_cutoff;
        coord_type m_output_fftfreq_start;
        coord_type m_output_fftfreq_span;
        index_type m_max_shell_index;
    };


    template<
        noa::Remap REMAP, bool IS_GPU = false,
        typename Input, typename Index, typename Ctf,
        typename Output, typename Weight, typename Options>
    void launch_rotational_average(
        Input&& input, const noa::Shape4<Index>& input_shape, Ctf&& input_ctf,
        Output&& output, Weight&& weight, noa::i64 n_shells, const Options& options
    ) {
        using input_value_t = nt::value_type_t<Input>; // FIXME nt::const_value_type_t<Input>
        using output_value_t = nt::value_type_t<Output>;
        using weight_value_t = nt::value_type_t<Weight>;
        using coord_t = nt::value_type_t<output_value_t>;
        constexpr auto EWISE_OPTION = noa::EwiseOptions{.generate_cpu = not IS_GPU, .generate_gpu = IS_GPU};

        // Output must be zeroed out.
        const auto output_view = output.view();
        if (not options.add_to_output)
            ewise<EWISE_OPTION>({}, output_view, noa::Zero{});

        // When computing the average, the weights must be valid.
        auto weight_view = weight.view();
        noa::Array<weight_value_t> weight_buffer;
        if (options.average) {
            if (weight_view.is_empty()) {
                weight_buffer = zeros<weight_value_t>(output_view.shape(), noa::ArrayOption{output.device(), noa::Allocator::DEFAULT_ASYNC});
                weight_view = weight_buffer.view();
            } else if (not options.add_to_output) {
                ewise<EWISE_OPTION>({}, weight_view, noa::Zero{});
            }
        }

        using output_accessor_t = noa::AccessorRestrictContiguous<output_value_t, 2, Index>;
        using weight_accessor_t = noa::AccessorRestrictContiguous<weight_value_t, 2, Index>;
        auto output_accessor = output_accessor_t(output_view.get(), noa::Strides1<Index>::from_value(output_view.strides()[0]));
        auto weight_accessor = weight_accessor_t(weight_view.get(), noa::Strides1<Index>::from_value(weight_view.strides()[0]));

        const auto input_fftfreq = options.input_fftfreq.template as<coord_t>();
        const auto output_fftfreq = options.output_fftfreq.template as<coord_t>();
        const auto iwise_shape = input.shape().template as<Index>();
        const auto input_strides = input.strides().template as<Index>();

        if (input_shape.ndim() == 2) {
            auto ctf = noa::guts::to_batched_parameter<true>(input_ctf);

            using input_accessor_t = noa::AccessorRestrict<input_value_t, 3, Index>;
            auto op = RotationalAverage
                <REMAP, 2, coord_t, Index, input_accessor_t, output_accessor_t, weight_accessor_t, decltype(ctf)>(
                input_accessor_t(input.get(), input_strides.filter(0, 2, 3)), input_shape.filter(2, 3),
                ctf, output_accessor, weight_accessor, static_cast<Index>(n_shells),
                input_fftfreq, output_fftfreq);

            iwise_3d(iwise_shape.filter(0, 2, 3), op, noa::Stream::current(output.device()).cuda());
        }

        // Some shells can be 0, so use DivideSafe.
        if (options.average) {
            if (weight_buffer.is_empty()) {
                ewise<EWISE_OPTION>(wrap(output_view, weight), std::forward<Output>(output), noa::DivideSafe{});
            } else {
                ewise<EWISE_OPTION>(wrap(output_view, std::move(weight_buffer)), std::forward<Output>(output), noa::DivideSafe{});
            }
        }
    }

    template<noa::Remap REMAP,
             nt::readable_varray_decay Input,
             nt::writable_varray_decay Output,
             typename Ctf,
             typename Weight>
        requires (REMAP.is_xx2h() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    [[gnu::noinline]] void rotational_average_anisotropic(
        Input&& input,
        const noa::Shape4<noa::i64>& input_shape,
        Ctf&& input_ctf,
        Output&& output,
        Weight&& weights = {},
        noa::geometry::RotationalAverageOptions options = {}
    ) {
        const auto n_shells = output.shape().pop_front().n_elements();
        launch_rotational_average<REMAP, true>(
            std::forward<Input>(input), input_shape.as<noa::i64>(),
            std::forward<Ctf>(input_ctf),
            std::forward<Output>(output),
            std::forward<Weight>(weights),
            n_shells, options);
    }
}

namespace {
    using namespace ::qn;
    using namespace ::qn::ctf;

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
            const SplineGridCubic<f64, 1>& initial_phase_shift,
            const SplineGridCubic<f64, 1>& initial_astigmatism_value,
            const SplineGridCubic<f64, 1>& initial_astigmatism_angle,
            const SetOptions<Vec<f64, 2>>& relative_bounds,
            f64 smallest_defocus_allowed
        ) {
            // Set the parameter sizes.
            m_parameters[ROTATION].m_ssize = 1;
            m_parameters[TILT].m_ssize = 1;
            m_parameters[PITCH].m_ssize = 1;
            m_parameters[DEFOCUS].m_ssize = metadata.ssize();
            m_parameters[PHASE_SHIFT].m_ssize = initial_phase_shift.ssize();
            m_parameters[ASTIGMATISM_VALUE].m_ssize = initial_astigmatism_value.ssize();
            m_parameters[ASTIGMATISM_ANGLE].m_ssize = initial_astigmatism_angle.ssize();

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

            // Save the default values in case the parameter isn't fitted.
            for (auto& e: initial_phase_shift.span)
                m_initial_phase_shift.push_back(e);
            for (auto& e: initial_astigmatism_value.span)
                m_initial_astigmatism_value.push_back(e);
            for (auto& e: initial_astigmatism_angle.span)
                m_initial_astigmatism_angle.push_back(e);

            // Initialize the defocus values.
            for (auto&& [defocus, slice]: noa::zip(defoci(), metadata))
                defocus = slice.defocus.value;

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

        const MetadataStack& m_metadata;
        const Grid& m_grid;
        const Patches& m_patches;

        Parameters m_parameters{};
        Memoizer m_memoizer{};

        // Shape
        i64 m_n; // number of images
        i64 m_p; // number of patches per image
        i64 m_t; // number of patches per stack, n*p
        i64 m_s; // size of the 1d spectra

        // Patches and their ctfs.
        noa::Linspace<f64> m_fftfreq_range_2d{};
        noa::Linspace<f64> m_fftfreq_range_1d{};
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
        View<f32> m_spectra;         // (t,1,1,w)
        View<f32> m_spectra_weights; // (t,1,1,w)
        View<f32> m_spectra_average; // (n,1,1,w)

        Array<f32> m_gaussian_filter;

        Array<f64> m_nccs; // (3,1,1,n)
        bool m_are_rotational_averages_ready{false};
        Background m_background;

        std::vector<f64> m_gradient_buffer;

    public:
        Fitter(
            const MetadataStack& metadata,
            const Grid& grid,
            const Patches& patches,
            const ns::CTFIsotropic<f64>& average_ctf,
            const Vec<f64, 2>& fftfreq_range,
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
            //
            m_parameters = Parameters(
                metadata, phase_shift, astigmatism_value, astigmatism_angle,
                relative_bounds, Background::smallest_defocus_for_fitting(fftfreq_range, average_ctf)
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

            m_memoizer = Memoizer(m_parameters.ssize(), 5),

            // Quick access of the dimensions.
            m_n = patches.n_slices();
            m_p = patches.n_patches_per_slice();
            m_t = patches.n_patches_per_stack(); // n * p

            // When computing the rotational average, don't render the low frequencies before fftfreq_range[0].
            const auto spectrum_size = patches.shape().width() / 2 + 1;
            const auto fftfreq_step = fftfreq_range[1] / static_cast<f64>(spectrum_size - 1);
            const auto start_index = noa::round(fftfreq_range[0] / fftfreq_step);
            m_s = spectrum_size - static_cast<i64>(start_index);
            m_fftfreq_range_1d = {start_index * fftfreq_step, fftfreq_range[1]};
            m_fftfreq_range_2d = {0, fftfreq_range[1]};

            // Prepare for the rotational averages.
            // Since accesses are per row, use a pitched layout for better performance on the GPU.
            const auto options = patches.rfft_ps().options();
            const auto options_managed = ArrayOption{options}.set_allocator(Allocator::MANAGED);

            m_buffer = Array<f32>({m_t + m_t + m_n, 1, 1, m_s}, options_managed); // FIXME MANAGED_PITCHED
            m_spectra = m_buffer.view().subregion(ni::Slice{0, m_t}); // (t,1,1,s)
            m_spectra_weights = m_buffer.view().subregion(ni::Slice{m_t, m_t + m_t}); // (t,1,1,s)
            m_spectra_average = m_buffer.view().subregion(ni::Slice{m_t + m_t}); // (n,1,1,s)

            // Allocate for the CTFs. Everything needs to be dereferenceable.
            // Initialize CTFs with the microscope parameters.
            // The defocus and phase-shift are going to be overwritten.
            m_ctfs = Array<CTFIsotropic64>(m_n, options_managed);
            m_ctfs_isotropic = Array<CTFIsotropic64>(m_t, options_managed);
            m_ctfs_anisotropic = Array<CTFAnisotropic64>(m_t, options_managed);
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
            m_nccs = noa::Array<f64>({3, 1, 1, m_n});

            // Precompute the spline range and weights.
            m_tilt_range = metadata.tilt_range();
            m_time_range = metadata.time_range().as<f64>();
            m_phase_shift_weights = Array<f64>({1, 1, phase_shift.ssize(), m_n});
            m_astig_value_weights = Array<f64>({1, 1, astigmatism_value.ssize(), m_n});
            m_astig_angle_weights = Array<f64>({1, 1, astigmatism_angle.ssize(), m_n});

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
            for (i64 i{}; i < m_n; ++i) {
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
                const auto chunk = m_patches.chunk_slice(i);
                const auto ctfs_anisotropic = m_ctfs_anisotropic.span_1d().subregion(chunk);
                const auto ctfs_isotropic = m_ctfs_isotropic.span_1d().subregion(chunk);
                const auto slice_spacing = Vec<f64, 2>::from_value(ictfs[i].pixel_size());
                const auto slice_angles = noa::deg2rad(m_metadata[i].angles) + angle_offsets;
                const auto patch_centers = m_grid.patches_centers();

                for (i64 j{}; j < m_p; ++j) {
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
            auto t = Logger::trace_scope_time("cost");
            update_ctfs();

            // Compute the 1d spectra, if needed.
            if (not m_are_rotational_averages_ready) {
                const auto patches_shape = m_patches.shape().push_front(Vec{m_t, i64{1}});

                if (m_parameters[ASTIGMATISM_VALUE].is_fitted()) {
                    // This is the most expensive step of the optimization, and since the astigmatism
                    // is part of the rotational average, it needs to be recomputed everytime.
                    noa::Event start, end;
                    start.record(Stream::current(m_spectra.device()));
                    rotational_average_anisotropic<"h2h">(
                    // ng::rotational_average_anisotropic<"h2h">(
                        m_patches.rfft_ps(), patches_shape, m_ctfs_anisotropic.view(),
                        m_spectra, m_spectra_weights, {
                            .input_fftfreq = m_fftfreq_range_2d,
                            .output_fftfreq = m_fftfreq_range_1d,
                        });

                    end.record(Stream::current(m_spectra.device()));
                    end.synchronize();
                    fmt::println("ra took={}", noa::Event::elapsed(start, end));
                } else {
                    ng::rotational_average<"h2h">(
                        m_patches.rfft_ps(), patches_shape,
                        m_spectra, m_spectra_weights, {
                            .input_fftfreq = m_fftfreq_range_2d,
                            .output_fftfreq = m_fftfreq_range_1d,
                        });
                    m_are_rotational_averages_ready = true;
                }
            }

            // noa::Event start, end;
            // start.record(Stream::current(m_spectra.device()));
            // Compute the average spectrum of each image to compute the per-image backgrounds.
            const auto spectra_average_smoothed = m_spectra_weights.subregion(ni::Slice{0, m_n}); // reuse buffer
            ng::fuse_spectra( // (n*p,1,1,s) -> (n,1,1,s)
                m_spectra, m_fftfreq_range_1d, m_ctfs_isotropic.view(),
                m_spectra_average, m_fftfreq_range_1d, m_ctfs.view(),
                spectra_average_smoothed
            );
            // end.record(Stream::current(m_spectra.device()));
            // end.synchronize();
            // fmt::println("fuse took={}", noa::Event::elapsed(start, end));

            ns::convolve(
                m_spectra_average, spectra_average_smoothed,
                m_gaussian_filter.view(), {.border = noa::Border::REFLECT}
            );

            // Prepare direct access.
            const auto spectrum_np = m_spectra.span().filter(0, 3).as_contiguous();
            const auto ctf_np = m_ctfs_isotropic.span_1d();
            const auto spectrum_n = spectra_average_smoothed.span().filter(0, 3).as_contiguous();
            const auto ctf_n = m_ctfs.span_1d();
            const auto ncc_n = m_nccs.subregion(nccs_index).span_1d();
            const auto baseline = m_spectra_weights.subregion(ni::Slice{m_n, m_n + 1}).span_1d();

            // Wait for the compute device. Everything below is done on the CPU.
            m_spectra.eval();

            // Compute the background based on the average spectrum of the image.
            // Then for every patch of the image, compute the NCC between the
            // background-subtracted spectrum and the simulated CTF.
            auto fftfreq_range = Vec{m_fftfreq_range_1d.start, m_fftfreq_range_1d.stop};
            f64 sum{};
            for (i64 i{}; i < m_n; ++i) { // per image
                // As opposed to the fitting range, the background is reevaluated every time.
                // This is mostly useful for the astigmatism search; fitting the background helps to identify
                // cases where the Thon rings are averaged out due to a wrong astigmatism.
                m_background.fit(spectrum_n[i], fftfreq_range, ctf_n[i]);
                m_background.sample(baseline, fftfreq_range);

                // Get the image spectra and CTFs.
                const auto chunk = m_patches.chunk_slice(i);
                const auto spectrum_p = spectrum_np.subregion(chunk);
                const auto ctf_p = ctf_np.subregion(chunk);

                f64 ncc{};
                for (i64 j{}; j < m_p; ++j) // per patch
                    ncc += normalized_cross_correlation(
                        spectrum_p[j], ctf_p[j], fftfreq_range, m_fitting_ranges[i], baseline);
                ncc /= static_cast<f64>(m_p);
                ncc_n[i] = ncc;
                sum += ncc;
            }
            return sum / static_cast<f64>(m_n);
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
            for (i64 i{}; i < span.ssize(); ++i) {
                f64 cost_minus_delta{0};
                f64 cost_plus_delta{0};
                for (i64 j{}; j < m_n; ++j) {
                    f64 weight{};
                    if (span.ssize() == 1) {
                        // If there's a single variable, it affects every image.
                        weight = 1;
                    } else if (span.ssize() == m_n) {
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
                cost_minus_delta /= static_cast<f64>(m_n);
                cost_plus_delta /= static_cast<f64>(m_n);
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
            auto cartesian = noa::like<f32>(m_patches.rfft_ps());
            noa::cast(m_patches.rfft_ps(), cartesian);

            auto polar = noa::like(cartesian);
            auto shells = Array<f32>(cartesian.shape().set<2>(1), polar.options());

            const auto patches_shape = m_patches.shape().push_front(Vec{m_t, i64{1}});
            for (i32 i{}; i < 15; ++i) {
                noa::Event start, end;
                start.record(Stream::current(m_spectra.device()));
                // rotational_average_anisotropic<"h2h">(
                //     m_patches.rfft_ps(), patches_shape, m_ctfs_anisotropic.view(),
                //     m_spectra, m_spectra_weights, {
                //         .input_fftfreq = m_fftfreq_range_2d,
                //         .output_fftfreq = m_fftfreq_range_1d,
                //     });

                ng::spectrum2polar<"h2fc">(cartesian, patches_shape, polar, {.interp = noa::Interp::NEAREST});
                noa::reduce_axes_ewise(polar, f32{}, shells, noa::Plus{});

                end.record(Stream::current(m_spectra.device()));
                end.synchronize();
                fmt::println("ra took={}", noa::Event::elapsed(start, end));
            }

            panic();


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
            target_ctf.set_defocus(average_defocus / static_cast<f64>(m_n));
            target_ctf.set_phase_shift(min_phase_shift);

            // Compute the average spectrum of each image.
            const auto patches_shape = m_patches.shape().push_front(Vec{m_t, i64{1}});
            ng::rotational_average_anisotropic<"h2h">(
                m_patches.rfft_ps(), patches_shape, m_ctfs_anisotropic.view(),
                m_spectra, m_spectra_weights, {
                    .input_fftfreq = m_fftfreq_range_2d,
                    .output_fftfreq = m_fftfreq_range_1d,
                });
            auto spectra_average_smoothed = m_spectra_weights.subregion(ni::Slice{0, m_n});
            ng::fuse_spectra( // (p*n,1,1,w) -> (n,1,1,w)
                m_spectra, m_fftfreq_range_1d, m_ctfs_isotropic.view(),
                m_spectra_average, m_fftfreq_range_1d, m_ctfs.view(),
                spectra_average_smoothed
            );
            m_spectra_average.to(spectra_average_smoothed); // FIXME
            // ns::convolve(
            //     m_spectra_average, spectra_average_smoothed, m_gaussian_filter.view(),
            //     {.border = noa::Border::REFLECT}
            // );

            // Wait for the compute device and prepare for direct access.
            auto buffer = m_spectra_average.view().reinterpret_as_cpu();
            auto phased = buffer.subregion(0);
            auto spectrum = buffer.subregion(1);
            auto weights = buffer.subregion(2);
            noa::fill(spectrum, 0);
            noa::fill(weights, 0);

            const auto fftfreq_range = Vec{m_fftfreq_range_1d.start, m_fftfreq_range_1d.stop};
            const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(m_s - 1);
            const auto spectrum_n = spectra_average_smoothed.span().filter(0, 3).as_contiguous();
            const auto ctf_n = m_ctfs.span_1d();

            for (i64 i{}; i < m_n; ++i) {
                // Fit the background of each image and tune based on the local NCC
                // between the background-subtracted spectrum and the simulated CTF.
                auto fitting_range = m_background.fit_and_tune_fitting_range(spectrum_n[i], fftfreq_range, ctf_n[i]);
                m_fitting_ranges[i] = fitting_range;

                // That's technically all we need, however, for diagnostics, reconstruct the spectrum of the stack:

                // Scale to the target CTF.
                m_background.subtract(spectrum_n[i], spectrum_n[i], fftfreq_range);
                ng::phase_spectra(
                    View(spectrum_n[i]), m_fftfreq_range_1d, ctf_n[i],
                    phased, m_fftfreq_range_1d, target_ctf, {.interp = noa::Interp::CUBIC}
                );
                for (auto j: noa::irange(2)) {
                    auto phase = ctf_n[i].phase_at(fitting_range[j]);
                    fitting_range[j] = target_ctf.fftfreq_at(phase);
                }

                // Before adding this spectrum to the average, get the L2-norm within the fitting range.
                f32 l2_norm{};
                for (i64 j{}; const auto& e: phased.span_1d()) {
                    const f64 fftfreq = static_cast<f64>(j++) * fftfreq_step + fftfreq_range[0];
                    if (fitting_range[0] <= fftfreq and fftfreq <= fitting_range[1])
                        l2_norm += e * e;
                }
                l2_norm = std::sqrt(l2_norm);

                // Exclude regions outside the fitting range from the average.
                for (i64 j{}; auto&& [p, w, s]: noa::zip(phased.span_1d(), weights.span_1d(), spectrum.span_1d())) {
                    const f64 fftfreq = static_cast<f64>(j++) * fftfreq_step + fftfreq_range[0];
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

            save_plot_xy(m_fftfreq_range_1d, spectrum, output_directory / "fitting_range.txt", {
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
            SplineGridCubic<f64, 1> astigmatism_angle
        ) {
            phase_shift.update_from_span(m_parameters.phase_shift().span);
            astigmatism_value.update_from_span(m_parameters.astigmatism_value().span);
            astigmatism_angle.update_from_span(m_parameters.astigmatism_angle().span);

            // Update metadata.
            const auto defoci = m_parameters.defoci();
            const auto angle_offsets = noa::rad2deg(m_parameters.angles());
            Logger::trace("angle_offsets={}, astig={::.4f}, {::.4f}", // FIXME
                angle_offsets, m_parameters.astigmatism_value().span, m_parameters.astigmatism_angle().span);
            for (i64 i{}; i < metadata.ssize(); ++i) {
                auto& slice = metadata[i];
                slice.angles = MetadataSlice::to_angle_range(slice.angles + angle_offsets);
                slice.defocus.value = defoci[i];
            }
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
        const Vec<f64, 2>& fftfreq_range,
        ns::CTFIsotropic<f64>& isotropic_ctf,
        const FitRefineOptions& options
    ) {
        auto timer = Logger::info_scope_time("Refine CTF fitting");

        auto fitting_ranges = Array<Vec<f64, 2>>(patches.n_slices());

        // Phase shift.
        auto phase_shift_buffer = Array<f64>(1);
        auto phase_shift = SplineGridCubic<f64, 1>(phase_shift_buffer.span_1d());
        phase_shift.span[0] = metadata[0].phase_shift;

        // Astigmatism. Importantly, use an interpolating spline since we will end up fitting per-image.
        auto astigmatism_buffer = noa::zeros<f64>({2, 1, 1, 5});
        auto astigmatism_value = SplineGridCubic<f64, 1>(astigmatism_buffer.span().subregion(0).as_1d());
        auto astigmatism_angle = SplineGridCubic<f64, 1>(astigmatism_buffer.span().subregion(1).as_1d());
        astigmatism_angle.span[0] = 45.;
        astigmatism_angle.span[1] = 45.;
        astigmatism_angle.span[2] = 45.;
        astigmatism_angle.span[3] = 45.;
        astigmatism_angle.span[4] = 45.;

        Vec<f64, 3> final_angle_offsets{};

        auto run_optimization = [&](
            nlopt_algorithm algorithm, i32 max_number_of_evaluations,
            const Parameters::SetOptions<Vec<f64, 2>>& relative_bounds
        ) {
            auto fitter = Fitter(
                metadata, grid, patches, isotropic_ctf,
                fftfreq_range, fitting_ranges.span_1d(),
                phase_shift, astigmatism_value, astigmatism_angle, relative_bounds
            );
            fitter.update_fitting_range(options.output_directory);
            fitter.fit(algorithm, max_number_of_evaluations);
            // fitter.update_fitting_range(options.output_directory); // FIXME angle_offsets
            fitter.update_metadata(metadata, phase_shift, astigmatism_value, astigmatism_angle);
        };

        // 1. Refine the stage-angles, the per-image defocus and average phase-shift.
        run_optimization(NLOPT_LD_LBFGS, 30, {
            .rotation = options.fit_rotation ? deg2rad(Vec{-5., 5.}) : Vec{0., 0.},
            .tilt =  options.fit_tilt ? deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .pitch = options.fit_pitch ? deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .phase_shift = options.fit_phase_shift ? Vec{0., noa::deg2rad(120.)} : Vec{0., 0.},
            .defocus = Vec{-1., 1.},
        });

        save_plot_xy(
            metadata | stdv::transform([&](auto& s) { return s.angles[1]; }),
            metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
            options.output_directory / "defocus_fit.txt", {
                .title = "Per-tilt defocus",
                .x_name = "Tilts (degrees)",
                .y_name = "Defocus (μm)",
                .label = "Defocus - Refine fit",
            });

        // 2. Fit an average astigmatism.
        Logger::trace("Enable astigmatism");
        run_optimization(NLOPT_GD_STOGO, 30, {
            .phase_shift = options.fit_phase_shift ? noa::deg2rad(Vec{-20., 20.}) : Vec{0., 0.},
            .defocus = Vec{-0.3, 0.3},
            .astigmatism_value = {-0.6, 0.6},
            .astigmatism_angle = {0, noa::deg2rad(90.)},
        });

        save_plot_xy(
            metadata | stdv::transform([&](auto& s) { return s.angles[1]; }),
            metadata | stdv::transform([](auto& s) { return s.defocus.value; }),
            options.output_directory / "defocus_fit.txt", {
                .title = "Per-tilt defocus",
                .x_name = "Tilts (degrees)",
                .y_name = "Defocus (μm)",
                .label = "Defocus - Astig fit",
            });

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
