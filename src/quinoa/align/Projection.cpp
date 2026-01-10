#include <noa/FFT.hpp>
#include <noa/Xform.hpp>
#include <noa/Signal.hpp>
#include <noa/IO.hpp>

#include "quinoa/CommonFOV.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/Types.hpp"
#include "quinoa/Utilities.hpp"

#include "quinoa/align/Projection.hpp"

namespace {
    using namespace qn;

    Path debug_dir;

    template<usize N = 192>
    struct BitMask {
        static_assert(noa::is_multiple_of(N, 8));
        static constexpr usize WORD_COUNT = 32;
        static constexpr usize N_WORDS = N / 32;
        u32 buffer[N_WORDS]{};

        struct Ref {
            u32& word;
            u32 mask;

            constexpr explicit operator bool() const { return (word & mask) != 0; }
            constexpr auto operator=(bool value) -> Ref& {
                if (value) word |= mask; else word &= ~mask;
                return *this;
            }
        };

        constexpr auto operator[](nt::integer auto i) const -> bool {
            noa::bounds_check(N, i);
            const u32 data = buffer[i / 32];
            const u32 mask = 1 << (i % 32);
            return data & mask;
        }

        constexpr auto operator[](nt::integer auto i) noexcept -> Ref {
            noa::bounds_check(N, i);
            return Ref{buffer[i / 32], 1u << (i % 32)};
        }
    };

    auto is_reference_included(
        const Metadata::Image& target,
        const Metadata::Image& candidate_reference,
        f64 max_tilt_difference
    ) {
        return std::abs(target.angles[1] - candidate_reference.angles[1]) <= max_tilt_difference;
    }

    // This is only used for the Fourier extraction step.
    // Given the iwise-index w and the blackman window size, return the fftfreq offset. For instance,
    // window_size=9: w=[0..8] -> [-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2]
    template<nt::integer Int, nt::real Coord>
    constexpr NOA_FHD Coord w_index_to_fftfreq_offset(Int w, Int window_size, Coord spectrum_size) {
        return static_cast<Coord>(w - window_size / 2) / spectrum_size;
    }

    template<bool FAST = false, typename T>
    NOA_FHD auto windowed_sinc_at(T fftfreq, T fftfreq_sinc, T fftfreq_blackman) -> T {
        // https://www.desmos.com/calculator/tu5b8aqg2e
        constexpr T PI = noa::Constant<T>::PI;
        fftfreq *= PI;
        const auto blackman_cutoff = fftfreq / fftfreq_blackman;
        #ifdef __CUDA_ARCH__
        if constexpr (FAST and std::same_as<T, f32>) {
            const auto blackman = 0.42f + 0.5f * __cosf(blackman_cutoff) + 0.08f * __cosf(2 * blackman_cutoff);
            const auto x = fftfreq / fftfreq_sinc;
            const auto sinc = x == 0 ? 1 : __sinf(x) / x;
            return sinc * blackman;
        }
        #endif
        const auto blackman =
            static_cast<T>(0.42) +
            static_cast<T>(0.5) * noa::cos(blackman_cutoff) +
            static_cast<T>(0.08) * noa::cos(2 * blackman_cutoff);
        const auto sinc = noa::sinc(fftfreq / fftfreq_sinc);
        return sinc * blackman;
    }

    class Sampler {
    public:
        using input_span_type = SpanContiguous<const c32, 3, i32>;
        using input_interp_type = nx::InterpolatorSpectrum<2, "hc2h", nx::Interp::LINEAR, input_span_type>;

        input_interp_type input_slices{};
        i32 n_input_slices{};

        SpanContiguous<c32, 2, i32> output_slice{};
        SpanContiguous<f32, 2, i32> output_weights{};

        SpanContiguous<const f32, 1, i32> windowed_sinc{};
        SpanContiguous<nx::Quaternion<f32>> insertion_inverse_rotation{};
        Mat<f32, 3, 3> extraction_forward_rotation{};
        Vec<f32, 2> f_shape{};
        BitMask<> reference_mask{};

        f32 volume_z{};
        f32 insert_fftfreq_sinc{};
        f32 insert_fftfreq_blackman{};
        i32 extract_blackman_size{};

        [[nodiscard]] NOA_HD auto sample_input_slices(const Vec<f32, 3>& fftfreq_3d) const {
            c32 value{};
            f32 weight{};
            for (i32 i{}; i < n_input_slices; ++i) {
                if (not reference_mask[i])
                    continue; // this slice isn't included, skip it

                // Project the 3d frequency onto the input central-slice.
                const auto fftfreq_3d_slice = insertion_inverse_rotation[i].rotate(fftfreq_3d);
                const auto fftfreq_from_slice = fftfreq_3d_slice[0]; // distance along the normal

                // Add the contribution of the central-slice if it affects the current frequency.
                if (noa::abs(fftfreq_from_slice) < insert_fftfreq_blackman) {
                    const auto sinc = windowed_sinc_at<true>(
                        fftfreq_from_slice, insert_fftfreq_sinc, insert_fftfreq_blackman
                    );
                    const auto frequency_yx = fftfreq_3d_slice.pop_front() * f_shape;
                    value += input_slices.interpolate_spectrum_at(frequency_yx, i) * sinc;
                    weight += sinc;
                }
            }
            return Pair{value, weight};
        }

        NOA_HD constexpr void operator()(i32 y, i32 x) const {
            // Compute the 3d fftfreq within the volume.
            const auto frequency_2d = nf::index2frequency<false, true>(Vec{y, x}, output_slice.shape());
            const auto fftfreq_2d = frequency_2d.as<f32>() / f_shape;
            const auto fftfreq_3d = extraction_forward_rotation * fftfreq_2d.push_front(0);

            c32 ovalue{};
            f32 oweight{};
            for (i32 w{}; w < extract_blackman_size; ++w) {
                // Offset the volume z for the z-windowed-sinc.
                const auto fftfreq_z_offset = w_index_to_fftfreq_offset(w, extract_blackman_size, volume_z);
                auto fftfreq_3d_w = fftfreq_3d;
                fftfreq_3d_w[0] += fftfreq_z_offset;

                if (dot(fftfreq_3d_w, fftfreq_3d_w) > 0.25f)
                    continue;

                // Sample the virtual volume at the required fftfreq.
                const auto [value, weight] = sample_input_slices(fftfreq_3d_w);

                // z-windowed sinc.
                const auto convolution_weight = windowed_sinc[w];
                ovalue += value * convolution_weight;
                oweight += weight * convolution_weight;
            }

            output_slice(y, x) = ovalue / noa::max(oweight, 1.f);
            output_weights(y, x) = oweight;
        }
    };

    struct WeightCorrection {
        SpanContiguous<const f32, 2, i32> weights_padded_rfft{};
        SpanContiguous<f32, 2, i32> target_rfft{};

        NOA_HD void operator()(const Vec<i32, 2>& indices) const {
            // The weights are oversampled by 2 compared to the target.
            // To get the weight at a given target frequency, a simple nearest-ish interpolation is fine.
            const auto target_frequency = nf::index2frequency<true, true>(indices, target_rfft.shape());
            const auto weight_frequency = target_frequency * 2;
            const auto weight = weights_padded_rfft(weight_frequency);

            // Downweight the target at frequencies that aren't fully sampled in the projected reference.
            target_rfft(indices) *= noa::min(weight, 1.f);
        }
    };

    class Projector {
    private:
        View<f32> m_reference_padded;
        View<c32> m_references_padded_rfft;
        View<c32> m_projected_padded_rfft;
        View<f32> m_weights_padded_rfft;
        View<f32> m_target_and_projected;
        View<c32> m_target_and_projected_rfft;
        View<f32> m_image_buffer;
        View<f32> m_xmap;
        View<f32> m_xmap_centered;

        isize m_n_references{};
        Array<f32> m_windowed_sinc;
        Array<nx::Quaternion<f32>> m_insertion_inverse_rotations;
        Sampler m_sampler;
        std::vector<Metadata::Image> m_references_metadata;

    public:
        Projector(
            const Array<f32>& reference_padded,
            const Array<c32>& references_padded_rfft,
            const Array<c32>& projected_padded_rfft,
            const Array<f32>& weights_padded_rfft,
            const Array<f32>& target_and_projected,
            const Array<c32>& target_and_projected_rfft,
            const Array<f32>& image_buffer,
            const Array<f32>& xmap,
            const Array<f32>& xmap_centered,
            const ProjectionMatchingParameters& parameters
        ) :
            m_reference_padded(reference_padded.view()),
            m_references_padded_rfft(references_padded_rfft.view()),
            m_projected_padded_rfft(projected_padded_rfft.view()),
            m_weights_padded_rfft(weights_padded_rfft.view()),
            m_target_and_projected(target_and_projected.view()),
            m_target_and_projected_rfft(target_and_projected_rfft.view()),
            m_image_buffer(image_buffer.view()),
            m_xmap(xmap.view()),
            m_xmap_centered(xmap_centered.view())
        {
            // Allocate for the quaternions encoding the 3d rotation of the input central-slices.
            const auto device = references_padded_rfft.device();
            const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
            m_insertion_inverse_rotations = Array<nx::Quaternion<f32>>(references_padded_rfft.shape()[0], options);

            // Prepare the w-windowed-sinc convolution filter.
            const auto volume_z = static_cast<f64>(size_padded());
            const auto& esinc = parameters.extraction_sinc;
            const auto [extract_blackman_size, extract_window_total_weight] = nx::details::z_window_spec<i32>(
                esinc.fftfreq_sinc, esinc.fftfreq_blackman, volume_z);

            m_windowed_sinc = Array<f32>(extract_blackman_size, options);
            for (i32 i{}; auto& e: m_windowed_sinc.span_1d()) {
                const auto fftfreq_z_offset = nx::details::w_index_to_fftfreq_offset(i++, extract_blackman_size, volume_z);
                const auto convolution_weight =
                    nx::details::windowed_sinc(fftfreq_z_offset, esinc.fftfreq_sinc, esinc.fftfreq_blackman) /
                    extract_window_total_weight;
                e = static_cast<f32>(convolution_weight);
            }

            // Initialize the sampler.
            const auto references = m_references_padded_rfft.span_contiguous<const c32, 3, i32>();
            const auto shape_2d = shape_padded().pop_front<2>().as<i32>();
            m_sampler = Sampler{
                .input_slices = Sampler::input_interp_type(references, shape_2d),
                .output_slice = m_projected_padded_rfft.span_contiguous<c32, 2, i32>(),
                .output_weights = m_weights_padded_rfft.span_contiguous<f32, 2, i32>(),
                .windowed_sinc = m_windowed_sinc.span_1d<const f32, i32>(),
                .insertion_inverse_rotation = m_insertion_inverse_rotations.span_1d(),
                .f_shape = shape_2d.vec.as<f32>(),
                .volume_z = static_cast<f32>(volume_z),
                .insert_fftfreq_sinc = static_cast<f32>(parameters.insertion_sinc.fftfreq_sinc),
                .insert_fftfreq_blackman = static_cast<f32>(parameters.insertion_sinc.fftfreq_blackman),
                .extract_blackman_size = extract_blackman_size,
            };
        }

        [[nodiscard]] auto project_and_correlate_next(
            const View<const f32>& stack,
            const Metadata::Image& reference_metadata,
            const Metadata::Image& target_metadata,
            const CommonFOV& common_fov,
            const ProjectionMatchingParameters& parameters
        ) -> Vec<f64, 2> {
            preprocess_new_reference(stack, reference_metadata, common_fov, parameters);
            compute_projected_reference(target_metadata, parameters.max_tilt_difference);

            // Find the shift between the target and the projected-reference.
            // TODO It seems that we need to iterate to have a stable shift. It's probably due to the
            // selected the best shifts. Iterating here is not an issue performance-wise (most of the cost is the
            // projection itself, which is done by now) and usually it needs 5 (at low tilts) to 75-125 (at high tilts)
            // iterations to converge.

            auto final_shifts = Vec<f64, 2>{};
            auto updated_target_metadata = target_metadata;
            constexpr i64 MAX_N_ITERATIONS = 1; // FIXME
            i64 iteration{};
            for (; iteration < MAX_N_ITERATIONS; iteration += 1) {
                const auto shift = cross_correlate(stack, updated_target_metadata, common_fov, parameters);
                updated_target_metadata.shifts += shift;
                final_shifts += shift;
                Logger::trace("shift={::.4f}", shift);

                if (abs(shift) < parameters.shift_tolerance)
                    break;
            }
            if (iteration == MAX_N_ITERATIONS)
                Logger::warn("cross_correlate didn't converge");
            return final_shifts;
        }

    public:
        [[nodiscard]] auto size_padded() const noexcept -> isize {
            return m_references_padded_rfft.shape().height();
        }
        [[nodiscard]] auto shape_padded() const noexcept -> Shape4 {
            return {1, 1, size_padded(), size_padded()};
        }
        [[nodiscard]] auto shape_original() const noexcept -> Shape4 {
            return m_image_buffer.shape().filter(2, 3).push_front<2>(1);
        }
        [[nodiscard]] auto center_original() const noexcept -> Vec<f64, 2> {
            return (shape_original().filter(2, 3).vec / 2).as<f64>();
        }
        [[nodiscard]] auto padding() const noexcept -> Vec<isize, 4> {
            return (shape_padded() - shape_original()).vec;
        }

        void preprocess_new_reference(
            const View<const f32>& stack,
            const Metadata::Image& reference_metadata,
            const CommonFOV& common_fov,
            const ProjectionMatchingParameters& parameters
        ) {
            // Register the new central-slice.
            m_n_references += 1;
            m_sampler.n_input_slices = static_cast<i32>(m_n_references);
            m_references_metadata.push_back(reference_metadata);

            // Keep only the common field-of-view.
            common_fov.apply_fov(
                stack.subregion(reference_metadata.index), m_image_buffer, reference_metadata, {
                    .smooth_edge_percent = parameters.smooth_edge_percent,
                    .add_shifts = true,
                });

            // Prepare the central-slice.
            // 1. Zero-pad the image and place it at the end of the main buffer with the other references.
            // 1. The Fourier insertion is done on centered central-slices, so remap.
            // 2. Place the rotation center of the image at the origin.
            const auto reference_padded_rfft = m_references_padded_rfft.subregion(m_n_references - 1);
            noa::resize(m_image_buffer, m_reference_padded, {}, padding());
            nf::r2c(m_reference_padded, reference_padded_rfft);
            nf::remap("h2hc", reference_padded_rfft, reference_padded_rfft, shape_padded());
            ns::phase_shift_2d<"hc">(
                reference_padded_rfft, reference_padded_rfft, shape_padded(),
                (-center_original() - reference_metadata.shifts).as<f32>()
            );

            // Transform to place the central-slice inside the 3d Fourier (virtual) volume.
            const auto insertion_angles = noa::deg2rad(reference_metadata.angles);
            m_insertion_inverse_rotations.span_1d()[m_n_references - 1] = nx::matrix2quaternion((
                nx::rotate_x(insertion_angles[2]) *
                nx::rotate_y(insertion_angles[1]) *
                nx::rotate_z(-insertion_angles[0])
            ).transpose()).as<f32>();
        }

        void compute_projected_reference(const Metadata::Image& target_metadata, f64 max_tilt_difference) {
            // Transform of the central-slice to extract from the 3d Fourier (virtual) volume.
            const auto extraction_angles = noa::deg2rad(target_metadata.angles);
            m_sampler.extraction_forward_rotation = (
                nx::rotate_x(extraction_angles[2]) *
                nx::rotate_y(extraction_angles[1]) *
                nx::rotate_z(-extraction_angles[0])
            ).as<f32>();

            // Exclude certain references based on their tilt difference with the target.
            m_sampler.reference_mask = {};
            for (i32 i{}; const auto& metadata: m_references_metadata)
                m_sampler.reference_mask[i++] = is_reference_included(target_metadata, metadata, max_tilt_difference);

            // Sample the central-slice from the virtual volume.
            const auto padded_size = static_cast<i32>(size_padded());
            noa::iwise(Shape{padded_size, padded_size}.rfft(), m_weights_padded_rfft.device(), m_sampler);

            ns::phase_shift_2d<"h">(
                m_projected_padded_rfft,
                m_projected_padded_rfft,
                shape_padded(),
                center_original().as<f32>(),
                0.5
            );

            // Go back to real space and removing the zero-padding.
            nf::c2r(m_projected_padded_rfft, m_reference_padded);
            noa::resize(m_reference_padded, m_target_and_projected.subregion(1), {}, -padding());
        }

        auto cross_correlate(
            const View<const f32>& stack,
            const Metadata::Image& target_metadata,
            const CommonFOV& common_fov,
            const ProjectionMatchingParameters& parameters
        ) -> Vec<f64, 2> {
            // Mask the target and projected image with the common FOV.
            common_fov.apply_fov(
                stack.subregion(target_metadata.index),
                m_target_and_projected.subregion(0),
                target_metadata, {
                    .smooth_edge_percent = parameters.smooth_edge_percent,
                    .add_shifts = true,
                });
            common_fov.apply_fov(
                m_target_and_projected.subregion(1), target_metadata, {
                .smooth_edge_percent = parameters.smooth_edge_percent,
                .add_shifts = true,
            });
            // TODO normalize within the mask? have an operator doing everything at once
            noa::write_image(m_target_and_projected, debug_dir / fmt::format("target_and_projected_{:0>2}.mrc", target_metadata.index), {.dtype = "f16"});

            nf::r2c(m_target_and_projected, m_target_and_projected_rfft);

            // Downweight the frequencies that are not well-sampled in the projected reference.
            // TODO how useful is this?
            // FIXME weights are padded! have a operator that does the sampling conversion
            // noa::ewise(
            //     m_weights_padded_rfft, target_rfft,
            //     []NOA_HD(const f32& w, c32& t) { t *= noa::min(w, 1.f); }
            // );

            // Shift the projected reference onto the target.
            const auto origin_to_target_center = center_original() + target_metadata.shifts;
            ns::phase_shift_2d<"h">(
                m_target_and_projected_rfft.subregion(1),
                m_target_and_projected_rfft.subregion(1),
                m_image_buffer.shape(),
                target_metadata.shifts.as<f32>(),
                0.5
            );

            //
            noa::write_image(
                m_target_and_projected_rfft,
                debug_dir / fmt::format("target_and_projected_rfft_{:0>2}.mrc", target_metadata.index),
                {.dtype = "c32"}
            );

            // Cross-correlate.
            // The resulting shift is by how much the target is from the projected reference.
            ns::cross_correlation_map<"h2fc">(
                m_target_and_projected_rfft.subregion(0),
                m_target_and_projected_rfft.subregion(1), m_xmap, {
                    .mode = ns::Correlation::CONVENTIONAL,
                    .ifft_norm = nf::Norm::NONE
                });
            auto shift = find_peak<"fc">(m_xmap, m_xmap_centered, {
                .distortion_angle_deg = target_metadata.angles[0],
                .max_shift_percent = 0.15,
            }).first;
            noa::write_image(m_xmap, debug_dir / fmt::format("xmap_{:0>2}.mrc", target_metadata.index), {.dtype = "f16"});
            // noa::write_image(m_xmap_centered, debug_dir / "xmap_centered.mrc", {.dtype = noa::io::DataType::F16});

            return shift;
        }
    };

    class Sampler2 {
    public:
        using input_span_type = SpanContiguous<const c32, 3, i32>;
        using input_interp_type = nx::InterpolatorSpectrum<2, "hc2h", nx::Interp::LINEAR, input_span_type>;

        input_interp_type input_slices{};
        i32 n_input_slices{};

        SpanContiguous<c32, 2, i32> output_slice{};
        SpanContiguous<f32, 2, i32> output_weights{};

        SpanContiguous<const f32, 1, i32> windowed_sinc{};
        SpanContiguous<nx::Quaternion<f32>> insertion_inverse_rotation{};
        Mat<f32, 3, 3> extraction_forward_rotation{};
        Vec<f32, 2> f_shape{};
        BitMask<> reference_mask{};

        f32 volume_z{};
        f32 insert_fftfreq_sinc{};
        f32 insert_fftfreq_blackman{};
        i32 extract_blackman_size{};

        [[nodiscard]] NOA_HD auto sample_input_slices(const Vec<f32, 3>& fftfreq_3d) const {
            c32 value{};
            f32 weight{};
            for (i32 i{}; i < n_input_slices; ++i) {
                if (not reference_mask[i])
                    continue; // this slice isn't included, skip it

                // Project the 3d frequency onto the input central-slice.
                const auto fftfreq_3d_slice = insertion_inverse_rotation[i].rotate(fftfreq_3d);
                const auto fftfreq_from_slice = fftfreq_3d_slice[0]; // distance along the normal

                // Add the contribution of the central-slice if it affects the current frequency.
                if (noa::abs(fftfreq_from_slice) < insert_fftfreq_blackman) {
                    const auto sinc = windowed_sinc_at<true>(
                        fftfreq_from_slice, insert_fftfreq_sinc, insert_fftfreq_blackman
                    );
                    const auto frequency_yx = fftfreq_3d_slice.pop_front() * f_shape;
                    value += input_slices.interpolate_spectrum_at(frequency_yx, i) * sinc;
                    weight += sinc;
                }
            }
            return Pair{value, weight};
        }

        NOA_HD constexpr void operator()(i32 y, i32 x) const {
            // Compute the 3d fftfreq within the volume.
            const auto frequency_2d = nf::index2frequency<false, true>(Vec{y, x}, output_slice.shape());
            const auto fftfreq_2d = frequency_2d.as<f32>() / f_shape;
            const auto fftfreq_3d = extraction_forward_rotation * fftfreq_2d.push_front(0);

            c32 ovalue{};
            f32 oweight{};
            for (i32 w{}; w < extract_blackman_size; ++w) {
                // Offset the volume z for the z-windowed-sinc.
                const auto fftfreq_z_offset = w_index_to_fftfreq_offset(w, extract_blackman_size, volume_z);
                auto fftfreq_3d_w = fftfreq_3d;
                fftfreq_3d_w[0] += fftfreq_z_offset;

                if (dot(fftfreq_3d_w, fftfreq_3d_w) > 0.25f)
                    continue;

                // Sample the virtual volume at the required fftfreq.
                const auto [value, weight] = sample_input_slices(fftfreq_3d_w);

                // z-windowed sinc.
                const auto convolution_weight = windowed_sinc[w];
                ovalue += value * convolution_weight;
                oweight += weight * convolution_weight;
            }

            output_slice(y, x) = ovalue;
            output_weights(y, x) = oweight;
        }
    };

    class Projector2 {
    private:
        Array<f32> m_reference_padded;
        Array<c32> m_references_padded_rfft;
        Array<c32> m_projected_padded_rfft;
        Array<c32> m_projected2_padded_rfft;
        Array<f32> m_weights_padded_rfft;

        Array<f32> m_target_padded;
        Array<c32> m_target_padded_rfft;

        Array<f32> m_target_and_projected;
        Array<c32> m_target_and_projected_rfft;

        Array<f32> m_image_buffer;
        Array<f32> m_xmap;
        Array<f32> m_xmap_centered;

        isize m_n_references{};
        Array<f32> m_windowed_sinc;
        Array<nx::Quaternion<f32>> m_insertion_inverse_rotations;
        Sampler2 m_sampler;
        std::vector<Metadata::Image> m_references_metadata;

    public:
        Projector2(
            const View<f32>& stack,
            const ProjectionMatchingParameters& parameters
        ) {

            auto device = stack.device();
            auto n_slices = stack.shape()[0];
            {
                auto shape_2d = stack.shape().filter(2, 3);

                const auto size_padded = nf::next_fast_size(noa::max(shape_2d) * 2);
                const auto maximum_n_references = n_slices;
                const auto shape = Shape4{1, 1, shape_2d[0], shape_2d[1]};
                const auto padded_shape = Shape4{1, 1, size_padded, size_padded};
                const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};

                m_image_buffer = Array<f32>(shape, options);
                m_xmap = Array<f32>(shape, options);
                m_xmap_centered = Array<f32>({1, 1, 128, 128}, options);

                m_reference_padded = Array<f32>(padded_shape, options);
                m_target_padded = Array<f32>(padded_shape, options);
                m_projected_padded_rfft = Array<c32>(padded_shape.rfft(), options);
                m_projected2_padded_rfft = Array<c32>(padded_shape.rfft(), options);

                m_weights_padded_rfft = Array<f32>(padded_shape.rfft(), options);
                m_references_padded_rfft = Array<c32>(padded_shape.set<0>(maximum_n_references).rfft(), options);
                m_target_padded_rfft = Array<c32>(padded_shape.rfft(), options);

                m_target_and_projected = Array<f32>(shape.set<0>(2), options);
                m_target_and_projected_rfft = Array<c32>(shape.set<0>(2).rfft(), options);
            }

            // Allocate for the quaternions encoding the 3d rotation of the input central-slices.
            const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
            m_insertion_inverse_rotations = Array<nx::Quaternion<f32>>(n_slices - 1, options);

            // Prepare the w-windowed-sinc convolution filter.
            const auto volume_z = static_cast<f64>(size_padded());
            const auto [extract_blackman_size, extract_window_total_weight] = nx::details::z_window_spec<i32>(
                parameters.extraction_sinc.fftfreq_sinc, parameters.extraction_sinc.fftfreq_blackman, volume_z);

            m_windowed_sinc = Array<f32>(extract_blackman_size, options);
            for (i32 i{}; auto& e: m_windowed_sinc.span_1d()) {
                const auto fftfreq_z_offset = nx::details::w_index_to_fftfreq_offset(i++, extract_blackman_size, volume_z);
                const auto convolution_weight =
                    nx::details::windowed_sinc(fftfreq_z_offset, parameters.extraction_sinc.fftfreq_sinc, parameters.extraction_sinc.fftfreq_blackman) /
                    extract_window_total_weight;
                e = static_cast<f32>(convolution_weight);
            }

            // Initialize the sampler.
            const auto references = m_references_padded_rfft.span_contiguous<const c32, 3, i32>();
            const auto shape_2d = shape_padded().pop_front<2>().as<i32>();
            m_sampler = Sampler2{
                .input_slices = Sampler::input_interp_type(references, shape_2d),
                .output_slice = m_projected_padded_rfft.span_contiguous<c32, 2, i32>(),
                .output_weights = m_weights_padded_rfft.span_contiguous<f32, 2, i32>(),
                .windowed_sinc = m_windowed_sinc.span_1d<const f32, i32>(),
                .insertion_inverse_rotation = m_insertion_inverse_rotations.span_1d(),
                .f_shape = shape_2d.vec.as<f32>(),
                .volume_z = static_cast<f32>(volume_z),
                .insert_fftfreq_sinc = static_cast<f32>(parameters.insertion_sinc.fftfreq_sinc),
                .insert_fftfreq_blackman = static_cast<f32>(parameters.insertion_sinc.fftfreq_blackman),
                .extract_blackman_size = extract_blackman_size,
            };
        }

        [[nodiscard]] auto project_and_correlate_next(
            const View<const f32>& stack,
            const Metadata::Image& reference_metadata,
            const Metadata::Image& target_metadata,
            const CommonFOV& common_fov,
            const ProjectionMatchingParameters& parameters
        ) -> Vec<f64, 2> {
            preprocess_new_reference(stack, reference_metadata, common_fov, parameters.smooth_edge_percent);
            compute_projected_reference(target_metadata, parameters.max_tilt_difference);
            return cross_correlate(stack, target_metadata, common_fov, parameters);
        }

    public:
        [[nodiscard]] auto size_padded() const noexcept -> isize {
            return m_references_padded_rfft.shape().height();
        }
        [[nodiscard]] auto shape_padded() const noexcept -> Shape4 {
            return {1, 1, size_padded(), size_padded()};
        }
        [[nodiscard]] auto shape_original() const noexcept -> Shape4 {
            return m_image_buffer.shape().filter(2, 3).push_front<2>(1);
        }
        [[nodiscard]] auto center_original() const noexcept -> Vec<f64, 2> {
            return (shape_original().filter(2, 3).vec / 2).as<f64>();
        }
        [[nodiscard]] auto padding() const noexcept -> Vec<isize, 4> {
            return (shape_padded() - shape_original()).vec;
        }

        void preprocess_new_reference(
            const View<const f32>& stack,
            const Metadata::Image& reference_metadata,
            const CommonFOV& common_fov,
            f64 smooth_edge_percent
        ) {
            // Register the new central-slice.
            m_n_references += 1;
            m_sampler.n_input_slices = static_cast<i32>(m_n_references);
            m_references_metadata.push_back(reference_metadata);

            // Keep only the common field-of-view.
            common_fov.apply_fov(
                stack.subregion(reference_metadata.index), m_image_buffer.view(), reference_metadata, {
                    .smooth_edge_percent = smooth_edge_percent,
                    .add_shifts = true,
                });

            // Prepare the central-slice. [references..., reference, target, ...]
            const auto reference_padded_rfft = m_references_padded_rfft.subregion(m_n_references - 1);
            noa::resize(m_image_buffer, m_reference_padded, {}, padding());
            nf::r2c(m_reference_padded, reference_padded_rfft);
            nf::remap("h2hc", reference_padded_rfft, reference_padded_rfft, shape_padded());
            ns::phase_shift_2d<"hc">(
                reference_padded_rfft.subregion(0), reference_padded_rfft.subregion(0), shape_padded(),
                (-center_original() - reference_metadata.shifts).as<f32>()
            );

            // Transform to place the central-slice inside the 3d Fourier (virtual) volume.
            const auto insertion_angles = noa::deg2rad(reference_metadata.angles);
            m_insertion_inverse_rotations.span_1d()[m_n_references - 1] = nx::matrix2quaternion((
                nx::rotate_x(insertion_angles[2]) *
                nx::rotate_y(insertion_angles[1]) *
                nx::rotate_z(-insertion_angles[0])
            ).transpose()).as<f32>();
        }

        void compute_projected_reference(const Metadata::Image& target_metadata, f64 max_tilt_difference) {
            // Transform of the central-slice to extract from the 3d Fourier (virtual) volume.
            const auto extraction_angles = noa::deg2rad(target_metadata.angles);
            m_sampler.extraction_forward_rotation = (
                nx::rotate_x(extraction_angles[2]) *
                nx::rotate_y(extraction_angles[1]) *
                nx::rotate_z(-extraction_angles[0])
            ).as<f32>();

            // Exclude certain references based on their tilt difference with the target.
            m_sampler.reference_mask = {};
            for (i32 i{}; const auto& metadata: m_references_metadata)
                m_sampler.reference_mask[i++] = is_reference_included(target_metadata, metadata, max_tilt_difference);

            // Sample the central-slice from the virtual volume.
            const auto padded_size = static_cast<i32>(size_padded());
            noa::iwise(Shape{padded_size, padded_size}.rfft(), m_weights_padded_rfft.device(), m_sampler);
        }

        auto cross_correlate(
            const View<const f32>& stack,
            const Metadata::Image& target_metadata,
            const CommonFOV& common_fov,
            const ProjectionMatchingParameters& parameters
        ) -> Vec<f64, 2> {
            // center the projected slice onto the target
            ns::phase_shift_2d<"h">(
                m_projected_padded_rfft, m_projected_padded_rfft, shape_padded(),
                (center_original() + target_metadata.shifts).as<f32>(), 0.5
            );

            // Zero-pad the target.
            common_fov.apply_fov(
                stack.subregion(target_metadata.index), m_image_buffer.view(), target_metadata, {
                .smooth_edge_percent = parameters.smooth_edge_percent,
                .add_shifts = true,
            });
            noa::write_image(m_image_buffer, debug_dir / fmt::format("target_{:0>2}.mrc", target_metadata.index)); // , {.dtype = "f16"}
            noa::resize(m_image_buffer, m_target_padded, {}, padding());
            nf::r2c(m_target_padded, m_target_padded_rfft);

            // Compute the weighted projection with and without the target.
            noa::ewise(
                noa::wrap(m_target_padded_rfft, m_weights_padded_rfft),
                noa::wrap(m_projected_padded_rfft, m_projected2_padded_rfft),
                []NOA_HD(c32 t, f32 w, c32& projected, c32& projected2) {
                    projected2 = (projected + t) / (w + 1.f);
                    projected  = projected / noa::max(w, 1.f);
                }
            );

            // Go back to real space and removing the zero-padding.
            nf::c2r(m_projected_padded_rfft, m_reference_padded);
            noa::resize(m_reference_padded, m_target_and_projected.subregion(0), {}, -padding());
            nf::c2r(m_projected2_padded_rfft, m_reference_padded);
            noa::resize(m_reference_padded, m_target_and_projected.subregion(1), {}, -padding());

            // mask
            common_fov.apply_fov(
                m_target_and_projected.view().subregion(0), target_metadata, {
                .smooth_edge_percent = parameters.smooth_edge_percent,
                .add_shifts = true,
            });
            common_fov.apply_fov(
                m_target_and_projected.view().subregion(1), target_metadata, {
                .smooth_edge_percent = parameters.smooth_edge_percent,
                .add_shifts = true,
            });
            noa::write_image(m_target_and_projected, debug_dir / fmt::format("target_and_projected_{:0>2}.mrc", target_metadata.index)); // , {.dtype = "f16"}

            // Cross-correlate.
            // The resulting shift is by how much the target is from the projected reference.
            // nf::r2c(m_target_and_projected, m_target_and_projected_rfft);
            // ns::cross_correlation_map<"h2fc">(
            //     m_target_and_projected_rfft.subregion(0),
            //     m_target_and_projected_rfft.subregion(1), m_xmap, {
            //         .mode = ns::Correlation::CONVENTIONAL,
            //         .ifft_norm = nf::Norm::NONE
            //     });
            // auto shift = find_peak<"fc">(m_xmap.view(), m_xmap_centered.view(), {
            //     .distortion_angle_deg = target_metadata.angles[0],
            //     .max_shift_percent = 0.15,
            // }).first;
            // noa::write_image(m_xmap, debug_dir / fmt::format("xmap_{:0>2}.mrc", target_metadata.index), {.dtype = "f16"});

            return {}; //shift
        }
    };
}

namespace qn {
    ProjectionMatcher::ProjectionMatcher(isize n_slices, const Shape2& shape_2d, Device device) {

        // TODO try higher interpolation and reduce size?
        const auto size_padded = nf::next_fast_size(noa::max(shape_2d) * 2);

        const auto maximum_n_references = n_slices - 1;
        const auto shape = Shape4{1, 1, shape_2d[0], shape_2d[1]};
        const auto padded_shape = Shape4{1, 1, size_padded, size_padded};
        const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};

        const auto n0 = Allocator::bytes_currently_allocated(device);

        m_image_buffer = Array<f32>(shape, options);
        m_xmap = Array<f32>(shape, options);
        m_xmap_centered = Array<f32>({1, 1, 128, 128}, options);

        m_reference_padded = Array<f32>(padded_shape, options);
        m_projected_padded_rfft = Array<c32>(padded_shape.rfft(), options);
        m_weights_padded_rfft = Array<f32>(padded_shape.rfft(), options);
        m_references_padded_rfft = Array<c32>(padded_shape.set<0>(maximum_n_references).rfft(), options);

        m_target_and_projected = Array<f32>(shape.set<0>(2), options);
        m_target_and_projected_rfft = Array<c32>(shape.set<0>(2).rfft(), options);

        const auto n1 = Allocator::bytes_currently_allocated(device);

        Logger::trace(
            "Projection matching:\n"
            "  image_shape={}\n"
            "  spectrum_size={}\n"
            "  n_bytes_allocated={:.2f}GB (device={}, allocator={})",
            shape, size_padded, static_cast<f64>(n1 - n0) * 1e-9,
            device, options.allocator
        );
    }

    void ProjectionMatcher::update_shifts(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const ProjectionMatchingParameters& parameters
    ) const {
        auto t = Logger::info_scope_time("Projection-matching: shift alignment");

        auto common_fov = CommonFOV(
            stack.shape().filter(2, 3),
            metadata
        );
        auto projector = Projector(
            m_reference_padded,
            m_references_padded_rfft,
            m_projected_padded_rfft,
            m_weights_padded_rfft,
            m_target_and_projected,
            m_target_and_projected_rfft,
            m_image_buffer,
            m_xmap,
            m_xmap_centered,
            parameters
        );

        // FIXME
        debug_dir = parameters.debug_directory;
        Stream::current({}).set_thread_limit(8);

        // Projection matching, using the lowest tilt as the initial reference,
        // aligning from low-to-high tilts. When a tilt is aligned, it is added
        // to the set of reference images used to compute the projected reference.
        auto projection_metadata = metadata;
        projection_metadata.sort("absolute_tilt");

        for (isize target_index = 1; target_index < projection_metadata.ssize(); ++target_index) {
            const auto& new_reference_slice = projection_metadata[target_index - 1];
            auto& target_slice = projection_metadata[target_index];

            target_slice.shifts += projector.project_and_correlate_next(
                stack, new_reference_slice, target_slice,
                common_fov, parameters
            );
        }

        save_plot_xy(
            projection_metadata | stdv::transform([](auto& slice) { return slice.angles[1]; }),
            projection_metadata | stdv::transform([](auto& slice) { return slice.shifts[1]; }),
                debug_dir / "shifts.txt");

        metadata.update_from(projection_metadata, {.update_shifts = true});
    }

    void update_shifts2(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const ProjectionMatchingParameters& parameters
    ) {
        auto t = Logger::info_scope_time("Projection-matching: shift alignment");

        auto common_fov = CommonFOV(
            stack.shape().filter(2, 3),
            metadata
        );
        auto projector = Projector2(stack, parameters);

        // FIXME
        debug_dir = parameters.debug_directory;
        Stream::current({}).set_thread_limit(8);

        // Projection matching, using the lowest tilt as the initial reference,
        // aligning from low-to-high tilts. When a tilt is aligned, it is added
        // to the set of reference images used to compute the projected reference.
        auto projection_metadata = metadata;
        projection_metadata.sort("absolute_tilt"); // FIXME time?

        for (isize target_index = 1; target_index < projection_metadata.ssize(); ++target_index) {
            const auto& new_reference_slice = projection_metadata[target_index - 1];
            auto& target_slice = projection_metadata[target_index];

            target_slice.shifts += projector.project_and_correlate_next(
                stack, new_reference_slice, target_slice,
                common_fov, parameters
            );
        }

        // save_plot_xy(
        //     projection_metadata | stdv::transform([](auto& slice) { return slice.angles[1]; }),
        //     projection_metadata | stdv::transform([](auto& slice) { return slice.shifts[1]; }),
        //         debug_dir / "shifts.txt");
        //
        // metadata.update_from(projection_metadata, {.update_shifts = true});
    }
}
