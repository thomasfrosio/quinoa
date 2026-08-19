#include <noa/FFT.hpp>
#include <noa/Xform.hpp>
#include <noa/Signal.hpp>
#include <noa/IO.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Optimizer.hpp"
#include "quinoa/Plot.hpp"
#include "quinoa/Types.hpp"
#include "quinoa/Utilities.hpp"

#include "quinoa/align/CommonFOV.hpp"
#include "quinoa/align/Projection.hpp"

namespace {
    using namespace qn;

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
    template<nt::integer I, nt::real T>
    constexpr NOA_FHD auto w_index_to_fftfreq_offset(I w, I window_size, T spectrum_size) -> T {
        return static_cast<T>(w - window_size / 2) / spectrum_size;
    }

    // Windowed-sinc. This function assumes fftfreq <= fftfreq_blackman,
    // above that the blackman window will start again.
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

    // Odd-sized blackman window for the sampling of the central-slice.
    template<nt::integer I, nt::real T>
    auto blackman_window_size(T fftfreq_blackman, T spectrum_size) -> I {
        // Given a blackman window in range [0, fftfreq_blackman] and a spectrum logical-size
        // (the z size in our case), what is the size of the blackman window, in elements.
        // For instance:
        //  spectrum_size=10, fftfreq_blackman=0.23
        //  rfftfreq=[0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        //  rfftfreq_samples=4.6->5, window_size=11
        //  computed_window=[-0.25,-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2,0.25]
        auto rfftfreq_samples = static_cast<f64>(spectrum_size) * static_cast<f64>(fftfreq_blackman);
        if (noa::allclose(rfftfreq_samples, 1.)) {
            // Due to floating-point precision errors, the default value (1/spectrum_size)
            // may be slightly greater than 1. In this case, we really mean 1.
            rfftfreq_samples = round(rfftfreq_samples);
        } else {
            rfftfreq_samples = ceil(rfftfreq_samples); // include last fraction
        }
        const auto rfftfreq_samples_int = std::max(I{1}, static_cast<I>(rfftfreq_samples));
        auto window_size = 2 * (rfftfreq_samples_int) + 1;

        // Truncate the edges because at these indices, the window is 0, so there's no need to compute it.
        // So using the same example, computed_window_fftfreq_offset=[-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2]
        return window_size - 2;
    }

    // This is only used for the Fourier extraction step.
    // Compute the sum of the z-window, so that it can be directly applied to the extracted values,
    // thereby correcting for the multiplicity on the fly.
    template<nt::integer Int, nt::real Real>
    auto w_window_spec(Real fftfreq_sinc, Real fftfreq_blackman, Real spectrum_size) -> Pair<Int, Real> {
        auto window_size = blackman_window_size<Int>(fftfreq_blackman, spectrum_size);
        Real sum{};
        for (Int i{}; i < window_size; ++i) {
            const auto fftfreq = w_index_to_fftfreq_offset(i, window_size, spectrum_size);
            sum += windowed_sinc_at(fftfreq, fftfreq_sinc, fftfreq_blackman);
        }
        return {window_size, sum};
    }

    class Sampler {
    public:
        using input_span_type = SpanContiguous<const c32, 3, i32>;
        using input_interp_type = nx::InterpolatorSpectrum<2, nf::Layout::H2H, nx::Interp::LINEAR, input_span_type>;

        input_interp_type reference_slices{};
        SpanContiguous<c32, 2, i32> projected_slice{};

        SpanContiguous<const f32, 1, i32> w_windowed_sinc{};
        SpanContiguous<const u8, 1, i32> reference_indices{};
        SpanContiguous<const nx::Quaternion<f32>, 1, i32> reference_rotations{};
        Mat<f32, 3, 3> target_rotation{};
        Vec<f32, 2> f_shape{};

        f32 volume_z{};
        f32 insert_fftfreq_sinc{};
        f32 insert_fftfreq_blackman{};
        i32 extract_blackman_size{};

        [[nodiscard]] NOA_HD auto sample_virtual_volume_at(const Vec<f32, 3>& fftfreq_3d) const {
            c32 value{};
            f32 weight{};
            for (i32 i{}; i < reference_indices.n_elements(); ++i) {
                // Project the 3d frequency onto the input central-slice.
                const auto fftfreq_3d_slice = reference_rotations[i].rotate(fftfreq_3d);
                const auto fftfreq_from_slice = fftfreq_3d_slice[0]; // distance along the normal

                // If the slice affects the current frequency, add its contribution.
                if (noa::abs(fftfreq_from_slice) < insert_fftfreq_blackman) {
                    const auto sinc = windowed_sinc_at<true>(
                        fftfreq_from_slice, insert_fftfreq_sinc, insert_fftfreq_blackman
                    );
                    const auto frequency_yx = fftfreq_3d_slice.pop_front() * f_shape;
                    value += reference_slices.interpolate_spectrum_at(frequency_yx, reference_indices[i]) * sinc;
                    weight += sinc;
                }
            }
            return Pair{value, weight};
        }

        NOA_HD void operator()(i32 v, i32 u) const {
            // Compute the 3d fftfreq within the volume.
            const auto frequency_2d = nf::index2frequency<false, true>(Vec{v, u}, projected_slice.shape());
            const auto fftfreq_2d = frequency_2d.as<f32>() / f_shape;
            const auto fftfreq_3d = target_rotation * fftfreq_2d.push_front(0);

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
                const auto [value, weight] = sample_virtual_volume_at(fftfreq_3d_w);

                // z-windowed sinc.
                const auto convolution_weight = w_windowed_sinc[w];
                ovalue += value * convolution_weight;
                oweight += weight * convolution_weight;
            }

            // Weighted back-projection. Downweight frequencies that are sampled more than once.
            projected_slice(v, u) = ovalue / noa::max(abs(oweight), 1.f);
        }
    };

    struct PrepareReferenceAndTarget {
        SpanContiguous<const f32, 3, i32> stack{};
        SpanContiguous<f32, 2, i32> reference_padded{};
        SpanContiguous<f32, 2, i32> target_padded{};

        ParallelogramMask reference_mask{};
        ParallelogramMask target_mask{};
        Vec<i32, 2> right_edge{};
        i32 reference_index{};
        i32 target_index{};

        NOA_HD void operator()(i32 y, i32 x) const {
            // If inside the padding, set to zero.
            const auto indices = Vec{y, x};
            if (indices.any_ge(right_edge)) {
                reference_padded(indices) = 0;
                target_padded(indices) = 0;
                return;
            }

            // Otherwise copy and apply the mask.
            auto reference = reference_mask(y, x);
            if (reference > 1e-6f)
                reference *= stack(reference_index, y, x);
            reference_padded(indices) = reference;

            auto target = target_mask(y, x);
            if (target > 1e-6f)
                target *= stack(target_index, y, x);
            target_padded(indices) = target;
        }
    };

    struct PrepareTargetAndProjected {
        SpanContiguous<const f32, 3, i32> target_and_projected_padded{};
        SpanContiguous<f32, 3, i32> target_and_projected{};
        ParallelogramMask target_mask{};

        NOA_HD void operator()(i32 y, i32 x) const {
            // Crop and apply the mask.
            const auto mask = target_mask(y, x);
            if (mask > 1e-6f) {
                target_and_projected(0, y, x) = target_and_projected_padded(0, y, x) * mask;
                target_and_projected(1, y, x) = target_and_projected_padded(1, y, x) * mask;
            } else {
                target_and_projected(0, y, x) = 0;
                target_and_projected(1, y, x) = 0;
            }
        }
    };

    struct CrossCorrelate {
        SpanContiguous<const c32, 2, i32> projected_rfft{};
        SpanContiguous<c32, 2, i32> target_rfft{};

        NOA_HD void operator()(i32 y, i32 x) const {
            const auto frequency = nf::index2frequency<false, true>(Vec{y, x}, projected_rfft.shape().filter(0));
            const auto phase_shift = static_cast<f32>(product(1 - 2 * abs(frequency % 2))); // shift by +shape/2

            auto& lhs = target_rfft(y, x);
            auto rhs = projected_rfft(y, x);

            auto cc = lhs * conj(rhs);
            cc /= noa::sqrt(abs(lhs) * abs(rhs)) + 1e-6f;
            cc *= phase_shift; // produce the centered xmap

            // TODO bandpass?
            lhs = cc;
        }
    };

    template<typename T>
    struct ZNCC {
        SpanContiguous<const f32, 2, i32> lhs{};
        SpanContiguous<const f32, 2, i32> rhs{};
        ParallelogramMask mask{};

        using reduced_type = Vec<T, 6>;

        NOA_HD void operator()(i32 y, i32 x, reduced_type& reduced) {
            const auto m = mask(y, x);
            reduced[0] += static_cast<T>(lhs(y, x) * m);
            reduced[1] += static_cast<T>(rhs(y, x) * m);
            reduced[2] += static_cast<T>(m);
            reduced[3] += reduced[0] * reduced[0];
            reduced[4] += reduced[1] * reduced[1];
            reduced[5] += reduced[0] * reduced[1];
        }

        static NOA_HD void join(const reduced_type& reduced, reduced_type& joined) {
            joined += reduced;
        }

        using remove_default_post = bool;
        static NOA_HD void post(const reduced_type& stats, f64& zncc) {
            const auto denom_x = stats[3] - (stats[0] * stats[0]) / stats[2];
            const auto denom_y = stats[4] - (stats[1] * stats[1]) / stats[2];
            auto denom = denom_x * denom_y;
            if (denom <= 0) {
                zncc = 0.;
            } else {
                const auto num = stats[5] - (stats[0] * stats[1]) / stats[2];
                denom = noa::sqrt(denom);
                zncc = num / denom;
            }
        }
    };

    struct Projector {
        Array<c32> m_buffer_padded_rfft; // [references, ..., reference, target, projected]
        Array<f32> m_buffer_padded; // 2 padded images
        Array<f32> m_buffer; // 2 images
        Array<c32> m_buffer_rfft; // 2 slices
        Array<f32> m_xmap_centered; // small xmap

        std::vector<Metadata::Image> m_references_metadata;
        std::vector<Mat<f64, 3, 3>> m_references_metadata_rotations;

        PrepareReferenceAndTarget m_prepare_reference_and_target;

        Array<f32> m_windowed_sinc;

        Array<u8> m_reference_indices;
        Array<u8> m_reference_indices_device;
        Array<nx::Quaternion<f32>> m_reference_rotations;
        Array<nx::Quaternion<f32>> m_reference_rotations_device;

        Sampler m_sampler;

    public:
        Projector() = default;

        explicit Projector(isize n_slices, const Shape2& shape_2d, Device device) {
            const auto n0 = Allocator::bytes_currently_allocated(device);

            // TODO try higher interpolation with reduced padding?
            const auto size_padded = nf::next_fast_size(noa::max(shape_2d) * 2);
            const auto shape = Shape4{1, 1, shape_2d[0], shape_2d[1]};
            const auto padded_shape = Shape4{1, 1, size_padded, size_padded};

            // If there's enough memory use device-only as it seems more performant on some systems.
            const bool has_enough_space = [&] {
                const auto available = device.memory_capacity().free;
                const auto n = size_padded * size_padded * (n_slices + 5);
                const auto rough_estimate = static_cast<usize>(n) * sizeof(f32) * 2;
                return available >= rough_estimate;
            }();
            const auto options = ArrayOption{
                .device = device,
                .allocator = has_enough_space ? Allocator::ASYNC : Allocator::MANAGED,
            };

            m_buffer_padded_rfft = Array<c32>(padded_shape.rfft().set<0>(n_slices + 3), options); // +target, +projected x2
            m_buffer_padded = Array<f32>(padded_shape.set<0>(2), options);
            m_buffer = Array<f32>(shape.set<0>(2), options);
            m_buffer_rfft = Array<c32>(shape.rfft().set<0>(2), options);

            // Small xmap centered on the peak, needs to be dereferenceable.
            m_xmap_centered = Array<f32>({1, 1, 64, 64}, {device, Allocator::MANAGED});

            // TODO Try in-place FFTs?
            if (device.is_gpu()) {
                // All the FFTs.
                const auto fft_options = nf::FFTOptions{.record_and_share_workspace = true};
                nf::r2c(m_buffer_padded.view(), m_buffer_padded_rfft.view().subregion(Slice{0, 2}), fft_options);
                nf::c2r(m_buffer_padded_rfft.view().subregion(Slice{0, 2}), m_buffer_padded.view(), fft_options);
                nf::r2c(m_buffer.view(), m_buffer_rfft.view(), fft_options);
                nf::c2r(m_buffer_rfft.view().subregion(0), m_buffer.view().subregion(0), fft_options);

                const auto workspace = Array<std::byte>(nf::workspace_left_to_allocate(device), options);
                const auto n_plans_set = nf::set_workspace(device, std::move(workspace));
                if (auto left = nf::workspace_left_to_allocate(device); n_plans_set == 0 or left > 0) {
                    Logger::warn(
                        "Failed to set the FFT workspace. A new workspace will have to be allocated, possibly increasing the memory requirements significantly. Please report this. shape={}, workspace_left_to_allocate={}bytes, n_plans_set={}",
                        shape, left, n_plans_set);
                }
            }

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

        void initialize(const View<const f32>& stack, const ProjectionMatchingParameters& parameters) {
            // Reset.
            m_references_metadata.clear();
            m_references_metadata_rotations.clear();

            // Allocate for the quaternions encoding the 3d rotation of the input central-slices.
            const auto max_n_slices = stack.shape()[0] - 1;
            const auto device = m_buffer.device();
            const auto options = ArrayOption{.device = device, .allocator = Allocator::ASYNC};

            m_reference_indices = Array<u8>(max_n_slices);
            m_reference_rotations = Array<nx::Quaternion<f32>>(max_n_slices);
            if (device.is_gpu()) {
                m_reference_indices_device = Array<u8>(max_n_slices, options);
                m_reference_rotations_device = Array<nx::Quaternion<f32>>(max_n_slices, options);
            } else {
                m_reference_indices_device = m_reference_indices;
                m_reference_rotations_device = m_reference_rotations;
            }

            // Prepare the w-windowed-sinc convolution filter.
            const auto shape_padded = m_buffer_padded.shape().pop_front<2>();
            const auto volume_z = static_cast<f64>(shape_padded[0]);
            const auto& esinc = parameters.extraction_sinc;
            const auto [extract_blackman_size, extract_window_total_weight] = w_window_spec<i32>(
                esinc.fftfreq_sinc, esinc.fftfreq_blackman, volume_z);

            m_windowed_sinc = Array<f32>(extract_blackman_size);
            for (i32 i{}; auto& e: m_windowed_sinc.span_1d()) {
                const auto fftfreq_z_offset = w_index_to_fftfreq_offset(i++, extract_blackman_size, volume_z);
                const auto convolution_weight = windowed_sinc_at(fftfreq_z_offset, esinc.fftfreq_sinc, esinc.fftfreq_blackman);
                e = static_cast<f32>(convolution_weight);
            }
            m_windowed_sinc = std::move(m_windowed_sinc).to(options);

            // Initialize operators.
            const auto shape = m_buffer.shape().vec.pop_front<2>().as<i32>();
            m_prepare_reference_and_target = PrepareReferenceAndTarget{
                .stack = stack.span_contiguous<const f32, 3, i32>(),
                .right_edge = shape,
            };

            const auto references = m_buffer_padded_rfft.span_contiguous<const c32, 3, i32>();
            m_sampler = Sampler{
                .reference_slices = Sampler::input_interp_type(references, shape_padded.as<i32>()),
                .w_windowed_sinc = m_windowed_sinc.span_1d<const f32, i32>(),
                .f_shape = shape_padded.vec.as<f32>(),
                .volume_z = static_cast<f32>(volume_z),
                .insert_fftfreq_sinc = static_cast<f32>(parameters.insertion_sinc.fftfreq_sinc),
                .insert_fftfreq_blackman = static_cast<f32>(parameters.insertion_sinc.fftfreq_blackman),
                .extract_blackman_size = extract_blackman_size,
            };
        }

        auto project_and_correlate_next(
            const Metadata::Image& reference_metadata,
            const Metadata::Image& target_metadata,
            const CommonFOV& common_fov,
            bool compute_score,
            f64 smooth_edge_percent,
            f64 max_tilt_difference
        ) {
            // Prepare the reference and target.
            const auto reference_and_target_padded = m_buffer_padded.view();
            m_prepare_reference_and_target.reference_padded = reference_and_target_padded.subregion(0).span_contiguous<f32, 2, i32>();
            m_prepare_reference_and_target.target_padded = reference_and_target_padded.subregion(1).span_contiguous<f32, 2, i32>();

            const auto fov_options = FOVMaskOptions{
                .smooth_edge_percent = smooth_edge_percent,
                .add_shifts = true,

                // In theory, removing the region that are not visible in the lower tilts should help.
                // However, it ends up removing too much perpendicular to the tilt-axis, where the image tilts
                // already limit the available signal. While we could mask based only on the nearby images (the images
                // that provide most of the signal) so that the tilt difference isn't that big, we would effectively
                // end up aligning images on different tomograms. Clearly not an ideal situation either way, but NOT
                // applying the tilt/pitch to the mask seem to produce better alignments (beads are more symmetric)
                // by preventing high-tilt images to drift away too much orthogonal to the tilt-axis.
                .add_tilt_and_pitch = false,
            };
            m_prepare_reference_and_target.reference_mask = common_fov.set_fov(reference_metadata, fov_options);
            m_prepare_reference_and_target.target_mask = common_fov.set_fov(target_metadata, fov_options);
            m_prepare_reference_and_target.reference_index = reference_metadata.index;
            m_prepare_reference_and_target.target_index = target_metadata.index;

            const auto device = m_buffer.device();
            const auto shape_padded_2d = reference_and_target_padded.shape().filter(2, 3).as<i32>();
            const auto shape_padded = shape_padded_2d.as<isize>().push_front<2>(1);
            noa::iwise(shape_padded_2d, device, m_prepare_reference_and_target);

            // Register the new central-slice.
            const auto insertion_angles = noa::deg2rad(reference_metadata.angles);
            m_references_metadata.push_back(reference_metadata);
            m_references_metadata_rotations.push_back((
                nx::rotate_x(+insertion_angles[2]) *
                nx::rotate_y(+insertion_angles[1]) *
                nx::rotate_z(-insertion_angles[0]) // the virtual volume has the tilt-axis aligned onto the y-axis.
            ).transpose()); // volume to central slice

            // Compute the central-slice of the new reference (centered onto the origin) and the target.
            // The buffer is organized as [references..., reference, target, projected].
            const auto n_references = std::ssize(m_references_metadata);
            const auto reference_padded_rfft = m_buffer_padded_rfft.subregion(n_references - 1);
            const auto reference_and_target_padded_rfft = m_buffer_padded_rfft.subregion(Slice{n_references - 1, n_references + 1});
            const auto shape_2d = m_buffer.shape().filter(2, 3).as<i32>();
            const auto original_center = (shape_2d.vec / 2).as<f64>();
            nf::r2c(reference_and_target_padded, reference_and_target_padded_rfft);
            ns::phase_shift_2d<"h">(
                reference_padded_rfft, reference_padded_rfft, shape_padded,
                (-original_center - reference_metadata.shifts).as<f32>()
            );

            // Filter out the references that are not included for this projection (based on the tilt difference).
            // This makes the quaternions contiguous in memory and removes the need to 'continue' the per-slice loop
            // when sampling. Without this, the kernel performance significantly drops as more slices are added,
            // even when using a per-slice bitmask stored directly in the operator.
            usize count{};
            const auto reference_indices = m_reference_indices.span_1d();
            const auto reference_rotations = m_reference_rotations.span_1d();
            for (usize i{}; i < m_references_metadata.size(); ++i) {
                if (is_reference_included(target_metadata, m_references_metadata[i], max_tilt_difference)) {
                    reference_rotations[count] = nx::matrix2quaternion(m_references_metadata_rotations[i]).as<f32>();
                    reference_indices[count] = noa::safe_cast<u8>(i);
                    ++count;
                }
            }
            if (m_reference_indices_device.device().is_gpu()) {
                m_reference_indices.view().to(m_reference_indices_device.view());
                m_reference_rotations.view().to(m_reference_rotations_device.view());
            }
            m_sampler.reference_indices = m_reference_indices_device.span_contiguous<const u8, 1, i32>().subregion(Slice{0, count});
            m_sampler.reference_rotations = m_reference_rotations_device.span_contiguous<const nx::Quaternion<f32>, 1, i32>().subregion(Slice{0, count});

            // auto& stream = Stream::current(device);
            // auto start = noa::Event{};
            // auto end = noa::Event{};
            // start.record(stream);
            // end.record(stream);
            // end.synchronize();
            // Logger::trace("sampling1 {}", Event::elapsed(start, end));

            // Sample the central-slice from the virtual volume.
            // This is the most expensive step of this function, especially for thin samples.
            auto target_and_projected_padded_rfft = m_buffer_padded_rfft.view().subregion(Slice{n_references, n_references + 2});
            auto projected_padded_rfft = target_and_projected_padded_rfft.subregion(1);
            const auto extraction_angles = noa::deg2rad(target_metadata.angles);
            m_sampler.projected_slice = projected_padded_rfft.span_contiguous<c32, 2, i32>();
            m_sampler.target_rotation = (
                 nx::rotate_x(+extraction_angles[2]) *
                 nx::rotate_y(+extraction_angles[1]) *
                 nx::rotate_z(-extraction_angles[0])
             ).as<f32>();
            noa::iwise(shape_padded_2d.rfft(), device, m_sampler);

            // Keep a copy of the projected central-slice for the ZNCC.
            auto projected_padded_rfft_copy = m_buffer_padded_rfft.view().subregion(n_references + 2);
            if (compute_score)
                projected_padded_rfft.to(projected_padded_rfft_copy);

            // Center the projected slice onto the target.
            ns::phase_shift_2d<"h">(
                projected_padded_rfft, projected_padded_rfft, shape_padded,
                (original_center + target_metadata.shifts).as<f32>(), 0.5
            );

            // Back to real-space.
            const auto target_and_projected_padded = m_buffer_padded.view();
            nf::c2r(target_and_projected_padded_rfft, target_and_projected_padded);

            // Remove the padding and mask again to remove small projection/weighting artifacts.
            const auto target_and_projected = m_buffer.view();
            noa::iwise(shape_2d, device, PrepareTargetAndProjected{
                .target_and_projected_padded = target_and_projected_padded.span_contiguous<const f32, 3, i32>(),
                .target_and_projected = target_and_projected.span_contiguous<f32, 3, i32>(),
                .target_mask = common_fov.set_fov(target_metadata, fov_options),
            });

            // if (not Logger::s_debug_path.empty()) {
            //     auto filename = Logger::s_debug_path / fmt::format("tp_{:0>2}.mrc", target_metadata.index);
            //     noa::write_image(target_and_projected, filename);
            // }

            // Phase-like (i.e., mutual) cross-correlation.
            // Note that using the conventional CC isn't as good; the peaks are less sharp
            // up to the point that, for some samples, high tilts failed to pick reliably.
            const auto target_and_projected_rfft = m_buffer_rfft.view();
            nf::r2c(target_and_projected, target_and_projected_rfft, {.norm = nf::Norm::NONE});
            noa::iwise(shape_2d.rfft(), device, CrossCorrelate{
                .projected_rfft = target_and_projected_rfft.subregion(1).span_contiguous<const c32, 2, i32>(),
                .target_rfft = target_and_projected_rfft.subregion(0).span_contiguous<c32, 2, i32>(), // save xmap
            });

            // Find the best CC peak. The resulting shift should be added to the target.
            const auto centered_xmap_rfft = target_and_projected_rfft.subregion(0);
            const auto centered_xmap = target_and_projected.subregion(0);
            nf::c2r(centered_xmap_rfft, centered_xmap, {.norm = nf::Norm::ORTHO}); // nice scale
            const auto shift = find_peak<"fc">(centered_xmap, m_xmap_centered.view(), {
                .distortion_angle_deg = target_metadata.angles[0],
                .max_shift_percent = 0.15,
            }).first;

            // if (not debug_dir.empty()) {
            //     auto filename = debug_dir / fmt::format("xmap_{:0>2}.mrc", target_metadata.index);
            // if (not Logger::s_debug_path.empty()) {
            //     auto filename = Logger::s_debug_path / fmt::format("xmap_{:0>2}.mrc", target_metadata.index);
            //     noa::write_image(centered_xmap, filename);
            //     auto filename2 = Logger::s_debug_path / fmt::format("xmap_centered_{:0>2}.mrc", target_metadata.index);
            //     noa::write_image(m_xmap_centered, filename2);
            // }

            // Compute the ZNCC.
            f64 zncc{};
            if (compute_score) {
                // Apply the shift.
                auto final_metadata = target_metadata;
                final_metadata.shifts += shift;
                const auto fov = common_fov.set_fov(final_metadata, fov_options);

                // Move the projection to its newly found center.
                ns::phase_shift_2d<"h">(
                    projected_padded_rfft_copy, projected_padded_rfft_copy, shape_padded,
                    (original_center + final_metadata.shifts).as<f32>(), 0.5
                );

                // Remove the padding and mask again to remove small projection/weighting artifacts.
                // The target_padded is still valid from the previous CC.
                nf::c2r(projected_padded_rfft_copy, target_and_projected_padded.subregion(1));
                noa::iwise(shape_2d, device, PrepareTargetAndProjected{
                    .target_and_projected_padded = target_and_projected_padded.span_contiguous<const f32, 3, i32>(),
                    .target_and_projected = target_and_projected.span_contiguous<f32, 3, i32>(),
                    .target_mask = fov,
                });

                // if (not Logger::s_debug_path.empty()) {
                //     auto filename = Logger::s_debug_path / fmt::format("tpf_{:0>2}.mrc", target_metadata.index);
                //     noa::write_image(target_and_projected, filename);
                // }

                // Compute the zero-normalized cross-correlation score within the mask.
                noa::reduce_iwise(shape_2d, device, ZNCC<f64>::reduced_type{}, zncc, ZNCC<f64>{
                    .lhs = target_and_projected.subregion(0).span_contiguous<f32, 2, i32>(),
                    .rhs = target_and_projected.subregion(1).span_contiguous<f32, 2, i32>(),
                    .mask = fov,
                });
            }

            return Pair{shift, zncc};
        }
    };

    // Keep the implementation hidden from the header.
    // Each thread has its own projector to support for multi-GPU processing (one thread per GPU).
    thread_local Projector projector{};
}

namespace qn {
    ProjectionMatcher::ProjectionMatcher(isize n_slices, const Shape2& shape_2d, Device device) {
        projector = Projector(n_slices, shape_2d, device);
    }

    ProjectionMatcher::~ProjectionMatcher() {
        projector = Projector{};
    }

    [[nodiscard]] auto ProjectionMatcher::spectrum_size() const -> isize {
        return projector.m_buffer_padded.shape().height();
    }

    auto ProjectionMatcher::update_shifts(
        const View<f32>& stack,
        Metadata::Stack& metadata,
        const ProjectionMatchingParameters& settings
    ) const -> f64 {
        auto t = Logger::trace_scope_time("ProjectionMatcher::update_shifts");
        // Logger::s_debug_path = "/dls/ebic/data/staff-scratch/thomas2/datasets/kyprianos/quinoa"; // FIXME

        projector.initialize(stack, settings);

        // Projection matching, using the lowest tilt as the initial reference,
        // aligning from low-to-high tilts. When a tilt is aligned, it is added
        // to the set of reference images used to compute the projected reference.
        auto projection_metadata = metadata;
        projection_metadata.sort("time"); // TODO

        f64 zncc{};
        const auto common_fov = CommonFOV(stack.shape().filter(2, 3), projection_metadata);
        for (isize target_index = 1; target_index < projection_metadata.ssize(); ++target_index) {
            const auto& new_reference_slice = projection_metadata[target_index - 1];
            auto& target_slice = projection_metadata[target_index];

            const auto [peak_shift, peak_value] = projector.project_and_correlate_next(
                new_reference_slice, target_slice, common_fov,
                false, settings.smooth_edge_percent, settings.max_tilt_difference
            );
            target_slice.shifts += peak_shift;
            zncc += static_cast<f64>(peak_value);
        }
        zncc /= static_cast<f64>(projection_metadata.ssize() - 1);

        if (settings.update_metadata)
            metadata.update_from(projection_metadata, {.update_shifts = true});
        return zncc;
    }
}
