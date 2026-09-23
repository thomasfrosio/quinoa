#include "quinoa/Logger.hpp"
#include "quinoa/ctf/CTF.hpp"

#include "quinoa/postprocessing/FilterStack.hpp"
#include "quinoa/postprocessing/Utilities.hpp"

namespace {
    using namespace qn;

    struct RampFilter {
    private:
        static constexpr f32 ALPHA = 0.00195f;
        static constexpr f32 MATCH_ADD = 0.3f;

        // Ramp.
        Vec<f32, 2> m_normal_to_tilt_axis;
        bool m_ramp_filter;

        // Fake SIRT lowpass.
        f32 m_exponent;

    public:
        RampFilter(
            bool ramp_filter,
            Vec<f32, 2> normal_to_tilt_axis,
            i32 fake_sirt_iterations
        ) :
            m_normal_to_tilt_axis(normal_to_tilt_axis),
            m_ramp_filter(ramp_filter),
            m_exponent(fake_sirt_exponent(fake_sirt_iterations))
        {}

        [[nodiscard]] NOA_HD auto fake_sirt_(const Vec<f32, 2>& fftfreq_2d) const -> f32 {
            const auto fftfreq = noa::sqrt(noa::dot(fftfreq_2d, fftfreq_2d));
            return fake_sirt_filter(fftfreq, m_exponent);
        }

        [[nodiscard]] NOA_HD auto ramp_filter_(const Vec<f32, 2>& fftfreq_2d) const -> f32 {
            const f32 fftfreq_x = noa::abs(noa::dot(m_normal_to_tilt_axis, fftfreq_2d));
            // return 2.f * fftfreq_x * (0.55f + 0.45f * noa::cos(6.2831852f * fftfreq_x)); // aretomo
            // return 2.f * fftfreq_x * (1.92f + 1.57f * noa::cos(6.15f * fftfreq_x)); // aretomo-like rescaled to [0,1]
            return fftfreq_x; // simple linear ramp
        }

        NOA_HD auto operator()(const Vec<f32, 2>& fftfreq_2d, isize) const -> f32 {
            f32 filter{1};
            if (m_ramp_filter)
                filter *= ramp_filter_(fftfreq_2d);
            if (m_exponent != 0.f)
                filter *= fake_sirt_(fftfreq_2d);
            return filter;
        }
    };

    struct CTFPhaseFlip {
        SpanContiguous<const c32, 2> image_padded_rfft; // (h,w)
        SpanContiguous<c32, 3> strips_padded_rfft; // (s,h,w)
        ns::CTFAnisotropic<f32> ctf;
        Vec<f32, 2> fftfreq_norm;
        f32 phase_flip_strength;
        f32 defocus_start_um;
        f32 defocus_step_um;
        isize strip_offset;

        NOA_HD void operator()(isize s, isize y, isize x) {
            constexpr bool CENTERED = false;
            constexpr bool RFFT = true;
            const auto frequency = nf::index2frequency<CENTERED, RFFT>(Vec{y, x}, image_padded_rfft.shape().filter(0));
            const auto fftfreq_2d = frequency.as<f32>() * fftfreq_norm;

            // Get the CTF of the current strip.
            const auto defocus = ctf.defocus();
            ctf.set_defocus({
                .value = defocus_start_um + defocus_step_um * static_cast<f32>(strip_offset + s),
                .astigmatism = defocus.astigmatism,
                .angle = defocus.angle,
            });

            const auto fftfreq_sqd = noa::dot(fftfreq_2d, fftfreq_2d);
            const auto fftfreq = noa::sqrt(fftfreq_sqd);
            const auto value = -noa::sin(ctf.phase_at(fftfreq_2d)); // ctf.value_at but without envelope

            // Compute and the filter.
            // Note that this CTF model, like in RELION or Warp, goes positive first, so the CTF multiplication
            // below does not inverse the visible contrast of the images and instead keeps particles dark.
            const auto regularization = phase_flip_strength * noa::exp(2 * fftfreq_sqd);
            const auto phase_flip = noa::sign(value);
            const auto wiener_like = ((noa::abs(value) + regularization) / (1 + regularization)) * phase_flip;
            const auto filter = noa::exp(ctf.bfactor() / 4 * fftfreq) / wiener_like;
            strips_padded_rfft(s, y, x) = image_padded_rfft(y, x) * filter;
        }
    };

    struct RecomposeFilteredImage {
        SpanContiguous<const f32, 3> strips_padded; // (s,h+p,w+p)
        SpanContiguous<f32, 2> image; // (h,w)

        Vec<isize, 2> image_center;
        Vec<isize, 2> left_padding;
        Vec<f32, 3> image_plane_normal;
        f32 spacing_nm;
        f32 z_offset_start_nm;
        f32 z_step_nm;

        isize strip_start;
        isize strip_end;

        NOA_HD void operator()(isize i, isize j) const {
            // Get the z-position at this index of the image.
            const auto indices = Vec{i, j};
            const auto coordinates = (indices - image_center).as<f32>();

            const auto& [c, b, a] = image_plane_normal;
            const auto volume_z_coordinate = -(a * coordinates[1] + b * coordinates[0]) / c;
            const auto volume_z_coordinate_nm = volume_z_coordinate * spacing_nm;

            // Get the closest z-strip.
            const auto strip = (volume_z_coordinate_nm - z_offset_start_nm) / z_step_nm;
            const auto strip_index = static_cast<isize>(noa::round(strip));

            // If the chunk contains that z-strip, save it into the image.
            if (strip_index >= strip_start and strip_index < strip_end) {
                const auto padded_indices = indices + left_padding;
                const auto chunk_index = strip_index - strip_start;
                image(indices) = strips_padded(padded_indices.push_front(chunk_index));
            }
        }
    };
}

namespace qn {
    StackFilterer::StackFilterer(
        StackLoader&& loader,
        Metadata::Stack& metadata,
        bool use_aligned_stack,
        nx::Interp aligned_stack_interpolation,
        bool ramp_filter,
        i32 fake_sirt_iterations
    ) {
        Logger::trace(
            "Stack filterer:\n"
            "  align_stack={} (interp={})\n"
            "  ramp_filter={}\n"
            "  fake_sirt_iterations={}\n"
            "  ctf=false",
            use_aligned_stack, aligned_stack_interpolation,
            ramp_filter, fake_sirt_iterations
        );

        const auto image_shape = loader.slice_shape().push_front<2>(1);
        const auto images_shape = image_shape.set<0>(metadata.ssize());
        const auto options = ArrayOption{.device = loader.compute_device(), .allocator = Allocator::ASYNC};

        const auto b0 = Allocator::bytes_currently_allocated(options.device);
        const auto t0 = Logger::trace_scope_time("StackFilter preprocessing");

        m_images_filtered = Array<f32>(images_shape, options);
        const bool filtering = ramp_filter or fake_sirt_iterations;
        const auto image_rfft = filtering ? Array<c32>(image_shape.rfft(), options) : Array<c32>{};

        const auto center = (loader.slice_shape().vec / 2).as<f64>();
        const auto buffer = use_aligned_stack ? Array<f32>(image_shape, options) : Array<f32>{};

        for (i32 i{}; auto& m: metadata) {
            auto image = m_images_filtered.view().subregion(i);

            if (use_aligned_stack) {
                loader.read_slice(buffer.view(), m.index_file);

                const auto inverse_transform = (
                    nx::translate(center) *
                    nx::rotate<true>(noa::deg2rad(-m.angles[0])) *
                    nx::translate(-center - m.shifts)
                ).inverse().as<f32>();

                nx::transform_2d(buffer.view(), image, inverse_transform, {
                    .interp = aligned_stack_interpolation,
                    .border = noa::Border::ZERO,
                });

                // Images are aligned from now.
                m.shifts = 0.;
                m.angles[0] = 0.;
            } else {
                loader.read_slice(image, m.index_file);
            }

            // Images are loaded as sorted in the stack.
            m.index = i++;

            if (filtering) {
                nf::r2c(image, image_rfft);
                const auto normal_to_tilt_axis = nx::rotate(noa::deg2rad(-m.angles[0]))[1].as<f32>();
                ns::filter_spectrum_2d<"h">(image_rfft, image_rfft, image.shape(), RampFilter(
                    ramp_filter, normal_to_tilt_axis, fake_sirt_iterations
                ));
                nf::c2r(image_rfft, image);
            }
        }

        const auto b1 = Allocator::bytes_currently_allocated(options.device);
        Logger::trace(
            "  allocated {:.3f}GB (device={}, {})",
            static_cast<f64>(b1 - b0) / 1e9, options.device, options.allocator
        );
        loader = StackLoader{}; // images are loaded, free buffers
    }

    StackFilterer::StackFilterer(
        StackLoader&& loader,
        Metadata::Stack& metadata,
        bool use_aligned_stack,
        nx::Interp aligned_stack_interpolation,
        bool ramp_filter,
        i32 fake_sirt_iterations,
        const CTFIsotropic64& ctf,
        f64 volume_thickness_nm,
        f64 z_step_nm,
        f64 phase_flip_strength
    ) :
        m_metadata{&metadata},
        m_ctf{&ctf},
        m_z_step_nm{z_step_nm},
        m_phase_flip_strength{static_cast<f32>(phase_flip_strength)} {
        const auto image_shape = loader.slice_shape();
        const auto spacing_nm = ctf.pixel_size() * 1e-1;

        // Compute the z-offset at the center of the volume.
        // This will be used to compute the relative z-offset of z-sections.
        const auto n_sections = std::round(volume_thickness_nm / z_step_nm);
        check(noa::is_odd(static_cast<isize>(n_sections)));
        m_volume_z_center_nm = (n_sections / 2) * z_step_nm;

        // Get the maximum defocus and the number of strips.
        f64 max_z_range_nm{};
        f64 max_defocus_nm{};
        isize min_n_strips{std::numeric_limits<isize>::max()};
        isize max_n_strips{};
        for (const auto& image: metadata) {
            const auto [start_offset_nm, n_strips] = divide_image_in_z_strips(
                image_shape, image.angles, spacing_nm, z_step_nm
            );
            const auto end_strip_nm = start_offset_nm + z_step_nm * static_cast<f64>(n_strips - 1);
            const auto z_range_nm = end_strip_nm - start_offset_nm;
            const auto image_defocus_nm = (image.defocus.value + std::abs(image.defocus.astigmatism)) * 1e3;
            const auto highest_defocus_nm = image_defocus_nm - start_offset_nm; // underfocus negative

            max_z_range_nm = std::max(max_z_range_nm, z_range_nm);
            max_defocus_nm = std::max(max_defocus_nm, highest_defocus_nm);
            min_n_strips = std::min(min_n_strips, n_strips);
            max_n_strips = std::max(max_n_strips, n_strips);
        }
        max_defocus_nm += volume_thickness_nm / 2;

        // Get the size requirement for the maximum defocus.
        // The following is adapted from Russo & Henderson, 2018.
        // This is just a rough (under)estimate of what the CTF does.
        // https://www.desmos.com/calculator/w1dlw58f8t
        // 8A resolution, 4um defocus -> delocalization is ~50pix
        // 4A resolution, 4um defocus -> delocalization is ~200pix
        const f64 wavelength_nm = ns::relativistic_electron_wavelength(ctf.voltage() * 1e3) * 1e9;
        const f64 resolution_nm = spacing_nm * 2;
        const f64 delocalization_nm = 2 * max_defocus_nm * wavelength_nm / resolution_nm;
        const f64 delocalization_pix = std::round(delocalization_nm / spacing_nm);
        const isize aliasing_free_size = [&] {
            auto ictf = ctf;
            ictf.set_defocus(max_defocus_nm * 1e-3);
            return ctf::aliasing_free_size(ictf, Vec{0., 0.5});
        }();

        // Compute a satisfying shape given these limits.
        constexpr f64 PADDING_FACTOR = 1.2;
        const auto minimum_padding = static_cast<isize>(delocalization_pix);
        auto image_padded_shape = Shape{(image_shape.vec.as<f64>() * PADDING_FACTOR).as<isize>()};
        image_padded_shape = noa::max(image_padded_shape, image_shape + minimum_padding);
        image_padded_shape = noa::max(image_padded_shape, aliasing_free_size);
        image_padded_shape = nf::next_fast_shape(image_padded_shape);

        // Allocating and processing all strips at once can require a lot of memory for high-resolution and
        // high-tilt images. To decrease the memory requirement, process the strips in chunks so that we only
        // need to allocate a chunk_size of strips.
        const auto [n_chunks, chunk_size, keep_spectra_on_device] = reduce_memory_requirements(
            image_shape, image_padded_shape, metadata.ssize(), max_n_strips,
            loader.compute_device().memory_capacity().free
        );

        Logger::trace(
            "Stack filterer:\n"
            "  align_stack={} (interp={})\n"
            "  ramp_filter={}\n"
            "  fake_sirt_iterations={}\n"
            "  ctf=true\n"
            "    defocus_resolution={:.3f}nm|{}pix\n"
            "    max_z_range_in_image={:.2f}nm\n"
            "    max_defocus_in_stack={:.2f}nm\n"
            "    strips=[min={}, max={}, chunk={}, n_chunks={}]\n"
            "    max_delocalization={:.3f}nm|{}pix\n"
            "    aliasing_free_size={}\n"
            "    padded_shape={} (shape={}, ratio={::.2f})",
            use_aligned_stack, aligned_stack_interpolation,
            ramp_filter, fake_sirt_iterations,
            z_step_nm, std::round(z_step_nm / spacing_nm),
            max_z_range_nm, max_defocus_nm, min_n_strips, max_n_strips, chunk_size, n_chunks,
            delocalization_nm, delocalization_pix, aliasing_free_size, image_padded_shape, image_shape,
            image_padded_shape.vec.as<f64>() / image_shape.vec.as<f64>()
        );

        allocate_and_prepare_spectra(
            std::move(loader), metadata, use_aligned_stack, aligned_stack_interpolation,
            ramp_filter, fake_sirt_iterations,
            image_shape, image_padded_shape, keep_spectra_on_device, chunk_size
        );
    }

    void StackFilterer::allocate_and_prepare_spectra(
        StackLoader&& loader,
        Metadata::Stack& metadata,
        bool use_aligned_stack,
        nx::Interp aligned_stack_interpolation,
        bool ramp_filter,
        i32 fake_sirt_iterations,
        const Shape2& image_shape,
        const Shape2& image_padded_shape,
        bool keep_spectra_on_device,
        isize chunk_size
    ) {
        const auto images_shape = image_shape.push_front(Vec{metadata.ssize(), isize{1}});
        const auto images_padded_shape = image_padded_shape.push_front(Vec{metadata.ssize(), isize{1}});
        const auto image_padded_strips_shape = image_padded_shape.push_front(Vec{chunk_size, isize{1}});
        const auto options = ArrayOption{.device = loader.compute_device(), .allocator = Allocator::MANAGED};

        const auto b0 = Allocator::bytes_currently_allocated(options.device);
        const auto t0 = Logger::trace_scope_time("StackFilter preprocessing");

        noa::tie(m_images_padded, m_images_padded_rfft) = nf::empty<f32>(
            images_padded_shape, keep_spectra_on_device ?
                options : ArrayOption{.device = Device{}, .allocator = Allocator::DEFAULT}
        );

        // To reduce memory requirements on the device,
        // process images one by one and store everything on the host.
        Array<f32> resize_buffer;
        Array<c32> resize_buffer_rfft;
        if (not keep_spectra_on_device) {
            noa::tie(resize_buffer, resize_buffer_rfft) =
                nf::empty<f32>(image_padded_shape.push_front<2>(1), options);
        }

        const auto center = (loader.slice_shape().vec / 2).as<f64>();
        const auto io_buffer = Array<f32>(image_shape.push_front<2>(1), options);
        const auto xform_buffer = use_aligned_stack ? noa::like(io_buffer) : Array<f32>{};

        for (i32 i{}; auto& slice: metadata) {
            if (use_aligned_stack) {
                loader.read_slice(xform_buffer.view(), slice.index_file);
                const auto inverse_transform = (
                    nx::translate(center) *
                    nx::rotate<true>(noa::deg2rad(-slice.angles[0])) *
                    nx::translate(-center - slice.shifts)
                ).inverse().as<f32>();

                nx::transform_2d(xform_buffer.view(), io_buffer.view(), inverse_transform, {
                    .interp = aligned_stack_interpolation,
                    .border = noa::Border::ZERO,
                });
                slice.shifts = 0.;
                slice.angles[0] = 0.;
            } else {
                loader.read_slice(io_buffer.view(), slice.index_file);
            }

            if (keep_spectra_on_device) {
                noa::resize(io_buffer, m_images_padded.subregion(i));
            } else {
                noa::resize(io_buffer, resize_buffer);
                nf::r2c(resize_buffer, resize_buffer_rfft, {.norm = nf::Norm::FORWARD});
                if (ramp_filter) {
                    const auto normal_to_tilt_axis = nx::rotate(noa::deg2rad(-slice.angles[0]))[1].as<f32>();
                    ns::filter_spectrum_2d<"h">(resize_buffer_rfft, resize_buffer_rfft, resize_buffer.shape(),
                        RampFilter(ramp_filter, normal_to_tilt_axis, fake_sirt_iterations)
                    );
                }
                resize_buffer_rfft.to(m_images_padded_rfft.subregion(i));
            }

            // Images are loaded as sorted in the stack.
            slice.index = i++;
        }
        m_images_padded_rfft.eval();
        loader = StackLoader{};

        if (keep_spectra_on_device) {
            nf::r2c(m_images_padded, m_images_padded_rfft, {.norm = nf::Norm::FORWARD, .cache_plan = false});
            if (ramp_filter) {
                check(metadata.has_single_rotation(), "TODO Varying rotation. Add per-image ramp filter.");
                const auto normal_to_tilt_axis = nx::rotate(noa::deg2rad(-metadata[0].angles[0]))[1].as<f32>();
                ns::filter_spectrum_2d<"h">(m_images_padded_rfft, m_images_padded_rfft, m_images_padded.shape(),
                    RampFilter(ramp_filter, normal_to_tilt_axis, fake_sirt_iterations)
                );
            }
        }

        // Allocate remaining buffers.
        m_images_filtered = Array<f32>(images_shape, options);
        noa::tie(m_strips_padded, m_strips_padded_rfft) = nf::empty<f32>(image_padded_strips_shape, options);
        if (not keep_spectra_on_device)
            m_image_padded_rfft = Array<c32>(image_padded_shape.rfft().push_front<2>(1), options);

        const auto b1 = Allocator::bytes_currently_allocated(options.device);
        Logger::trace(
            "  allocated {:.3f}GB (device={}, {})",
            static_cast<f64>(b1 - b0) / 1e9, options.device, options.allocator
        );
    }

    auto StackFilterer::divide_image_in_z_strips(
        const Shape2& image_shape,
        const Vec<f64, 3>& image_angles,
        f64 spacing_nm,
        f64 z_step_nm
    ) -> Pair<f64, isize> {
        // The image is divided into z-strips centered at the image center. For instance, if z_step_nm=15,
        // the z-axis is divided such as: [..., -45, -30, -15,  +0, +15, +30, +45, ...]nm
        // These point to the z-height center of each strip and relative to the image center. In this case,
        // the central strip maps the [-7.5, +7.5]nm range, so every projected coordinate that falls within
        // that range should be assigned to that strip.
        //
        // The image center points at the average defocus. Furthermore, the defocus is underfocus positive,
        // as such strips with positive z-offsets are above the rotation axis (closer to focus). As a result,
        // to compute the defocus of a strip, we need to subtract the z-offset of the strip to the average defocus.

        // Get the 4 image edges.
        const auto top_right_edge = image_shape.vec - 1;
        const auto image_edges = Vec{
            Vec<f64, 2>::from_values(0, 0),
            Vec<f64, 2>::from_values(0, top_right_edge[1]),
            Vec<f64, 2>::from_values(top_right_edge[0], 0),
            Vec<f64, 2>::from_values(top_right_edge[0], top_right_edge[1]),
        };

        // Image plane coefficients to get the z-offset at image coordinate.
        const auto angles = noa::deg2rad(image_angles);
        const auto plane_rotation = (
            nx::rotate_z(angles[0]) *
            nx::rotate_y(angles[1]) *
            nx::rotate_x(angles[2])
        );
        const auto [c, b, a] = plane_rotation * Vec{1., 0., 0.};

        // Compute the z-range within the image.
        auto minmax = Vec<f64, 2>{}; // in nm
        const auto image_center = (image_shape.vec / 2).as<f64>();
        for (const auto& image_edge: image_edges) {
            const auto coordinates = image_edge - image_center;
            const auto z_distance = -(b * coordinates[0] + a * coordinates[1]) / c;
            const auto z_distance_nm = z_distance * spacing_nm;
            minmax[0] = std::min(minmax[0], z_distance_nm);
            minmax[1] = std::max(minmax[1], z_distance_nm);
        }

        // Get the corresponding z-strips for that image.
        const auto first_strip_nm = noa::round(minmax[0] / z_step_nm) * z_step_nm;
        const auto last_strip_nm = noa::round(minmax[1] / z_step_nm) * z_step_nm;
        const auto n_strips = (last_strip_nm - first_strip_nm) / z_step_nm + 1;
        return {first_strip_nm, static_cast<isize>(std::round(n_strips))};
    }

    auto StackFilterer::reduce_memory_requirements(
        const Shape2& image_shape,
        const Shape2& image_padded_shape,
        isize n_images,
        isize max_n_strips,
        usize n_bytes_free
    ) -> Tuple<isize, isize, bool> {
        // The strategy is the following:
        //  1. If GPU memory is low, try 2 chunks. This should be enough for most cases and
        //     decreases the overall memory needed (host and device). The runtime overhead is minimal.
        //  2. If this is still not enough, keep spectra on the host.
        //  3. If this is still not enough, divide in more chunks.
        const auto images_shape = image_shape.push_front(Vec{n_images, isize{1}});
        const auto images_padded_shape = image_padded_shape.push_front(Vec{n_images, isize{1}});
        const auto images_bytes = static_cast<usize>(images_shape.n_elements()) * sizeof(f32);
        const auto images_padded_bytes = static_cast<usize>(images_padded_shape.rfft().n_elements()) * sizeof(c32);

        bool keep_spectra_on_device{true};
        isize n_chunks{1};

        // We may not be able to query the device stats, in which case
        // better to hope for the best and process in one chunk.
        if (n_bytes_free == 0)
            return noa::make_tuple(n_chunks, max_n_strips, keep_spectra_on_device);

        isize chunk_size{};
        for (; n_chunks < max_n_strips; ++n_chunks) {
            // until 1 strip per chunk
            auto base = images_bytes;
            if (n_chunks <= 2) {
                base += images_padded_bytes;
            } else {
                // Before trying to divide into 3 chunks,
                // try moving the spectra back to the host.
                if (keep_spectra_on_device)
                    n_chunks = 2;
                keep_spectra_on_device = false;
            }

            chunk_size = static_cast<isize>(std::ceil(static_cast<f64>(max_n_strips) / static_cast<f64>(n_chunks)));
            const auto n_elements = chunk_size * image_padded_shape.rfft().n_elements();
            const auto n_bytes = static_cast<usize>(n_elements) * sizeof(c32);
            const auto n_bytes_total = static_cast<usize>(static_cast<f64>(n_bytes) * 2); // x2 for FFT plans

            if (n_bytes_free < base + n_bytes_total)
                continue;
            if (n_bytes_total > static_cast<usize>(10e9))
                continue;
            break;
        }
        return noa::make_tuple(n_chunks, chunk_size, keep_spectra_on_device);
    }

    void StackFilterer::compute_irffts(isize n_strips, bool record, isize n_groups) const {
        // We compute many FFTs with different batch sizes. In CUDA, this leads to computing many plans, and while
        // the memory consumption can be minimized by sharing the workspace across these plans, the overhead of
        // computing the plans in the first place makes it quite inefficient. In fact, it is faster to group
        // the batch size and compute larger arrays, as long as the number of plans decreases.

        // Group batch size into groups.
        const auto maximum_n_strips = m_strips_padded.shape()[0];
        const auto group_size = noa::next_multiple_of(maximum_n_strips, n_groups) / n_groups;
        const auto index = static_cast<isize>(noa::ceil(static_cast<f64>(n_strips) / static_cast<f64>(group_size)));
        const auto slice = Slice{0, index * group_size};

        // Prepare for this transform, asking to share the workspace.
        nf::c2r(m_strips_padded_rfft.view().subregion(slice), m_strips_padded.view().subregion(slice), {
            .norm = noa::fft::Norm::FORWARD,
            .record_and_share_workspace = record,
        });

        // Synchronizing after the transform reduces the latency (the host is waiting for the GPU with ioctl
        // twice as long without the synchronization point), which significantly improves the overall performance.
        // This is somewhat surprising, and I don't really understand why this happens.
        if (not record)
            m_strips_padded.eval();
    }

    void StackFilterer::prepare_irffts() const {
        if (m_z_step_nm <= 0) // no ctf
            return;

        // Create and cache the plans for every FFT about to be run.
        // These FFTs also share the same workspace, so they have to run on the same stream.
        for (isize i{1}; i < m_strips_padded.shape()[0]; ++i)
            compute_irffts(i, true);

        const auto device = m_strips_padded_rfft.device();
        const auto workspace_bytes = nf::workspace_left_to_allocate(device);
        if (workspace_bytes > 0) {
            const auto options = ArrayOption{.device = device, .allocator = Allocator::ASYNC};
            nf::set_workspace(device, Array<std::byte>(workspace_bytes, options));
        }
    }

    [[nodiscard]] auto StackFilterer::compute_filtered_stack(isize z) const -> Array<f32> {
        if (m_z_step_nm <= 0) // no ctf
            return m_images_filtered;

        const auto device = m_images_filtered.device();
        const auto spacing_nm = m_ctf->pixel_size() * 1e-1;
        const auto image_shape = m_images_filtered.shape().filter(2, 3);
        const auto image_padded_shape = m_images_padded.shape().filter(2, 3);
        const auto left_padding = image_padded_shape.vec / 2 - image_shape.vec / 2;
        const auto fftfreq_norm = 1. / image_padded_shape.vec.as<f64>();

        // Compute the z-offset at the center of the current z-section, relative to the volume center.
        const f64 z_offset_section_center_nm =
            (m_z_step_nm * static_cast<f64>(z) + m_z_step_nm / 2) - m_volume_z_center_nm;

        for (const auto& image: *m_metadata) {
            // Compute defocus-strips.
            const auto [z_offset_start_nm, n_strips] = divide_image_in_z_strips(
                image_shape, image.angles, spacing_nm, m_z_step_nm
            );
            auto ictf = ns::CTFAnisotropic(*m_ctf);
            ictf.set_defocus(image.defocus); // sets the astigmatism

            const auto z_offset_of_lowest_strip_um = (z_offset_section_center_nm + z_offset_start_nm) * 1e-3;
            const auto defocus_start = image.defocus.value - z_offset_of_lowest_strip_um; // underfocus negative
            const auto defocus_step = -m_z_step_nm * 1e-3; // underfocus negative

            // Recompose the filtered tile from the defocus-strips.
            const auto angles = noa::deg2rad(image.angles);
            const auto plane_rotation = (
                nx::rotate_z(angles[0]) *
                nx::rotate_y(angles[1]) *
                nx::rotate_x(angles[2])
            );
            const auto plane_normal = (plane_rotation * Vec{1., 0., 0.}).as<f32>();

            // Make the input spectrum available on the device.
            auto image_padded_rfft = m_images_padded_rfft.view().subregion(image.index);
            if (image_padded_rfft.device() != device)
                image_padded_rfft = image_padded_rfft.to(m_image_padded_rfft.view());

            // If the device has enough memory to hold all strips at once, this is a single pass.
            const auto chunk_size = m_strips_padded.shape()[0];
            for (isize i{}; i < n_strips; i += chunk_size) {
                const auto ichunk_size = std::min(chunk_size, n_strips - i);
                const auto ichunk = Slice{0, ichunk_size};

                noa::iwise(image_padded_shape.rfft().push_front(ichunk_size), device, CTFPhaseFlip{
                    .image_padded_rfft = image_padded_rfft.span().filter(2, 3).as_contiguous(),
                    .strips_padded_rfft = m_strips_padded_rfft.span().subregion(ichunk).filter(0, 2, 3).as_contiguous(),
                    .ctf = ictf.as<f32>(),
                    .fftfreq_norm = fftfreq_norm.as<f32>(),
                    .phase_flip_strength = m_phase_flip_strength,
                    .defocus_start_um = static_cast<f32>(defocus_start),
                    .defocus_step_um = static_cast<f32>(defocus_step),
                    .strip_offset = i,
                });

                compute_irffts(ichunk_size);

                const auto image_filtered = m_images_filtered.span().subregion(image.index).filter(2, 3).as_contiguous();
                noa::iwise(image_filtered.shape(), device, RecomposeFilteredImage{
                    .strips_padded = m_strips_padded.span().subregion(ichunk).filter(0, 2, 3).as_contiguous(),
                    .image = image_filtered,
                    .image_center = image_shape.vec / 2,
                    .left_padding = left_padding,
                    .image_plane_normal = plane_normal,
                    .spacing_nm = static_cast<f32>(spacing_nm),
                    .z_offset_start_nm = static_cast<f32>(z_offset_start_nm),
                    .z_step_nm = static_cast<f32>(m_z_step_nm),
                    .strip_start = i,
                    .strip_end = i + ichunk_size,
                });
            }
        }

        return m_images_filtered;
    }

    [[nodiscard]] auto StackFilterer::compute_filtered_stack() const -> Array<f32> {
        if (m_z_step_nm <= 0) // no ctf
            return m_images_filtered;

        const auto central_z_f64 = m_volume_z_center_nm / m_z_step_nm;
        const auto central_z = static_cast<i64>(std::floor(central_z_f64));
        return compute_filtered_stack(central_z);
    }

    auto filter_stack(
        StackLoader&& stack,
        Metadata& metadata,
        const FilterStackSettings& settings
    ) -> Array<f32> {
        const auto timer = Logger::info_scope_time("Filtering stack");

        const auto spacing = mean(stack.stack_spacing());
        const auto spacing_nm = spacing * 1e-1;
        const auto ctf = CTFIsotropic64::Parameters{
            .pixel_size = spacing,
            .defocus = 0.,
            .voltage = metadata.sample.voltage,
            .amplitude = metadata.sample.amplitude,
            .cs = metadata.sample.cs,
            .phase_shift = 0,
            .bfactor = settings.ctf_bfactor,
            .scale = 1.,
        }.to_ctf();

        const auto setup = reconstruction_thickness(spacing_nm, settings.ctf_defocus_step_nm, metadata.sample.thickness);
        const auto device = stack.compute_device();

        auto stack_filterer = StackFilterer{};
        if (settings.correct_ctf) {
            stack_filterer = StackFilterer(
                std::move(stack), metadata.stack,
                settings.prealign_stack, settings.prealign_stack_interpolation,
                settings.ramp_filter, settings.fake_sirt_iterations,
                ctf, setup.thickness_nm, setup.z_step_nm, settings.ctf_phase_flip_strength
            );
        } else {
            stack_filterer = StackFilterer(
                std::move(stack), metadata.stack,
                settings.prealign_stack, settings.prealign_stack_interpolation,
                settings.ramp_filter, settings.fake_sirt_iterations
            );
        }

        const auto b0 = Allocator::bytes_currently_allocated(device);
        stack_filterer.prepare_irffts();
        const auto b1 = Allocator::bytes_currently_allocated(device);
        Logger::trace(
            "FFT workspace allocated: {}={:.2f}GB",
            device, static_cast<f64>(b1 - b0) * 1e-9
        );

        return stack_filterer.compute_filtered_stack();
    }
}
