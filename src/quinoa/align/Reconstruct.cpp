#include "quinoa/Logger.hpp"
#include "quinoa/ctf/CTF.hpp"
#include "quinoa/align/Reconstruct.hpp"

namespace {
    using namespace qn;

    Path debug_path{};

    struct FilterImages {
        NOA_HD auto operator()(const Vec<f32, 2>& fftfreq_2d, i64) const -> f32 {
            // Directly from Aretomo3.
            const auto fftfreq = noa::sqrt(noa::dot(fftfreq_2d, fftfreq_2d));
            return 2.f * fftfreq * (0.55f + 0.45f * noa::cos(6.2831852f * fftfreq));
        }
    };

    struct FilterPaddedImages {
        SpanContiguous<const c32, 2> image_padded_rfft; // (h,w)
        SpanContiguous<c32, 3> strips_padded_rfft; // (s,h,w)
        ns::CTFAnisotropic<f32> ctf;
        Vec<f32, 2> fftfreq_norm;
        f32 phase_flip_strength;
        f32 defocus_start_um;
        f32 defocus_step_um;
        isize strip_offset;

        // SpanContiguous<f32, 3> debug_filter; // (s,h,w)

        NOA_HD void operator()(isize s, isize y, isize x) {
            const auto frequency = nf::index2frequency<false, true>(Vec{y, x}, image_padded_rfft.shape().filter(0));
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
            // const auto regularization = phase_flip_strength * noa::exp(2 * fftfreq_sqd);
            // const auto phase_flip = noa::sign(value);
            // const auto wiener_like = (1 + regularization) / (noa::abs(value) + regularization) * phase_flip;
            // const auto b_decay = noa::exp(ctf.bfactor() / 4 * fftfreq_sqd);
            // strips_padded_rfft(s, y, x) = image_padded_rfft(y, x) * wiener_like * b_decay;

            const auto regularization = phase_flip_strength * noa::exp(2 * fftfreq_sqd);
            const auto phase_flip = noa::sign(value);
            const auto wiener_like = (noa::abs(value) + regularization) / (1 + regularization) * phase_flip;
            const auto filter = noa::exp(ctf.bfactor() / 4 * fftfreq) / wiener_like;
            strips_padded_rfft(s, y, x) = image_padded_rfft(y, x) * filter;
            // debug_filter(s, y, x) = filter;
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

    class Filterer {
    public:
        Filterer() = default;
        Filterer(
            StackLoader&& loader,
            Metadata::Stack& metadata,
            bool ramp_filter
        ) {
            Logger::trace(
                "Filtering:\n"
                "  ramp_filter={}\n"
                "  ctf=false",
                ramp_filter
            );

            const auto image_shape = loader.slice_shape().push_front<2>(1);
            const auto images_shape = image_shape.set<0>(metadata.ssize());
            const auto options = ArrayOption{.device = loader.compute_device(), .allocator = Allocator::ASYNC};

            m_images_filtered = Array<f32>(images_shape, options);
            const auto image_rfft = ramp_filter ? Array<c32>(image_shape.rfft(), options) : Array<c32>{};

            for (i32 i{}; auto& m: metadata) {
                auto image = m_images_filtered.view().subregion(i);
                loader.read_slice(image, m.index_file);
                m.index = i++;

                if (ramp_filter) {
                    nf::r2c(image, image_rfft);
                    ns::filter_spectrum_2d<"h">(image_rfft, image_rfft, image.shape(), FilterImages{});
                    nf::c2r(image_rfft, image);
                }
            }
            loader = StackLoader{};
        }

        Filterer(
            StackLoader&& loader,
            const Metadata::Stack& metadata,
            bool ramp_filter,
            const CTFIsotropic64& ctf,
            f64 volume_thickness_nm,
            f64 z_step_nm,
            f64 phase_flip_strength
        ) :
            m_metadata{&metadata},
            m_ctf{&ctf},
            m_z_step_nm{z_step_nm},
            m_phase_flip_strength{static_cast<f32>(phase_flip_strength)}
        {
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
                "Filtering:\n"
                "  ramp_filter={}\n"
                "  ctf=true\n"
                "  defocus_resolution={:.3f}nm|{}pix\n"
                "  max_z_range={:.2f}nm\n"
                "  max_defocus={:.2f}nm\n"
                "  strips=[min={}, max={}, chunk={}, n_chunks={}]\n"
                "  max_delocalization={:.3f}nm|{}pix\n"
                "  aliasing_free_size={}\n"
                "  padded_shape={} (shape={}, ratio={::.2f})",
                ramp_filter, z_step_nm, std::round(z_step_nm / spacing_nm),
                max_z_range_nm, max_defocus_nm, min_n_strips, max_n_strips, chunk_size, n_chunks,
                delocalization_nm, delocalization_pix, aliasing_free_size, image_padded_shape, image_shape,
                image_padded_shape.vec.as<f64>() / image_shape.vec.as<f64>()
            );

            allocate_and_prepare_spectra(
                std::move(loader), image_shape, image_padded_shape, metadata,
                keep_spectra_on_device, chunk_size, ramp_filter
            );
        }

        void allocate_and_prepare_spectra(
            StackLoader&& loader,
            const Shape2& image_shape,
            const Shape2& image_padded_shape,
            const Metadata::Stack& metadata,
            bool keep_spectra_on_device,
            isize chunk_size,
            bool ramp_filter
        ) {
            const auto images_shape = image_shape.push_front(Vec{metadata.ssize(), isize{1}});
            const auto images_padded_shape = image_padded_shape.push_front(Vec{metadata.ssize(), isize{1}});
            const auto image_padded_strips_shape = image_padded_shape.push_front(Vec{chunk_size, isize{1}});
            const auto options = ArrayOption{.device = loader.compute_device(), .allocator = Allocator::MANAGED};

            noa::tie(m_images_padded, m_images_padded_rfft) = nf::empty<f32>(
                images_padded_shape, keep_spectra_on_device ? options :
                ArrayOption{.device = Device{}, .allocator = Allocator::DEFAULT}
            );

            // To reduce memory requirements on the device,
            // process images one by one and store everything on the host.
            Array<f32> resize_buffer;
            Array<c32> resize_buffer_rfft;
            if (not keep_spectra_on_device) {
                noa::tie(resize_buffer, resize_buffer_rfft) =
                    nf::empty<f32>(image_padded_shape.push_front<2>(1), options);
            }

            const auto io_buffer = Array<f32>(image_shape.push_front<2>(1), options);
            for (const auto& slice: metadata) {
                loader.read_slice(io_buffer.view(), slice.index_file);
                if (keep_spectra_on_device) {
                    noa::resize(io_buffer, m_images_padded.subregion(slice.index));
                } else {
                    noa::resize(io_buffer, resize_buffer);
                    nf::r2c(resize_buffer, resize_buffer_rfft, {.norm = nf::Norm::FORWARD});
                    if (ramp_filter)
                        ns::filter_spectrum_2d<"h">(resize_buffer_rfft, resize_buffer_rfft, resize_buffer.shape(), FilterImages{});
                    resize_buffer_rfft.to(m_images_padded_rfft.subregion(slice.index));
                }
            }
            m_images_padded_rfft.eval();
            loader = StackLoader{};

            if (keep_spectra_on_device) {
                nf::r2c(m_images_padded, m_images_padded_rfft, {.norm = nf::Norm::FORWARD, .cache_plan = false});
                if (ramp_filter)
                    ns::filter_spectrum_2d<"h">(m_images_padded_rfft, m_images_padded_rfft, m_images_padded.shape(), FilterImages{});
            }

            // Allocate remaining buffers.
            m_images_filtered = Array<f32>(images_shape, options);
            noa::tie(m_strips_padded, m_strips_padded_rfft) = nf::empty<f32>(image_padded_strips_shape, options);
            if (not keep_spectra_on_device)
                m_image_padded_rfft = Array<c32>(image_padded_shape.rfft().push_front<2>(1), options);
        }

        static auto divide_image_in_z_strips(
            const Shape2& image_shape,
            const Vec<f64, 3>& image_angles,
            f64 spacing_nm,
            f64 z_step_nm
        ) -> Pair<f64, isize> {
            // The image is divided into z-strips centered at the image center. For instance, if z_step_nm=15,
            // the z-axis is divided such as: [..., -45, -30, -15,  +0, +15, +30, +45, ...]nm
            // These point to the z-height center of each strip and relative to the image center. In this case,
            // the central strip maps the [-7.5, +7.5]nm range, so every projected coordinate that falls within
            // that range should be assigned to this strip.
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

        static auto reduce_memory_requirements(
            const Shape2& image_shape,
            const Shape2& image_padded_shape,
            isize n_images,
            isize max_n_strips,
            usize n_bytes_free
        ) -> Tuple<isize, isize, bool> {
            // The strategy is the following:
            //  1. If GPU memory is low, try 2 chunks. This should be enough for most cases and
            //     decreases the overall memory needed (host and device). The overhead is minimal.
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

        void compute_irffts(isize n_strips, bool record = false, isize n_groups = 4) const {
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

        void prepare_irffts() const {
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

        [[nodiscard]] auto compute_filtered_stack(isize z) const -> Array<f32> {
            if (m_z_step_nm <= 0) // no ctf
                return m_images_filtered;

            Array<f32> debug_filter = noa::like<f32>(m_strips_padded_rfft);

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
                // ictf.set_bfactor(-0 / (cos(noa::deg2rad(image.angles[1]) * 0.01745) + 0.001f));
                // Logger::trace("bfactor={}", ictf.bfactor());

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

                    noa::iwise(image_padded_shape.rfft().push_front(ichunk_size), device, FilterPaddedImages{
                        .image_padded_rfft = image_padded_rfft.span().filter(2, 3).as_contiguous(),
                        .strips_padded_rfft = m_strips_padded_rfft.span().subregion(ichunk).filter(0, 2, 3).as_contiguous(),
                        .ctf = ictf.as<f32>(),
                        .fftfreq_norm = fftfreq_norm.as<f32>(),
                        .phase_flip_strength = m_phase_flip_strength,
                        .defocus_start_um = static_cast<f32>(defocus_start),
                        .defocus_step_um = static_cast<f32>(defocus_step),
                        .strip_offset = i,

                        // .debug_filter = debug_filter.span().subregion(ichunk).filter(0, 2, 3).as_contiguous(),
                    });
                    // noa::write_image(debug_filter.subregion(0), debug_path / fmt::format("debug_filter_chunk{:02}.mrc", i), {.dtype = "f16"}); // FIXME

                    compute_irffts(ichunk_size);

                    // noa::write_image(m_strips_padded.subregion(0), debug_path / fmt::format("debug_strips_chunk{:02}.mrc", i), {.dtype = "f32"}); // FIXME


                    const auto image_filtered = m_images_filtered.span().subregion(image.index).filter(2, 3).as_contiguous();
                    noa::iwise(image_filtered.shape(), device, RecomposeFilteredImage{
                        .strips_padded = m_strips_padded.span().subregion(ichunk).filter(0, 2, 3).as_contiguous(),
                        .image = image_filtered,
                        .image_center = image_shape.vec / 2,
                        .left_padding = left_padding,
                        // .z_projection_nm = z_projection_nm,
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

        [[nodiscard]] auto compute_filtered_stack() const -> Array<f32> {
            if (m_z_step_nm <= 0) // no ctf
                return m_images_filtered;

            const auto central_z_f64 = m_volume_z_center_nm / m_z_step_nm;
            const auto central_z = static_cast<i64>(std::floor(central_z_f64));
            return compute_filtered_stack(central_z);
        }

    public:
        const Metadata::Stack* m_metadata{};
        const CTFIsotropic64* m_ctf{};
        f64 m_z_step_nm{};
        f64 m_volume_z_center_nm{};
        f32 m_phase_flip_strength{};

        Array<c32> m_image_padded_rfft;
        Array<f32> m_images_padded;
        Array<c32> m_images_padded_rfft;
        Array<f32> m_strips_padded;
        Array<c32> m_strips_padded_rfft;
        Array<f32> m_images_filtered;
    };
}

namespace {
    using namespace qn;

    auto volume2image_matrices(
        const Metadata::Stack& metadata,
        bool correct_rotation,
        const Shape2& image_shape,
        const Shape3& volume_shape
    ) -> Array<Mat<f64, 2, 4>> {
        const auto image_center = (image_shape.vec / 2).as<f64>();
        const auto volume_center = (volume_shape.vec / 2).as<f64>();
        const auto volume2image_matrices = Array<Mat<f64, 2, 4>>(metadata.ssize());
        for (auto&& [image, volume2image]: noa::zip(metadata, volume2image_matrices.span_1d())) {
            const auto angles = noa::deg2rad(image.angles);
            const auto final_rotation = correct_rotation ? 0. : angles[0];
            volume2image = (
                nx::translate((image_center + image.shifts).push_front(0)) *
                nx::rotate_z<true>(angles[0]) *
                nx::rotate_y<true>(angles[1]) *
                nx::rotate_x<true>(angles[2]) *
                nx::rotate_z<true>(-final_rotation) *
                nx::translate(-volume_center)
            ).filter_rows(1, 2);
        }

        // Matrices relating 3d positions in the tomogram to 2d positions in the images.
        return volume2image_matrices;
    }

    // Project subvolume center back to image-space and
    // extract the twice-enlarged tile origin and residual shifts.
    auto extract_tile_large_window(
        f64 tile_large_center,
        const Vec<f64, 3>& subvolume_center_coordinates,
        const Mat<f64, 2, 4>& volume2image) {
        const auto tile_center_coordinate = volume2image * subvolume_center_coordinates.push_back(1);
        const auto tile_large_origin_coordinate = tile_center_coordinate - tile_large_center;
        const auto tile_large_origin_truncated = noa::floor(tile_large_origin_coordinate);
        const auto tile_large_origin = tile_large_origin_truncated.as<i32>();
        const auto tile_residual_shift = tile_large_origin_coordinate - tile_large_origin_truncated;

        // tilt_large_center + tile_residual_shift points to the center of
        // the padded tile after extraction at tile_large_origin.
        return Pair{tile_large_origin, tile_residual_shift};
    }

    struct ExtractPaddedTiles {
        SpanContiguous<const f32, 3> images;

        // For the Fourier reconstruction, this is tiles_large_padded.
        // For the real-space reconstruction, this is tiles_large.
        SpanContiguous<f32, 3> output_tiles;

        SpanContiguous<const Vec<i32, 2>> tile_large_origins;
        isize tile_large_center;
        f32 taper_radius;
        f32 taper_smoothness;

        NOA_HD void operator()(isize i, isize y, isize x) const {
            const auto tile_large_indices = Vec{y, x};
            const auto tile_coordinates = tile_large_indices - tile_large_center;
            const auto image_indices = tile_large_origins[i].as<isize>() + tile_large_indices;

            f32 value{};
            if (noa::is_inbound(images.shape().pop_front(), image_indices)) {
                // Compute the smooth taper.
                f32 taper{1};
                for (i32 j{}; j < 2; ++j) {
                    const auto tile_coordinate = static_cast<f32>(noa::abs(tile_coordinates[j]));
                    if (tile_coordinate >= taper_radius + taper_smoothness) {
                        taper = 0;
                    } else if (tile_coordinate >= taper_radius) {
                        constexpr auto PI = noa::Constant<f32>::PI;
                        const auto distance = (tile_coordinate - taper_radius) / taper_smoothness;
                        taper *= noa::cos(PI * distance) * 0.5f + 0.5f;
                    }
                }
                value = taper;

                // For the Fourier reconstruction, a significant part of the output tile is after the taper
                // because the oversampling is done by zero-padding the enlarged tiles. Therefore, only read
                // the input image if it's not zeroed-out by the taper and therefore in the padding.
                if (taper > 1e-6f)
                    value *= images(image_indices.push_front(i));
            }
            output_tiles(i, y, x) = value;
        }
    };

    template<nx::Interp INTERP>
    struct BackwardProjection {
    public:
        static constexpr auto BORDER = noa::Border::ZERO;
        using input_span_t = SpanContiguous<const f32, 3>;
        using interpolator_t = nx::Interpolator<2, INTERP, BORDER, input_span_t>;
        using matrices_span_t = SpanContiguous<const Mat<f32, 2, 4>>;

    public:
        interpolator_t tiles_large{}; // (n,h,w)
        matrices_span_t projection_matrices{}; // (n)
        SpanContiguous<f32, 3> subvolume_large{}; // (d,h,w)

    public:
        constexpr void operator()(const Vec<isize, 3>& indices) const {
            const auto volume_coordinates = indices.as<f32>().push_back(1);

            f32 value{};
            for (isize i{}; i < projection_matrices.ssize(); ++i) {
                const auto image_coordinates = projection_matrices[i] * volume_coordinates;
                value += tiles_large.interpolate_at(image_coordinates, i);
            }
            subvolume_large(indices) = value;
        }
    };

    template<typename T>
    void reconstruct_z_section(const T& reconstructor, const View<const f32>& stack, const View<f32>& z_section, isize z) {
        const auto [sy, sx] = reconstructor.subvolume_shape.filter(1, 2);
        const auto [ny, nx] = reconstructor.grid_shape.filter(1, 2);

        if (stack.device().is_cpu()) {
            // The subvolumes are relatively small, so to better distribute resources,
            // distribute subvolumes to threads directly, and set each subvolume to one thread.
            auto& stream = Stream::current({});
            auto n_threads = stream.thread_limit();
            stream.set_thread_limit(1);
            parallel_for(reconstructor.n_threads, Shape{ny, nx}, [&](isize tid, isize y, isize x) {
                const auto subvolume = reconstructor.reconstruct_subvolume(stack, z, y, x, tid);
                auto dst = z_section.subregion(
                    Ellipsis{},
                    Slice{y * sy, y * sy + sy},
                    Slice{x * sx, x * sx + sx}
                );
                auto src = subvolume.subregion(Ellipsis{}, Slice{0, dst.shape()[2]}, Slice{0, dst.shape()[3]});
                src.to(dst);
            });
            stream.set_thread_limit(n_threads);
        } else {
            // Store the subvolumes into the row buffer.
            // Once the row of subvolumes is computed, transfer to the host.
            for (isize y{}; y < ny; ++y) {
                for (isize x{}; x < nx; ++x) {
                    const auto subvolume = reconstructor.reconstruct_subvolume(stack, z, y, x);
                    const auto dst = reconstructor.subvolume_row.view().subregion(Ellipsis{}, Slice{x * sx, x * sx + sx});

                    auto src = subvolume.subregion(Ellipsis{}, Slice{0, dst.shape()[2]}, Slice{0, dst.shape()[3]});
                    src.to(dst);
                }
                auto dst = z_section.subregion(Ellipsis{}, Slice{y * sy, y * sy + sy}, Full{});
                reconstructor.subvolume_row.view().subregion(Ellipsis{}, Slice{0, dst.shape()[2]}, Full{}).to(dst);
            }
        }
    }

    class Reconstructor {
    public:
        nx::Interp interp;
        i32 n_threads{};

        isize actual_oversampling_factor{};
        Shape3 subvolume_shape{};
        Shape3 subvolume_large_shape{};
        Shape3 subvolume_large_padded_shape{};
        Shape4 grid_shape{};

        f64 tile_large_center;
        Array<Vec<i32, 2>> tile_large_origins;
        Array<Mat<f32, 2, 4>> tile_large_padded_matrices;

        Array<f32> subvolume_row;
        Array<c32> tiles_large_buffer;
        Array<c32> tiles_large_padded_buffer;
        Array<c32> subvolume_large_buffer;
        Array<c32> subvolume_large_padded_buffer;

    public:
        Reconstructor() = default;
        Reconstructor(
            const Shape2& image_shape,
            const Shape3& volume_shape,
            const Metadata::Stack& metadata,
            const Device& device,
            isize oversampling_factor,
            bool correct_rotation,
            nx::Interp interpolation,
            f64 spacing_nm,
            f64 z_step_nm
        ) :
            interp{interpolation}
        {
            const auto z_step = static_cast<isize>(z_step_nm / spacing_nm);
            const auto n_sections = volume_shape[0] / z_step;
            check(noa::is_odd(z_step) and noa::is_odd(n_sections));

            // 1. To support large reconstructions with oversampling, the volume is divided into subvolumes.
            // To backproject a subvolume, the input tiles should be large enough to map all voxels of the subvolume
            // from any angle. More specifically, if the tilt-axis is along Y, tiles should be sqrt(2)=1.41 times
            // larger than the largest dimension of the subvolume. If the tilt-axis is not aligned, tiles should be
            // sqrt(3)=1.73 times larger.
            //
            // 2. To prevent aliasing, we oversample both the tiles and subvolume. Oversampling real-space tiles is
            // done by zero-padding in Fourier space and thus requires the real-space tiles to have smoothed edges to
            // remove/reduce the Gibbs phenomenon.
            //
            // As such, we extract tiles twice as large and apply a smooth zero-taper to keep edges at zero. Then,
            // we oversample the padded tiles and backproject them. The resulting subvolume is then downsampled,
            // and the central subvolume is extracted and placed back into the volume. In other words, we backproject
            // subvolumes 4 times larger (x2 padding, x2 oversampling) than the final subvolume.
            const auto tile_size = nf::next_fast_size(std::max(z_step, isize{64}));
            const auto tile_large_size = tile_size * 2;
            actual_oversampling_factor = oversampling_factor;
            const auto tile_large_padded_size = tile_large_size * actual_oversampling_factor;

            subvolume_shape = Shape{z_step, tile_size, tile_size};
            subvolume_large_shape = Shape{z_step * 2, tile_large_size, tile_large_size};
            subvolume_large_padded_shape = Shape{z_step * 2 * actual_oversampling_factor, tile_large_padded_size, tile_large_padded_size};

            grid_shape = Shape{
                n_sections,
                noa::divide_up(image_shape[0], tile_size),
                noa::divide_up(image_shape[1], tile_size),
                metadata.ssize()
            };

            // Compute the transformation for each subvolume.
            const auto subvolume_center = (subvolume_shape.vec / 2).as<f64>();
            const auto subvolume_large_center = (subvolume_large_shape.vec / 2).as<f64>();
            tile_large_center = static_cast<f64>(tile_large_size / 2);
            tile_large_origins = Array<Vec<i32, 2>>(grid_shape);
            tile_large_padded_matrices = Array<Mat<f32, 2, 4>>(grid_shape);

            const auto volume2image = volume2image_matrices(metadata, correct_rotation, image_shape, volume_shape);
            const auto volume2image_1d = volume2image.span_1d();

            for (isize z{}; z < grid_shape[0]; ++z) {
                for (isize y{}; y < grid_shape[1]; ++y) {
                    for (isize x{}; x < grid_shape[2]; ++x) {
                        const auto subvolume_origin = Vec{z, y, x} * subvolume_shape.vec;
                        const auto subvolume_center_coordinates = subvolume_origin.as<f64>() + subvolume_center;

                        for (isize t{}; t < grid_shape[3]; ++t) {
                            const auto [tile_large_origin, tile_residual_shift] = extract_tile_large_window(
                                tile_large_center, subvolume_center_coordinates, volume2image_1d[t]);

                            // Compute the backward projection matrix (this is done on the enlarged and oversampled tiles).
                            // Note that the enlarged-tiles/subvolumes are even-sized, so the center is preserved
                            // during oversampling, meaning we can just scale the center with the oversampling factor.
                            const auto angles = noa::deg2rad(metadata[t].angles);
                            const auto final_rotation = correct_rotation ? 0. : angles[0];
                            const auto scale = static_cast<f64>(actual_oversampling_factor);

                            tile_large_origins.span()(z, y, x, t) = tile_large_origin;
                            tile_large_padded_matrices.span()(z, y, x, t) = ( // volume->image
                                nx::translate((tile_large_center + tile_residual_shift).push_front(0) * scale) *
                                nx::rotate_z<true>(angles[0]) *
                                nx::rotate_y<true>(angles[1]) *
                                nx::rotate_x<true>(angles[2]) *
                                nx::rotate_z<true>(-final_rotation) *
                                nx::translate(-subvolume_large_center * scale)
                            ).filter_rows(1, 2).as<f32>(); // (y, x)
                        }
                    }
                }
            }
            if (device.is_gpu()) {
                const auto options_async = ArrayOption{.device = device, .allocator = Allocator::ASYNC};
                tile_large_origins = std::move(tile_large_origins).to(options_async);
                tile_large_padded_matrices = std::move(tile_large_padded_matrices).to(options_async);
            }

            // Compute device.
            // On the GPU, use the tmp row buffer.
            // On the CPU, distribute subvolumes to threads. Since each thread needs its own buffers,
            // limit the number of threads to keep the memory usage reasonable. 10 threads need about 0.5GB.
            n_threads = device.is_gpu() ? 1 : std::max(Stream::current(device).thread_limit(), 8);

            // Allocate buffers.
            // Note that for the CPU mode, each thread needs its own buffer.
            // To retrieve the buffer (as a real and complex view), use the *_pair(tid) functions.
            const auto bd = Vec<isize, 2>::from_values(n_threads, grid_shape[3]);
            const auto tile_large_shape = Shape{tile_large_size, tile_large_size};
            const auto tile_large_padded_shape = Shape{tile_large_padded_size, tile_large_padded_size};
            const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};

            tiles_large_buffer = Array<c32>(tile_large_shape.rfft().push_front(bd), options);
            subvolume_large_buffer = Array<c32>(subvolume_large_shape.rfft().push_front(n_threads), options);
            if (actual_oversampling_factor > 1) {
                tiles_large_padded_buffer = Array<c32>(tile_large_padded_shape.rfft().push_front(bd), options);
                subvolume_large_padded_buffer = Array<c32>(subvolume_large_padded_shape.rfft().push_front(n_threads), options);
            }
            if (device.is_gpu())
                subvolume_row = Array<f32>(subvolume_shape.set<2>(volume_shape[2]).push_front(1), options);
        }

        [[nodiscard]] auto tiles_large_pair(isize tid) const {
            const auto& [td, th, tw] = subvolume_large_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = tiles_large_buffer.view().subregion(tid).permute({1, 0, 2, 3});
            pair.first = nf::alias_to_real(pair.second, Shape{grid_shape[3], isize{1}, th, tw});
            return pair;
        }

        [[nodiscard]] auto tiles_large_padded_pair(isize tid) const {
            if (actual_oversampling_factor == 1)
                return tiles_large_pair(tid);
            const auto& [td, th, tw] = subvolume_large_padded_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = tiles_large_padded_buffer.view().subregion(tid).permute({1, 0, 2, 3});
            pair.first = nf::alias_to_real(pair.second, Shape{grid_shape[3], isize{1}, th, tw});
            return pair;
        }

        [[nodiscard]] auto subvolume_large_pair(isize tid) const {
            const auto& [td, th, tw] = subvolume_large_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = subvolume_large_buffer.view().subregion(tid);
            pair.first = nf::alias_to_real(pair.second, Shape{isize{1}, td, th, tw});
            return pair;
        }

        [[nodiscard]] auto subvolume_large_padded_pair(isize tid) const {
            if (actual_oversampling_factor == 1)
                return subvolume_large_pair(tid);
            const auto& [td, th, tw] = subvolume_large_padded_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = subvolume_large_padded_buffer.view().subregion(tid);
            pair.first = nf::alias_to_real(pair.second, Shape{isize{1}, td, th, tw});
            return pair;
        }

        void prepare_rffts() const {
            if (actual_oversampling_factor > 1) {
                auto [tiles_large, tiles_large_rfft] = tiles_large_pair(0);
                auto [tiles_large_padded, tiles_large_padded_rfft] = tiles_large_padded_pair(0);
                auto [subvolume_large, subvolume_large_rfft] = subvolume_large_pair(0);
                auto [subvolume_large_padded, subvolume_large_padded_rfft] = subvolume_large_padded_pair(0);

                nf::r2c(tiles_large, tiles_large_rfft, {.record_and_share_workspace = true});
                nf::c2r(tiles_large_padded_rfft, tiles_large_padded, {.record_and_share_workspace = true});
                nf::r2c(subvolume_large_padded, subvolume_large_padded_rfft, {.record_and_share_workspace = true});
                nf::c2r(subvolume_large_rfft, subvolume_large, {.record_and_share_workspace = true});
            }
        }

        NOA_NOINLINE auto reconstruct_subvolume(
            const View<const f32>& input_stack,
            isize z, isize y, isize x, isize tid = 0
        ) const -> View<f32> {
            const auto [tiles_large, tiles_large_rfft] = tiles_large_pair(tid);
            const auto [tiles_large_padded, tiles_large_padded_rfft] = tiles_large_padded_pair(tid);
            const auto [subvolume_large, subvolume_large_rfft] = subvolume_large_pair(tid);
            const auto [subvolume_large_padded, subvolume_large_padded_rfft] = subvolume_large_padded_pair(tid);

            // Extract the twice-enlarged tiles and apply the zero-taper at the same time.
            // The backprojected region of the tile is, at most, sqrt(3)=1.73, so each edges has
            // an extra 6.5% of padding that isn't backprojected. We use the last 5% for the taper,
            // which should be enough to remove oversampling artifacts.
            const auto output_tiles = tiles_large.span_contiguous<f32, 3>();
            noa::iwise(output_tiles.shape(), tiles_large.device(), ExtractPaddedTiles{
                .images = input_stack.span_contiguous<const f32, 3>(),
                .output_tiles = output_tiles,
                .tile_large_origins = tile_large_origins.span().subregion(z, y, x).as_1d(),
                .tile_large_center = static_cast<isize>(tile_large_center),
                .taper_radius = static_cast<f32>(tile_large_center * 0.95),
                .taper_smoothness = static_cast<f32>(tile_large_center * 0.05),
            });

            // Oversample, if necessary.
            if (actual_oversampling_factor > 1) {
                nf::r2c(tiles_large, tiles_large_rfft);
                nf::resize<"h">(
                    tiles_large_rfft, tiles_large.shape(),
                    tiles_large_padded_rfft, tiles_large_padded.shape()
                );
                nf::c2r(tiles_large_padded_rfft, tiles_large_padded);
            }

            // Prefilter, if necessary.
            // TODO We could also prefilter before the oversampling.
            if (interp == nx::Interp::CUBIC_BSPLINE)
                nx::cubic_bspline_prefilter(tiles_large_padded, tiles_large_padded);

            // Backward project.
            const auto input = tiles_large_padded.span().filter(0, 2, 3).as_contiguous();
            const auto output = subvolume_large_padded.span().filter(1, 2, 3).as_contiguous();
            const auto matrices = tile_large_padded_matrices.span().subregion(z, y, x).as_1d();
            if (interp == nx::Interp::CUBIC_BSPLINE) {
                using operator_t = BackwardProjection<nx::Interp::CUBIC_BSPLINE>;
                noa::iwise(
                    output.shape(), subvolume_large_padded.device(),
                    operator_t{
                        .tiles_large = operator_t::interpolator_t(input, input.shape().pop_front()),
                        .projection_matrices = matrices,
                        .subvolume_large = output,
                    });
            } else if (interp == nx::Interp::LINEAR) {
                using operator_t = BackwardProjection<nx::Interp::LINEAR>;
                noa::iwise(
                    output.shape(), subvolume_large_padded.device(),
                    operator_t{
                        .tiles_large = operator_t::interpolator_t(input, input.shape().pop_front()),
                        .projection_matrices = matrices,
                        .subvolume_large = output,
                    });
            } else {
                panic("Unsupported interpolation mode");
            }

            // Downsample, if necessary.
            if (actual_oversampling_factor > 1) {
                nf::r2c(subvolume_large_padded, subvolume_large_padded_rfft);
                nf::resize<"h">(
                    subvolume_large_padded_rfft, subvolume_large_padded.shape(),
                    subvolume_large_rfft, subvolume_large.shape()
                );
                nf::c2r(subvolume_large_rfft, subvolume_large);
            }

            // Return a view of the subvolume (excluding the padding).
            const auto left_padding = subvolume_large_shape.vec / 2 - subvolume_shape.vec / 2;
            auto subvolume = subvolume_large.view().subregion(0,
                Slice{left_padding[0], left_padding[0] + subvolume_shape[0]},
                Slice{left_padding[1], left_padding[1] + subvolume_shape[1]},
                Slice{left_padding[2], left_padding[2] + subvolume_shape[2]}
            );
            return subvolume;
        }
    };

    template<nx::Interp INTERP>
    struct FourierInsertInterpolate {
        using input_span_t = SpanContiguous<const c32, 3, i32>;
        using interpolator_t = nx::InterpolatorSpectrum<2, nf::Layout::H2H, INTERP, input_span_t>;
        using output_span_t = SpanContiguous<c32, 3, i32>;

        using rotation_span_t = SpanContiguous<const nx::Quaternion<f32>, 1, i32>;
        using shift_span_t = SpanContiguous<const Vec<f32, 2>, 1, i32>;

    public:
        interpolator_t tiles_large_padded_rfft;
        rotation_span_t tile_rotations;
        shift_span_t tile_shifts;
        output_span_t subvolume_large_padded;

        f32 fftfreq_step;
        f32 fftfreq_sinc;
        f32 fftfreq_blackman;

    public:
        NOA_FHD auto volume2slice(const Vec<f32, 3>& fftfreq_3d, i32 slice_index) const {
            const auto fftfreq_3d_ = tile_rotations[slice_index].rotate(fftfreq_3d);
            return Pair{fftfreq_3d_[0], fftfreq_3d_.pop_front()};
        }

        NOA_HD void operator()(i32 oz, i32 oy, i32 ox) const noexcept {
            const auto frequency = nf::index2frequency<false, true>(Vec{oz, oy, ox}, subvolume_large_padded.shape());
            const auto fftfreq_3d = frequency.as<f32>() * fftfreq_step;
            if (noa::dot(fftfreq_3d, fftfreq_3d) > 0.25f) {
                subvolume_large_padded(oz, oy, ox) = 0;
                return;
            }

            c32 value{};
            f32 weights{};
            for (i32 i{}; i < tile_rotations.shape()[0]; ++i) {
                const auto [fftfreq_z, fftfreq_2d] = volume2slice(fftfreq_3d, i);

                if (abs(fftfreq_z) <= fftfreq_blackman) { // the slice affects the voxel
                    const auto window = nx::details::windowed_sinc(fftfreq_z, fftfreq_sinc, fftfreq_blackman);
                    const auto frequency_2d = fftfreq_2d / fftfreq_step;
                    value += tiles_large_padded_rfft.interpolate_spectrum_at(frequency_2d, i) * window;
                    weights += window;
                }
            }
            subvolume_large_padded(oz, oy, ox) = value / noa::max(noa::abs(weights), 1.f);
        }
    };

    class ReconstructorFourier {
    public:
        nx::Interp interp;
        i32 n_threads{};

        isize actual_oversampling_factor{};
        isize tile_large_padded_size{};
        f64 tile_large_center{};
        Shape3 subvolume_shape{};
        Vec<f64, 3> subvolume_center{};
        Shape3 subvolume_large_padded_shape{};

        Shape4 grid_shape{};

        Array<Vec<i32, 2>> tile_large_origins;
        Array<nx::Quaternion<f32>> tile_rotations;
        Array<Vec<f32, 2>> tile_large_shifts;

        Array<f32> subvolume_row;
        Array<c32> tiles_large_padded_buffer;
        Array<c32> subvolume_large_padded_buffer;

    public:
        ReconstructorFourier() = default;
        ReconstructorFourier(
            const Shape2& image_shape,
            const Shape3& volume_shape,
            const Metadata::Stack& metadata,
            const Device& device,
            isize oversampling_factor,
            bool correct_rotation,
            nx::Interp interpolation,
            f64 spacing_nm,
            f64 z_step_nm
        ) :
            interp{interpolation}
        {
            const auto z_step = static_cast<isize>(z_step_nm / spacing_nm);
            const auto n_sections = volume_shape[0] / z_step;
            check(noa::is_odd(z_step) and noa::is_odd(n_sections));

            // 1. Similar to the real-space reconstruction, the reconstruction is divided into subvolumes and
            // to backproject a subvolume, the input tiles should be twice as large as the subvolume.
            //
            // 2. Then, the twice-enlarged tiles are zero-tapered and zero-padded to match the desired oversampling
            // factor. This zero-padding is done on the right-side to simplify the phase-shifts, and a minimum of 3x
            // oversampling is enforced (3x the subvolume size) to ensure that the edges of the enlarged tiles don't
            // warp around when rotating in Fourier space.
            const auto tile_size = nf::next_fast_size(std::max(z_step, isize{64}));
            const auto tile_large_size = tile_size * 2;
            actual_oversampling_factor = std::max(isize{3}, oversampling_factor);
            tile_large_padded_size = tile_size * actual_oversampling_factor;
            subvolume_large_padded_shape = Shape3::from_value(tile_large_padded_size);

            // Importantly, while we reconstruct twice-enlarged zero-padded cubes, we center them in z
            // so that the center of the cubes matches the original subvolume z-center, aka z_step/2.
            subvolume_shape = Shape{z_step, tile_size, tile_size};
            subvolume_center = (subvolume_shape.vec / 2).as<f64>();

            grid_shape = Shape{
                n_sections,
                noa::divide_up(image_shape[0], tile_size),
                noa::divide_up(image_shape[1], tile_size),
                metadata.ssize()
            };

            // Compute the rotation of the central-slices.
            tile_rotations = Array<nx::Quaternion<f32>>(grid_shape[3]);
            for (auto&& [image, rotation]: noa::zip(metadata, tile_rotations.span_1d())) {
                const auto angles = noa::deg2rad(image.angles);
                const auto final_rotation = correct_rotation ? 0. : angles[0];
                rotation = nx::matrix2quaternion( // volume->slice
                    nx::rotate_z(angles[0]) *
                    nx::rotate_y(angles[1]) *
                    nx::rotate_x(angles[2]) *
                    nx::rotate_z(-final_rotation)
                ).as<f32>();
            }

            // Compute the shifts for every subvolume.
            tile_large_center = static_cast<f64>(tile_large_size / 2);
            tile_large_origins = Array<Vec<i32, 2>>(grid_shape);
            tile_large_shifts = Array<Vec<f32, 2>>(grid_shape);

            const auto volume2image = volume2image_matrices(metadata, correct_rotation, image_shape, volume_shape);
            const auto volume2image_1d = volume2image.span_1d();

            for (isize z{}; z < grid_shape[0]; ++z) {
                for (isize y{}; y < grid_shape[1]; ++y) {
                    for (isize x{}; x < grid_shape[2]; ++x) {
                        const auto subvolume_origin = Vec{z, y, x} * subvolume_shape.vec;
                        const auto subvolume_center_coordinates = subvolume_origin.as<f64>() + subvolume_center;

                        for (isize t{}; t < grid_shape[3]; ++t) {
                            const auto& [tile_padded_origin, tile_residual_shift] = extract_tile_large_window(
                                tile_large_center, subvolume_center_coordinates, volume2image_1d[t]);

                            tile_large_origins.span()(z, y, x, t) = tile_padded_origin;
                            tile_large_shifts.span()(z, y, x, t) = -(tile_large_center + tile_residual_shift).as<f32>();
                        }
                    }
                }
            }
            if (device.is_gpu()) {
                const auto options_async = ArrayOption{.device = device, .allocator = Allocator::ASYNC};
                tile_large_origins = std::move(tile_large_origins).to(options_async);
                tile_large_shifts = std::move(tile_large_shifts).to(options_async);
                tile_rotations = std::move(tile_rotations).to(options_async);
            }

            // Compute device.
            // On the GPU, use the tmp row buffer.
            // On the CPU, distribute subvolumes to threads. Since each thread needs its own buffers,
            // limit the number of threads to keep the memory usage reasonable. 10 threads need about 0.5GB.
            n_threads = device.is_gpu() ? 1 : std::max(Stream::current(device).thread_limit(), 8);

            // Allocate buffers.
            // Note that for the CPU mode, each thread needs its own buffer.
            // To retrieve the buffer (as a real and complex view), use the *_pair(tid) functions.
            const auto bd = Vec<isize, 2>::from_values(n_threads, grid_shape[3]);
            const auto tile_large_padded_shape = Shape2::from_value(tile_large_padded_size);
            const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};

            tiles_large_padded_buffer = Array<c32>(tile_large_padded_shape.rfft().push_front(bd), options);
            subvolume_large_padded_buffer = Array<c32>(subvolume_large_padded_shape.rfft().push_front(n_threads), options);
            if (device.is_gpu())
                subvolume_row = Array<f32>(subvolume_shape.set<2>(volume_shape[2]).push_front(1), options);
        }

        [[nodiscard]] auto tiles_large_padded_pair(isize tid) const {
            const auto& ts = tile_large_padded_size;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = tiles_large_padded_buffer.view().subregion(tid).permute({1, 0, 2, 3});
            pair.first = nf::alias_to_real(pair.second, Shape{grid_shape[3], isize{1}, ts, ts});
            return pair;
        }

        [[nodiscard]] auto subvolume_large_padded_pair(isize tid) const {
            const auto& ts = tile_large_padded_size;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = subvolume_large_padded_buffer.view().subregion(tid);
            pair.first = nf::alias_to_real(pair.second, Shape{isize{1}, ts, ts, ts});
            return pair;
        }

        void prepare_rffts() const {
            auto [tiles_large_padded, tiles_large_padded_rfft] = tiles_large_padded_pair(0);
            auto [subvolume_large_padded, subvolume_large_padded_rfft] = subvolume_large_padded_pair(0);

            nf::r2c(tiles_large_padded, tiles_large_padded_rfft, {.record_and_share_workspace = true});
            nf::c2r(subvolume_large_padded_rfft, subvolume_large_padded, {.record_and_share_workspace = true});
        }

        NOA_NOINLINE auto reconstruct_subvolume(
            const View<const f32>& input_stack,
            isize z, isize y, isize x, isize tid = 0
        ) const -> View<f32> {
            const auto [tiles_large_padded, tiles_large_padded_rfft] = tiles_large_padded_pair(tid);
            const auto [subvolume_large_padded, subvolume_large_padded_rfft] = subvolume_large_padded_pair(tid);
            const auto device = tiles_large_padded.device();

            // Extract and taper the padded tiles.
            // For the Fourier reconstruction, the tiles are padded by twice for the backprojection
            // and an additional zero-padding is done on the right for the interpolation in Fourier space.
            const auto output_tiles = tiles_large_padded.span_contiguous<f32, 3>();
            noa::iwise(output_tiles.shape(), device, ExtractPaddedTiles{
                .images = input_stack.span_contiguous<const f32, 3>(),
                .output_tiles = output_tiles,
                .tile_large_origins = tile_large_origins.span().subregion(z, y, x).as_1d(),
                .tile_large_center = static_cast<isize>(tile_large_center),
                .taper_radius = static_cast<f32>(tile_large_center * 0.95),
                .taper_smoothness = static_cast<f32>(tile_large_center * 0.05), // FIXME
            });
            // noa::write_image(tiles_large_padded, debug_path / "output_tiles.mrc", {.dtype = "f16"});

            // Compute the rotation-centered central-slices.
            nf::r2c(tiles_large_padded, tiles_large_padded_rfft);
            ns::phase_shift_2d<"h">(
                tiles_large_padded_rfft, tiles_large_padded_rfft,
                tiles_large_padded.shape(), tile_large_shifts.view().subregion(z, y, x)
            );

            // Insert the central-slices.
            const auto central_slices = tiles_large_padded_rfft.span_contiguous<const c32, 3, i32>();
            if (interp == nx::Interp::LINEAR) {
                using operator_t = FourierInsertInterpolate<nx::Interp::LINEAR>;
                noa::iwise(subvolume_large_padded_rfft.shape().pop_front().as<i32>(), device, operator_t{
                    .tiles_large_padded_rfft = operator_t::interpolator_t(central_slices, central_slices.shape().pop_front()),
                    .tile_rotations = tile_rotations.span_1d().as_index<i32>(),
                    .tile_shifts = tile_large_shifts.span().subregion(z, y, x).as_1d().as_index<i32>(),
                    .subvolume_large_padded = subvolume_large_padded_rfft.span_contiguous<c32, 3, i32>(),
                    .fftfreq_step = 1 / static_cast<f32>(tile_large_padded_size),
                    .fftfreq_sinc = 1 / static_cast<f32>(tile_large_padded_size),
                    .fftfreq_blackman = 8 / static_cast<f32>(tile_large_padded_size),
                });
            } else if (interp == nx::Interp::LANCZOS6) {
                using operator_t = FourierInsertInterpolate<nx::Interp::LANCZOS6>;
                noa::iwise(subvolume_large_padded_rfft.shape().pop_front().as<i32>(), device, operator_t{
                    .tiles_large_padded_rfft = operator_t::interpolator_t(central_slices, central_slices.shape().pop_front()),
                    .tile_rotations = tile_rotations.span_1d().as_index<i32>(),
                    .tile_shifts = tile_large_shifts.span().subregion(z, y, x).as_1d().as_index<i32>(),
                    .subvolume_large_padded = subvolume_large_padded_rfft.span_contiguous<c32, 3, i32>(),
                    .fftfreq_step = 1 / static_cast<f32>(tile_large_padded_size),
                    .fftfreq_sinc = 1 / static_cast<f32>(tile_large_padded_size),
                    .fftfreq_blackman = 8 / static_cast<f32>(tile_large_padded_size),
                });
            } else {
                panic("Unsupported interpolation mode");
            }

            // Compute the reconstructed subvolume.
            ns::phase_shift_3d<"h">(
                subvolume_large_padded_rfft, subvolume_large_padded_rfft,
                subvolume_large_padded.shape(), subvolume_center.as<f32>()
            );
            nf::c2r(subvolume_large_padded_rfft, subvolume_large_padded);
            // noa::write_image(subvolume_large_padded, debug_path / "subvolume_large_padded.mrc", {.dtype = "f32"});

            // Return a view of the subvolume.
            auto subvolume = subvolume_large_padded.view().subregion(0,
                Slice{0, subvolume_shape[0]},
                Slice{0, subvolume_shape[1]},
                Slice{0, subvolume_shape[2]}
            );
            return subvolume;
        }
    };
}

namespace {
    struct ReconstructionThickness {
        f64 z_step_nm;
        isize z_step;
        f64 z_padding;
        f64 thickness_nm;
        isize thickness;
    };

    auto reconstruction_thickness(f64 spacing_nm, f64 defocus_step_nm, f64 sample_thickness_nm, f64 z_padding_percent = 0) {
        // For simplicity, make the defocus resolution (in pixels) an odd integer multiple of the pixel size.
        auto z_step = static_cast<isize>(std::floor(defocus_step_nm / spacing_nm));
        z_step += noa::is_even(z_step);
        const auto z_step_nm = static_cast<f64>(z_step) * spacing_nm;

        // Get volume thickness and number of z-sections (of size z_step).
        // To guarantee that the volume center is at the center of a z-section,
        // make the volume thickness an odd multiple of z_step.
        const f64 sample_thickness = sample_thickness_nm / spacing_nm;
        const f64 z_padding = sample_thickness * z_padding_percent / spacing_nm;
        auto volume_thickness = static_cast<isize>(std::round(sample_thickness + z_padding));
        volume_thickness = noa::next_multiple_of(volume_thickness, z_step);
        if (noa::is_even(volume_thickness / z_step))
            volume_thickness += z_step;
        const auto volume_thickness_nm = static_cast<f64>(volume_thickness) * spacing_nm;

        return ReconstructionThickness{
            .z_step_nm = z_step_nm,
            .z_step = z_step,
            .z_padding = z_padding,
            .thickness_nm = volume_thickness_nm,
            .thickness = volume_thickness,
        };
    }
}

namespace qn {
    auto filter_stack(
        StackLoader&& stack,
        Metadata& metadata,
        const FilterStackSettings& settings
    ) -> Array<f32> {
        const auto timer = Logger::info_scope_time("Filtering stack");

        // debug_path = "/dls/ebic/data/staff-scratch/thomas2/tmp/quinoa/10304/pm11"; // FIXME

        const auto spacing = mean(stack.stack_spacing());
        const auto spacing_nm = spacing * 1e-1;
        const auto ctf = CTFIsotropic64::Parameters{
            .pixel_size = spacing,
            .defocus = 0.,
            .voltage = metadata.sample.voltage,
            .amplitude = metadata.sample.amplitude,
            .cs = metadata.sample.cs,
            .phase_shift = 0,
            .bfactor = settings.bfactor,
            .scale = 1.,
        }.to_ctf();

        const auto recons = reconstruction_thickness(spacing_nm, settings.defocus_step_nm, metadata.sample.thickness);
        const auto device = stack.compute_device();
        const auto bytes_start = Allocator::bytes_currently_allocated(device);

        const auto filterer = not settings.correct_ctf ?
            Filterer(std::move(stack), metadata.stack, settings.ramp_filter) :
            Filterer(std::move(stack), metadata.stack, settings.ramp_filter,
                     ctf, recons.thickness_nm, recons.z_step_nm, settings.phase_flip_strength);

        nf::clear_cache(device);
        nf::set_cache_limit(10, device);
        filterer.prepare_irffts();

        const auto bytes_send = Allocator::bytes_currently_allocated(device);
        Logger::trace("allocated: {}={:.2f}GB", device, static_cast<f64>(bytes_send - bytes_start) * 1e-9);

        auto filtered_stack = filterer.compute_filtered_stack();

        nf::clear_cache(device);
        return filtered_stack;
    }

    auto reconstruct_tomogram(
        StackLoader&& stack,
        Metadata& metadata,
        const FilterStackSettings& filter_settings,
        const ReconstructTomogramSettings& settings
    ) -> Array<f32> {
        auto timer = Logger::info_scope_time("Reconstructing tomogram");

        auto run = [&]<typename T>() {
            const auto spacing = mean(stack.stack_spacing());
            const auto spacing_nm = spacing * 1e-1;
            const auto ctf = CTFIsotropic64({
                .pixel_size = spacing,
                .defocus = 0.,
                .voltage = metadata.sample.voltage,
                .amplitude = metadata.sample.amplitude,
                .cs = metadata.sample.cs,
                .phase_shift = 0,
                .bfactor = 0,
                .scale = 1.,
            });

            const auto setup = reconstruction_thickness(
                spacing_nm, filter_settings.defocus_step_nm,
                metadata.sample.thickness, settings.z_padding_percent
            );
            const auto image_shape = stack.slice_shape();
            const auto volume_shape = image_shape.push_front(setup.thickness);

            const auto device = stack.compute_device();
            const auto bytes_start = Allocator::bytes_currently_allocated(device);

            const auto filterer = not filter_settings.correct_ctf ?
                Filterer(std::move(stack), metadata.stack, filter_settings.ramp_filter) :
                Filterer(std::move(stack), metadata.stack, filter_settings.ramp_filter,
                         ctf, setup.thickness_nm, setup.z_step_nm, filter_settings.phase_flip_strength);

            const auto reconstructor = T(
                image_shape, volume_shape, metadata.stack, device,
                settings.oversampling_factor, settings.correct_rotation,
                settings.interp, spacing_nm, setup.z_step_nm
            );

            Logger::trace(
                "Tomogram reconstruction:\n"
                "  shape={}"
                "  spacing={:.1f}A (resolution={:.1f}A)\n"
                "  thickness={:.1f}nm (specimen={:.1f}nm, z_padding={:.1f}nm|{:.1f}pix)\n"
                "  algorithm={} (interp={}, oversampling_factor={})\n"
                "  subvolume_shape={} (actual_subvolume_shape={})\n"
                "  grid_shape={} (n_subvolumes={})\n"
                "  device={}{}",
                volume_shape,
                spacing_nm * 10, spacing_nm * 20,
                setup.thickness_nm, metadata.sample.thickness, setup.z_padding * spacing_nm, setup.z_padding,
                settings.algorithm, settings.interp, reconstructor.actual_oversampling_factor,
                reconstructor.subvolume_shape, reconstructor.subvolume_large_padded_shape,
                reconstructor.grid_shape.pop_back(), reconstructor.grid_shape.pop_back().n_elements(),
                device, device.is_cpu() ? fmt::format(" (n_threads={})", reconstructor.n_threads) : ""
            );

            // Prepare for FFTs with various batch sizes.
            // In CPU, this only precomputes the plans and isn't necessary.
            // In CUDA, while this is also optional, it allows sharing the workspace
            // across all transforms, possibly saving a lot of memory.
            nf::clear_cache(device);
            nf::set_cache_limit(10, device);
            reconstructor.prepare_rffts();
            filterer.prepare_irffts(); // allocates the workspace

            const auto bytes_end = Allocator::bytes_currently_allocated(device);
            Logger::trace("allocated: {}={:.2f}GB", device, static_cast<f64>(bytes_end - bytes_start) * 1e-9);

            // Reconstruct the (possibly CTF-corrected) tomogram.
            auto tomogram = Array<f32>(volume_shape.push_front(1));
            const auto sz = reconstructor.subvolume_shape[0];
            const auto nz = reconstructor.grid_shape[0];
            for (isize z{}; z < nz; ++z) {
                auto t1 = Logger::trace_scope_time("z={:02}/{:02}", z + 1, nz);
                const auto filtered_stack = filterer.compute_filtered_stack(z);
                const auto volume_z_section = tomogram.view().subregion(0, Slice{z * sz, z * sz + sz});
                reconstruct_z_section(reconstructor, filtered_stack.view(), volume_z_section, z);
            }

            nf::clear_cache(device);
            return tomogram;
        };

        if (settings.algorithm == "fourier-wbp")
            return run.operator()<ReconstructorFourier>();
        if (settings.algorithm == "real-bp")
            return run.operator()<Reconstructor>();

        panic("Unknown reconstruction algorithm: {}", settings.algorithm);
    }

    void post_processing(
        const Path& input_stack,
        const Metadata& metadata,
        const PostProcessingSettings& settings,
        const FilterStackSettings& filter_settings,
        const ReconstructTomogramSettings& reconstruct_settings
    ) {
        auto loader = StackLoader(input_stack, {
            .compute_device = settings.compute_device,
            .allocator = Allocator::DEFAULT_ASYNC,
            .precise_cutoff = true,
            .rescale_target_resolution = settings.target_resolution,
            .rescale_min_size = settings.min_size,
            .bandpass{
                .highpass_cutoff = 0.01,
                .highpass_width = 0.01,
                .lowpass_cutoff = 0.49,
                .lowpass_width = 0.01,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .exposure_filter_voltage = metadata.sample.voltage,
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.02,
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });

        const auto spacing = mean(loader.stack_spacing());
        const auto basename = input_stack.stem().string();

        auto meta = metadata;
        meta.set_spacing(spacing);
        meta.stack.sort("tilt").reset_indices();

        if (settings.save_aligned_stack) {
            const auto filename = settings.output_directory / fmt::format("{}_stack.mrc", basename);
            save_stack(loader, filename, meta.stack, {
                .correct_rotation = settings.stack_correct_rotation,
                .cache_loader = settings.save_tomogram,
                .interp = settings.stack_interp,
                .border = noa::Border::ZERO,
                .dtype = settings.stack_dtype,
            });
        }

        if (settings.save_tomogram) {
            const auto filename = settings.output_directory / fmt::format("{}_tomogram.mrc", basename);
            debug_path = settings.output_directory; // FIXME
            const auto tomogram = reconstruct_tomogram(std::move(loader), meta, filter_settings, reconstruct_settings);
            noa::write_image(tomogram, filename, {
                .spacing = Vec<f64, 3>::from_value(spacing),
                .dtype = settings.tomogram_dtype,
            });
            Logger::trace("{} saved", filename);
        }
    }
}
