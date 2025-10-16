#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>
#include <noa/Utils.hpp>
#include <noa/IO.hpp>

#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/CTF.hpp"
#include "quinoa/PostProcessing.hpp"

namespace {
    using namespace qn;

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
        i64 strip_offset;

        NOA_HD void operator()(i64 s, i64 y, i64 x) {
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
            const auto value = -noa::sin(ctf.phase_at(fftfreq_2d)); // ctf.value_at but without envelope

            // Compute and the filter.
            const auto regularization = phase_flip_strength * noa::exp(2 * fftfreq_sqd);
            const auto phase_flip = -noa::sign(value); // negate for negative contrast
            const auto wiener_like = (1 + regularization) / (noa::abs(value) + regularization) * phase_flip;
            const auto b_decay = noa::exp(ctf.bfactor() / 4 * fftfreq_sqd);
            strips_padded_rfft(s, y, x) = image_padded_rfft(y, x) * wiener_like * b_decay;
        }
    };

    struct RecomposeFilteredImage {
        SpanContiguous<const f32, 3> strips_padded; // (s,h+p,w+p)
        SpanContiguous<f32, 2> image; // (h,w)

        Vec<i64, 2> image_center;
        Vec<i64, 2> left_padding;
        Vec<f32, 3> z_projection_nm;
        f32 z_offset_start_nm;
        f32 z_step_nm;

        i64 strip_start;
        i64 strip_end;

        NOA_HD void operator()(i64 i, i64 j) const {
            // Get the z-position at this index of the image.
            const auto indices = Vec{i, j};
            const auto coordinates = (indices - image_center).as<f32>();
            const auto volume_z_coordinate_nm = noa::dot(z_projection_nm, coordinates.push_front(0));

            // Get the closest z-strip.
            const auto strip = (volume_z_coordinate_nm - z_offset_start_nm) / z_step_nm;
            const auto strip_index = static_cast<i64>(noa::round(strip));

            // If the chunk contains that z-strip, save it into the image.
            if (strip_index >= strip_start and strip_index < strip_end) {
                const auto padded_indices = indices + left_padding;
                const auto chunk_index = strip_index - strip_start;
                image(indices) = strips_padded(padded_indices.push_front(chunk_index));
            }
        }
    };

    struct ExtractPaddedTiles {
        SpanContiguous<const f32, 3> images;
        SpanContiguous<f32, 3> tiles_padded;
        SpanContiguous<const Vec<i32, 2>> tile_padded_origins;
        Vec<i64, 2> tile_padded_center;
        f32 taper_radius;
        f32 taper_smoothness;

        NOA_HD void operator()(i64 i, i64 y, i64 x) const {
            const auto tile_padded_indices = Vec{y, x};
            const auto tile_coordinates = tile_padded_indices - tile_padded_center;
            const auto image_indices = tile_padded_origins[i].as<i64>() + tile_padded_indices;

            // Get the image value, or zero if OOB.
            f32 value{};
            if (ni::is_inbound(images.shape().pop_front(), image_indices)) {
                value = images(image_indices.push_front(i));

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
                value *= taper;
            }
            tiles_padded(i, y, x) = value;
        }
    };

    template<noa::Interp INTERP>
    struct BackwardProjection {
    public:
        static constexpr auto BORDER = noa::Border::ZERO;
        using input_span_t = SpanContiguous<const f32, 3>;
        using interpolator_t = noa::Interpolator<2, INTERP, BORDER, input_span_t>;
        using matrices_span_t = SpanContiguous<const Mat<f32, 2, 4>>;

    public:
        interpolator_t tiles_padded{}; // (n,h,w)
        matrices_span_t projection_matrices{}; // (n)
        SpanContiguous<f32, 3> subvolume{}; // (d,h,w)

    public:
        constexpr void operator()(const Vec<i64, 3>& indices) const {
            const auto volume_coordinates = indices.as<f32>().push_back(1);

            f32 value{};
            for (i64 i{}; i < projection_matrices.ssize(); ++i) {
                // TODO Add volume deformations with the addition of a SplineGrid updating volume_coordinates.
                const auto image_coordinates = projection_matrices[i] * volume_coordinates;
                value += tiles_padded.interpolate_at(image_coordinates, i);
            }
            subvolume(indices) = value;
        }
    };
}

namespace {
    using namespace qn;

    class Filterer {
    public:
        Filterer() = default;

        Filterer(
            StackLoader& loader,
            const MetadataStack& metadata
        ) {
            Logger::trace(
                "Filtering:\n"
                "  mode=radial-weight"
            );

            const auto image_shape = loader.slice_shape();
            const auto images_shape = image_shape.push_front(Vec{metadata.ssize(), i64{1}});
            const auto [buffer, buffer_rfft] = nf::empty<f32>(images_shape, {
                .device = loader.compute_device(),
                .allocator = Allocator::ASYNC,
            });

            for (i64 i{}; auto& slice: metadata)
                loader.read_slice(buffer.view().subregion(i++), slice.index_file);
            loader.clear_cache(); // remove stack saved on the host

            nf::r2c(buffer, buffer_rfft);
            ns::filter_spectrum_2d<"h2h">(buffer_rfft, buffer_rfft, buffer.shape(), FilterImages{});
            nf::c2r(buffer_rfft, buffer);

            m_images_filtered = buffer;
        }

        Filterer(
            StackLoader& loader,
            const MetadataStack& metadata,
            const CTFIsotropic64& ctf,
            f64 volume_thickness_nm,
            f64 z_step_nm,
            i64 phase_flip_strength
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
            check(noa::is_odd(static_cast<i64>(n_sections)));
            m_volume_z_center_nm = (n_sections / 2) * z_step_nm;

            // Get the maximum defocus and n_strips.
            f64 max_z_range_nm{};
            f64 max_defocus_nm{};
            i64 min_n_strips{std::numeric_limits<i64>::max()};
            i64 max_n_strips{};
            for (auto& slice: metadata) {
                const auto [start_offset_nm, n_strips] = divide_image_in_z_strips(
                    image_shape, slice.angles, spacing_nm, z_step_nm
                );
                const auto end_strip_nm = start_offset_nm + z_step_nm * static_cast<f64>(n_strips - 1);
                const auto z_range_nm = end_strip_nm - start_offset_nm;
                const auto image_defocus_nm = (slice.defocus.value + std::abs(slice.defocus.astigmatism)) * 1e3;
                const auto highest_defocus_nm = image_defocus_nm - start_offset_nm; // underfocus negative

                max_z_range_nm = std::max(max_z_range_nm, z_range_nm);
                max_defocus_nm = std::max(max_defocus_nm, highest_defocus_nm);
                min_n_strips = std::min(min_n_strips, n_strips);
                max_n_strips = std::max(max_n_strips, n_strips);
            }
            max_defocus_nm += volume_thickness_nm / 2;

            // Get the size requirement for the maximum defocus.
            // The following is adapted from Russo & Henderson, 2018.
            // This is just a rough (under)estimate of what the filtering does.
            // https://www.desmos.com/calculator/w1dlw58f8t
            // 8A resolution, 4um defocus -> delocalization is ~50pix
            // 4A resolution, 4um defocus -> delocalization is ~200pix
            const f64 wavelength_nm = ns::relativistic_electron_wavelength(ctf.voltage() * 1e3) * 1e9;
            const f64 resolution_nm = spacing_nm * 2;
            const f64 delocalization_nm = 2 * max_defocus_nm * wavelength_nm / resolution_nm;
            const f64 delocalization_pix = std::round(delocalization_nm / spacing_nm);
            const i64 aliasing_free_size = [&] {
                auto ictf = ctf;
                ictf.set_defocus(max_defocus_nm * 1e-3);
                return ctf::aliasing_free_size(ictf, Vec{0., 0.5});
            }();

            // Compute a satisfying shape given these limits.
            constexpr f64 PADDING_FACTOR = 1.2;
            const auto minimum_padding = static_cast<i64>(delocalization_pix);
            auto image_padded_shape = Shape{(image_shape.vec.as<f64>() * PADDING_FACTOR).as<i64>()};
            image_padded_shape = noa::max(image_padded_shape, image_shape + minimum_padding);
            image_padded_shape = noa::max(image_padded_shape, aliasing_free_size);
            image_padded_shape = nf::next_fast_shape(image_padded_shape);

            // Allocating and processing all strips at once can require a lot of memory for high-resolution and
            // high-tilt images. To decrease the memory requirement, process the strips in chunks so that we only
            // need to allocate a chunk size of strips.
            const auto [n_chunks, chunk_size, keep_spectra_on_device] = reduce_memory_requirements(
                image_shape, image_padded_shape, metadata.ssize(), max_n_strips,
                loader.compute_device().memory_capacity().free
            );

            Logger::trace(
                "Filtering:\n"
                "  mode=ctf\n"
                "  max_z_range={:.2f}nm\n"
                "  max_defocus={:.2f}nm\n"
                "  strips=[min={}, max={}, chunk={}, n_chunks={}]\n"
                "  max_delocalization={:.3f}nm ({}pix)\n"
                "  aliasing_free_size={}\n"
                "  padded_shape={} (shape={}, ratio={::.2f})",
                max_z_range_nm, max_defocus_nm, min_n_strips, max_n_strips, chunk_size, n_chunks,
                delocalization_nm, delocalization_pix, aliasing_free_size, image_padded_shape, image_shape,
                image_padded_shape.vec.as<f64>() / image_shape.vec.as<f64>()
            );

            allocate_and_prepare_spectra(
                loader, image_shape, image_padded_shape, metadata,
                keep_spectra_on_device, chunk_size
            );
        }

        static auto divide_image_in_z_strips(
            const Shape<i64, 2>& image_shape,
            const Vec<f64, 3>& image_angles,
            f64 spacing_nm,
            f64 z_step_nm
        ) -> Pair<f64, i64> {
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

            // Projection matrix to get the z-height of image coordinates in volume space.
            // Note that the image shifts should not be applied; the CTF fitting is done on
            // tiles extracted from the original images/frames.
            const auto angles = noa::deg2rad(image_angles);
            const auto z_projection = (
                ng::scale(Vec<f64, 3>::from_value(spacing_nm)) *
                ng::rotate_x(angles[2]) *
                ng::rotate_y(angles[1]) *
                ng::rotate_z(angles[0])
            )[0];

            // Compute the z-range within the image.
            auto minmax = Vec<f64, 2>{}; // in nm
            const auto image_center = (image_shape.vec / 2).as<f64>();
            for (const auto& image_edge: image_edges) {
                const auto z_distance_nm = noa::dot(z_projection, (image_edge - image_center).push_front(0));
                minmax[0] = std::min(minmax[0], z_distance_nm);
                minmax[1] = std::max(minmax[1], z_distance_nm);
            }

            // Get the corresponding z-strips for that image.
            auto first_strip_nm = noa::round(minmax[0] / z_step_nm) * z_step_nm;
            auto last_strip_nm = noa::round(minmax[1] / z_step_nm) * z_step_nm;
            auto n_strips = (last_strip_nm - first_strip_nm) / z_step_nm + 1;
            return {first_strip_nm, static_cast<i64>(std::round(n_strips))};
        }

        static auto reduce_memory_requirements(
            const Shape<i64, 2>& image_shape,
            const Shape<i64, 2>& image_padded_shape,
            i64 n_images,
            i64 max_n_strips,
            size_t n_bytes_free
        ) -> Tuple<i64, i64, bool> {
            // The strategy is the following:
            //  1. If GPU memory is low, try 2 chunks. This should be enough for most cases and decreases the overall
            //     memory needed (host and device). The overhead is minimal.
            //  2. If this is still not enough, keep spectra on the host.
            //  3. If this is still not enough, divide in more chunks.
            const auto images_shape = image_shape.push_front(Vec{n_images, i64{1}});
            const auto images_padded_shape = image_padded_shape.push_front(Vec{n_images, i64{1}});
            const auto images_bytes = static_cast<size_t>(images_shape.n_elements()) * sizeof(f32);
            const auto images_padded_bytes = static_cast<size_t>(images_padded_shape.rfft().n_elements()) * sizeof(c32);

            bool keep_spectra_on_device{true};

            // We may not be able to query the device stats, in which case
            // better to hope for the best and process in one chunk.
            if (n_bytes_free == 0)
                return noa::make_tuple(i64{1}, max_n_strips, keep_spectra_on_device);

            i64 chunk_size{};
            i64 n_chunks{1};
            for (; n_chunks < max_n_strips; ++n_chunks) { // until 1 strip per chunk
                auto base = images_bytes;
                if (n_chunks <= 2) {
                    base += images_padded_bytes;
                } else {
                    // Before trying to divide into 3 chunks, try moving the spectra back to the host.
                    if (keep_spectra_on_device)
                        n_chunks = 2;
                    keep_spectra_on_device = false;
                }

                chunk_size = static_cast<i64>(std::ceil(static_cast<f64>(max_n_strips) / static_cast<f64>(n_chunks)));
                const auto n_elements = n_chunks * image_padded_shape.rfft().n_elements();
                const auto n_bytes = static_cast<size_t>(n_elements) * sizeof(c32);
                const auto n_bytes_total = static_cast<size_t>(static_cast<f64>(n_bytes) * 2); // x2 for FFT plans
                if (n_bytes_free > base + n_bytes_total)
                    break;
            }
            return noa::make_tuple(n_chunks, chunk_size, keep_spectra_on_device);
        }

        void allocate_and_prepare_spectra(
            StackLoader& loader,
            const Shape<i64, 2>& image_shape,
            const Shape<i64, 2>& image_padded_shape,
            const MetadataStack& metadata,
            bool keep_spectra_on_device,
            i64 chunk_size
        ) {
            const auto images_shape = image_shape.push_front(Vec{metadata.ssize(), i64{1}});
            const auto images_padded_shape = image_padded_shape.push_front(Vec{metadata.ssize(), i64{1}});
            const auto image_padded_strips_shape = image_padded_shape.push_front(Vec{chunk_size, i64{1}});
            const auto options = ArrayOption{.device = loader.compute_device(), .allocator = Allocator::MANAGED};

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

            const auto io_buffer = Array<f32>(image_shape.push_front<2>(1), options);
            for (auto& slice: metadata) {
                loader.read_slice(io_buffer.view(), slice.index_file);
                if (keep_spectra_on_device) {
                    noa::resize(io_buffer, m_images_padded.subregion(slice.index));
                } else {
                    noa::resize(io_buffer, resize_buffer);
                    nf::r2c(resize_buffer, resize_buffer_rfft, {.norm = nf::Norm::FORWARD});
                    resize_buffer_rfft.to(m_images_padded_rfft.subregion(slice.index));
                }
            }
            loader.clear_cache(); // remove stack saved on the host
            if (keep_spectra_on_device)
                nf::r2c(m_images_padded, m_images_padded_rfft, {.norm = nf::Norm::FORWARD, .cache_plan = false});

            // Allocate remaining buffers.
            m_images_filtered = Array<f32>(images_shape, options);
            noa::tie(m_strips_padded, m_strips_padded_rfft) = nf::empty<f32>(image_padded_strips_shape, options);
            if (not keep_spectra_on_device)
                m_image_padded_rfft = Array<c32>(images_padded_shape.rfft(), options);
        }

        void compute_irffts(i64 n_strips, bool record = false, i64 n_groups = 4) {
            // We compute many FFTs with different batch sizes. In CUDA, this leads to computing many plans, and while
            // the memory consumption can be almost entirely eliminated by sharing the workspace across these plans,
            // the overhead of computing the plans in the first place is too important. In fact, it is faster to group
            // the batch size and compute larger arrays, as long as the number of plans decreases.

            // Group batch size into groups.
            const auto maximum_n_strips = m_strips_padded.shape()[0];
            const auto group_size = noa::next_multiple_of(maximum_n_strips, n_groups) / n_groups;
            const auto index = static_cast<i64>(noa::ceil(static_cast<f64>(n_strips) / static_cast<f64>(group_size)));
            const auto slice = ni::Slice{0, index * group_size};

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

        void prepare_irffts() {
            if (m_z_step_nm <= 0) // no ctf
                return;

            // Create and cache the plans for every FFT about to be run.
            // These FFTs also share the same workspace, so they have to run on the same stream.
            for (i64 i{1}; i < m_strips_padded.shape()[0]; ++i)
                compute_irffts(i, true);
        }

        NOA_NOINLINE auto compute_filtered_stack(i64 z) -> View<f32> {
            if (m_z_step_nm <= 0) // no ctf
                return m_images_filtered.view();

            const auto spacing_nm = m_ctf->pixel_size() * 1e-1;
            const auto image_shape = m_images_filtered.shape().filter(2, 3);
            const auto image_padded_shape = m_images_padded.shape().filter(2, 3);
            const auto left_padding = image_padded_shape.vec / 2 - image_shape.vec / 2;
            const auto fftfreq_norm = 1. / image_padded_shape.vec.as<f64>();

            // Compute the z-offset at the center of the current z-section, relative to the volume center.
            const f64 z_offset_section_center_nm =
                (m_z_step_nm * static_cast<f64>(z) + m_z_step_nm / 2) - m_volume_z_center_nm;

            auto& stream = Stream::current(m_images_padded_rfft.device());
            auto start = noa::Event{};
            auto end = noa::Event{};
            start.record(stream);

            for (auto& slice: *m_metadata) {
                // Compute defocus-strips.
                const auto [z_offset_start_nm, n_strips] = divide_image_in_z_strips(
                    image_shape, slice.angles, spacing_nm, m_z_step_nm
                );
                auto ictf = ns::CTFAnisotropic(*m_ctf);
                ictf.set_defocus(slice.defocus); // sets the astigmatism
                const auto z_offset_of_lowest_strip_um = (z_offset_section_center_nm + z_offset_start_nm) * 1e-3;
                const auto defocus_start = slice.defocus.value - z_offset_of_lowest_strip_um; // underfocus negative
                const auto defocus_step = -m_z_step_nm * 1e-3; // underfocus negative

                // Recompose the filtered tile from the defocus-strips.
                const auto angles = noa::deg2rad(slice.angles);
                const auto z_projection_nm = (
                    ng::scale(Vec<f64, 3>::from_value(spacing_nm)) *
                    ng::rotate_x(angles[2]) *
                    ng::rotate_y(angles[1]) *
                    ng::rotate_z(angles[0])
                )[0].as<f32>();

                // Make the input spectrum available on the device.
                auto image_padded_rfft = m_images_padded_rfft.view().subregion(slice.index);
                if (image_padded_rfft.device() != m_images_filtered.device())
                    image_padded_rfft = image_padded_rfft.to(m_image_padded_rfft.view());

                const auto chunk_size = m_strips_padded.shape()[0];
                for (i64 i{}; i < n_strips; i += chunk_size) {
                    const auto ichunk_size = std::min(chunk_size, n_strips - i);
                    const auto ichunk = ni::Slice{0, ichunk_size};

                    noa::iwise(image_padded_shape.rfft().push_front(ichunk_size), m_images_filtered.device(), FilterPaddedImages{
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

                    const auto image_filtered = m_images_filtered.span().subregion(slice.index).filter(2, 3).as_contiguous();
                    noa::iwise(image_filtered.shape(), m_images_filtered.device(), RecomposeFilteredImage{
                        .strips_padded = m_strips_padded.span().subregion(ichunk).filter(0, 2, 3).as_contiguous(),
                        .image = image_filtered,
                        .image_center = image_shape.vec / 2,
                        .left_padding = left_padding,
                        .z_projection_nm = z_projection_nm,
                        .z_offset_start_nm = static_cast<f32>(z_offset_start_nm),
                        .z_step_nm = static_cast<f32>(m_z_step_nm),
                        .strip_start = i,
                        .strip_end = i + ichunk_size,
                    });
                }
            }

            end.record(stream);
            end.synchronize();
            Logger::trace("took = {}", noa::Event::elapsed(start, end));

            return m_images_filtered.view();
        }

    public:
        const MetadataStack* m_metadata{};
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

    class Reconstructor {
    public:
        Reconstructor(
            const Shape<i64, 2>& image_shape,
            const Shape<i64, 3>& volume_shape,
            const MetadataStack& metadata,
            const Device& device,
            bool oversample,
            bool correct_rotation,
            noa::Interp interp,
            f64 spacing_nm,
            f64 z_step_nm
        ) :
            m_oversample{oversample},
            m_interp{interp}
        {
            // To support large reconstructions with oversampling, the volume is divided into subvolumes.
            // To backproject a subvolume, the input tiles should be large enough to map all voxels of the subvolume
            // from any angle. More specifically, if the tilt-axis is along Y, tiles should be sqrt(2)=1.41 times
            // larger than the largest dimension of the subvolume. If the tilt-axis is not aligned, tiles should be
            // sqrt(3)=1.73 times larger.
            // To prevent aliasing, we oversample both the tiles and subvolume. Oversampling in Fourier space
            // requires smoothed edges, which is done by padding and applying a smooth zero-taper on that padding.
            // Combining these two points results in extracting tiles twice as large and applying a smooth zero-taper
            // to keep edges at zero. These are then oversampled and backprojected. The resulting subvolume is then
            // downsampled, and the central subvolume (without the padding) is extracted and placed back into the
            // volume. In other words, we backproject subvolumes 4 times larger (x2 padding, x2 oversampling) than
            // the final subvolume.
            const auto tile_size = i64{64}; // TODO max z_step with this
            const auto tile_padded_size = tile_size * 2;
            const auto tile_padded_os_size = tile_padded_size * (1 + oversample);
            const auto tile_shape = Shape{tile_size, tile_size};
            const auto tile_padded_shape = Shape{tile_padded_size, tile_padded_size};
            const auto tile_padded_os_shape = Shape{tile_padded_os_size, tile_padded_os_size};

            const auto z_step = static_cast<i64>(z_step_nm / spacing_nm);
            const auto n_sections = volume_shape[0] / z_step;
            check(noa::is_odd(z_step) and noa::is_odd(n_sections));
            m_subvolume_shape = Shape{z_step, tile_size, tile_size};
            m_subvolume_padded_shape = Shape{z_step * 2, tile_padded_size, tile_padded_size};
            m_subvolume_padded_os_shape = Shape{z_step * 2 * (1 + oversample), tile_padded_os_size, tile_padded_os_size};
            m_left_padding = m_subvolume_padded_shape.vec / 2 - m_subvolume_shape.vec / 2;
            m_grid_shape = Shape{
                n_sections,
                noa::divide_up(image_shape[0], tile_shape[0]),
                noa::divide_up(image_shape[1], tile_shape[1]),
                metadata.ssize()
            };
            const auto& [nz, ny, nx, nt] = m_grid_shape;

            // Compute device.
            // On the GPU, use the tmp row buffer.
            // On the CPU, distribute subvolumes to threads. Since each thread needs its own buffers,
            // limit the number of threads to keep the memory usage reasonable. 10 threads need about 0.5GB.
            const auto is_gpu = device.is_gpu();
            const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
            m_n_threads = is_gpu ? i64{1} : i64{8};//std::max(Stream::current(device).thread_limit(), i64{10});

            // Forward matrices relating 3d positions in the tomogram to 2d positions in the images.
            // Then, compute the padded tile origins and projection matrices for the oversampled tiles.
            const auto image_center = (image_shape.vec / 2).as<f64>();
            const auto volume_center = (volume_shape.vec / 2).as<f64>();
            const auto forward_projection_matrices = Array<Mat<f64, 2, 4>>(nt);
            const auto forward_projection_matrices_1d = forward_projection_matrices.span_1d();
            for (auto&& [slice, forward_projection_matrix]: noa::zip(metadata, forward_projection_matrices_1d)) {
                const auto angles = noa::deg2rad(slice.angles);
                const auto final_rotation = correct_rotation ? 0. : angles[0];
                forward_projection_matrix = (
                    ng::translate((image_center + slice.shifts).push_front(0)) *
                    ng::rotate_z<true>(angles[0]) *
                    ng::rotate_y<true>(angles[1]) *
                    ng::rotate_x<true>(angles[2]) *
                    ng::rotate_z<true>(-final_rotation) *
                    ng::translate(-volume_center)
                ).filter_rows(1, 2);
            }

            const auto subvolume_center = (m_subvolume_shape.vec / 2).as<f64>();
            const auto subvolume_padded_center = (m_subvolume_padded_shape.vec / 2).as<f64>();
            const auto tile_padded_center = (tile_padded_shape / 2).vec.as<f64>();
            m_tile_padded_origins = Array<Vec<i32, 2>>(m_grid_shape);
            m_tile_padded_os_matrices = Array<Mat<f32, 2, 4>>(m_grid_shape);
            for (i64 z{}; z < nz; ++z) {
                for (i64 y{}; y < ny; ++y) {
                    for (i64 x{}; x < nx; ++x) {
                        // Compute the center of the subvolume.
                        const auto subvolume_origin = Vec{z, y, x} * m_subvolume_shape.vec;
                        const auto subvolume_center_coordinates = subvolume_origin.as<f64>() + subvolume_center;

                        for (i64 t{}; t < nt; ++t) {
                            // Project subvolume center back to image-space.
                            // Extract the padded tile origin and residual shifts for the extraction.
                            const auto tile_padded_center_coordinate = forward_projection_matrices_1d[t] * subvolume_center_coordinates.push_back(1);
                            const auto tile_padded_origin_coordinate = tile_padded_center_coordinate - tile_padded_center;
                            const auto tile_padded_origin_truncated = noa::floor(tile_padded_origin_coordinate);
                            const auto tile_padded_origin = tile_padded_origin_truncated.as<i32>();
                            const auto tile_residual_shift = tile_padded_origin_coordinate - tile_padded_origin_truncated;
                            m_tile_padded_origins(z, y, x, t) = tile_padded_origin;

                            // Compute the backward projection matrix (this is done on the oversampled padded tiles).
                            // Note that the padded tiles/subvolumes are even-sized, so the center is preserved during
                            // oversampling.
                            const auto angles = noa::deg2rad(metadata[t].angles);
                            const auto final_rotation = correct_rotation ? 0. : angles[0];
                            const auto scale = static_cast<f64>(1 + oversample);
                            m_tile_padded_os_matrices(z, y, x, t) = ( // volume->image
                                ng::translate((tile_padded_center + tile_residual_shift).push_front(0) * scale) *
                                ng::rotate_z<true>(angles[0]) *
                                ng::rotate_y<true>(angles[1]) *
                                ng::rotate_x<true>(angles[2]) *
                                ng::rotate_z<true>(-final_rotation) *
                                ng::translate(-subvolume_padded_center * scale)
                            ).filter_rows(1, 2).as<f32>(); // (y, x)
                        }
                    }
                }
            }
            if (is_gpu) {
                const auto options_async = ArrayOption{.device = device, .allocator = Allocator::ASYNC};
                m_tile_padded_origins = std::move(m_tile_padded_origins).to(options_async);
                m_tile_padded_os_matrices = std::move(m_tile_padded_os_matrices).to(options_async);
            }

            Logger::trace(
                "Reconstruction:\n"
                "  interp={}\n"
                "  oversampling={}\n"
                "  subvolume_shape={} (padded={}{})\n"
                "  grid_shape={} (n_subvolumes={})\n"
                "  mode={}",
                interp, oversample,
                m_subvolume_shape, m_subvolume_padded_shape,
                oversample ? fmt::format(", padded_os={}", m_subvolume_padded_os_shape) : "",
                m_grid_shape.pop_back(), m_grid_shape.pop_back().n_elements(),
                device.is_cpu() ? fmt::format("cpu (n_threads={})", m_n_threads) : "gpu"
            );

            // Allocate buffers.
            // Note that for the CPU mode, each thread needs its own buffer.
            // To retrieve the buffer (as a real and complex view), use the *_pair(tid) functions.
            m_tiles_padded_rfft = Array<c32>(tile_padded_shape.rfft().push_front(Vec{m_n_threads, nt}), options);
            m_subvolume_padded_rfft = Array<c32>(m_subvolume_padded_shape.rfft().push_front(m_n_threads), options);
            if (oversample) {
                m_tiles_padded_os_rfft = Array<c32>(tile_padded_os_shape.rfft().push_front(Vec{m_n_threads, nt}), options);
                m_subvolume_padded_os_rfft = Array<c32>(m_subvolume_padded_os_shape.rfft().push_front(m_n_threads), options);
            }
            if (is_gpu)
                m_subvolume_row = Array<f32>(m_subvolume_shape.set<2>(volume_shape[2]).push_front(1), options);
        }

        [[nodiscard]] auto tiles_padded_pair(i64 tid) const {
            const auto& [td, th, tw] = m_subvolume_padded_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = m_tiles_padded_rfft.view().subregion(tid).permute({1, 0, 2, 3});
            pair.first = nf::alias_to_real(pair.second, Shape{m_grid_shape[3], i64{1}, th, tw});
            return pair;
        }

        [[nodiscard]] auto tiles_padded_os_pair(i64 tid) const {
            if (not m_oversample)
                return tiles_padded_pair(tid);
            const auto& [td, th, tw] = m_subvolume_padded_os_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = m_tiles_padded_os_rfft.view().subregion(tid).permute({1, 0, 2, 3});
            pair.first = nf::alias_to_real(pair.second, Shape{m_grid_shape[3], i64{1}, th, tw});
            return pair;
        }

        [[nodiscard]] auto subvolume_padded_pair(i64 tid) const {
            const auto& [td, th, tw] = m_subvolume_padded_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = m_subvolume_padded_rfft.view().subregion(tid);
            pair.first = nf::alias_to_real(pair.second, Shape{i64{1}, td, th, tw});
            return pair;
        }

        [[nodiscard]] auto subvolume_padded_os_pair(i64 tid) const {
            if (not m_oversample)
                return subvolume_padded_pair(tid);
            const auto& [td, th, tw] = m_subvolume_padded_os_shape;
            auto pair = Pair<View<f32>, View<c32>>{};
            pair.second = m_subvolume_padded_os_rfft.view().subregion(tid);
            pair.first = nf::alias_to_real(pair.second, Shape{i64{1}, td, th, tw});
            return pair;
        }

        void prepare_rffts() const {
            if (m_oversample) {
                auto [tiles_padded, tiles_padded_rfft] = tiles_padded_pair(0);
                auto [tiles_padded_os, tiles_padded_os_rfft] = tiles_padded_os_pair(0);
                auto [subvolume_padded, subvolume_padded_rfft] = subvolume_padded_pair(0);
                auto [subvolume_padded_os, subvolume_padded_os_rfft] = subvolume_padded_os_pair(0);

                nf::r2c(tiles_padded, tiles_padded_rfft, {.record_and_share_workspace = true});
                nf::c2r(tiles_padded_os_rfft, tiles_padded_os, {.record_and_share_workspace = true});
                nf::r2c(subvolume_padded_os, subvolume_padded_os_rfft, {.record_and_share_workspace = true});
                nf::c2r(subvolume_padded_rfft, subvolume_padded, {.record_and_share_workspace = true});
            }
        }

        NOA_NOINLINE auto reconstruct_subvolume(
            const View<const f32>& input_stack,
            i64 z, i64 y, i64 x, i64 tid = 0
        ) -> View<f32> {
            const auto [tiles_padded, tiles_padded_rfft] = tiles_padded_pair(tid);
            const auto [tiles_padded_os, tiles_padded_os_rfft] = tiles_padded_os_pair(tid);
            const auto [subvolume_padded, subvolume_padded_rfft] = subvolume_padded_pair(tid);
            const auto [subvolume_padded_os, subvolume_padded_os_rfft] = subvolume_padded_os_pair(tid);

            // Extract and taper the padded tiles.
            // The backprojected region of the tile is, at most, sqrt(3)=1.73, so each edges has
            // an extra 6.5% of padding that isn't backprojected. We use the last 5% for the taper,
            // which should be enough to remove oversampling artifacts.
            const auto tile_padded_center = (tiles_padded.shape().vec.filter(2, 3) / 2).as<f32>();
            const auto taper_radius = noa::min(tile_padded_center.as<f32>()) * 0.95f;
            const auto taper_smoothness = noa::min(tile_padded_center.as<f32>()) * 0.05f;
            noa::iwise(tiles_padded.shape().filter(0, 2, 3), tiles_padded.device(), ExtractPaddedTiles{
                .images = input_stack.span().filter(0, 2, 3).as_contiguous(),
                .tiles_padded = tiles_padded.span().filter(0, 2, 3).as_contiguous(),
                .tile_padded_origins = m_tile_padded_origins.span().subregion(z, y, x).as_1d(),
                .tile_padded_center = tile_padded_center.as<i64>(),
                .taper_radius = taper_radius,
                .taper_smoothness = taper_smoothness,
            });

            // Oversample, if necessary.
            if (m_oversample) {
                nf::r2c(tiles_padded, tiles_padded_rfft);
                nf::resize<"h2h">(
                    tiles_padded_rfft, tiles_padded.shape(),
                    tiles_padded_os_rfft, tiles_padded_os.shape()
                );
                nf::c2r(tiles_padded_os_rfft, tiles_padded_os);
            }

            // Prefilter, if necessary.
            // TODO We could also prefilter before the oversampling.
            if (m_interp == noa::Interp::CUBIC_BSPLINE)
                noa::cubic_bspline_prefilter(tiles_padded_os, tiles_padded_os);

            // Backward project.
            const auto input = tiles_padded_os.span().filter(0, 2, 3).as_contiguous();
            const auto output = subvolume_padded_os.span().filter(1, 2, 3).as_contiguous();
            const auto matrices = m_tile_padded_os_matrices.span().subregion(z, y, x).as_1d();
            if (m_interp == noa::Interp::CUBIC_BSPLINE) {
                using operator_t = BackwardProjection<noa::Interp::CUBIC_BSPLINE>;
                noa::iwise(
                    output.shape(), subvolume_padded_os.device(),
                    operator_t{
                        .tiles_padded = operator_t::interpolator_t(input, input.shape().pop_front()),
                        .projection_matrices = matrices,
                        .subvolume = output,
                    });
            } else if (m_interp == noa::Interp::LINEAR) {
                using operator_t = BackwardProjection<noa::Interp::LINEAR>;
                noa::iwise(
                    output.shape(), subvolume_padded_os.device(),
                    operator_t{
                        .tiles_padded = operator_t::interpolator_t(input, input.shape().pop_front()),
                        .projection_matrices = matrices,
                        .subvolume = output,
                    });
            }

            // Downsample, if necessary.
            if (m_oversample) {
                nf::r2c(subvolume_padded_os, subvolume_padded_os_rfft);
                nf::resize<"h2h">(
                    subvolume_padded_os_rfft, subvolume_padded_os.shape(),
                    subvolume_padded_rfft, subvolume_padded.shape()
                );
                nf::c2r(subvolume_padded_rfft, subvolume_padded);
            }

            // Return a view of the subvolume (excluding the padding).
            auto subvolume = subvolume_padded.view().subregion(0,
                ni::Slice{m_left_padding[0], m_left_padding[0] + m_subvolume_shape[0]},
                ni::Slice{m_left_padding[1], m_left_padding[1] + m_subvolume_shape[1]},
                ni::Slice{m_left_padding[2], m_left_padding[2] + m_subvolume_shape[2]}
            );
            return subvolume;
        }

        void reconstruct_z_section(const View<const f32>& stack, const View<f32>& z_section, i64 z) {
            const auto [sy, sx] = m_subvolume_shape.filter(1, 2);
            const auto [ny, nx] = m_grid_shape.filter(1, 2);

            // TODO If there's no oversampling, we can reconstruct the volume without using tiles.
            if (stack.device().is_cpu()) {
                // The subvolumes are relatively small, so to better distribute resources,
                // distribute subvolumes to threads directly, and set each subvolume to one thread.
                auto& stream = Stream::current({});
                auto n_threads = stream.thread_limit();
                stream.set_thread_limit(1);
                parallel_for(m_n_threads, Shape{ny, nx}, [&](i64 tid, i64 y, i64 x) {
                    const auto subvolume = reconstruct_subvolume(stack, z, y, x, tid);
                    auto dst = z_section.subregion(
                        ni::Ellipsis{},
                        ni::Slice{y * sy, y * sy + sy},
                        ni::Slice{x * sx, x * sx + sx}
                    );
                    auto src = subvolume.subregion(ni::Ellipsis{}, ni::Slice{0, dst.shape()[2]}, ni::Slice{0, dst.shape()[3]});
                    src.to(dst);
                });
                stream.set_thread_limit(n_threads);
            } else {
                // Store the subvolumes into the row buffer.
                // Once the row of subvolumes is computed, transfer to the host.
                for (i64 y{}; y < ny; ++y) {
                    for (i64 x{}; x < nx; ++x) {
                        const auto subvolume = reconstruct_subvolume(stack, z, y, x);
                        const auto dst = m_subvolume_row.view().subregion(ni::Ellipsis{}, ni::Slice{x * sx, x * sx + sx});

                        auto src = subvolume.subregion(ni::Ellipsis{}, ni::Slice{0, dst.shape()[2]}, ni::Slice{0, dst.shape()[3]});
                        // Logger::trace("y={}, x={}, src={}, dst={}", y, x, src.shape(), dst.shape());

                        src.to(dst);
                    }
                    // panic();
                    auto dst = z_section.subregion(ni::Ellipsis{}, ni::Slice{y * sy, y * sy + sy}, ni::Full{});
                    m_subvolume_row.view().subregion(ni::Ellipsis{}, ni::Slice{0, dst.shape()[2]}, ni::Full{}).to(dst);
                }
            }
        }

        [[nodiscard]] auto z_range() const { return Pair{m_subvolume_shape[0], m_grid_shape[0]}; }

    private:
        bool m_oversample;
        noa::Interp m_interp;
        i64 m_n_threads{};

        Vec<i64, 3> m_left_padding{};
        Shape<i64, 3> m_subvolume_shape{};
        Shape<i64, 3> m_subvolume_padded_shape{};
        Shape<i64, 3> m_subvolume_padded_os_shape{};
        Shape<i64, 4> m_grid_shape{};

        Array<Vec<i32, 2>> m_tile_padded_origins;
        Array<Mat<f32, 2, 4>> m_tile_padded_os_matrices;
        Array<f32> m_subvolume_row;

        Array<c32> m_tiles_padded_rfft;
        Array<c32> m_tiles_padded_os_rfft;
        Array<c32> m_subvolume_padded_rfft;
        Array<c32> m_subvolume_padded_os_rfft;
    };

    void reconstruct_tomogram(
        StackLoader& stack,
        const MetadataStack& metadata,
        const Path& filename,
        const PostProcessingTomogramParameters& parameters
    ) {
        auto timer = Logger::info_scope_time("Reconstructing tomogram");

        const f64 spacing = mean(stack.stack_spacing());
        const f64 spacing_nm = spacing * 1e-1;
        auto ctf = ns::CTFIsotropic<f64>({
            .pixel_size = spacing,
            .defocus = 0.,
            .voltage = parameters.voltage,
            .amplitude = parameters.amplitude,
            .cs = parameters.cs,
            .phase_shift = 0,
            .bfactor = 0,
            .scale = 1.,
        });

        // For simplicity, make the defocus resolution (in pixels) an odd integer multiple of the pixel size.
        auto z_step = static_cast<i64>(std::floor(parameters.defocus_step_nm / spacing_nm));
        z_step += noa::is_even(z_step);
        const auto z_step_nm = static_cast<f64>(z_step) * spacing_nm;
        Logger::trace("defocus_resolution={:.3f}nm ({}pix)", z_step_nm, z_step);

        // Get volume thickness and number of z-sections (of size z_step).
        // To guarantee that the volume center is at the center of a z-section,
        // make the volume thickness an odd multiple of z_step.
        const f64 sample_thickness = parameters.sample_thickness_nm / spacing_nm;
        const f64 z_padding = parameters.sample_thickness_nm * parameters.z_padding_percent / spacing_nm;
        auto volume_thickness = static_cast<i64>(std::round(sample_thickness + z_padding));
        volume_thickness = noa::next_multiple_of(volume_thickness, z_step);
        if (noa::is_even(volume_thickness / z_step))
            volume_thickness += z_step;
        const auto volume_thickness_nm = static_cast<f64>(volume_thickness) * spacing_nm;
        const auto image_shape = stack.slice_shape();
        const auto volume_shape = image_shape.push_front(volume_thickness);

        Logger::trace(
            "Volume:\n"
            "  spacing={:.3f}A (resolution={:.3f}A)\n"
            "  thickness={:.3f}nm (specimen={:.3f}nm, z_padding={:.1f}pix)\n"
            "  shape={}",
            spacing_nm * 10, spacing_nm * 20,
            static_cast<f64>(volume_thickness) * spacing_nm, parameters.sample_thickness_nm, z_padding,
            volume_shape
        );

        const auto device = stack.compute_device();
        const auto bytes_start = Allocator::bytes_currently_allocated(device);

        // Filtering.
        auto filterer = not parameters.correct_ctf ? Filterer(stack, metadata) : Filterer(
            stack, metadata, ctf, volume_thickness_nm, z_step_nm,
            parameters.phase_flip_strength
        );
        auto reconstructor = Reconstructor(
            image_shape, volume_shape, metadata, device,
            parameters.oversample, parameters.correct_rotation, parameters.interp,
            spacing_nm, z_step_nm
        );
        stack = {}; // clear cache

        // Prepare for FFTs with various batch sizes.
        // In CPU, this only precomputes the plans and isn't necessary.
        // In CUDA, while this is also optional, it allows sharing the workspace
        // across all transforms, possibly saving a lot of memory.
        nf::clear_cache(device);
        nf::set_cache_limit(10, device);
        filterer.prepare_irffts();
        reconstructor.prepare_rffts();
        // TODO Add nf::allocate_workspace();

        const auto bytes_send = Allocator::bytes_currently_allocated(device);
        Logger::trace("allocator - bytes_currently_allocated={:.2f}GB", static_cast<f64>(bytes_send - bytes_start) * 1e-9);

        auto dir = filename.parent_path();

        // Reconstruct the (possibly CTF-corrected) tomogram.
        auto volume = Array<f32>(volume_shape.push_front(1));
        const auto [sz, nz] = reconstructor.z_range();
        for (i64 z{}; z < nz; ++z) {
            auto t1 = Logger::trace_scope_time("z={:02}/{:02}", z, nz);
            const auto filtered_stack = filterer.compute_filtered_stack(z);
            const auto volume_z_section = volume.view().subregion(0, ni::Slice{z * sz, z * sz + sz});
            reconstructor.reconstruct_z_section(filtered_stack, volume_z_section, z);
        }

        nf::clear_cache(device);
        noa::write(volume, Vec<f64, 3>::from_value(spacing_nm * 10), filename, {.dtype = parameters.dtype});
        Logger::trace("{} saved", filename);
    }
}

namespace qn {
    void post_processing(
        const Path& input_stack,
        const MetadataStack& metadata,
        const PostProcessingParameters& parameters,
        const PostProcessingStackParameters& stack_parameters,
        const PostProcessingTomogramParameters& tomogram_parameters
    ) {
        // Set up the input stack.
        auto loader = StackLoader(input_stack, {
            .compute_device = parameters.compute_device,
            .allocator = Allocator::DEFAULT_ASYNC,
            .precise_cutoff = true,
            .rescale_target_resolution = parameters.target_resolution,
            .rescale_min_size = parameters.min_size,
            .bandpass{
                .highpass_cutoff = 0.01,
                .highpass_width = 0.01,
                .lowpass_cutoff = 0.49,
                .lowpass_width = 0.01,
            },
            .bandpass_mirror_padding_factor = 0.5,
            .normalize_and_standardize = true,
            .smooth_edge_percent = 0.02,
            .zero_pad_to_fast_fft_shape = false,
            .zero_pad_to_square_shape = false,
        });

        // Rescale to new spacing and make sure images are sorted by their tilt angles.
        auto postprocessing_metadata = metadata;
        postprocessing_metadata.rescale_shifts(loader.file_spacing(), loader.stack_spacing());
        postprocessing_metadata.sort("tilt");
        postprocessing_metadata.reset_indices();

        const auto basename = input_stack.stem().string();

        if (stack_parameters.save_aligned_stack) {
            const auto filename = parameters.output_directory / fmt::format("{}_aligned.mrc", basename);
            save_stack(loader, filename, postprocessing_metadata, {
                .correct_rotation = stack_parameters.correct_rotation,
                .cache_loader = tomogram_parameters.save_tomogram,
                .interp = stack_parameters.interp,
                .border = noa::Border::ZERO,
                .dtype = stack_parameters.dtype,
            });
        }

        if (tomogram_parameters.save_tomogram) {
            const auto filename = parameters.output_directory / fmt::format("{}_tomogram.mrc", basename);
            reconstruct_tomogram(loader, postprocessing_metadata, filename, tomogram_parameters);
        }
    }
}
