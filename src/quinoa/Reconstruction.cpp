#include <noa/FFT.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>
#include <noa/Utils.hpp>
#include <noa/IO.hpp>

#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/Reconstruction.hpp"
#include "quinoa/CTF.hpp"

namespace {
    using namespace qn;

    auto divide_image_in_z_strips(
        const Shape<i64, 2>& image_shape,
        const Vec<f64, 3>& image_angles, // in degrees
        f64 spacing_nm,
        f64 z_step_nm
    ) {
        // Projection matrix to get the Z-height of image coordinates in volume space.
        // Note that the image shifts should not be applied; the CTF fitting is done on
        // tiles extracted from the raw images/frames.
        const auto angles = noa::deg2rad(image_angles);
        const auto z_projection = (
            ng::rotate_x(angles[2]) *
            ng::rotate_y(angles[1]) *
            ng::rotate_z(angles[0])
        ).filter_rows(0);

        // Compute the z-offset range within the image  by transforming the 4 edges on the image to volume space.
        const auto image_center = (image_shape.vec / 2).as<f64>();
        const auto edge = image_shape - 1;
        const auto image_edges = Vec{
            Vec<f64, 2>::from_values(0,       0),
            Vec<f64, 2>::from_values(0,       edge[1]),
            Vec<f64, 2>::from_values(edge[0], 0),
            Vec<f64, 2>::from_values(edge[0], edge[1]),
        };

        auto minmax = Vec<f64, 2>{}; // in pixels
        for (const auto& image_edge : image_edges) {
            const auto z_distance = (z_projection * (image_edge - image_center).push_front(0))[0];
            minmax[0] = std::min(minmax[0], z_distance);
            minmax[1] = std::max(minmax[1], z_distance);
        }

        // The z-offset refers to the offset relative to the image center, where the z-offset is 0. The z-range of the
        // image is divided into strips. Explicitly make the number of strips odd to always sample the 0 offset at
        // the image center. For instance, for 11 strips, we get strips=[-5, 5], with the image center at strip=0.
        // The z-offset of the ith strip is i * z_step_nm.
        auto strips = noa::round(minmax * spacing_nm / z_step_nm).as<i64>();
        strips = noa::max(noa::abs(strips));
        strips[0] *= -1;

        // When fitting the CTF, the image center points at the average defocus, which matches the strip range.
        // Note, however, that the strips are in volume space, where the z-offset of the first strip is negative (below
        // the rotation axis) and the z-offset of the last strip is positive (above the rotation axis). Therefore, to
        // compute the defocus of a strip, we need to subtract the z-offset of the strip to the average defocus. This
        // is because the defocus is underfocus positive.
        return strips;
    }

    auto analyse_defocus_range(
        const Shape<i64, 2>& image_shape,
        const MetadataStack& metadata,
        const ns::CTFIsotropic<f64>& ctf,
        f64 defocus_step_nm,
        f64 specimen_thickness_nm
    ) {
        const auto spacing_nm = ctf.pixel_size() * 1e-1;

        i64 min_n_strips{};
        i64 max_n_strips{};
        f64 max_defocus_range{};
        f64 max_defocus{};
        for (auto& slice: metadata) {
            const auto strips = divide_image_in_z_strips(image_shape, slice.angles, spacing_nm, defocus_step_nm);
            const auto n_strips = strips[1] - strips[0] + 1;
            check(noa::is_odd(n_strips), "strips={}, n_strips={}", strips, n_strips);
            const auto defocus_offsets = strips.as<f64>() * defocus_step_nm;
            const auto defocus_range = defocus_offsets[1] - defocus_offsets[0];
            const auto slice_defocus_nm = (slice.defocus.value + std::abs(slice.defocus.astigmatism)) * 1e3;
            const auto highest_defocus_nm = slice_defocus_nm + defocus_offsets[1];

            min_n_strips = std::min(min_n_strips, n_strips);
            max_n_strips = std::max(max_n_strips, n_strips);
            max_defocus_range = std::max(max_defocus_range, defocus_range);
            max_defocus = std::max(max_defocus, highest_defocus_nm);
        }
        max_defocus += specimen_thickness_nm / 2;

        Logger::trace(
           "Analysing z gradient within images:\n"
           "  n_strips_within_an_image=[min={}, max={}]\n"
           "  max_z_range_within_an_image={:.2f}nm",
           min_n_strips, max_n_strips, max_defocus_range
       );

        // When multiplying with the CTF, the signal is delocalized (relocalized, actually). Since we filter the
        // entire image and take strips from it, we need to make sure the strip edges are zero-padded so that the
        // signal from one edge is not delocalized to the other edge.

        // The following is adapted from Russo & Henderson, 2018:
        // https://www.desmos.com/calculator/w1dlw58f8t
        // 8A resolution, 4um defocus -> delocalization is ~50pix
        // 4A resolution, 4um defocus -> delocalization is ~200pix
        const f64 wavelength = ns::relativistic_electron_wavelength(ctf.voltage() * 1000) * 1e9;
        const f64 spacing = ctf.pixel_size() * 1e-1;
        const f64 resolution = spacing * 2;
        const f64 delocalization_nm = max_defocus * 2 * wavelength / resolution;
        const f64 delocalization_pix = std::round(delocalization_nm / spacing);

        const auto minimum_padding = static_cast<i64>(delocalization_pix);
        const auto padded_image_shape = nf::next_fast_shape(image_shape + minimum_padding);
        Logger::trace(
            "Zero-padding for signal delocalization at image edges:\n"
            "  max_defocus={:.1f}nm\n"
            "  max_delocalization={:.3f}nm ({}pix)\n"
            "  zero_padding={}\n"
            "  original_image_shape={}\n"
            "  padded_image_shape={}",
            max_defocus, delocalization_nm, delocalization_pix,
            minimum_padding, image_shape, padded_image_shape
        );

        return noa::make_tuple(max_n_strips, padded_image_shape);
    }

    struct WienerFilter {
        ns::CTFAnisotropic<f32> ctf;
        f32 defocus_start_um;
        f32 defocus_step_um;

        constexpr auto operator()(const Vec<f32, 2>& fftfreq, i64 batch) -> f32 {
            return 1;

            // Get the CTF of the current strip.
            const auto& [_, astigmatism, angle] = ctf.defocus();
            const auto defocus = defocus_start_um + defocus_step_um * static_cast<f32>(batch);
            ctf.set_defocus({.value = defocus, .astigmatism = astigmatism, .angle = angle});
            return ctf.value_at(fftfreq);
        }
    };

    struct RecomposeImage {
        SpanContiguous<const f32, 3> filtered_images;
        SpanContiguous<f32, 2> recomposed_image;
        Mat<f32, 1, 3> z_projection_nm{};
        f32 z_offset_start_nm;
        f32 z_step_nm;

        NOA_HD void operator()(i64 i, i64 j) const {
            // Get the z-position at this index of the image.
            const auto image_center = (recomposed_image.shape().vec / 2).as<f32>();
            const auto image_coordinates = Vec<f32, 2>::from_values(i, j) - image_center;
            const auto volume_z_coordinate_nm = (z_projection_nm * image_coordinates.push_front(0))[0];

            // Get the closest z-strip.
            const auto z_strip = (volume_z_coordinate_nm - z_offset_start_nm) / z_step_nm;
            auto z_strip_index = static_cast<i64>(noa::round(z_strip));

            // Regions with a z-offset lower or higher than the lowest or highest strip are just clamped.
            // Since we compute the number of strips based on the z-range in the image, this should always
            // be in range, but do this is just in case we change the logic to get the number of strips.
            z_strip_index = noa::clamp(z_strip_index, 0, filtered_images.shape()[0] - 1);

            // Save the corresponding pixel at the closest strip.
            recomposed_image(i, j) = filtered_images(z_strip_index, i, j);
        }
    };

    void compute_ctf_corrected_stack(
       const View<const c32>& padded_images_rfft,
       const View<f32>& padded_images_strip_buffer,
       const View<c32>& padded_images_strip_buffer_rfft,
       const View<f32>& output_images,
       const MetadataStack& metadata,
       const ns::CTFIsotropic<f64>& ctf,
       f64 z_step_nm,
       f64 z_offset_nm
   ) {
        auto t = Logger::trace_scope_time("Computing corrected stack");

        const auto spacing_nm = ctf.pixel_size() * 1e-1;
        const auto image_shape = output_images.shape().filter(2, 3);
        const auto remove_padding_right = ni::make_subregion<4>(
            ni::Ellipsis{}, ni::Slice{0, image_shape[0]}, ni::Slice{0, image_shape[1]}
        );

        auto& stream = Stream::current(padded_images_rfft.device());
        // noa::Event start0, stop0;
        // start0.record(stream);

        for (auto& slice: metadata) {
            // Recompute how many strips are needed for this image.
            const auto strips = divide_image_in_z_strips(image_shape, slice.angles, spacing_nm, z_step_nm);
            const auto n_strips = strips[1] - strips[0] + 1;

            // Select the necessary subregions.
            auto padded_images_strip_rfft = padded_images_strip_buffer_rfft.subregion(ni::Slice{0, n_strips});
            auto padded_images_strip = padded_images_strip_buffer.subregion(ni::Slice{0, n_strips});
            auto output_image = output_images.subregion(slice.index);
            auto padded_image_rfft = ni::broadcast(padded_images_rfft.subregion(slice.index), padded_images_strip_rfft.shape());

            // The z offset (relative to the image center) at the lowest strip.
            // Note that this offset should be subtracted to the defocus; positive z-offset means closer to focus.
            const auto z_offset_start_nm = z_step_nm * static_cast<f64>(strips[0]);
            const auto z_start_nm = z_offset_nm + z_offset_start_nm;

            // Compute the filtered padded images for each step.
            auto ictf = ns::CTFAnisotropic(ctf);
            ictf.set_defocus(slice.defocus); // set the astigmatism

            noa::Event start1, stop1;
            // start1.record(stream);
            ns::filter_spectrum_2d<"h2h">(
                padded_image_rfft, padded_images_strip_rfft, padded_images_strip.shape(),
                WienerFilter{
                    .ctf = ictf.as<f32>(),
                    .defocus_start_um = static_cast<f32>(slice.defocus.value - z_start_nm * 1e-3),
                    .defocus_step_um = static_cast<f32>(z_step_nm * 1e-3),
                });
            // stop1.record(stream);
            // stop1.synchronize();
            // Logger::trace("filter took {}", noa::Event::elapsed(start1, stop1));

            start1.record(stream);
            // nf::c2r(padded_images_strip_rfft, padded_images_strip);
            // TODO Try grouping every by halft and see if it's better than running everything all the time
            // TODO Why does syncing make it faster?
            nf::c2r(padded_images_strip_buffer_rfft, padded_images_strip_buffer, {.norm = noa::fft::Norm::FORWARD});
            stop1.record(stream);
            stop1.synchronize();
            // Logger::trace("fft took {}, n={}", noa::Event::elapsed(start1, stop1), n_strips);

            // Recompose the filtered strips into a single image.
            const auto angles = noa::deg2rad(slice.angles);
            const auto z_projection_nm = (
                ng::scale(Vec<f64, 3>::from_value(spacing_nm)) *
                ng::rotate_x(angles[2]) *
                ng::rotate_y(angles[1]) *
                ng::rotate_z(angles[0])
            ).filter_rows(0).as<f32>();

            // start1.record(stream);
            auto images_filtered_strip = padded_images_strip.view().subregion(remove_padding_right);
            noa::iwise(image_shape, images_filtered_strip.device(), RecomposeImage{
                .filtered_images = images_filtered_strip.span().filter(0, 2, 3).as_contiguous(),
                .recomposed_image = output_image.span().filter(2, 3).as_contiguous(),
                .z_projection_nm = z_projection_nm,
                .z_offset_start_nm = static_cast<f32>(z_offset_start_nm),
                .z_step_nm = static_cast<f32>(z_step_nm),
            });
            // stop1.record(stream);
            // stop1.synchronize();
            // Logger::trace("recompose took {}", noa::Event::elapsed(start1, stop1));
        }

        // stop0.record(stream);
        // stop0.synchronize();
        // Logger::trace("all took {}", noa::Event::elapsed(start0, stop0));
    }

    template<noa::Interp INTERP>
    struct BackwardProjection {
    public:
        static constexpr auto BORDER = noa::Border::ZERO;
        using input_span_t = SpanContiguous<const f32, 3>;
        using interpolator_t = noa::Interpolator<2, INTERP, BORDER, input_span_t>;
        using matrices_span_t = SpanContiguous<const Mat<f32, 2, 4>>;

    public:
        interpolator_t images{}; // (n,h,w)
        matrices_span_t projection_matrices{}; // (n)
        SpanContiguous<f32, 3> volume{}; // (d,h,w)
        f32 z_offset{};

    public:
        constexpr void operator()(const Vec<i64, 3>& indices) const {
            auto volume_coordinates = indices.as<f32>().push_back(1);
            volume_coordinates[0] += z_offset;

            // TODO Add volume deformations with the addition of a SplineGrid updating volume_coordinates.

            f32 value{};
            for (i64 i{}; i < projection_matrices.ssize(); ++i) {
                const auto image_coordinates = projection_matrices[i] * volume_coordinates;
                value += images.interpolate_at(image_coordinates, i);
            }
            volume(indices) = value;
        }
    };

    // Simple backward projection of images into a volume, without any filtering.
    void backward_project(
        const View<f32>& input_images, // (n,1,h,w)
        const View<const Mat<f32, 2, 4>>& projection_matrices, // (1,1,1,n)
        const View<f32>& output_volume, // (1,d,h,w)
        i64 z_offset,
        noa::Interp interpolation_method
    ) {
        auto t = Logger::trace_scope_time("Backprojecting");

        // auto& stream = Stream::current(input_images.device());
        // noa::Event start0, stop0;
        // start0.record(stream);

        if (interpolation_method == noa::Interp::CUBIC_BSPLINE)
            noa::cubic_bspline_prefilter(input_images, input_images);

        const auto input_span = input_images.span().filter(0, 2, 3).as_contiguous();
        const auto output_span = output_volume.span().filter(1, 2, 3).as_contiguous();

        if (interpolation_method == noa::Interp::CUBIC_BSPLINE) {
            using operator_t = BackwardProjection<noa::Interp::CUBIC_BSPLINE>;
            noa::iwise(
                output_span.shape(), output_volume.device(),
                operator_t{
                    .images = operator_t::interpolator_t(input_span, input_span.shape().pop_front()),
                    .projection_matrices = projection_matrices.span_1d(),
                    .volume = output_span,
                    .z_offset = static_cast<f32>(z_offset),
                });
        } else if (interpolation_method == noa::Interp::LINEAR) {
            using operator_t = BackwardProjection<noa::Interp::LINEAR>;
            noa::iwise(
                output_span.shape(), output_volume.device(),
                operator_t{
                    .images = operator_t::interpolator_t(input_span, input_span.shape().pop_front()),
                    .projection_matrices = projection_matrices.span_1d(),
                    .volume = output_span,
                    .z_offset = static_cast<f32>(z_offset),
                });
        }

        // stop0.record(stream);
        // stop0.synchronize();
        // Logger::trace("backproject took {}", noa::Event::elapsed(start0, stop0));
    }
}

namespace qn {
    auto tomogram_reconstruction(
        const View<f32>& stack,
        const MetadataStack& metadata,
        const ns::CTFIsotropic<f64>& ctf,
        const TomogramReconstructionParameters& parameters
    ) -> Array<f32> {
        auto timer = Logger::info_scope_time("Reconstructing tomogram");

        // For simplicity, make the defocus resolution an odd integer multiple of the pixel size.
        const f64 spacing_nm = ctf.pixel_size() * 1e-1;
        auto z_step = static_cast<i64>(std::floor(parameters.defocus_step_nm / spacing_nm));
        z_step += noa::is_even(z_step);
        const auto z_step_nm = static_cast<f64>(z_step) * spacing_nm;
        Logger::trace("defocus_resolution={:.3f}nm ({}pix)", z_step_nm, z_step);

        // Get image steps and necessary padding.
        const auto image_shape = stack.shape().filter(2, 3);
        const auto image_center = (image_shape.vec / 2).as<f64>();
        const auto [max_n_strips, padded_image_shape] = analyse_defocus_range(
            image_shape, metadata, ctf, z_step_nm, parameters.sample_thickness_nm
        );

        // Get volume thickness and number of z-sections (of size z_step).
        // To guarantee that the volume center is at the center of a z-section,
        // make the volume thickness an odd multiple of z_step.
        const f64 sample_thickness = parameters.sample_thickness_nm / spacing_nm;
        const f64 z_padding = parameters.sample_thickness_nm * parameters.z_padding_percent / spacing_nm;
        auto volume_thickness = static_cast<i64>(std::round(sample_thickness + z_padding));
        volume_thickness = noa::next_multiple_of(volume_thickness, z_step);
        if (noa::is_even(volume_thickness / z_step))
            volume_thickness += z_step;
        const auto n_sections = volume_thickness / z_step;
        const auto volume_z_center = volume_thickness / 2;

        Logger::trace(
            "Volume settings:\n"
            "  spacing={:.3f}A (resolution={:.3f}A)\n"
            "  specimen_thickness={:.3f}nm\n"
            "  volume_thickness={:.3f}nm ({}pix, z_padding={:.1f}pix)\n"
            "  n_z_sections={}",
            spacing_nm * 10, spacing_nm * 20, parameters.sample_thickness_nm,
            static_cast<f64>(volume_thickness) * spacing_nm, volume_thickness, z_padding,
            n_sections
        );

        // Shapes.
        const auto options = ArrayOption{.device = stack.device(), .allocator = Allocator::MANAGED};
        const auto padded_stack_shape = padded_image_shape.push_front(Vec{stack.shape()[0], i64{1}});
        const auto padded_filter_shape = padded_image_shape.push_front(Vec{max_n_strips, i64{1}});
        const auto z_section_shape = image_shape.push_front(Vec{i64{1}, z_step});
        const auto volume_shape = image_shape.push_front(Vec{i64{1}, volume_thickness});

        const auto n_bytes_to_allocate =
            static_cast<size_t>(padded_stack_shape.rfft().n_elements()) * sizeof(c32) +
            static_cast<size_t>(padded_filter_shape.rfft().n_elements()) * sizeof(c32) +
            static_cast<size_t>(z_section_shape.n_elements()) * sizeof(f32);
        Logger::trace(
            "Allocating {:.2f}GB of temporary buffers on {} ({})",
            static_cast<f64>(n_bytes_to_allocate) * 1e-9, options.device, options.allocator
        );

        // Prepare the input spectrum. It will be filtered for each z-step.
        const auto [padded_images, padded_images_rfft] = nf::empty<f32>(padded_stack_shape, options);
        const auto padding_right = (padded_image_shape - image_shape).vec.push_front<2>(0);
        noa::resize(stack.view(), padded_images.view(), {}, padding_right);
        nf::r2c(padded_images.view(), padded_images_rfft.view(), {.norm = noa::fft::Norm::FORWARD});

        const auto corrected_stack = stack;

        // Allocate for the temporary buffers.
        const auto [buffer, buffer_rfft] = nf::empty<f32>(padded_filter_shape, options);
        const auto z_section = Array<f32>(z_section_shape, options);

        // Compute the projection matrices.
        const auto volume_center = image_center.push_front(static_cast<f64>(volume_z_center));
        const auto matrices = Array<Mat<f32, 2, 4>>(stack.shape()[0], options);
        for (auto&& [slice, matrix]: noa::zip(metadata, matrices.span_1d())) {
            auto angles = noa::deg2rad(slice.angles);
            matrix = ( // (image->volume).inverse()
                ng::translate(volume_center) *
                ng::linear2affine(ng::rotate_z(+angles[0])) * // TODO .to_affine()
                ng::linear2affine(ng::rotate_x(-angles[2])) * // TODO .to_affine()
                ng::linear2affine(ng::rotate_y(-angles[1])) * // TODO .to_affine()
                ng::linear2affine(ng::rotate_z(-angles[0])) * // TODO .to_affine()
                ng::translate(-(image_center + slice.shifts).push_front(0))
            ).inverse().filter_rows(1, 2).as<f32>(); // (y, x)
        }

        // backward_project(stack, matrices.view(), volume.view(), 0, noa::Interp::LINEAR);
        // noa::write(volume, parameters.output_directory / "volume.mrc");
        // panic();
        // noa::fill(volume, 0);

        // The full tomogram, on the CPU. Z-sections are progressively copied into it.
        auto volume = Array<f32>(volume_shape);

        for (i64 z{}; z < volume_thickness; z += z_step) {
            // Correct the stack for the defocus at the center of the current z-section.
            // At the volume z-center, this offset is exactly zero.
            const auto z_section_center = z_step / 2;
            const auto z_offset = z + z_section_center - volume_z_center;
            const auto z_offset_nm = static_cast<f64>(z_offset) * spacing_nm;
            compute_ctf_corrected_stack(
                padded_images_rfft.view(), buffer.view(), buffer_rfft.view(),
                corrected_stack, metadata, ctf, z_step_nm, z_offset_nm
            );
            // noa::write(stack, parameters.output_directory / "stack.mrc");
            // noa::write(corrected_stack, parameters.output_directory / "corrected_stack.mrc");

            backward_project(corrected_stack.view(), matrices.view(), z_section.view(), z, parameters.interp);

            // Copy the section into the volume.
            // TODO Use different stream
            z_section.to(volume.view().subregion(0, ni::Slice{z, z + z_step}));
            // TODO noa::copy(z_section, volume.view().subregion(0, ni::Slice{z, z + z_step}), {.sync_gpu_to_cpu = false});
        }

        noa::write(volume, Vec<f64, 3>::from_value(spacing_nm * 10), parameters.output_directory / "tomogram.mrc", {.dtype = noa::io::Encoding::F16});

        return volume;
    }
}
