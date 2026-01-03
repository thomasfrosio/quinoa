#include <noa/Runtime.hpp>
#include <noa/Signal.hpp>
#include <noa/Xform.hpp>
#include <noa/FFT.hpp>
#include <noa/IO.hpp>

#include "Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/CommonFOV.hpp"
#include "quinoa/Types.hpp"
#include "quinoa/Utilities.hpp"
#include "quinoa/Stack.hpp"

namespace {
    using namespace qn;

    template<usize N = 160>
    struct BitMask {
        static_assert(noa::is_multiple_of(N, 8));
        static constexpr usize WORD_COUNT = 32;
        static constexpr usize N_WORDS = N / 32;
        u32 buffer[N_WORDS]{};

        constexpr BitMask() = default;
        constexpr explicit BitMask(const std::vector<bool>& mask) {
            for (usize i{}; i < mask.size(); ++i)
                if (mask[i])
                    buffer[i / 32] |= 1 << (i % 32);
        }

        constexpr auto operator[](nt::integer auto i) const -> bool {
            noa::bounds_check(N, i);
            const u32 data = buffer[i / 32];
            const u32 mask = 1 << (i % 32);
            return data & mask;
        }
    };

    struct MaskReferences {
        SpanContiguous<const f32, 3> input_images;
        SpanContiguous<f32, 3> output_images;
        SpanContiguous<const ParallelogramMask, 1> parallelograms;
        SpanContiguous<const i32, 1> input_indices;

        NOA_HD void operator()(isize i, isize h, isize w) const {
            auto value = parallelograms[i](h, w);
            if (value > 1e-6f)
                value *= input_images(input_indices[i], h, w);
            output_images(i, h, w) = value;
        }
    };

    struct MaskTarget {
        SpanContiguous<f32, 2> input;
        SpanContiguous<f32, 2> output;
        ParallelogramMask parallelogram;

        NOA_HD void operator()(isize h, isize w) const {
            auto value = parallelogram(h, w);
            if (value > 1e-6f)
                value *= input(h, w);
            output(h, w) = value;
        }
    };

    struct FilterImages {
        NOA_HD auto operator()(const Vec<f32, 2>& fftfreq_2d, isize) const -> f32 {
            // Directly from Aretomo3.
            const auto fftfreq = noa::sqrt(noa::dot(fftfreq_2d, fftfreq_2d));
            return 2.f * fftfreq * (0.55f + 0.45f * noa::cos(6.2831852f * fftfreq));
        }
    };

    // Adapted and simplified version of noa::geometry::BackwardForwardProject.
    class BackwardForwardProject {
    public:
        using index_type = i32;
        using coord_3d_type = Vec<f32, 3>;
        using input_span_type = SpanContiguous<const f32, 3, index_type>;
        using interpolator_type = nx::Interpolator<2, nx::Interp::LINEAR, noa::Border::ZERO, input_span_type>;

        interpolator_type input_images;
        SpanContiguous<const Mat<f32, 2, 4>> backward_matrices;

        SpanContiguous<f32, 2> output_image;
        Mat<f32, 3, 4> forward_matrix;

        coord_3d_type volume_shape{};
        coord_3d_type volume_center{};
        index_type projection_window_radius{};

    public:
        static constexpr auto forward_projection_window_size(
            const Shape<i64, 3>& volume_shape_,
            const Mat<f32, 3, 4>& projection_matrix
        ) -> i64 {
            const auto projection_axis = projection_matrix.col(0);
            const auto projection_window_center = (volume_shape_.vec / 2).as<f32>();

            auto distance_to_volume_edge = Vec<f32, 3>::from_value(std::numeric_limits<f32>::max());
            for (auto i: noa::irange(3))
                if (abs(projection_axis[i]) > 0) // not parallel to the ith-plane
                    distance_to_volume_edge[i] = abs(projection_window_center[i] / projection_axis[i]);

            const auto index = argmin(distance_to_volume_edge);
            check(index == 0);
            const auto projection_window_radius_ = static_cast<i64>(ceil(distance_to_volume_edge[index]));
            return (projection_window_radius_ + 1) * 2 + 1;
        }

        static constexpr auto image2volume(
            const Vec<f32, 3>& image_coordinates,
            const Vec<f32, 3>& projection_window_center,
            const Mat<f32, 3, 4>& projection_matrix
        ) -> Vec<f32, 3> {
            const auto projection_axis = projection_matrix.col(0);
            const auto projection_matrix_no_z = projection_matrix.filter_columns(1, 2, 3);

            // Transform from image space to volume space.
            // Since we need the transformed yx-plane, extract the 0yx transformed vector from the matrix-vector product.
            Vec<f32, 3> plane_0yx = projection_matrix_no_z * image_coordinates.pop_front().push_back(1);
            auto volume_coordinates = plane_0yx + projection_axis * image_coordinates[0];

            // Compute and add the distance along the projection axis,
            // from the yx-plane to the selected plane centered on the window center.
            // This is from https://en.m.wikipedia.org/wiki/Line-plane_intersection
            const f32 distance = (projection_window_center[0] - plane_0yx[0]) / projection_axis[0];
            volume_coordinates += projection_axis * distance;

            return volume_coordinates;
        }

        constexpr void operator()(index_type z, index_type y, index_type x) const {
            const auto image_coordinates = coord_3d_type::from_values(z - projection_window_radius, y, x);
            const auto volume_coordinates = image2volume(image_coordinates, volume_center, forward_matrix);

            // Return early if the current coordinate falls outside the virtual volume.
            if (volume_coordinates[0] < -1 or volume_coordinates[0] > volume_shape[0] or
                volume_coordinates[1] < -1 or volume_coordinates[1] > volume_shape[1] or
                volume_coordinates[2] < -1 or volume_coordinates[2] > volume_shape[2])
                return;

            // Backward project: sample the input images at the current coordinate.
            f32 value{};
            for (index_type j{}; j < backward_matrices.ssize(); ++j) {
                const auto input_coordinates = backward_matrices[j] * volume_coordinates.push_back(1);
                value += input_images.interpolate_at(input_coordinates, j);
            }

            // Forward project: project along the z-axis (in image space).
            noa::details::atomic_add(output_image, value, y, x);
        }
    };

    struct Projector {
        Array<Mat<f32, 2, 4>> m_backward_matrices; // (n)
        Array<f32> m_reference_images;
        Array<f32> m_reference_and_target;
        Array<c32> m_reference_and_target_rfft;
        Array<f32> m_xmap;
        Array<f32> m_xmap_centered;
        Array<ParallelogramMask> m_parallelogram_references;
        Array<i32> m_reference_indices;
        Array<f32> m_volume;

        Projector(isize maximum_number_of_images, Device device) {
            // m_backward_matrices = Array<Mat<f32, 2, 4>>(maximum_number_of_images, options);
            // m_forward_matrices = Array<Mat<f32, 3, 4>>(maximum_number_of_images, options);
        }

        void compute_projected_reference(
            const View<const f32>& input_images,
            const Metadata::Stack& input_metadata,
            const View<f32>& output_image,
            const Metadata::Image& output_metadata,
            isize volume_thickness
        ) const {
            noa::fill(output_image, 0);

            //
            const auto image_shape = input_images.shape().filter(2, 3);
            const auto volume_shape = image_shape.push_front(volume_thickness);
            const auto volume_center = (volume_shape / 2).vec.as<f64>();
            const auto n_inputs = input_metadata.ssize();

            // Backward projection matrices.
            auto backward_matrices = m_backward_matrices.span_1d().subregion(Slice{0, n_inputs});
            for (auto&& [backward_matrix, slice]: noa::zip(backward_matrices, input_metadata)) {
                const auto angles = noa::deg2rad(slice.angles);
                backward_matrix = (
                    nx::translate((volume_center.pop_front() + slice.shifts).push_front(0)) *
                    nx::rotate_z<true>(angles[0]) *
                    nx::rotate_y<true>(angles[1]) *
                    nx::rotate_x<true>(angles[2]) *
                    nx::translate(-volume_center)
                ).inverse().filter_rows(1, 2).as<f32>();
            }
            // nx::backward_project_3d(input_images, m_volume, m_backward_matrices.flat(0).subregion(ni::Slice{0, n_inputs}));

            // Forward projection matrices.
            const auto forward_matrix = (
                nx::translate((volume_center.pop_front() + output_metadata.shifts).push_front(0)) *
                nx::rotate_z<true>(noa::deg2rad(output_metadata.angles[0])) *
                nx::rotate_y<true>(noa::deg2rad(output_metadata.angles[1])) *
                nx::rotate_x<true>(noa::deg2rad(output_metadata.angles[2])) *
                nx::translate(-volume_center)
            ).pop_back().as<f32>();

            const auto projection_window = BackwardForwardProject::forward_projection_window_size(
                volume_shape, forward_matrix
            );

            Logger::trace("forward_projection_window_size={}", projection_window);
            noa::iwise(Shape{projection_window, image_shape[0], image_shape[1]}, output_image.device(), BackwardForwardProject{
                .input_images = BackwardForwardProject::interpolator_type(
                    input_images.span<const f32, 3, i32>().as_contiguous(), image_shape.as<i32>()),
                .backward_matrices = backward_matrices,
                .output_image = output_image.span().filter(2, 3).as_contiguous(),
                .forward_matrix = forward_matrix,
                .volume_shape = volume_shape.vec.as<f32>(),
                .volume_center = volume_center.as<f32>(),
                .projection_window_radius = static_cast<i32>(projection_window / 2 + 1),
            });
        }

        void update_shifts(const View<f32>& stack, Metadata::Stack& metadata, f64 thickness_um) {
            auto t = Logger::trace_scope_time("update shifts");

            const auto device = stack.device();
            const auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
            m_backward_matrices = Array<Mat<f32, 2, 4>>(metadata.ssize(), options);
            m_reference_images = Array<f32>(stack.shape(), options); // FIXME -1
            m_reference_and_target = Array<f32>(stack.shape().set<0>(2), options);
            m_reference_and_target_rfft = Array<c32>(stack.shape().set<0>(2).rfft(), options);
            m_xmap =  Array<f32>(stack.shape().set<0>(1), options);
            m_xmap_centered = Array<f32>({1, 1, 128, 128}, options);
            m_parallelogram_references = Array<ParallelogramMask>(metadata.ssize(), options);
            m_reference_indices = Array<i32>(metadata.ssize(), options);
            m_volume = Array<f32>(stack.shape().set<0>(1).set<1>(130), options);

            Metadata::Stack references;

            auto fov = CommonFOV{};
            auto fov_mask_options = FOVMaskOptions{
                .smooth_edge_percent = 0.05,
                .add_shifts = false,
            };

            auto& stream = Stream::current(device);
            auto start = noa::Event{};
            auto end = noa::Event{};

            for (isize i{1}; i < metadata.ssize(); ++i) {
                // Set up the references and target metadata.
                references.images.push_back(metadata[i - 1]);
                for (auto&& [reference, index]: noa::zip(references, m_reference_indices.span_1d()))
                    index = reference.index;

                auto reference_images = m_reference_images.view().subregion(Slice{0, references.ssize()});

                // Mask the references.
                start.record(stream);
                fov.set_geometry(stack.shape().filter(2, 3), metadata);
                for (auto&& [slice, parallelogram] : noa::zip(references, m_parallelogram_references.span_1d()))
                    parallelogram = fov.set_fov(slice, fov_mask_options);
                noa::iwise(reference_images.shape().filter(0, 2, 3), device, MaskReferences{
                    .input_images = stack.span().filter(0, 2, 3).as_contiguous(),
                    .output_images = reference_images.span().filter(0, 2, 3).as_contiguous(),
                    .parallelograms = m_parallelogram_references.span_1d(),
                    .input_indices = m_reference_indices.span_1d(),
                });
                end.record(stream);
                end.synchronize();
                Logger::trace("reference mask took {}", noa::Event::elapsed(start, end));
                // noa::write_image(reference_images, "~/Tmp/quinoa/pm01/reference_images.mrc");

                start.record(stream);
                compute_projected_reference(
                    reference_images, references,
                    m_reference_and_target.view().subregion(0), metadata[i],
                    100
                );
                end.record(stream);
                end.synchronize();
                Logger::trace("projection took {}", noa::Event::elapsed(start, end));
                // noa::write_image(m_volume, fmt::format("~/Tmp/quinoa/pm01/volume_{:>02}.mrc", i));

                start.record(stream);
                noa::iwise(m_reference_and_target.shape().filter(2, 3), device, MaskTarget{
                    .input = stack.span().subregion(metadata[i].index).filter(2, 3).as_contiguous(),
                    .output = m_reference_and_target.span().subregion(1).filter(2, 3).as_contiguous(),
                    .parallelogram = fov.set_fov(metadata[i], fov_mask_options),
                });
                end.record(stream);
                end.synchronize();
                Logger::trace("target mask took {}", noa::Event::elapsed(start, end));

                nf::r2c(m_reference_and_target, m_reference_and_target_rfft);
                ns::bandpass<"h2h">(
                    m_reference_and_target_rfft, m_reference_and_target_rfft,
                    m_reference_and_target.shape(), {
                        .highpass_cutoff = 0.03, .highpass_width = 0.03,
                        .lowpass_cutoff = 0.4, .lowpass_width = 0.1,
                    }
                    );
                nf::c2r(m_reference_and_target_rfft, m_reference_and_target);
                noa::normalize_per_batch(m_reference_and_target, m_reference_and_target, {.mode = noa::Norm::L2});
                // noa::write_image(m_reference_and_target, fmt::format("~/Tmp/quinoa/pm01/reference_target_{:>02}.mrc", i));

                start.record(stream);
                nf::r2c(m_reference_and_target, m_reference_and_target_rfft);
                ns::cross_correlation_map<"h2fc">(
                    m_reference_and_target_rfft.subregion(0),
                    m_reference_and_target_rfft.subregion(1),
                    m_xmap
                );
                auto shift = qn::find_peak<"fc2fc">(m_xmap.view(), m_xmap_centered.view(), {
                    .distortion_angle_deg = metadata[i].angles[0],
                    .max_shift_percent = 0.1,
                }).first;
                Logger::trace("shift={}", shift);
                end.record(stream);
                end.synchronize();
                Logger::trace("rfft/ccmap/peak took {}", noa::Event::elapsed(start, end));
                // noa::write_image(m_xmap, fmt::format("~/Tmp/quinoa/pm01/xmap_{:>02}.mrc", i));
                // noa::write_image(m_xmap_centered, "~/Tmp/qui   noa/pm01/xmap_centered.mrc");

                // metadata[i].shifts -= shift;
                // if (i == 2)
                //     panic();
            }
        }
    };
}

namespace qn {
    void simple_projection_matching(
        const View<f32>& stack, Metadata::Stack& metadata, f64 thickness_um, const Path& output_directory) {
        auto projector = Projector(stack.shape()[0], stack.device());

        auto metadata_sorted = metadata;
        metadata_sorted.sort("absolute_tilt");

        projector.update_shifts(stack, metadata_sorted, thickness_um);

        metadata_sorted.sort("tilt");
        save_stack(stack, {}, metadata_sorted, output_directory / "aligned2.mrc");

        panic();
    }
}
