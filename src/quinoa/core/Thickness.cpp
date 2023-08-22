#include <noa/Memory.hpp>
#include <noa/Geometry.hpp>
#include <noa/Signal.hpp>
#include <noa/FFT.hpp>
#include <noa/Math.hpp>

#include "quinoa/core/Ewise.hpp"
#include "quinoa/core/Thickness.hpp"
#include "quinoa/core/Optimizer.hpp"

namespace {
    using namespace qn;

    auto compute_insertion_geometry_(
            const MetadataStack& metadata,
            const Vec2<f64>& slice_center
    ) -> std::pair<Array<Vec2<f32>>, Array<Float33>> {

        auto phase_shifts = noa::memory::empty<Vec2<f32>>(metadata.ssize());
        auto matrices = noa::memory::empty<Float33>(metadata.ssize());

        const auto phase_shifts_span = phase_shifts.span();
        const auto matrices_span = matrices.span();

        i64 i = 0;
        for (const MetadataSlice& slice: metadata.slices()) {
            phase_shifts_span[i] = -(slice_center + slice.shifts).as<f32>();

            auto angles = noa::math::deg2rad(slice.angles);
            matrices_span[i] = noa::geometry::euler2matrix(
                    Vec3<f64>{-angles[0], angles[1], angles[2]},
                    /*axes=*/ "zyx", /*intrinsic=*/ false)
                    .transpose().as<f32>();
            ++i;
        }

        return {phase_shifts, matrices};
    }

    auto compute_profile_(
            const View<f32>& stack,
            const MetadataStack& metadata,
            const Path& debug_directory
    ) -> Array<f32> {
        // Zero pad.
        const auto padded_size = noa::fft::next_fast_size(noa::math::max(stack.shape().filter(2, 3)) * 2);
        const auto slice_padded_shape = Shape4<i64>{1, 1, padded_size, padded_size};
        const auto stack_padded_shape = slice_padded_shape.set<0>(stack.shape()[0]);

        // Assumes the stack is tapered and mean-normalised.
        const auto [stack_padded, stack_padded_rfft] = noa::fft::empty<f32>(stack_padded_shape, stack.options());
        noa::memory::resize(stack, stack_padded);

        noa::io::save(stack_padded, debug_directory / "stack_padded.mrc");

        // Input slices to insert.
        noa::fft::r2c(stack_padded, stack_padded_rfft, noa::fft::NORM_DEFAULT, false);
        noa::fft::remap(fft::H2HC, stack_padded_rfft, stack_padded_rfft, stack_padded_shape); // require even size
        auto [phase_shifts, stack_inv_matrices] = compute_insertion_geometry_(
                metadata, MetadataSlice::center<f64>(slice_padded_shape));
        if (stack.device() != phase_shifts.device()) {
            phase_shifts = phase_shifts.to(stack.options());
            stack_inv_matrices = stack_inv_matrices.to(stack.options());
        }

        // Output extracted slice.
        const auto [slice_padded, slice_padded_rfft] = noa::fft::empty<f32>(slice_padded_shape, stack.options());
        const auto slice_padded_rfft_weight = noa::memory::like<f32>(slice_padded_rfft);
        const auto slice_fwd_matrix = noa::geometry::rotate_x(noa::math::deg2rad(90.)).as<f32>();

        // Phase-shift to the centre of rotation.
        noa::signal::fft::phase_shift_2d<fft::HC2HC>(
                stack_padded_rfft, stack_padded_rfft, stack_padded_shape, phase_shifts);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
                stack_padded_rfft.view(), stack_padded_shape,
                slice_padded_rfft.view(), slice_padded_shape,
                Float22{}, stack_inv_matrices.view(),
                Float22{}, slice_fwd_matrix
        );
        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
                1.f, stack_padded_shape,
                slice_padded_rfft_weight.view(), slice_padded_shape,
                Float22{}, stack_inv_matrices.view(),
                Float22{}, slice_fwd_matrix
        );

        noa::ewise_binary(slice_padded_rfft, slice_padded_rfft_weight, slice_padded_rfft, qn::correct_multiplicity_t{});
        noa::signal::fft::phase_shift_2d<fft::H2H>(
                slice_padded_rfft, slice_padded_rfft, slice_padded_shape,
                MetadataSlice::center(slice_padded_shape));
        noa::fft::c2r(slice_padded_rfft, slice_padded, noa::fft::NORM_DEFAULT, false);

        noa::io::save(slice_padded, debug_directory / "slice_padded_profile.mrc");

        // Crop back to about the original shape.
        const auto original_size = padded_size / 2; // TODO This could be improved...
        return noa::memory::resize(slice_padded, {1, 1, original_size, original_size});
    }

    class WindowFitter {
    public:
        explicit WindowFitter(const View<f32>& variances) :
                m_variances(variances),
                m_window(variances.ssize()) {
            noa::math::normalize(variances, variances, NormalizationMode::L2_NORM);
            noa::ewise_binary(variances, noa::math::min(variances), variances, noa::minus_t{});
            save_vector_to_text(variances, "/home/thomas/Projects/quinoa/tests/ribo_ctf/debug_thickness/variances.txt");
        };

        static auto function_to_maximize(u32 n, const f64* parameters, f64* gradients, void* instance) {
            QN_CHECK(n == 2 && parameters && !gradients && instance, "invalid parameters");
            auto& self = *static_cast<WindowFitter*>(instance);
            const f64 center = parameters[0];
            const f64 radius = parameters[1];
            return self.cost_(center, radius);
        }

    private:
        static void compute_window_(Span<f32> window, f64 center, f64 radius) {
            auto line = noa::geometry::LineSmooth<f64, false, f64>(center, radius, 2.);
            for (i64 i = 0; i < window.ssize(); ++i)
                window[i] = static_cast<f32>(line(static_cast<f64>(i)));
        };

        f64 cost_(f64 center, f64 radius) {
            const auto variances = m_variances; // already normalized
            const auto window = m_window.view();
            compute_window_(window.span(), center, radius);
            noa::math::normalize(m_window, m_window, NormalizationMode::L2_NORM);
            save_vector_to_text(window, "/home/thomas/Projects/quinoa/tests/ribo_ctf/debug_thickness/window.txt");

            const f64 ncc = noa::reduce_binary( // or dot product
                    variances, window,
                    f64{}, [](f32 lhs, f32 rhs) {
                        return static_cast<f64>(lhs) * static_cast<f64>(rhs);
                    }, noa::plus_t{}, {});

            return ncc;
        }

    private:
        View<f32> m_variances;
        Array<f32> m_window;
    };

    auto fit_window(const View<f32>& variances, f64 spacing) -> Vec2<f64> {
        auto fitter = WindowFitter(variances.flat());

        // Estimates.
        //  - The center is in the middle of the reconstruction and isn't allowed to move much.
        //  - The radius is initialized to 50nm and has a range of [30,200] (60nm-400nm max thickness).
        const f64 initial_center = static_cast<f64>(variances.ssize()) / 2; // center estimate
        const f64 initial_radius = 500. / spacing; // 100nm thickness
        const std::array upper_bounds{initial_center + initial_center * 0.1, 2000. / spacing};
        const std::array lower_bounds{initial_center - initial_center * 0.1, 50. / spacing};

        Optimizer optimizer(NLOPT_LN_SBPLX, 2);
        optimizer.set_max_objective(WindowFitter::function_to_maximize, &fitter);
        optimizer.set_bounds(lower_bounds.data(), upper_bounds.data());

        std::array parameters{initial_center, initial_radius};
        const f64 ncc = optimizer.optimize(parameters.data());

        return {parameters[0], parameters[1]}; // in pixels
    }
}

namespace qn {
    auto estimate_sample_thickness(
            const View<f32>& stack,
            const MetadataStack& metadata,
            f64 spacing,
            const Path& debug_directory
    ) -> Vec2<f64> {
        QN_CHECK(stack.shape()[0] == metadata.ssize(), "Stack and metadata don't have the same number of slices");

        // Compute the profile of the tomogram (forward projection at elevation=90deg)
        // and compute the variance of the profile along the rows.
        const Array tomogram_profile = compute_profile_(stack, metadata, debug_directory);
        noa::io::save(tomogram_profile, debug_directory / "slice_profile.mrc");

        const auto variances = noa::math::var(tomogram_profile.view(), Vec4<bool>{0, 0, 0, 1}).to_cpu();
        save_vector_to_text(variances.view(), debug_directory / "variances.txt");

        // Fit a rectangular window to this vector.
        return fit_window(variances.view(), spacing);
    }
}
