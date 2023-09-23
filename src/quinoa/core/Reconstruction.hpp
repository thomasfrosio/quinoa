#include "quinoa/Types.h"
#include "quinoa/core/Metadata.h"
#include "quinoa/core/Stack.hpp"
#include "quinoa/core/Ewise.hpp"

namespace qn {
    void full_reconstruction(
        const Path& stack_path,
        MetadataStack metadata,
        const Path& reconstruction_path
    ) {
        const auto gpu_options = ArrayOption(Device("gpu"), Allocator::MANAGED);
        noa::Session::clear_fft_cache(gpu_options.device());

        // load the stack
        const auto loading_parameters = LoadStackParameters{
                /*compute_device=*/ gpu_options.device(),
                /*allocator=*/ gpu_options.allocator(),
                /*precise_cutoff=*/ true,
                /*rescale_target_resolution=*/ 24,
                /*rescale_min_size=*/ 1024,
                /*exposure_filter=*/ false,
                /*highpass_parameters=*/ {0.01, 0.01},
                /*lowpass_parameters=*/ {0.5, 0.05},
                /*normalize_and_standardize=*/ true,
                /*smooth_edge_percent=*/ 0.1f,
                /*zero_pad_to_fast_fft_shape=*/ false,
                /*zero_pad_to_square_shape*/ true,
        };

        auto loader = StackLoader(stack_path, loading_parameters);
        metadata.rescale_shifts(loader.file_spacing(), loader.stack_spacing());

        const auto current_shape = loader.slice_shape();
        const auto zero_padded_size = current_shape[0]; //noa::fft::next_fast_size(noa::math::max(current_shape) * 2);
        const auto padded_shape = Shape2<i64>{zero_padded_size, zero_padded_size};
        const auto border_right = (padded_shape - current_shape).vec().push_front<2>({0, 0});

        const auto original_center_1d = MetadataSlice::center(current_shape[0]);
        const auto original_center_2d = Vec2<f32>{original_center_1d, original_center_1d};
        const auto original_center_3d = Vec3<f32>{original_center_1d, original_center_1d, original_center_1d};

        const auto slice = noa::memory::empty<f32>(current_shape.push_front<2>({1, 1}), gpu_options);
        const auto slice_padded = slice;//noa::memory::empty<f32>(padded_shape.push_front<2>({1, 1}), gpu_options);

        const auto volume_padded_shape = Shape4<i64>{1, zero_padded_size, zero_padded_size, zero_padded_size};
        const auto border_right_3d = border_right.set<1>(border_right[3]);
        auto volume_padded_rfft = noa::memory::zeros<c32>(volume_padded_shape.rfft(), gpu_options);
        auto volume_padded_rfft_weights = noa::memory::zeros<f32>(volume_padded_shape.rfft(), gpu_options);

        // Windowed sinc.
        const f32 volume_size_f = static_cast<f32>(zero_padded_size);
        const f32 fftfreq_sinc = 1 / volume_size_f;
        const f32 fftfreq_blackman = 8 * fftfreq_sinc;

        for (const MetadataSlice& metadata_slice: metadata.slices()) {
            loader.read_slice(slice.view(), metadata_slice.index_file);


            if (slice.data() != slice_padded.data())
                noa::memory::resize(slice, slice_padded, {}, border_right);

            const auto slice_padded_rfft = noa::fft::r2c(slice_padded);
            noa::fft::remap(fft::H2HC, slice_padded_rfft, slice_padded_rfft, slice_padded.shape());
            noa::signal::fft::phase_shift_2d<fft::HC2HC>(
                    slice_padded_rfft, slice_padded_rfft, slice_padded.shape(),
                    -original_center_2d - metadata_slice.shifts.as<f32>());

            const Vec3<f64> insertion_angles = noa::math::deg2rad(metadata_slice.angles);
            const auto inv_rotation = noa::geometry::euler2matrix(
                    Vec3<f64>{-insertion_angles[0], insertion_angles[1], insertion_angles[2]},
                    "zyx", /*intrinsic=*/ false).transpose().as<f32>();

            noa::geometry::fft::insert_interpolate_3d<fft::HC2H>(
                    slice_padded_rfft, 1.f, slice_padded.shape(),
                    volume_padded_rfft, volume_padded_rfft_weights, volume_padded_shape,
                    Float22{}, inv_rotation, {fftfreq_sinc, fftfreq_blackman});
        }

        noa::ewise_binary(
                volume_padded_rfft, volume_padded_rfft_weights.release(), volume_padded_rfft,
                qn::correct_multiplicity_t{});

        noa::signal::fft::phase_shift_3d<fft::H2H>(
                volume_padded_rfft, volume_padded_rfft, volume_padded_shape,
                original_center_3d);

        if (volume_padded_rfft.device().is_gpu())
            volume_padded_rfft = volume_padded_rfft.to_cpu();
        const auto volume_padded = noa::fft::c2r(volume_padded_rfft, volume_padded_shape);
        const auto volume = noa::memory::resize(volume_padded, {}, -border_right_3d);
        noa::io::save(volume, reconstruction_path);
    }
}
