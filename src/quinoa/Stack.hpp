#pragma once

#include <noa/IO.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"

namespace qn {
    struct LoadStackParameters {
        Device compute_device;
        Allocator allocator;

        // Fourier cropping:
        bool precise_cutoff{true};
        f64 rescale_target_resolution{0};
        i64 rescale_min_size{0};
        i64 rescale_max_size{0};

        // Signal processing after cropping:
        bool exposure_filter{false};
        noa::signal::Bandpass bandpass{
            .highpass_cutoff = 0.10,
            .highpass_width = 0.10,
            .lowpass_cutoff = 0.45,
            .lowpass_width = 0.05,
        };
        f64 bandpass_mirror_padding_factor{0};

        // Image processing after cropping:
        bool normalize_and_standardize{true};
        f64 smooth_edge_percent{0.01};
        bool zero_pad_to_fast_fft_shape{true};
        bool zero_pad_to_square_shape{false};
    };

    class StackLoader {
    public:
        static void register_input_stack(const Path& filename);

    public:
        StackLoader() = default;

        /// Allocate buffers and set up the pre-processing and rescaling parameters.
        /// If the file doesn't exist, it will throw an exception.
        StackLoader(const Path& filename, const LoadStackParameters& parameters);

        void read_slice(const View<f32>& output_slice, i64 file_slice_index, bool cache = false);

        /// Loads the slices in "stack" in the same order as the order of the slices in "metadata".
        /// The .index field of the slices in "metadata" are reset to the [0..n) range.
        void read_stack(MetadataStack& metadata, const View<f32>& stack);
        auto read_stack(MetadataStack& metadata) -> Array<f32>;

        [[nodiscard]] auto compute_device() const noexcept -> Device { return m_parameters.compute_device; }
        [[nodiscard]] auto allocator() const noexcept -> Allocator { return m_parameters.allocator; }
        [[nodiscard]] auto file_spacing() const noexcept -> Vec2<f64> { return m_input_spacing; }
        [[nodiscard]] auto file_slice_shape() const noexcept -> Shape2<i64> { return m_input_slice_shape; }
        [[nodiscard]] auto stack_spacing() const noexcept -> Vec2<f64> { return m_output_spacing; }
        [[nodiscard]] auto slice_shape() const noexcept -> Shape2<i64> { return m_output_slice_shape; }
        [[nodiscard]] auto n_slices_in_file() const noexcept -> i64 { return m_file_slice_count; }

        [[nodiscard]] static auto registered_stack() noexcept -> View<const f32> { return s_input_stack.view(); }

        void clear_cache() { m_cache.clear(); }

    private:
        void read_slice_and_precision_pad_(i64 file_slice_index, const View<f32>& padded_slice);

    private:
        static Array<f32> s_input_stack; // register the input stack

        noa::io::ImageFile m_file{};
        i64 m_file_slice_count{};
        LoadStackParameters m_parameters{};

        Shape2<i64> m_input_slice_shape{};
        Shape2<i64> m_padded_slice_shape{};
        Shape2<i64> m_cropped_slice_shape{};
        Shape2<i64> m_bandpass_slice_shape{};
        Shape2<i64> m_output_slice_shape{};

        Vec2<f64> m_input_spacing{};
        Vec2<f64> m_output_spacing{};
        Vec2<f64> m_rescale_shift{};

        Array<f32> m_input_slice{}; // empty if no padding
        Array<f32> m_input_slice_io{}; // empty if compute is on the cpu, otherwise, this is cpu array
        Array<c32> m_padded_slice_rfft{};
        Array<c32> m_cropped_slice_rfft{};
        Array<c32> m_bandpass_slice_rfft{};
        std::vector<std::pair<i64, Array<f32>>> m_cache{}; // cache the pre-processed slices
    };

    struct LoadStackOutputs {
        Array<f32> stack;
        Vec2<f64> stack_spacing;
        Vec2<f64> file_spacing;
        Shape2<i64> file_slice_shape;
    };

    [[nodiscard]]
    inline auto load_stack(
        const Path& tilt_series_path,
        MetadataStack& tilt_series_metadata,
        const LoadStackParameters& parameters
    ) -> LoadStackOutputs {
        auto stack_loader = StackLoader(tilt_series_path, parameters);
        auto stack = stack_loader.read_stack(tilt_series_metadata);
        return {stack, stack_loader.stack_spacing(), stack_loader.file_spacing(), stack_loader.file_slice_shape()};
    }

    struct SaveStackParameters {
        bool correct_rotation{false};
        bool cache_loader{false};
        noa::Interp interp{noa::Interp::LINEAR};
        noa::Border border{noa::Border::ZERO};
        noa::io::Encoding::Type dtype = noa::io::Encoding::F32;
    };

    void save_stack(
        StackLoader& stack,
        const Path& filename,
        const MetadataStack& metadata,
        const SaveStackParameters& saving_parameters = {}
    );

    void save_stack(
        const View<const f32>& stack,
        const Vec<f64, 2>& spacing,
        const MetadataStack& metadata,
        const Path& filename,
        const SaveStackParameters& saving_parameters = {}
    );
}
