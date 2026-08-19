#pragma once

#include <noa/IO.hpp>
#include <noa/Signal.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Utilities.hpp"

namespace qn {
    struct LoadStackParameters {
        bool use_stack_register{true};
        Device compute_device;
        Allocator allocator;

        // Fourier cropping:
        bool precise_cutoff{true};
        f64 rescale_target_resolution{0};
        i64 rescale_min_size{0};
        i64 rescale_max_size{0};

        // Signal processing after cropping:
        ns::Bandpass bandpass{
            .highpass_cutoff = 0.10,
            .highpass_width = 0.10,
            .lowpass_cutoff = 0.45,
            .lowpass_width = 0.05,
        };
        f64 bandpass_mirror_padding_factor{0};
        f64 exposure_filter_voltage{0}; // kV, 0 turns off the exposure filter

        // Image processing after cropping:
        bool normalize_and_standardize{true};
        f64 smooth_edge_percent{0.01};
        bool zero_pad_to_fast_fft_shape{true};
        bool zero_pad_to_square_shape{false};

        bool allocate_fft_workspace{true};
    };

    class StackLoader {
    public:
        static auto register_input_stack(const Path& filename) -> Pair<Shape2, Vec<f64, 2>>;

    public:
        StackLoader() = default;

        /// Allocate buffers and set up the pre-processing and rescaling parameters.
        /// If the file doesn't exist, it will throw an exception.
        StackLoader(ni::ImageFile&& file, const LoadStackParameters& parameters);
        StackLoader(const Path& filename, const LoadStackParameters& parameters);

        /// Loads and preprocess the slice.
        /// \note If an image exists in the cache at the given file_slice_index, it is used regardless of the given
        ///       exposure. In other words, when caching images, the exposure is assumed to be unchanged (or ignored).
        void read_slice(const View<f32>& output_slice, isize file_slice_index, bool cache = false, f64 exposure = 0);

        /// Loads the slices in "stack" in the same order as the order of the slices in "metadata".
        /// The .index field of the slices in "metadata" are reset to the [0..n) range.
        void read_stack(Metadata::Stack& metadata, const View<f32>& stack);
        auto read_stack(Metadata::Stack& metadata) -> Array<f32>;

        [[nodiscard]] auto compute_device() const noexcept -> Device { return m_parameters.compute_device; }
        [[nodiscard]] auto allocator() const noexcept -> Allocator { return m_parameters.allocator; }
        [[nodiscard]] auto file_spacing() const noexcept -> Vec<f64, 2> { return m_input_spacing; }
        [[nodiscard]] auto file_slice_shape() const noexcept -> Shape2 { return m_input_slice_shape; }
        [[nodiscard]] auto stack_spacing() const noexcept -> Vec<f64, 2> { return m_output_spacing; }
        [[nodiscard]] auto slice_shape() const noexcept -> Shape2 { return m_output_slice_shape; }
        [[nodiscard]] auto n_slices_in_file() const noexcept -> isize { return m_file_slice_count; }

        void record_fft() const;
        void clear_cache() { m_cache.clear(); }

    private:
        void init_();
        auto input_slice_() const -> Pair<View<f32>, View<c32>>;
        auto padded_slice_() const -> Pair<View<f32>, View<c32>>;
        auto cropped_slice_() const -> Pair<View<f32>, View<c32>>;
        auto bandpass_slice_() const -> Pair<View<f32>, View<c32>>;

    private:
        static thread_local Array<std::byte> s_input_stack; // register the input stack
        static thread_local noa::io::DataType s_input_stack_dtype;

        ni::ImageFile m_file{};
        isize m_file_slice_count{};
        LoadStackParameters m_parameters{};
        bool m_swap_bd{};

        bool m_has_padding{};
        bool m_has_cropping{};
        bool m_has_filter{};

        Shape2 m_input_slice_shape{};
        Shape2 m_padded_slice_shape{};
        Shape2 m_cropped_slice_shape{};
        Shape2 m_bandpass_slice_shape{};
        Shape2 m_output_slice_shape{};

        Vec<f64, 2> m_input_spacing{};
        Vec<f64, 2> m_output_spacing{};
        Vec<f64, 2> m_rescale_shift{};

        Array<f32> m_io_slice{};
        Array<c32> m_input_slice_rfft{};
        Array<c32> m_padded_slice_rfft{};
        Array<c32> m_cropped_slice_rfft{};
        Array<c32> m_bandpass_slice_rfft{};
        std::vector<std::pair<isize, Array<f32>>> m_cache{}; // cache the pre-processed slices
    };

    struct LoadStackOutputs {
        Array<f32> stack;
        Vec<f64, 2> stack_spacing;
        Vec<f64, 2> file_spacing;
        Shape2 file_slice_shape;
    };

    [[nodiscard]]
    inline auto load_stack(
        const Path& tilt_series_path,
        Metadata::Stack& tilt_series_metadata,
        const LoadStackParameters& parameters
    ) -> LoadStackOutputs {
        auto stack_loader = StackLoader(tilt_series_path, parameters);
        auto stack = stack_loader.read_stack(tilt_series_metadata);
        return {stack, stack_loader.stack_spacing(), stack_loader.file_spacing(), stack_loader.file_slice_shape()};
    }

    [[nodiscard]]
    inline auto load_stack(
        const Path& tilt_series_path,
        Metadata& tilt_series_metadata,
        const LoadStackParameters& parameters
    ) ->  Array<f32> {
        auto stack_loader = StackLoader(tilt_series_path, parameters);
        auto stack = stack_loader.read_stack(tilt_series_metadata.stack);
        tilt_series_metadata.set_spacing(stack_loader.stack_spacing());
        return stack;
    }

    struct SaveStackParameters {
        bool correct_rotation{false};
        bool cache_loader{false};
        nx::Interp interp{nx::Interp::LINEAR};
        noa::Border border{noa::Border::ZERO};
        noa::io::DataType dtype = noa::io::DataType::F32;
    };

    void save_stack(
        StackLoader& stack,
        const Path& filename,
        const Metadata::Stack& metadata,
        const SaveStackParameters& saving_parameters = {}
    );

    void save_stack(
        const View<const f32>& stack,
        const Vec<f64, 2>& spacing,
        const Metadata::Stack& metadata,
        const Path& filename,
        const SaveStackParameters& saving_parameters = {}
    );
}
