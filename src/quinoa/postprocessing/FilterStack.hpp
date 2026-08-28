#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    struct FilterStackSettings{
        // Stack.
        bool prealign_stack{};
        nx::Interp prealign_stack_interpolation{};

        // Simple filters.
        bool ramp_filter{};
        i32 fake_sirt_iterations{};

        // CTF.
        bool correct_ctf{};
        f64 ctf_phase_flip_strength{};
        f64 ctf_defocus_step_nm{};
        f64 ctf_bfactor{};
    };

    auto filter_stack(
        StackLoader&& stack,
        Metadata& metadata,
        const FilterStackSettings& settings
    ) -> Array<f32>;

    class StackFilterer {
    public:
        StackFilterer() = default;

        // Loads and applies a simple filter to the stack.
        StackFilterer(
            StackLoader&& loader,
            Metadata::Stack& metadata, // loaded in current order, resets index, and possibly rotation and shifts
            bool use_aligned_stack,
            nx::Interp aligned_stack_interpolation,
            bool ramp_filter,
            i32 fake_sirt_iterations
        );

        // Loads the stack and prepare for CTF correction.
        StackFilterer(
            StackLoader&& loader,
            Metadata::Stack& metadata, // loaded in current order, resets index, and possibly rotation and shifts
            bool use_aligned_stack,
            nx::Interp aligned_stack_interpolation,
            bool ramp_filter,
            i32 fake_sirt_iterations,
            const CTFIsotropic64& ctf,
            f64 volume_thickness_nm,
            f64 z_step_nm,
            f64 phase_flip_strength
        );

        void allocate_and_prepare_spectra(
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
        );

        static auto divide_image_in_z_strips(
            const Shape2& image_shape,
            const Vec<f64, 3>& image_angles,
            f64 spacing_nm,
            f64 z_step_nm
        ) -> Pair<f64, isize>;

        static auto reduce_memory_requirements(
            const Shape2& image_shape,
            const Shape2& image_padded_shape,
            isize n_images,
            isize max_n_strips,
            usize n_bytes_free
        ) -> Tuple<isize, isize, bool>;

        void compute_irffts(isize n_strips, bool record = false, isize n_groups = 4) const;
        void prepare_irffts() const;

        [[nodiscard]] auto compute_filtered_stack(isize z) const -> Array<f32>;
        [[nodiscard]] auto compute_filtered_stack() const -> Array<f32>;

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
