#pragma once

#include "quinoa/Types.hpp"
#include "quinoa/Metadata.hpp"

namespace qn {
    class SpecimenThickness {
    public:
        SpecimenThickness() = default;

        SpecimenThickness(
            const Path& stack_filename,
            Metadata& metadata, // updated: .shifts
            Device device
        );

        [[nodiscard]] auto shared_buffer_bytes() const -> isize;
        void set_shared_buffer(const ArrayView<std::byte, 1>& shared_buffer);

        auto estimate(
            Metadata& metadata, // updated: stack.shifts, sample.thickness
            const Path& output_directory
        ) const -> f64;

    private:
        using value_type = f16;
        Array<f32, 3> m_tilt_series{};
        ArrayView<value_type, 3> m_tomogram{};
        isize m_volume_depth{};
        f64 m_spacing_nm{};
    };
}
