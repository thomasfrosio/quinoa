#pragma once

#include <noa/Geometry.h>
#include <noa/IO.h>
#include <noa/Memory.h>

#include "quinoa/core/Metadata.h"

namespace qn::geometry {
    /// Correct for the yaw (in-place rotation) and shifts, as encoded in the metadata,
    /// of each slice in the stack and save the transformed slices.
    /// \param[in] stack            Input tilt-series.
    /// \param[in] metadata         Metadata of \p stack. The slices are saved in the order specified by the metadata.
    ///                             Excluded views are still saved in the output file, but they are not transformed.
    /// \param pixel_size           Global HW pixel size.
    /// \param output_filename      Output filename.
    /// \param compute_device       Device where to do the computation. If it is a GPU, \p stack can be on any device,
    ///                             including the CPU. If it is a CPU, \p stack must be on the CPU as well.
    /// \param interpolation_mode   Any interpolation mode supported by this device.
    /// \param border_mode          Any border mode supported by this device.
    /// \see noa::geometry::transform2D() for more details on the supported layouts and interpolation/border modes.
    void transform(const Array<float>& stack,
                   const MetadataStack& metadata,
                   float2_t pixel_size,
                   const path_t& output_filename,
                   Device compute_device,
                   InterpMode interpolation_mode = InterpMode::INTERP_LINEAR,
                   BorderMode border_mode = BorderMode::BORDER_ZERO) {
        const size4_t slice_shape{1, 1, stack.shape()[2], stack.shape()[3]};
        const float2_t slice_center = float2_t(slice_shape.get(2)) / 2;

        io::ImageFile file(output_filename, io::WRITE);
        file.shape(size4_t{metadata.size(), 1, slice_shape[2], slice_shape[3]});
        file.pixelSize(float3_t{1, pixel_size[0], pixel_size[1]});

        Texture<float> texture(slice_shape, compute_device, interpolation_mode, border_mode);
        Array<float> buffer(slice_shape, {compute_device, Allocator::DEFAULT_ASYNC});
        Array<float> buffer_io = compute_device.gpu() ? memory::empty<float>(slice_shape) : buffer;

        for (size_t i = 0; i < metadata.size(); ++i) {
            const MetadataSlice& slice = metadata[i];
            if (slice.excluded) {
                file.writeSlice(stack.subregion(slice.index), i, false);
                continue;
            }

            const float33_t matrix{
                    noa::geometry::translate(slice_center) *
                    float33_t(noa::geometry::rotate(math::deg2rad(-slice.angles[0]))) *
                    noa::geometry::translate(-slice_center - slice.shifts)
            };

            texture.update(stack.subregion(slice.index));
            noa::geometry::transform2D(texture, buffer, math::inverse(matrix));

            if (compute_device.gpu())
                memory::copy(buffer, buffer_io);
            file.writeSlice(buffer_io, i, false);
        }
    }
}