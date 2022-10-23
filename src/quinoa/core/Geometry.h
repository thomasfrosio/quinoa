#pragma once

#include <noa/Geometry.h>
#include <noa/IO.h>
#include <noa/Memory.h>

#include "quinoa/core/Metadata.h"

namespace qn::geometry {
    /// Correct for the yaw (in-place rotation) and shifts, as encoded in the metadata, and save the transformed stack.
    /// \param[in] stack            Input stack.
    /// \param[in] metadata         Metadata of \p stack. The slices are saved in the order specified in the metadata.
    ///                             Excluded views are still saved in the output file, but they are not transformed.
    /// \param output_filename      Path were to save the transformed stack.
    /// \param compute_device       Device where to do the computation. If it is a GPU, \p stack can be on any device,
    ///                             including the CPU. If it is a CPU, \p stack must be on the CPU as well.
    /// \param pixel_size           Output HW pixel size.
    /// \param interpolation_mode   Any interpolation mode supported by this device.
    /// \param border_mode          Any border mode supported by this device.
    /// \see noa::geometry::transform2D() for more details on the supported layouts and interpolation/border modes.
    void transform(const Array<float>& stack,
                   const MetadataStack& metadata,
                   const path_t& output_filename,
                   Device compute_device,
                   float2_t pixel_size = float2_t{},
                   InterpMode interpolation_mode = InterpMode::INTERP_LINEAR_FAST,
                   BorderMode border_mode = BorderMode::BORDER_ZERO) {
        const size4_t slice_shape{1, 1, stack.shape()[2], stack.shape()[3]};
        const float2_t slice_center = MetadataSlice::center(slice_shape);

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

            const float33_t fwd_transform{
                    noa::geometry::translate(slice_center) *
                    float33_t(noa::geometry::rotate(math::deg2rad(-slice.angles[0]))) *
                    noa::geometry::translate(-slice_center - slice.shifts)
            };

            texture.update(stack.subregion(slice.index));
            noa::geometry::transform2D(texture, buffer, math::inverse(fwd_transform));

            if (compute_device.gpu())
                memory::copy(buffer, buffer_io);
            file.writeSlice(buffer_io, i, false);
        }
    }

    /// Backward and forward projection, accounting for the multiplicity, a slice at a time.
    /// \note The input and output slices are non-redundant, non-centered.
    class Projector {
    public:
        Projector() = default;

        /// Creates a projector.
        /// \details Temporary arrays are created. Once constructed, the projector doesn't allocate any memory.
        /// \param slice_shape  The shape of a single slice.
        /// \param slice_center The (rotation) center of the slices.
        /// \param grid_shape   The shape of the grid, i.e. the 3D region that is to be reconstructed.
        /// \param target_shape The shape of the actual 3D Fourier volume. If empty, defaults to \p grid_shape.
        /// \param options      Array options for the temporary arrays. The device is the compute device.
        Projector(dim4_t slice_shape,
                  float2_t slice_center,
                  dim4_t grid_shape,
                  dim4_t target_shape = {},
                  ArrayOption options = {})
                : m_grid_data_fft(memory::zeros<cfloat_t>(grid_shape.fft(), options)),
                  m_grid_weights_fft(memory::zeros<float>(grid_shape.fft(), options)),
                  m_weights_ones_fft(memory::ones<float>(slice_shape.fft(), options)),
                  m_weights_extract_fft(memory::empty<float>(slice_shape.fft(), options)),
                  m_grid_shape(grid_shape),
                  m_slice_shape(slice_shape),
                  m_target_shape(target_shape),
                  m_slice_center(slice_center) {
            QN_CHECK(slice_shape.ndim() == 2,
                     "The slice shape should describe a 2D slice, but got shape {}",
                     slice_shape);
        }

        /// Backward project a slice into the 3D Fourier volume.
        /// \param[in] slice_fft    Slice to insert.
        /// \param rotation         3x3 DHW forward rotation matrices.
        /// \param shift            Extra shift to apply before any other transformation.
        ///                         Note that the slices are already phase shifted to their rotation center.
        /// \param scale            2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
        void backward(const Array<cfloat_t>& slice_fft,
                      float33_t rotation,
                      float2_t shift = {},
                      float22_t scaling = {},
                      float cutoff = 0.5f) const {
            noa::signal::fft::shift2D<fft::H2H>(
                    slice_fft, slice_fft, m_slice_shape, -m_slice_center + shift);
            noa::geometry::fft::insert3D<fft::H2HC>(
                    slice_fft, m_slice_shape,
                    m_grid_data_fft, m_grid_shape,
                    scaling, rotation, cutoff, m_target_shape);
            noa::geometry::fft::insert3D<fft::H2HC>(
                    m_weights_ones_fft, m_slice_shape,
                    m_grid_weights_fft, m_grid_shape,
                    scaling, rotation, cutoff, m_target_shape);
        }

        /// Forward project a slice from the 3D Fourier volume.
        /// \param[out] slice_fft   Slice to insert.
        /// \param rotation         3x3 DHW forward rotation matrices.
        /// \param shift            Extra shift to apply after any other transformation.
        ///                         Note that the slices are already phase shifted from their rotation center.
        /// \param scale            2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
        void forward(Array<cfloat_t>& slice_fft,
                     float33_t rotation,
                     float2_t shift = {},
                     float22_t scaling = {},
                     float cutoff = 0.5f) const {
            noa::geometry::fft::extract3D<fft::HC2H>(
                    m_grid_data_fft, m_grid_shape,
                    slice_fft, m_slice_shape,
                    scaling, rotation, cutoff, m_target_shape);
            noa::geometry::fft::extract3D<fft::HC2H>(
                    m_grid_weights_fft, m_grid_shape,
                    m_weights_extract_fft, m_slice_shape,
                    scaling, rotation, cutoff, m_target_shape);
            signal::fft::shift2D<fft::H2H>(slice_fft, slice_fft, m_slice_shape, m_slice_center + shift);
            math::ewise(slice_fft, m_weights_extract_fft, 1e-3f, slice_fft,
                        math::divide_epsilon_t{});
        }

        /// Resetting the buffer to the initial state.
        void reset() const {
            noa::memory::fill(m_grid_data_fft, cfloat_t{0});
            noa::memory::fill(m_grid_weights_fft, float{0});
        }

        /// Clears all buffers to return to the empty state.
        void clear() {
            *this = Projector();
        }

    private:
        Array<cfloat_t> m_grid_data_fft;
        Array<float> m_grid_weights_fft;
        Array<float> m_weights_ones_fft;
        Array<float> m_weights_extract_fft;
        dim4_t m_grid_shape;
        dim4_t m_slice_shape;
        dim4_t m_target_shape;
        float2_t m_slice_center;
    };
}
