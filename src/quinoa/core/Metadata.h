#pragma once

#include "quinoa/Types.h"
#include "quinoa/Exception.h"
#include "quinoa/io/YAML.h"
#include "quinoa/io/Options.h"

namespace qn {
    /// Metadata of a 2D slice.
    /// \note For the transformation, the rotation center is fixed at n // 2, where n is the size of the axis.
    ///       Furthermore, the shifts should be applied before the rotation.
    struct MetadataSlice {
    public:
        static constexpr int32_t EXCLUDED_INDEX = -1;

    public:
        int32_t index{};    // Index [0, N) of the slice within its stack.
        float3_t angles{};  // Euler angles, in degrees, of the slice. ZYX extrinsic (yaw, tilt, pitch)
        float2_t shifts{};  // YX shifts, in pixels, of the slice.
        float exposure{};   // Cumulated exposure, in e-/A2.
    };

    struct TiltScheme {
        float starting_angle{};         // Angle, in degrees, of the first image that was collected.
        int32_t starting_direction{};   // Direction (>0 or <0) after collecting the first image.
        float angle_increment{};        // Angle increment, in degrees, between images.
        int32_t group{};                // Number of images that are collected before switching to the opposite side.
        bool exclude_start{};           // Exclude the first image from the first group.
        float per_view_exposure{};      // Per view exposure, in e-/A^2.

        [[nodiscard]] std::vector<MetadataSlice> generate(int32_t count, float rotation_angle = 0.f) const;
    };

    /// Metadata of a stack of 2D slices.
    class MetadataStack {
    public:
        MetadataStack() = default;

        /// Creates the metadata.
        /// This also exclude the views if needed (but does not squeeze).
        explicit MetadataStack(const Options& options);

        /// Creates the metadata from a mdoc file.
        explicit MetadataStack(const path_t& mdoc_filename);

        /// Creates the metadata from a tlt and exposure file.
        MetadataStack(const path_t& tlt_filename,
                      const path_t& exposure_filename,
                      float rotation_angle = 0);

        /// Creates the metadata from the tilt scheme.
        MetadataStack(TiltScheme tilt_scheme,
                      int32_t order_count,
                      float rotation_angle = 0);

    public: // Stack manipulations
        /// Marks slices as "excluded" (by setting the index to EXCLUDED_INDEX).
        /// Note however that these "excluded" slices are not actually excluded from any operation,
        /// and one should use squeeze() to remove the "excluded" slices from the container.
        MetadataStack& exclude(const std::vector<int32_t>& indexes_to_exclude) noexcept;

        /// Keeps (by setting the other slices index to EXCLUDED_INDEX) some slices.
        /// Note however that these "excluded" slices are not actually excluded from any operation,
        /// and one should use squeeze() to remove the "excluded" slices from the container.
        MetadataStack& keep(const std::vector<int32_t>& indexes_to_keep) noexcept;

        /// Removes the excluded slices from the container.
        MetadataStack& squeeze();

        /// Sorts the slices based on a given key.
        /// Valid keys: "index", "tilt", "exposure".
        MetadataStack& sort(std::string_view key, bool ascending = true);

        /// Sets the average shift across the stack to 0.
        MetadataStack& centerShifts();

    public: // Getters
        /// Retrieves the size of the container.
        [[nodiscard]] size_t size() const noexcept {
            return m_slices.size();
        }

        /// Returns a view of the container.
        [[nodiscard]] constexpr const std::vector<MetadataSlice>& slices() const noexcept {
            return m_slices;
        }

        /// Returns a copy of the slice indexes.
        [[nodiscard]] std::vector<int32_t> indexes() const {
            std::vector<int32_t> out;
            out.reserve(m_slices.size());
            for (auto& e: m_slices)
                out.emplace_back(e.index);
            return out;
        }

        /// Returns a copy of the slice angles.
        [[nodiscard]] std::vector<float3_t> angles() const {
            std::vector<float3_t> out;
            out.reserve(m_slices.size());
            for (auto& e: m_slices)
                out.emplace_back(e.angles);
            return out;
        }

        /// Returns a copy of the slice shifts.
        [[nodiscard]] std::vector<float2_t> shifts() const {
            std::vector<float2_t> out;
            out.reserve(m_slices.size());
            for (auto& e: m_slices)
                out.emplace_back(e.shifts);
            return out;
        }

        /// Returns a copy of the slice exposures.
        [[nodiscard]] std::vector<float> exposure() const {
            std::vector<float> out;
            out.reserve(m_slices.size());
            for (auto& e: m_slices)
                out.emplace_back(e.exposure);
            return out;
        }

        /// Find the index (as currently sorted in this instance)
        /// of the slice with the lowest absolute tilt angle.
        [[nodiscard]] size_t lowestTilt() const;

    public: // Setters
        template<typename T, typename = std::enable_if_t<noa::traits::is_almost_same_v<T, std::vector<MetadataSlice>>>>
        constexpr MetadataStack& slices(T&& slices) {
            m_slices = std::forward<T>(slices);
            return *this;
        }

        MetadataStack& indexes(const std::vector<int32_t>& indexes) {
            if (m_slices.empty()) {
                m_slices.resize(indexes.size());
            } else if (indexes.size() != size()) {
                QN_THROW("The input vector size ({}) doesn't match the size of the container ({})",
                         indexes.size(), size());
            }
            for (size_t i = 0; i < size(); ++i)
                m_slices[i].index = indexes[i];
            return *this;
        }

        MetadataStack& angles(const std::vector<float3_t>& angles) {
            if (m_slices.empty()) {
                m_slices.resize(angles.size());
            } else if (angles.size() != size()) {
                QN_THROW("The input vector size ({}) doesn't match the size of the container ({})",
                         angles.size(), size());
            }
            for (size_t i = 0; i < size(); ++i)
                m_slices[i].angles = angles[i];
            return *this;
        }

        MetadataStack& shifts(const std::vector<float2_t>& shifts) {
            if (m_slices.empty()) {
                m_slices.resize(shifts.size());
            } else if (shifts.size() != size()) {
                QN_THROW("The input vector size ({}) doesn't match the size of the container ({})",
                         shifts.size(), size());
            }
            for (size_t i = 0; i < size(); ++i)
                m_slices[i].shifts = shifts[i];
            return *this;
        }

        MetadataStack& exposure(const std::vector<float>& exposure) {
            if (m_slices.empty()) {
                m_slices.resize(exposure.size());
            } else if (exposure.size() != size()) {
                QN_THROW("The input vector size ({}) doesn't match the size of the container ({})",
                         exposure.size(), size());
            }
            for (size_t i = 0; i < size(); ++i)
                m_slices[i].exposure = exposure[i];
            return *this;
        }

    public: // Indexing
        /// Returns a view of the slice at \p idx, as currently sorted in this instance (see sort()).
        template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
        [[nodiscard]] constexpr MetadataSlice& operator[](T idx) noexcept {
            NOA_ASSERT(idx > 0 && idx < size());
            return m_slices[idx];
        }

        /// Returns a view of the slice at \p idx, as currently sorted in this instance (see sort()).
        template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
        [[nodiscard]] constexpr const MetadataSlice& operator[](T idx) const noexcept {
            NOA_ASSERT(idx > 0 && idx < size());
            return m_slices[idx];
        }

    private:
        void sortBasedOnIndexes_(bool ascending = true);
        void sortBasedOnTilt_(bool ascending = true);
        void sortBasedOnExposure_(bool ascending = true);

    private:
        std::vector<MetadataSlice> m_slices;
    };
}
