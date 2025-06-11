#pragma once

#include <noa/IO.hpp>
#include <noa/FFT.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Types.hpp"

namespace qn::details {
    auto has_plot_file_uuid(const Path& path) -> bool;
}

namespace qn {
    template<typename Real>
    void save_vector_to_text(View<Real> x, const Path& filename) {
        check(noa::indexing::is_contiguous_vector_batched_strided(x));

        // Make sure it is dereferenceable and ready to read.
        Array<std::remove_const_t<Real>> x_cpu;
        if (not x.is_dereferenceable()) {
            x_cpu = x.to_cpu();
            x = x_cpu.view();
        }
        x.eval();

        std::string format;
        for (auto i : noa::irange(x.shape().batch()))
            format += fmt::format("{}\n", fmt::join(x.subregion(i).span_1d(), ","));
        noa::write_text(format, filename);
    }

    struct SavePlotXYOptions {
        /// Name of the plot and axes.
        /// Leaving them empty is also valid.
        /// These are ignored when appending a plot.
        std::string title{};
        std::string x_name{};
        std::string y_name{};

        /// Label associated of the saved points/curve.
        /// If the final plot has no labels, a default sequence [0,n) is used.
        std::string label{};

        /// Whether the plot should be appended.
        /// Appending only works within the same run (see uuid).
        /// Rerunning the program will overwrite existing files (a backup will be saved).
        bool append{true};
    };

    /// Saves x-y values for plotting.
    /// \param x        x-values. Arange, Linspace, or a 1d range (e.g. std::span).
    /// \param y        y-values. A 1d range (e.g. std::span) or a batched range (e.g. View, Array, or Span).
    /// \param path     File path where to store the values.
    /// \param options  Plot options.
    template<typename T = noa::Arange<f64>, typename U>
    void save_plot_xy(
        const T& x,
        const U& y,
        const Path& path,
        const SavePlotXYOptions& options = {}
    ) {
        const bool has_uuid = details::has_plot_file_uuid(path);
        const bool append = options.append and has_uuid;
        auto text_file = noa::io::OutputTextFile(path, noa::io::Open{
            .write = true,
            .append = append,
            .backup = not append,
        });

        if (not append) {
            text_file.write(
                fmt::format("uuid={}\ntitle={}\nxname={}\nyname={}\n\n",
                            Logger::s_uuid, options.title, options.x_name, options.y_name));
        }

        auto print_span = []<typename S>(const S& span) -> std::string {
            if constexpr (nt::varray<S> or nt::span_nd<S, 4, 3, 2>) {
                // Convert to a contiguous 2d span.
                constexpr size_t N = S::SIZE;
                auto new_shape = Shape<i64, N>::from_value(1);
                new_shape[0] = span.shape()[0];
                new_shape[N - 1] = -1;
                auto spand_2d = span.span().reshape(new_shape).filter(0, N - 1).as_contiguous();

                std::string fmt{};
                for (auto i: noa::irange(spand_2d.shape()[0]))
                    fmt += fmt::format("{};", fmt::join(spand_2d[i], ","));
                fmt.at(fmt.size() - 1) = '\n';
                return fmt;
            } else {
                return fmt::format("{}\n", fmt::join(span, ","));
            }
        };

        if constexpr (nt::any_of<T, noa::Arange<f32>, noa::Arange<f64>>) {
            text_file.write("type=arange\n");
            text_file.write(fmt::format("label={}\n", options.label));
            text_file.write(fmt::format("x={},{}\n", x.start, x.step));
        } else if constexpr (nt::any_of<T, noa::Linspace<f32>, noa::Linspace<f64>>) {
            text_file.write("type=linspace\n");
            text_file.write(fmt::format("label={}\n", options.label));
            text_file.write(fmt::format("x={},{},{}\n", x.start, x.stop, x.endpoint));
        } else {
            text_file.write("type=scatter\n");
            text_file.write(fmt::format("label={}\n", options.label));
            text_file.write(fmt::format("x={}\n", fmt::join(x, ",")));
        }
        text_file.write(fmt::format("y={}\n\n", print_span(y)));
        Logger::trace("{} {}", path, append ? "appended" : "saved");
    }

    struct SavePlotShiftsOptions {
        std::string title{};
        std::string label{};
        bool append{true};
    };

    ///
    inline void save_plot_shifts(
        const MetadataStack& metadata,
        const Path& path,
        const SavePlotShiftsOptions& options = {}
    ) {
        const bool has_uuid = details::has_plot_file_uuid(path);
        const bool append = options.append and has_uuid;
        auto text_file = noa::io::OutputTextFile(path, noa::io::Open{
            .write = true,
            .append = append,
            .backup = not append,
        });

        if (not append) {
            text_file.write(fmt::format("uuid={}\ntitle={}\nxname=x-shifts (pixels)\nyname=y-shifts (pixels)\n\n",
                Logger::s_uuid, options.title));
        }

        text_file.write("type=scatter-shifts\n");
        text_file.write(fmt::format("label={}\n", options.label));
        text_file.write(fmt::format("indices={}\n", fmt::join(metadata | stdv::transform([](auto& slice){ return slice.index; }), ",")));
        text_file.write(fmt::format("tilts={:.2f}\n", fmt::join(metadata | stdv::transform([](auto& slice){ return slice.angles[1]; }), ",")));
        text_file.write(fmt::format("x={:.5f}\n", fmt::join(metadata | stdv::transform([](auto& slice){ return slice.shifts[0]; }), ",")));
        text_file.write(fmt::format("y={:.5f}\n\n", fmt::join(metadata | stdv::transform([](auto& slice){ return slice.shifts[1]; }), ",")));
        Logger::trace("{} {}", path, append ? "appended" : "saved");
    }

    template<typename T>
    concept vec_or_real = nt::vec_real<T> or nt::real<T>;

    template<vec_or_real T, vec_or_real U>
    constexpr auto resolution_to_fftfreq(const T& spacing, const U& resolution) {
        return spacing / resolution;
    }

    template<vec_or_real T, vec_or_real U>
    constexpr auto fftfreq_to_resolution(const T& spacing, const U& fftfreq) {
        return spacing / fftfreq;
    }

    inline auto fourier_crop_to_resolution(i64 size, f64 spacing, f64 resolution) {
        const f64 input_size = static_cast<f64>(size);
        const f64 target_spacing = resolution / 2;
        const f64 target_size = input_size * spacing / target_spacing;
        const i64 final_size = std::min(size, nf::next_fast_size(static_cast<i64>(std::round(target_size))));
        const f64 final_fftfreq = (static_cast<f64>(final_size) / input_size) / 2;
        return Pair{static_cast<i64>(final_size), final_fftfreq};
    }
}
