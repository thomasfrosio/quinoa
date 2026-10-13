#pragma once

#include <noa/IO.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Types.hpp"

namespace qn::details {
    auto has_plot_file_uuid(const Path& path) -> bool;
}

namespace qn {
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
    /// \param x        x-values. Arange, Linspace, or a 1D range (fmt::join-able).
    /// \param y        y-values. A ((B..)W) range as a Span or Array.
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
            if constexpr (nt::array<S> or nt::span_nd<S, 4, 3, 2>) {
                const auto spand_2d = span.span().reshape(Shape2{-1, span.shape().back()}).as_contiguous();

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

    struct SavePlotCTFFitOptions {
        std::string title{};

        bool plot_ctf = true;

        /// Whether the plot should be appended.
        /// Appending only works within the same run (see uuid).
        /// Rerunning the program will overwrite existing files (a backup will be saved).
        bool append{true};
    };

    void save_plot_ctf_fit(
        const noa::Linspace<f64>& fftfreq_range,
        const SpanContiguous<const f32, 2>& spectra,
        const SpanContiguous<const CTFIsotropic64, 1>& ctfs,
        const SpanContiguous<const f32, 2>& thickness_modulations,
        const Path& path,
        const SavePlotCTFFitOptions& options = {}
    );

    struct SavePlotShiftsOptions {
        std::string title{};
        std::string label{};
        bool append{true};
    };

    void save_plot_shifts(
        const Metadata::Stack& metadata,
        const Path& path,
        const SavePlotShiftsOptions& options = {}
    );
}
