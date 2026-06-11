#include "quinoa/Plot.hpp"

namespace qn::details {
    auto has_plot_file_uuid(const Path& path) -> bool {
        bool has_uuid{};
        auto path_expanded = path;
        noa::io::expand_user(path_expanded);
        if (fs::is_regular_file(path_expanded)) {
            std::string buffer;
            auto file = noa::io::InputTextFile(std::move(path_expanded), {.read = true});
            while (file.next_line_or_throw(buffer)) {
                const auto line = std::string_view(buffer);
                const size_t index = line.find('=');
                if (noa::details::trim(line.substr(0, index)) == "uuid") {
                    std::optional result = noa::details::parse<u64>(line.substr(index + 1));
                    check(result.has_value(), "Invalid UUID: {}", line);
                    has_uuid = result.value() == Logger::s_uuid;
                    break;
                }
            }
        }
        return has_uuid;
    }
}

namespace qn {
    void save_plot_ctf_fit(
        const noa::Linspace<f64>& fftfreq_range,
        const SpanContiguous<const f32, 2>& spectra,
        const SpanContiguous<const CTFIsotropic64, 1>& ctfs,
        const SpanContiguous<const f32, 2>& thickness_modulations,
        const Path& path,
        const SavePlotCTFFitOptions& options
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
                fmt::format("uuid={}\ntitle={}\nxname=fftfreq\nyname=amplitudes\n\n",
                            Logger::s_uuid, options.title));
        }

        const auto [n, w] = spectra.shape();
        text_file.write("type=ctf_fit\n");
        text_file.write(fmt::format("batch={}\n", n));
        text_file.write(fmt::format("linspace={},{},{},{}\n",
            fftfreq_range.start, fftfreq_range.stop, w, fftfreq_range.endpoint));

        auto buffer = Array<f32>(w);
        auto simulate_ctf = [&](ns::CTFIsotropic<f32> ctf, auto thickness_modulation) {
            ctf.set_bfactor(-50);
            auto span = buffer.span_1d();
            auto fftfreq_step = fftfreq_range.for_size(w).step;
            for (isize i{}; i < w; ++i) {
                auto fftfreq = fftfreq_range.start + static_cast<f64>(i) * fftfreq_step;
                auto lhs = ctf.value_at(fftfreq);
                lhs *= lhs;
                auto envelope = ctf.envelope_at(fftfreq);
                envelope *= envelope;
                lhs -= envelope / 2; // [0,1] -> [-0.5, 0.5]
                lhs *= thickness_modulation[i];
                span[i] = lhs;
            }
            return span;
        };

        for (isize i{}; i < n; ++i) {
            text_file.write(fmt::format("spectrum={::.4f}\n", spectra[i]));
            if (options.plot_ctf)
                text_file.write(fmt::format("ctf={}\n", simulate_ctf(ctfs[i].as<f32>(), thickness_modulations[i])));
        }
        text_file.write("\n");

        Logger::trace("{} {}", path, append ? "appended" : "saved");
    }

    void save_plot_shifts(
        const Metadata::Stack& metadata,
        const Path& path,
        const SavePlotShiftsOptions& options
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
}
