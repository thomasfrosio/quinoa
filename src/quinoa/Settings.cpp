#include <filesystem>
#include <optional>
#include <string_view>
#include <cxxopts.hpp>
#include <toml++/toml.hpp>

#include <noa/Session.hpp>
#include <noa/Utils.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Settings.hpp"

namespace fmt {
    template<typename T> struct formatter<toml::node_view<T>> : ostream_formatter {};
}

namespace {
    using namespace qn;

    void sanitize_table_(const toml::table& table, const std::string& path = {}) {
        using namespace std::string_view_literals;
        constexpr std::array VALID_PATHS{
            "files.mdoc_file"sv,
            "files.stack_file"sv,
            "files.output_directory"sv,
            "files.frames_directory"sv,
            "files.csv_file"sv,
            "experiment.tilt_axis"sv,
            "experiment.specimen_tilt"sv,
            "experiment.specimen_pitch"sv,
            "experiment.voltage"sv,
            "experiment.amplitude"sv,
            "experiment.cs"sv,
            "experiment.phase_shift"sv,
            "experiment.thickness"sv,
            "preprocessing.run"sv,
            "preprocessing.exclude_blank_views"sv,
            "preprocessing.exclude_stack_images"sv,
            "alignment.coarse.run"sv,
            "alignment.coarse.fit_rotation"sv,
            "alignment.coarse.fit_tilt"sv,
            "alignment.ctf.run"sv,
            "alignment.ctf.fit_rotation"sv,
            "alignment.ctf.fit_tilt"sv,
            "alignment.ctf.fit_pitch"sv,
            "alignment.ctf.fit_phase_shift"sv,
            "alignment.ctf.fit_astigmatism"sv,
            "alignment.ctf.fit_thickness"sv,
            "alignment.refine.run"sv,
            "alignment.refine.fit_thickness"sv,
            "postprocessing.run"sv,
            "postprocessing.resolution"sv,
            "postprocessing.stack.run"sv,
            "postprocessing.stack.correct_rotation"sv,
            "postprocessing.stack.interpolation"sv,
            "postprocessing.stack.dtype"sv,
            "postprocessing.tomogram.run"sv,
            "postprocessing.tomogram.correct_rotation"sv,
            "postprocessing.tomogram.interpolation"sv,
            "postprocessing.tomogram.dtype"sv,
            "postprocessing.tomogram.oversample"sv,
            "postprocessing.tomogram.correct_ctf"sv,
            "postprocessing.tomogram.z_padding_percent"sv,
            "postprocessing.tomogram.phase_flip_strength"sv,
            "compute.device"sv,
            "compute.n_threads"sv,
            "compute.register_stack"sv,
            "compute.log_level"sv,
        };

        for (auto [key, value]: table) {
            auto current_path = fmt::format("{}{}{}", path, path.empty() ? "" : ".", key.str());
            if (value.is_table()) {
                sanitize_table_(*value.as_table(), current_path);
            } else {
                bool has_it{};
                for (auto&& valid_path: VALID_PATHS) { // TODO stdv::find
                    if (valid_path == current_path) {
                        has_it = true;
                        break;
                    }
                }
                check(has_it, "{} is not a valid setting", current_path);
            }
        }
    }

    template<typename T>
    auto parse_number_(std::string_view name, const toml::table& table, T fallback) -> T {
        if (auto arg = table.at_path(name)) {
            check(arg.is_number(), "{}={} is not a number", name, arg);
            return arg.value<T>().value();
        }
        return fallback;
    }

    auto parse_string_(std::string_view name, const toml::table& table, std::string fallback) -> std::string {
        if (auto arg = table.at_path(name)) {
            check(arg.is_string(), "{}={} is not a string", name, arg);
            return noa::string::to_lower(noa::string::trim(arg.value<std::string>().value()));
        }
        return fallback;
    }

    auto parse_boolean_(std::string_view name, const toml::table& table, bool fallback) -> bool {
        if (auto arg = table.at_path(name)) {
            check(arg.is_boolean(), "{}={} is not a boolean", name, arg);
            return arg.value_exact<bool>().value();
        }
        return fallback;
    }

    auto parse_interp(std::string_view name, const toml::table& table, const std::string& fallback) {
        const auto stack_interp = parse_string_(name, table, fallback);
        if (stack_interp == "linear")
            return noa::Interp::LINEAR;
        if (stack_interp == "cubic-bspline")
            return noa::Interp::CUBIC_BSPLINE;
        panic(R"({} should be "linear" or "cubic-bspline", but got "{}")", name, stack_interp);
    }

    auto parse_dtype(std::string_view name, const toml::table& table, const std::string& fallback) {
        const auto stack_dtype = parse_string_(name, table, fallback);
        if (stack_dtype == "f16")
            return noa::io::Encoding::F16;
        if (stack_dtype == "f32")
            return noa::io::Encoding::F32;
        panic(R"({} should be "f16" or "f32", but got "{}")", name, stack_dtype);
    }

    auto parse_files_(const toml::table& table, const cxxopts::ParseResult& cl) -> Settings::Files {
        auto get_path = [&](
            std::string_view name_settings,
            const std::string& name_cl,
            Path& field, bool is_file,
            std::source_location location = std::source_location::current()
        ) {
            if (cl.contains(name_cl)) {
                field = cl[name_cl].as<Path>();
            } else if (auto node = table.at_path(name_settings)) {
                auto result = node.value<std::string>();
                check_at_location(location, result, "{}={} cannot be converted to a path", name_settings, node);
                field = Path(result.value());
            } else {
                return;
            }
            noa::io::expand_user(field);
            check_at_location(
                location, not is_file or fs::is_regular_file(field),
                "{}={}. File does not exist or permissions to read are not granted.",
                name_settings, field);
        };

        Settings::Files files;
        get_path("files.stack_file", "stack", files.stack_file, true);
        get_path("files.frames_directory", "frames", files.frames_directory, false);
        get_path("files.csv_file", "csv", files.csv_file, true);
        get_path("files.mdoc_file", "mdoc", files.mdoc_file, true);

        get_path("files.output_directory", "output", files.output_directory, false);
        if (files.output_directory.empty())
            files.output_directory = fs::current_path();

        return files;
    }

    auto parse_experiment_(const toml::table& table) -> Settings::Experiment {
        constexpr f64 UNSPECIFIED_VALUE = std::numeric_limits<f64>::max();

        Settings::Experiment experiment;

        // These are marked as unspecified because the metadata will need to know if the user entered.
        experiment.tilt_axis = parse_number_("experiment.tilt_axis", table, UNSPECIFIED_VALUE);
        experiment.specimen_tilt = parse_number_("experiment.specimen_tilt", table, UNSPECIFIED_VALUE);
        experiment.specimen_pitch = parse_number_("experiment.specimen_pitch", table, UNSPECIFIED_VALUE);
        experiment.phase_shift = parse_number_("experiment.phase_shift", table, UNSPECIFIED_VALUE);

        check(noa::allclose(UNSPECIFIED_VALUE, experiment.specimen_tilt) or
              std::abs(experiment.specimen_tilt) < 40,
              "experiment.specimen_tilt={} (degrees). Should be less than 40 degrees.",
              experiment.specimen_tilt);
        check(noa::allclose(UNSPECIFIED_VALUE, experiment.specimen_pitch) or
              std::abs(experiment.specimen_pitch) < 40,
              "experiment.specimen_pitch={} (degrees). Should be less than 40 degrees.",
              experiment.specimen_pitch);
        check(noa::allclose(UNSPECIFIED_VALUE, experiment.phase_shift) or
              (experiment.phase_shift >= 0 and experiment.phase_shift <= 150),
              "experiment.phase_shift={} (degrees). Should be between 0 and 150 degrees.",
              experiment.phase_shift);

        experiment.thickness = parse_number_("experiment.thickness", table, 0.);
        check(experiment.thickness >= 0 and experiment.thickness <= 550,
              "experiment.thickness={} (nm). Should be between 0nm and 550 nm.",
              experiment.thickness);

        experiment.voltage = parse_number_("experiment.voltage", table, 300.);
        experiment.amplitude = parse_number_("experiment.amplitude", table, 0.07);
        experiment.cs = parse_number_("experiment.cs", table, 2.7);

        check(noa::allclose(experiment.voltage, 100.) or
              noa::allclose(experiment.voltage, 200.) or
              noa::allclose(experiment.voltage, 300.),
              "experiment.voltage={} (kV). Should be 100kV, 200kV or 300kV.",
              experiment.voltage);
        check(experiment.amplitude >= 0 and experiment.amplitude <= 0.2,
              "experiment.amplitude={} (fraction). Should be between 0 and 0.2.",
              experiment.amplitude);
        check(experiment.cs >= 0 and experiment.cs <= 4,
              "experiment.cs={} (micrometers). Should be between 0 and 4 micrometers.",
              experiment.cs);


        return experiment;
    }

    auto parse_preprocessing_(const toml::table& table) -> Settings::Preprocessing {
        Settings::Preprocessing preprocessing;
        preprocessing.run = parse_boolean_("preprocessing.run", table, true);
        preprocessing.exclude_blank_views = parse_boolean_("preprocessing.exclude_blank_views", table, true);

        if (auto node = table.at_path("preprocessing.exclude_stack_indices")) {
            if (node.is_array()) {
                for (auto&& e: *node.as_array()) {
                    auto result = e.value<i64>();
                    check(result.has_value(), "Could not parse preprocessing.exclude_stack_indices={} as an array of indices", node);
                    preprocessing.exclude_stack_indices.push_back(*result);
                }
            } else if (node.is_integer()) {
                preprocessing.exclude_stack_indices.push_back(*node.value<i64>());
            } else {
                panic("Could not parse preprocessing.exclude_stack_indices={} as an array of indices", node);
            }
        }

        return preprocessing;
    }

    auto parse_alignment_(const toml::table& table) -> Settings::Alignment {
        Settings::Alignment alignment;

        alignment.coarse_run = parse_boolean_("alignment.coarse.run", table, true);
        alignment.coarse_fit_rotation = parse_boolean_("alignment.coarse.fit_rotation", table, true);
        alignment.coarse_fit_tilt = parse_boolean_("alignment.coarse.fit_tilt", table, true);

        alignment.ctf_run = parse_boolean_("alignment.ctf.run", table, true);
        alignment.ctf_fit_rotation = parse_boolean_("alignment.ctf.fit_rotation", table, false);
        alignment.ctf_fit_tilt = parse_boolean_("alignment.ctf.fit_tilt", table, true);
        alignment.ctf_fit_pitch = parse_boolean_("alignment.ctf.fit_pitch", table, true);
        alignment.ctf_fit_phase_shift = parse_boolean_("alignment.ctf.fit_phase_shift", table, false);
        alignment.ctf_fit_astigmatism = parse_boolean_("alignment.ctf.fit_astigmatism", table, true);
        alignment.ctf_fit_thickness = parse_boolean_("alignment.ctf.fit_thickness", table, false);

        alignment.refine_run = parse_boolean_("alignment.refine.run", table, true);
        alignment.refine_fit_thickness = parse_boolean_("alignment.refine.fit_thickness", table, true);

        return alignment;
    }

    auto parse_postprocessing_(const toml::table& table) -> Settings::PostProcessing {
        Settings::PostProcessing postprocessing;
        postprocessing.run = parse_boolean_("postprocessing.run", table, true);
        postprocessing.resolution = parse_number_("postprocessing.resolution", table, -1.);

        postprocessing.stack_run = parse_boolean_("postprocessing.stack.run", table, false);
        postprocessing.stack_correct_rotation = parse_boolean_("postprocessing.stack.correct_rotation", table, true);
        postprocessing.stack_interpolation = parse_interp("postprocessing.stack.interpolation", table, "linear");
        postprocessing.stack_dtype = parse_dtype("postprocessing.stack.dtype", table, "f32");

        postprocessing.tomogram_run = parse_boolean_("postprocessing.tomogram.run", table, true);
        postprocessing.tomogram_correct_rotation = parse_boolean_("postprocessing.tomogram.correct_rotation", table, true);
        postprocessing.tomogram_interpolation = parse_interp("postprocessing.tomogram.interpolation", table, "linear");
        postprocessing.tomogram_dtype = parse_dtype("postprocessing.tomogram.dtype", table, "f32");
        postprocessing.tomogram_oversample = parse_boolean_("postprocessing.tomogram.oversample", table, false);
        postprocessing.tomogram_correct_ctf = parse_boolean_("postprocessing.tomogram.correct_ctf", table, true);

        postprocessing.tomogram_z_padding_percent = parse_number_("postprocessing.tomogram.z_padding_percent", table, 10.);
        check(postprocessing.tomogram_z_padding_percent >= 0 and postprocessing.tomogram_z_padding_percent <= 200,
              "postprocessing:tomogram_z_padding_percent should be between 0 and 200, but got {}",
              postprocessing.tomogram_z_padding_percent);

        postprocessing.tomogram_phase_flip_strength = parse_number_("postprocessing.tomogram.phase_flip_strength", table, i64{8});
        check(postprocessing.tomogram_phase_flip_strength >= 0 and postprocessing.tomogram_phase_flip_strength <= 10,
              "postprocessing:tomogram_phase_flip_strength should be between 0 and 10, but got {}",
              postprocessing.tomogram_phase_flip_strength);

        return postprocessing;
    }

    auto parse_compute_(const toml::table& table) -> Settings::Compute {
        Settings::Compute compute;

        // device
        std::string device_name;
        if (auto device = table.at_path("compute.device")) {
            auto result = device.value<std::string>();
            check(result.has_value(), "compute.device={} is not convertible to a string", device);
            device_name = result.value();
        }
        if (device_name == "auto") {
            compute.device = Device::is_any_gpu() ? Device::most_free_gpu() : "cpu";
        } else {
            // Let the device parsing do its thing and possibly throw...
            compute.device = Device(device_name);
        }

        // n_threads
        if (auto n_threads = table.at_path("compute.n_threads")) {
            auto result = n_threads.value<i64>();
            check(result.has_value(), "compute:n_threads={} is not convertible to an integer", n_threads);
            compute.n_threads = result.value();
        } else {
            compute.n_threads = noa::clamp(static_cast<i64>(noa::cpu::Device::cores().logical), 1, 16);
        }

        // register_input_stack
        if (auto register_stack = table.at_path("compute.register_stack")) {
            auto result = register_stack.value<bool>();
            check(result.has_value(), "compute:register_stack={} is not convertible to a boolean", register_stack);
            compute.register_stack = result.value();
        } else {
            compute.register_stack = true;
        }

        // log_level
        if (auto log_level = table.at_path("compute.log_level")) {
            auto result = log_level.value<std::string>();
            check(result.has_value(), "compute:log_level={} is not convertible to a string", log_level);
            compute.log_level = result.value();

            constexpr std::array valid_levels{"off", "error", "warn", "status", "info", "trace", "debug"};
            check(stdr::find(valid_levels, compute.log_level) != valid_levels.end(),
                  "compute.log_level={} is not valid. Should be {}",
                  compute.log_level, valid_levels);
        } else {
            compute.log_level = "trace";
        }

        return compute;
    }
}

namespace qn {
    auto Settings::parse(int argc, const char* const* argv) -> bool {
        auto options = cxxopts::Options("quinoa", "Tilt-series alignment software.");

        options.add_options("data")
        ("mdoc", "mdoc file. Overwrites settings.files.mdoc_file.", cxxopts::value<Path>(), "file")
        ("stack", "Stack file. MRC or TIF file containing the tilt images sorted in ascending order by their tilt angles. Overwrites settings.files.stack_file.", cxxopts::value<Path>(), "file")
        ("frames", "Directory containing the frames. The filenames should match the mdoc file. Overwrites settings.files.frames_directory.", cxxopts::value<Path>(), "dir");

        options.add_options("general")
        ("output", "Directory where all outputs are saved. Will be created if it doesn't exist. Defaults to the current working directory. Overwrites settings.files.output_directory.", cxxopts::value<Path>(), "dir")
        ("settings", "TOML file containing the settings. Command-line arguments may overwrite settings from this file.", cxxopts::value<Path>(), "file");

        options.add_options("special")
        ("h,help", "Print this help message and exit.")
        ("init", "Generate a TOML file with all of the settings and exit.", cxxopts::value<bool>()->implicit_value("true"));

        cxxopts::ParseResult cl;
        try {
            cl = options.parse(argc, argv);
            if (cl.contains("help")) {
                fmt::println(fmt::runtime(options.help()));
                return false;
            } else if (not cl.unmatched().empty()) {
                panic("Invalid command line arguments: {}. Use --help.", cl.unmatched());
            } else if (cl.contains("init") and cl["init"].as<bool>()) {
                // TODO
                return false;
            }
        } catch (...) {
            panic("Failed to parse command line arguments");
        }

        const auto& settings_path = cl["settings"].as<Path>();
        try {
            auto settings = toml::table{};
            if (cl.contains("settings")) {
                settings = toml::parse_file(settings_path.native());
                sanitize_table_(settings);
            }
            files = parse_files_(settings, cl);
            experiment = parse_experiment_(settings);
            preprocessing = parse_preprocessing_(settings);
            alignment = parse_alignment_(settings);
            postprocessing = parse_postprocessing_(settings);
            compute = parse_compute_(settings);
        } catch (...) {
            panic("Failed to parse the settings TOML file at {}", settings_path);
        }
        return true;
    }
}
