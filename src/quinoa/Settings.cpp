#include <filesystem>
#include <optional>
#include <string_view>
#include <cxxopts.hpp>
#include <toml++/toml.hpp>
#include <glob/glob.hpp>

#include <noa/Session.hpp>

#include "quinoa/Types.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Settings.hpp"

namespace fmt {
    template<typename T> struct formatter<toml::node_view<T>> : ostream_formatter {};
}

namespace {
    using namespace qn;

    void sanitize_table_(const toml::table& table, const std::string& path = {}) {
        using namespace std::string_view_literals;
        constexpr std::array VALID_PATHS{
            "data.mdocs"sv,
            "data.stacks"sv,
            "data.rawtlts"sv,
            "data.stars"sv,
            "data.output"sv,
            "experiment.tilt_axis"sv,
            "experiment.add_specimen_tilt"sv,
            "experiment.add_specimen_pitch"sv,
            "experiment.voltage"sv,
            "experiment.amplitude"sv,
            "experiment.cs"sv,
            "experiment.phase_shift"sv,
            "experiment.thickness"sv,
            "preprocessing.run"sv,
            "preprocessing.exclude_blank_views"sv,
            "preprocessing.exclude_stack_indices"sv,
            "alignment.coarse.run"sv,
            "alignment.coarse.check_rotation"sv,
            "alignment.coarse.fit_rotation"sv,
            "alignment.coarse.fit_tilt"sv,
            "alignment.coarse.fit_pitch"sv,
            "alignment.ctf.run"sv,
            "alignment.ctf.patch_size_ang"sv,
            "alignment.ctf.patch_size_min_pix"sv,
            "alignment.ctf.resolution_range"sv,
            "alignment.ctf.n_images_in_initial_average"sv,
            "alignment.ctf.check_defocus_gradient"sv,
            "alignment.ctf.fit_rotation"sv,
            "alignment.ctf.fit_tilt"sv,
            "alignment.ctf.fit_pitch"sv,
            "alignment.ctf.fit_phase_shift"sv,
            "alignment.ctf.fit_astigmatism"sv,
            "alignment.ctf.fit_thickness"sv,
            "alignment.refine.run"sv,
            "alignment.refine.correct_ctf"sv,
            "alignment.refine.phase_flip_strength"sv,
            "alignment.refine.fit_rotation"sv,
            "alignment.refine.fit_tilt"sv,
            "alignment.refine.fit_pitch"sv,
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
            "postprocessing.tomogram.algorithm"sv,
            "postprocessing.tomogram.oversampling_factor"sv,
            "postprocessing.tomogram.ramp_filter"sv,
            "postprocessing.tomogram.correct_ctf"sv,
            "postprocessing.tomogram.z_padding_percent"sv,
            "postprocessing.tomogram.phase_flip_strength"sv,
            "compute.device"sv,
            "compute.n_threads"sv,
            "compute.register_stack"sv,
            "compute.log_level"sv,
            "compute.dry"sv,
            "compute.stop_at_first_error"sv,
        };

        for (auto [key, value]: table) {
            auto current_path = fmt::format("{}{}{}", path, path.empty() ? "" : ".", key.str());
            if (value.is_table()) {
                sanitize_table_(*value.as_table(), current_path);
            } else {
                bool has_it{};
                for (auto&& valid_path: VALID_PATHS) {
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
    auto parse_value_(std::string_view name, const toml::table& table) -> std::optional<T> {
        if (auto arg = table.at_path(name)) {
            if constexpr (nt::boolean<T>) {
                check(arg.is_boolean(), "{}={} is not a bool", name, arg);
                return arg.value<bool>().value();
            } else if constexpr (nt::scalar<T>) {
                check(arg.is_number(), "{}={} is not a {}", name, arg, noa::details::stringify<T>());
                return arg.value<T>().value();
            } else if constexpr (nt::string<T>) {
                check(arg.is_string(), "{}={} is not a string", name, arg);
                namespace nd = noa::details;
                return nd::to_lower(nd::trim(arg.value<std::string>().value()));
            }
        }
        return std::nullopt;
    }

    template<typename T>
    auto parse_value_(std::string_view name, const toml::table& table, const T& fallback) -> T {
        return parse_value_<T>(name, table).value_or(fallback);
    }

    template<typename T>
    auto parse_value_(
        std::string_view name_settings,
        const toml::table& table,
        const std::string& name_cl,
        const cxxopts::ParseResult& cl,
        T fallback
    ) -> T {
        if (cl.contains(name_cl))
            return cl[name_cl].as<T>();
        if (auto arg = table.at_path(name_settings)) {
            if constexpr (nt::boolean<T>) {
                check(arg.is_boolean(), "{}={} is not a bool", name_settings, arg);
                return arg.value_exact<bool>().value();
            } else if constexpr (nt::scalar<T>) {
                check(arg.is_number(), "{}={} is not a {}", name_settings, arg, noa::details::stringify<T>());
                return arg.value<T>().value();
            } else if constexpr (nt::string<T>) {
                check(arg.is_string(), "{}={} is not a string", name_settings, arg);
                namespace nd = noa::details;
                return nd::to_lower(nd::trim(arg.value<std::string>().value()));
            }
        }
        return fallback;
    }

    template<typename T, usize N>
    auto parse_values_(std::string_view name, const toml::table& table, const Vec<T, N>& fallback) -> Vec<T, N> {
        if (auto arg = table.at_path(name)) {
            check(arg.is_array(), "{}={} is not an array", name, arg);
            auto array = arg.as_array();
            check(array and array->size() == N, "{}={} is not an array of {} elements", name, arg, N);
            auto out = Vec<T, N>{};
            for (i32 i{}; auto&& e: *array) {
                if constexpr (nt::boolean<T>) {
                    check(e.is_boolean(), "{}={}, index {} is not a bool", name, arg, i);
                    out[i++] = e.value_exact<bool>().value();
                } else if constexpr (nt::scalar<T>) {
                    check(e.is_number(), "{}={}, index {} is not a {}", name, arg, i, noa::details::stringify<T>());
                    out[i++] = e.value<T>().value();
                }
            }
            return out;
        }
        return fallback;
    }

    auto parse_interp(std::string_view name, const toml::table& table, const std::string& fallback) {
        const auto stack_interp = parse_value_(name, table, fallback);
        if (stack_interp == "linear")
            return nx::Interp::LINEAR;
        if (stack_interp == "cubic-bspline")
            return nx::Interp::CUBIC_BSPLINE;
        if (stack_interp == "lanczos")
            return nx::Interp::LANCZOS6;
        panic(R"({} should be "linear" or "cubic-bspline", but got "{}")", name, stack_interp);
    }

    auto parse_dtype(std::string_view name, const toml::table& table, const std::string& fallback) {
        const auto stack_dtype = parse_value_(name, table, fallback);
        if (stack_dtype == "f16")
            return noa::io::DataType::F16;
        if (stack_dtype == "f32")
            return noa::io::DataType::F32;
        panic(R"({} should be "f16" or "f32", but got "{}")", name, stack_dtype);
    }

    auto parse_path(
        const toml::table& table,
        const cxxopts::ParseResult& cl,
        std::string_view name_settings,
        const std::string& name_cl
    ) {
        auto field = Path{};
        if (cl.contains(name_cl)) {
            field = cl[name_cl].as<Path>();
        } else if (auto node = table.at_path(name_settings)) {
            auto result = node.value<std::string>();
            check(result, "{}={} cannot be converted to a path", name_settings, node);
            field = Path(result.value());
        } else {
            return Path{};
        }
        ni::expand_user(field);
        return field;
    }

    auto parse_paths(
        const toml::table& table,
        const cxxopts::ParseResult& cl,
        std::string_view name_settings,
        const std::string& name_cl
    ) {
        auto patterns = std::vector<Path>{};
        if (cl.contains(name_cl)) {
            patterns = cl[name_cl].as<std::vector<Path>>();
        } else if (const auto node = table.at_path(name_settings)) {
            node.visit([&]<typename T>(T&& value) {
                if constexpr (toml::is_string<T>) {
                    patterns.emplace_back(*value);
                } else if constexpr (toml::is_array<T>) {
                    for (auto&& item: value) {
                        auto result = item.template value<std::string>();
                        // check_at_location(location, result, "{}={} cannot be converted to an array of path", name_settings, value);
                        patterns.emplace_back(result.value());
                    }
                } else {
                    // panic_at_location(location, "{}={} cannot be converted to a path", name_settings, value);
                }
            });
        }
        return patterns;
    }

    auto expand_glob(std::vector<Path>& paths, const std::string_view& append_to_dir) {
        auto resolved_paths = std::vector<Path>{};
        auto add_path = [&](Path& path) {
            if (stdr::find(resolved_paths, path) == resolved_paths.end())
                resolved_paths.emplace_back(std::move(path));
        };

        for (auto& path: paths) {
            for (auto& resolved: glob::glob(path.native())) {
                if (fs::is_directory(resolved) and not append_to_dir.empty()) {
                    // If this points to a directory, add the files inside this directory matching the extension(s).
                    for (Path& r: glob::glob((resolved / append_to_dir).native()))
                        add_path(r);
                } else {
                    add_path(resolved);
                }
            }
        }
        return resolved_paths;
    }

    auto match_using_stem(std::string_view stem, const std::vector<Path>& paths, std::string_view info, bool is_optional) {
        isize index{-1};
        for (usize i{}; i < paths.size(); ++i) {
            if (paths[i].stem() == stem) {
                if (index < 0) {
                    index = static_cast<isize>(i);
                } else {
                    panic("Matching ambiguity for stem={}. At least two candidates detected: {} and {}",
                          stem, paths[static_cast<usize>(index)], paths[i]);
                }
            }
        }
        check(is_optional or index >= 0, "Could not find the {} for {}", info, stem);
        if (paths.empty())
            return Path{};
        return paths[static_cast<usize>(index)];
    }

    auto parse_data_(const toml::table& table, const cxxopts::ParseResult& cl) -> std::vector<Series> {
        // Get the files.
        auto mdocs = parse_paths(table, cl, "data.mdocs", "mdocs");
        auto stacks = parse_paths(table, cl, "data.stacks", "stacks");
        auto rawtlts = parse_paths(table, cl, "data.rawtlts", "rawtlts");
        auto stars = parse_paths(table, cl, "data.stars", "stars");

        mdocs = expand_glob(mdocs, "*.mdoc");
        stacks = expand_glob(stacks, "*.{mrc,mrcs,st}");
        rawtlts = expand_glob(rawtlts, "*.{rawtlt,tlt}");
        stars = expand_glob(stars, "*.star");

        auto output_directory = parse_path(table, cl, "data.output", "output");
        if (output_directory.empty())
            output_directory = fs::current_path();

        // If there's a single file, don't match on the basename.
        const bool is_single_mode =
            mdocs.size() == 1 and stacks.size() == 1 and
            (rawtlts.empty() or rawtlts.size() == 1) and
            (stars.empty() or stars.size() == 1);

        auto data = std::vector<Series>{};
        if (is_single_mode) {
            data.push_back({
                .mdoc_file = std::move(mdocs[0]),
                .stack_file = std::move(stacks[0]),
                .rawtlt_file = std::move(rawtlts[0]),
                .star_file = std::move(stars[0]),
                .output_directory = std::move(output_directory),
            });
        } else {
            // Batch mode.
            // Assign one stack/rawtlt/star per mdoc based on the stem.
            for (auto& mdoc: mdocs) {
                const auto stem = Series::stem(mdoc).native();
                data.push_back({
                    .mdoc_file = std::move(mdoc),
                    .stack_file = match_using_stem(stem, stacks, "stack", false),
                    .rawtlt_file = match_using_stem(stem, rawtlts, "rawtlt", true),
                    .star_file = match_using_stem(stem, stars, "star", true),
                    .output_directory = output_directory,
                });
            }
        }
        return data;
    }

    auto parse_experiment_(const toml::table& table, const cxxopts::ParseResult& cl) -> Settings::Experiment {
        constexpr f64 UNSPECIFIED_VALUE = std::numeric_limits<f64>::max();

        Settings::Experiment experiment;

        // These are marked as unspecified because the metadata will need to know if the user entered.
        experiment.tilt_axis = parse_value_("experiment.tilt_axis", table, "tilt-axis", cl, UNSPECIFIED_VALUE);
        experiment.add_specimen_tilt = parse_value_("experiment.add_specimen_tilt", table, UNSPECIFIED_VALUE);
        experiment.add_specimen_pitch = parse_value_("experiment.add_specimen_pitch", table, UNSPECIFIED_VALUE);
        experiment.phase_shift = parse_value_("experiment.phase_shift", table, UNSPECIFIED_VALUE);

        check(noa::allclose(UNSPECIFIED_VALUE, experiment.add_specimen_tilt) or
              std::abs(experiment.add_specimen_tilt) < 40,
              "experiment.add_specimen_tilt={} (degrees). Should be less than 40 degrees.",
              experiment.add_specimen_tilt);
        check(noa::allclose(UNSPECIFIED_VALUE, experiment.add_specimen_pitch) or
              std::abs(experiment.add_specimen_pitch) < 40,
              "experiment.add_specimen_pitch={} (degrees). Should be less than 40 degrees.",
              experiment.add_specimen_pitch);
        check(noa::allclose(UNSPECIFIED_VALUE, experiment.phase_shift) or
              (experiment.phase_shift >= 0 and experiment.phase_shift <= 150),
              "experiment.phase_shift={} (degrees). Should be between 0 and 150 degrees.",
              experiment.phase_shift);

        experiment.thickness = parse_value_("experiment.thickness", table, UNSPECIFIED_VALUE);
        check(noa::allclose(UNSPECIFIED_VALUE, experiment.thickness) or
              (experiment.thickness >= 0 and experiment.thickness <= 550),
              "experiment.thickness={} (nm). Should be between 0nm and 550 nm.",
              experiment.thickness);

        experiment.voltage = parse_value_("experiment.voltage", table, UNSPECIFIED_VALUE);
        experiment.amplitude = parse_value_("experiment.amplitude", table, UNSPECIFIED_VALUE);
        experiment.cs = parse_value_("experiment.cs", table, UNSPECIFIED_VALUE);

        check(noa::allclose(UNSPECIFIED_VALUE, experiment.voltage) or
              noa::allclose(experiment.voltage, 100.) or
              noa::allclose(experiment.voltage, 200.) or
              noa::allclose(experiment.voltage, 300.),
              "experiment.voltage={} (kV). Should be 100kV, 200kV or 300kV.",
              experiment.voltage);
        check(noa::allclose(UNSPECIFIED_VALUE, experiment.amplitude) or
              (experiment.amplitude >= 0 and experiment.amplitude <= 0.2),
              "experiment.amplitude={} (fraction). Should be between 0 and 0.2.",
              experiment.amplitude);
        check(noa::allclose(UNSPECIFIED_VALUE, experiment.cs) or
              (experiment.cs >= 0 and experiment.cs <= 4),
              "experiment.cs={} (micrometers). Should be between 0 and 4 micrometers.",
              experiment.cs);

        return experiment;
    }

    auto parse_preprocessing_(const toml::table& table) -> Settings::Preprocessing {
        Settings::Preprocessing preprocessing;
        preprocessing.run = parse_value_("preprocessing.run", table, true);
        preprocessing.exclude_blank_views = parse_value_("preprocessing.exclude_blank_views", table, true);

        if (auto node = table.at_path("preprocessing.exclude_stack_indices")) {
            if (node.is_array()) {
                for (auto&& e: *node.as_array()) {
                    auto result = e.value<isize>();
                    check(result.has_value(), "Could not parse preprocessing.exclude_stack_indices={} as an array of indices", node);
                    preprocessing.exclude_stack_indices.push_back(*result);
                }
            } else if (node.is_integer()) {
                preprocessing.exclude_stack_indices.push_back(*node.value<isize>());
            } else {
                panic("Could not parse preprocessing.exclude_stack_indices={} as an array of indices", node);
            }
        }

        return preprocessing;
    }

    auto parse_alignment_(const toml::table& table) -> Settings::Alignment {
        Settings::Alignment alignment;

        alignment.coarse_run = parse_value_("alignment.coarse.run", table, true);
        alignment.coarse_check_rotation = parse_value_("alignment.coarse.check_rotation", table, true);
        alignment.coarse_fit_rotation = parse_value_("alignment.coarse.fit_rotation", table, true);
        alignment.coarse_fit_tilt = parse_value_("alignment.coarse.fit_tilt", table, true);
        alignment.coarse_fit_pitch = parse_value_("alignment.coarse.fit_pitch", table, true);

        alignment.ctf_run = parse_value_("alignment.ctf.run", table, true);
        alignment.ctf_check_defocus_gradient = parse_value_("alignment.ctf.check_defocus_gradient", table, true);
        alignment.ctf_fit_rotation = parse_value_("alignment.ctf.fit_rotation", table, false);
        alignment.ctf_fit_tilt = parse_value_("alignment.ctf.fit_tilt", table, true);
        alignment.ctf_fit_pitch = parse_value_("alignment.ctf.fit_pitch", table, true);
        alignment.ctf_fit_phase_shift = parse_value_("alignment.ctf.fit_phase_shift", table, false);
        alignment.ctf_fit_astigmatism = parse_value_("alignment.ctf.fit_astigmatism", table, true);
        alignment.ctf_fit_thickness = parse_value_("alignment.ctf.fit_thickness", table, false);
        alignment.ctf_patch_size_ang = parse_value_("alignment.ctf.patch_size_ang", table, 750.);
        alignment.ctf_patch_size_min_pix = parse_value_("alignment.ctf.patch_size_min_pix", table, 512);
        alignment.ctf_n_images_in_initial_average = parse_value_("alignment.ctf.n_images_in_initial_average", table, 3);
        alignment.ctf_resolution_range = parse_values_("alignment.ctf.resolution_range", table, Vec{30., 4.});

        alignment.refine_run = parse_value_("alignment.refine.run", table, true);
        alignment.refine_correct_ctf = parse_value_("alignment.refine.correct_ctf", table, true);
        alignment.refine_fit_rotation = parse_value_("alignment.refine.fit_rotation", table, true);
        alignment.refine_fit_tilt = parse_value_("alignment.refine.fit_tilt", table, true);
        alignment.refine_fit_pitch = parse_value_("alignment.refine.fit_pitch", table, true);
        alignment.refine_fit_thickness = parse_value_("alignment.refine.fit_thickness", table, true);

        alignment.refine_phase_flip_strength = parse_value_("alignment.refine.phase_flip_strength", table, 8.);
        check(alignment.refine_phase_flip_strength >= 0 and alignment.refine_phase_flip_strength <= 10,
              "postprocessing:tomogram_phase_flip_strength should be between 0 and 10, but got {}",
              alignment.refine_phase_flip_strength);

        return alignment;
    }

    auto parse_postprocessing_(const toml::table& table) -> Settings::PostProcessing {
        Settings::PostProcessing postprocessing;
        postprocessing.run = parse_value_("postprocessing.run", table, true);
        postprocessing.resolution = parse_value_("postprocessing.resolution", table, -1.);

        postprocessing.stack_run = parse_value_("postprocessing.stack.run", table, false);
        postprocessing.stack_correct_rotation = parse_value_("postprocessing.stack.correct_rotation", table, true);
        postprocessing.stack_interpolation = parse_interp("postprocessing.stack.interpolation", table, "linear");
        postprocessing.stack_dtype = parse_dtype("postprocessing.stack.dtype", table, "f32");

        postprocessing.tomogram_run = parse_value_("postprocessing.tomogram.run", table, true);
        postprocessing.tomogram_correct_rotation = parse_value_("postprocessing.tomogram.correct_rotation", table, true);
        postprocessing.tomogram_interpolation = parse_interp("postprocessing.tomogram.interpolation", table, "linear");
        postprocessing.tomogram_dtype = parse_dtype("postprocessing.tomogram.dtype", table, "f32");
        postprocessing.tomogram_oversampling_factor = parse_value_("postprocessing.tomogram.oversampling_factor", table, 2);
        postprocessing.tomogram_ramp_filter = parse_value_("postprocessing.tomogram.ramp_filter", table, true);
        postprocessing.tomogram_correct_ctf = parse_value_("postprocessing.tomogram.correct_ctf", table, true);

        postprocessing.tomogram_algorithm = parse_value_("postprocessing.tomogram.algorithm", table, std::string("fourier-wbp"));
        check(postprocessing.tomogram_algorithm == "fourier-wbp" or postprocessing.tomogram_algorithm == "real-bp",
              "postprocessing.tomogram_algorithm should be 'fourier-wbp' or 'real-bp', but got '{}'",
              postprocessing.tomogram_algorithm);

        postprocessing.tomogram_z_padding_percent = parse_value_("postprocessing.tomogram.z_padding_percent", table, 10.);
        check(postprocessing.tomogram_z_padding_percent >= 0 and postprocessing.tomogram_z_padding_percent <= 200,
              "postprocessing:tomogram_z_padding_percent should be between 0 and 200, but got {}",
              postprocessing.tomogram_z_padding_percent);

        postprocessing.tomogram_phase_flip_strength = parse_value_("postprocessing.tomogram.phase_flip_strength", table, 8.);
        check(postprocessing.tomogram_phase_flip_strength >= 0 and postprocessing.tomogram_phase_flip_strength <= 10,
              "postprocessing:tomogram_phase_flip_strength should be between 0 and 10, but got {}",
              postprocessing.tomogram_phase_flip_strength);

        return postprocessing;
    }

    auto parse_compute_(const toml::table& table, const cxxopts::ParseResult& cl) -> Settings::Compute {
        Settings::Compute compute;

        // device
        std::string device_name;
        if (auto device = table.at_path("compute.device")) {
            device.visit([&]<typename T>(T&& value) {
                if constexpr (toml::is_string<T>) {
                    if (*value == "auto") {
                        if (Device::is_any_gpu()) {
                            // Get all the available GPUs, placing the freest first.
                            compute.devices = Device::all(Device::GPU);
                            const auto most_free = Device::most_free_gpu();
                            stdr::stable_partition(compute.devices, [most_free](Device id) {
                                return id == most_free;
                            });
                        } else {
                            compute.devices.emplace_back("cpu");
                        }
                    } else {
                        compute.devices.emplace_back(*value); // let Device do the parsing
                    }
                } else if constexpr (toml::is_array<T>) {
                    for (auto&& item: value) {
                        auto result = item.template value<std::string>();
                        check(result.has_value(), "compute.device={} is not convertible to a string", device);
                        compute.devices.emplace_back(result.value()); // let Device do the parsing
                    }
                } else {
                    panic("compute.device={} is not convertible to a string or an array of strings", device);
                }
            });
        }

        if (const auto result = parse_value_<i32>("compute.n_threads", table))
            compute.n_threads = result.value();
        else
            compute.n_threads = noa::clamp(static_cast<i32>(noa::cpu::Device::cores().logical), 1, 16);

        if (const auto result = parse_value_<std::string>("compute.log_level", table)) {
            compute.log_level = result.value();
            constexpr auto valid_levels = std::array{"off", "error", "warn", "status", "info", "trace", "debug"};
            check(stdr::find(valid_levels, compute.log_level) != valid_levels.end(),
                  "compute.log_level={} is not valid. Should be {}",
                  compute.log_level, valid_levels);
        } else {
            compute.log_level = "trace";
        }

        compute.register_stack = parse_value_("compute.register_stack", table, true);
        compute.dry = parse_value_("compute.dry", table, "dry", cl, true);
        compute.stop_at_first_error = parse_value_("compute.stop_at_first_error", table, true);

        return compute;
    }
}

namespace qn {
    auto Settings::parse(int argc, const char* const* argv) -> std::vector<Series> {
        auto options = cxxopts::Options("quinoa", "Tilt-series alignment software.");
        options.add_options("data")
        ("mdocs", ".mdoc files or directories containing .mdoc files. Overwrites settings.data.mdocs.", cxxopts::value<std::vector<Path>>(), "pattern")
        ("stacks", "MRC or TIFF files containing the tilt images. If files are entered, they are each assigned to a mdoc based on their basename. If directories are entered, files with extension .mrc, .mrcs, or .st are searched and matched to a mdoc based on their basename. Overwrites settings.data.stacks.", cxxopts::value<std::vector<Path>>(), "pattern")
        ("rawtlts", "Files containing the tilt angles (one per line). If files are entered, they are each assigned to a mdoc based on their basename. If directories are entered, files with extension .rawtlt or .tlt are searched and matched to a mdoc based on their basename. Tilt angles should match the mdoc TiltAngle (within 0.1 degree). If not provided, stacks are assumed to be sorted in ascending order by their tilt angles and are required to have the same number of tilts as the mdoc. Overwrites settings.data.rawtlts.", cxxopts::value<std::vector<Path>>(), "pattern")
        ("tilt-axis", "Tilt-axis angle of the stacks (as specified in IMOD). If not provided, it is taken from the mdocs (not recommended). If unknown, see settings.alignment.coarse.check_rotation. Overwrites settings.experiment.tilt_axis.", cxxopts::value<f64>(), "float");

        options.add_options("general")
        ("output", "Base directory where the per-stack output directories are saved. Directories will be created if they don't exist. Defaults to the current working directory. Overwrites settings.data.output.", cxxopts::value<Path>(), "dir")
        ("settings", "TOML file containing the settings. Command-line arguments may overwrite settings from this file.", cxxopts::value<Path>(), "file");

        options.add_options("special")
        ("h,help", "Print this help message and exit.")
        ("dry", "Print an overview of the data about to be processed and exit.", cxxopts::value<bool>()->implicit_value("true"))
        ("init", "Generate a TOML file with all of the settings and exit.", cxxopts::value<bool>()->implicit_value("true"));

        cxxopts::ParseResult cl;
        try {
            cl = options.parse(argc, argv);
            if (cl.contains("help")) {
                fmt::println(fmt::runtime(options.help()));
                return {};
            } else if (not cl.unmatched().empty()) {
                panic("Invalid command line arguments: {}. Use --help.", cl.unmatched());
            } else if (cl.contains("init") and cl["init"].as<bool>()) {
                panic("TODO generate settings file");
            }
        } catch (...) {
            panic("Failed to parse the command line arguments");
        }

        auto series = std::vector<Series>{};
        try {
            auto settings = toml::table{};
            if (cl.contains("settings")) {
                settings = toml::parse_file(cl["settings"].as<Path>().native());
                sanitize_table_(settings);
            }
            series = parse_data_(settings, cl);
            experiment = parse_experiment_(settings, cl);
            preprocessing = parse_preprocessing_(settings);
            alignment = parse_alignment_(settings);
            postprocessing = parse_postprocessing_(settings);
            compute = parse_compute_(settings, cl);
        } catch (...) {
            panic("Failed to parse the settings file");
        }
        return series;
    }
}
