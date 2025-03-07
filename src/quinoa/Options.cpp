#include <filesystem>
#include <optional>
#include <string_view>

#include <noa/Session.hpp>

#include "quinoa/Metadata.hpp"
#include "quinoa/Options.hpp"
#include "quinoa/YAML.hpp"

// TODO I hate this. Search for some kind of C++26 CLA-reflection alternative available in C++20.

namespace {
    using namespace qn;

    template<size_t N>
    void sanitize_node_(YAML::Node node, const std::array<const char*, N>& expected) {
        std::string name;
        const auto predicate = [&name](const char* expected_name) {
            return std::string_view(expected_name) == name;
        };

        // First, make sure the entries are all recognized.
        for (const auto& e: node) {
            name = e.first.as<std::string>();
            if (std::find_if(expected.begin(), expected.end(), predicate) == expected.end())
                panic("Invalid parameter: {}", name);
        }

        // Then, add the missing entries.
        for (const auto& e: expected)
            if (not node[e].IsDefined())
                node[e] = YAML::Null;
    }

    template<typename T>
    auto parse_scalar_(
        std::string_view prefix,
        const YAML::Node& node,
        const std::string& parameter_name,
        T fallback,
        bool is_optional
    ) -> T {
        const auto parameter_node = node[parameter_name];
        if (parameter_node.IsScalar()) {
            return parameter_node.as<T>(fallback);
        } else if (parameter_node.IsNull()) {
            check(is_optional, "{}{} is not optional", prefix, parameter_name);
            return fallback;
        } else {
            panic("{}{} has an invalid type ({})", prefix, parameter_name, parameter_node.Type());
        }
    };

    auto parse_files_(YAML::Node files_node) -> Options::Files {
        constexpr std::array COMPONENTS{
            "input_directory",
            "input_stack",
            "input_tilt",
            "input_exposure",
            "input_csv",
            "output_directory",
        };
        sanitize_node_(files_node, COMPONENTS);

        Options::Files files;

        // input_directory
        Path input_directory;
        const auto input_directory_node = files_node["input_directory"];
        if (input_directory_node.IsScalar()) {
            input_directory = files_node["input_directory"].as<Path>();
            check(fs::is_directory(input_directory),
                  "files:input_directory={}. Does not refer to an existing directory",
                  input_directory);
        } else if (input_directory_node.IsNull()) {
            input_directory = fs::current_path();
        } else {
            panic("files:input_directory has an invalid type ({}). "
                  "It should be a path to an existing directory or be left empty (defaulting on the cwd)",
                  input_directory_node.Type());
        }
        const std::string basename = *(--(input_directory / "").parent_path().end());

        // input_stack
        const auto input_stack_node = files_node["input_stack"];
        if (input_stack_node.IsScalar()) {
            files.input_stack = input_stack_node.as<Path>();
            check(fs::is_regular_file(files.input_stack),
                  "files:input_stack={}. File does not exist or permissions to read are not granted",
                  files.input_stack);
        } else if (input_stack_node.IsNull()) {
            // search within the input directory
            const auto get_if_file_exists = [&](const Path& filename) {
                Path stack_filename = filename;
                if (fs::is_regular_file(stack_filename))
                    return stack_filename;
                return Path();
            };

            // Try to find the input stack.
            using namespace std::string_literals;
            for (auto& e: std::array{".st"s, ".mrc"s, ".mrcs"s}) {
                auto input_stack = get_if_file_exists(input_directory / (basename + e));
                if (not input_stack.empty()) {
                    check(files.input_stack.empty(),
                          "Multiple files within the input directory fits the pattern of the input stack. "
                          "Found: {} and {}", files.input_stack.filename(), input_stack.filename());
                    files.input_stack = input_stack;
                }
            }
            check(not files.input_stack.empty(),
                  "Could not find the input stack {}.(st|mrc|mrcs) in the input directory {}",
                  basename, input_directory);
        } else {
            panic("files:input_stack has an invalid type ({}). "
                  "It should be a path to an existing file or be left empty",
                  input_stack_node.Type());
        }

        // Optional input files.
        const auto get_optional_file = [&](
            const std::string& parameter_name,
            std::string_view extensions,
            std::source_location location = std::source_location::current()
        ) {
            Path file;
            const auto parameter_node = files_node[parameter_name];
            if (parameter_node.IsScalar()) {
                // If a value was passed, check that it's a valid path.
                file = parameter_node.as<Path>();
                if (not fs::is_regular_file(file)) {
                    panic_at_location(
                        location,
                        "files:{}={}. File does not exist or permissions to read are not granted",
                        parameter_name, file);
                }
            } else if (parameter_node.IsNull()) {
                // Otherwise, fallback to the input directory.
                for (auto extension: extensions | stdv::split(',')) {
                    const Path filename = input_directory / fmt::format("{}.{}", basename, extension);
                    if (fs::is_regular_file(filename)) {
                        file = filename;
                        break;
                    }
                }
            } else {
                panic("files:{} has an invalid type ({}). "
                      "It should be a path to an existing file or be left empty",
                      parameter_name, parameter_node.Type());
            }
            return file;
        };
        files.input_csv = get_optional_file("input_csv", "csv");
        files.input_tilt = get_optional_file("input_tilt", "tlt,tilt");
        files.input_exposure = get_optional_file("input_exposure", "exp,exposure");

        // output_directory
        const auto output_directory_node = files_node["output_directory"];
        if (output_directory_node.IsScalar()) {
            files.output_directory = output_directory_node.as<Path>();
        } else if (output_directory_node.IsNull()) {
            files.output_directory = fs::current_path();
        } else {
            panic("files:output_directory has an invalid type ({}). "
                  "It should be a convertible to a path or be left empty (defaulting on the cwd)",
                  output_directory_node.Type());
        }

        return files;
    }

    auto parse_experiment_(YAML::Node experiment_node) -> Options::Experiment {
        constexpr std::array QN_OPTIONS_EXPERIMENT{
            "collection_order",
            "rotation_offset",
            "tilt_offset",
            "pitch_offset",
            "voltage",
            "amplitude",
            "cs",
            "phase_shift",
            "astigmatism_value",
            "astigmatism_angle",
            "thickness",
        };
        constexpr std::array QN_OPTIONS_EXPERIMENT_COLLECTION_ORDER{
            "starting_angle",
            "starting_direction",
            "tilt_increment",
            "group_of",
            "exclude_start",
            "per_view_exposure",
        };

        sanitize_node_(experiment_node, QN_OPTIONS_EXPERIMENT);
        if (experiment_node["collection_order"].IsMap())
            sanitize_node_(experiment_node["collection_order"], QN_OPTIONS_EXPERIMENT_COLLECTION_ORDER);

        const auto parse_collection_order_to_optional = [&](
            const YAML::Node& order_node,
            std::source_location location = std::source_location::current()
        )-> std::optional<Options::Experiment::CollectionOrder> {
            if (order_node.IsMap()) {
                // Every parameter should be specified.
                for (auto parameter: QN_OPTIONS_EXPERIMENT_COLLECTION_ORDER) {
                    if (not order_node[parameter].IsScalar())
                        panic_at_location(location, "experiment:collection_order:{} has an invalid type ({})",
                                          parameter, order_node[parameter].Type());
                }
                return Options::Experiment::CollectionOrder{
                    order_node["starting_angle"].as<f64>(),
                    order_node["starting_direction"].as<i64>(),
                    order_node["tilt_increment"].as<f64>(),
                    order_node["group_of"].as<i64>(),
                    order_node["exclude_start"].as<bool>(),
                    order_node["per_view_exposure"].as<f64>(),
                };
            } else if (not order_node.IsNull()) {
                panic_at_location(
                    location,
                    "experiment:collection_order has an invalid type ({}). Should be a map or be left empty",
                    order_node.Type());
            } else {
                return std::nullopt;
            }
        };

        // Use max() to say that the angle offsets are not specified.
        constexpr f64 UNSPECIFIED_VALUE = std::numeric_limits<f64>::max();
        Options::Experiment experiment;
        experiment.collection_order = parse_collection_order_to_optional(experiment_node["collection_order"]);
        experiment.rotation_offset = parse_scalar_<f64>("experiment:", experiment_node, "rotation_offset", UNSPECIFIED_VALUE, true);
        experiment.tilt_offset = parse_scalar_<f64>("experiment:", experiment_node, "tilt_offset", UNSPECIFIED_VALUE, true);
        experiment.pitch_offset = parse_scalar_<f64>("experiment:", experiment_node, "pitch_offset", UNSPECIFIED_VALUE, true);
        experiment.voltage = parse_scalar_<f64>("experiment:", experiment_node, "voltage", 300., true);
        experiment.amplitude = parse_scalar_<f64>("experiment:", experiment_node, "amplitude", 0.07, true);
        experiment.cs = parse_scalar_<f64>("experiment:", experiment_node, "cs", 2.7, true);
        experiment.phase_shift = parse_scalar_<f64>("experiment:", experiment_node, "phase_shift", 0., true);
        experiment.astigmatism_value = parse_scalar_<f64>("experiment:", experiment_node, "astigmatism_value", 0., true);
        experiment.astigmatism_angle = parse_scalar_<f64>("experiment:", experiment_node, "astigmatism_angle", 0., true);
        experiment.thickness = parse_scalar_<f64>("experiment:", experiment_node, "thickness", UNSPECIFIED_VALUE, true);

        // Angle range (this is optional).
        experiment.phase_shift = MetadataSlice::to_angle_range(experiment.phase_shift);
        experiment.astigmatism_angle = MetadataSlice::to_angle_range(experiment.astigmatism_angle);

        // Sanitize.
        check(experiment.voltage >= 50 and experiment.voltage <= 450,
              "experiment::voltage={} (kV). Value is not supported",
              experiment.voltage);
        check(experiment.amplitude >= 0 and experiment.amplitude <= 0.2,
              "experiment::amplitude={} (fraction). Value is not supported",
              experiment.amplitude);
        check(experiment.cs >= 0 and experiment.cs <= 4,
              "experiment::cs={} (micrometers). Value is not supported",
              experiment.cs);
        check(experiment.phase_shift >= 0 and experiment.phase_shift <= 45,
              "experiment::phase_shift={} (degrees). Value is not supported",
              experiment.phase_shift);
        check(experiment.astigmatism_value >= 0 and experiment.astigmatism_value <= 0.5,
              "experiment::astigmatism_value={} (micrometers). Value is not supported",
              experiment.astigmatism_value);
        check(experiment.thickness == UNSPECIFIED_VALUE or
              (experiment.thickness > 20. and experiment.thickness < 400.),
              "experiment.thickness={}nm. Value is not supported",
              experiment.thickness);

        return experiment;
    }

    auto parse_preprocessing_(YAML::Node preprocessing_node) -> Options::Preprocessing {
        constexpr std::array QN_OPTIONS_PREPROCESSING{
            "run",
            "exclude_blank_views",
            "exclude_view_indexes",
        };
        sanitize_node_(preprocessing_node, QN_OPTIONS_PREPROCESSING);

        Options::Preprocessing preprocessing;
        preprocessing.run = parse_scalar_<bool>("preprocessing:", preprocessing_node, "run", true, true);
        preprocessing.exclude_blank_views = parse_scalar_<bool>("preprocessing:", preprocessing_node,
                                                                "exclude_blank_views", true, true);

        const YAML::Node exclude_view_indexes_node = preprocessing_node["exclude_view_indexes"];
        if (exclude_view_indexes_node.IsSequence())
            preprocessing.exclude_view_indexes = exclude_view_indexes_node.as<std::vector<i64>>();
        else if (exclude_view_indexes_node.IsScalar())
            preprocessing.exclude_view_indexes.emplace_back(exclude_view_indexes_node.as<i64>());
        else if (!exclude_view_indexes_node.IsNull()) {
            panic("preprocessing:exclude_view_indexes has an invalid type ({}). Should be a scalar or a sequence",
                  exclude_view_indexes_node.Type());
        }

        return preprocessing;
    }

    auto parse_alignment_(YAML::Node alignment_node) -> Options::Alignment {
        constexpr std::array QN_OPTIONS_ALIGNMENT{
            "run",
            "fit_rotation_offset",
            "fit_tilt_offset",
            "fit_pitch_offset",
            "fit_phase_shift",
            "fit_astigmatism",
            "fit_thickness",
            "do_coarse_alignment",
            "do_ctf_alignment",
            "do_refine_alignment",
        };
        sanitize_node_(alignment_node, QN_OPTIONS_ALIGNMENT);

        Options::Alignment alignment;
        alignment.run = parse_scalar_<bool>("alignment:", alignment_node, "run", true, true);
        alignment.fit_rotation_offset = parse_scalar_<bool>("alignment:", alignment_node, "fit_rotation_offset", true, true);
        alignment.fit_tilt_offset = parse_scalar_<bool>("alignment:", alignment_node, "fit_tilt_offset", true, true);
        alignment.fit_pitch_offset = parse_scalar_<bool>("alignment:", alignment_node, "fit_pitch_offset", true, true);
        alignment.fit_phase_shift = parse_scalar_<bool>("alignment:", alignment_node, "fit_phase_shift", false, true);
        alignment.fit_astigmatism = parse_scalar_<bool>("alignment:", alignment_node, "fit_astigmatism", false, true);
        alignment.fit_thickness = parse_scalar_<bool>("alignment:", alignment_node, "fit_thickness", true, true);

        alignment.do_coarse_alignment = parse_scalar_<bool>("alignment:", alignment_node, "do_coarse_alignment", true, true);
        alignment.do_ctf_alignment = parse_scalar_<bool>("alignment:", alignment_node, "do_ctf_alignment", true, true);
        alignment.do_refine_alignment = parse_scalar_<bool>("alignment:", alignment_node, "do_refine_alignment", true, true);
        return alignment;
    }

    auto parse_postprocessing_(YAML::Node postprocessing_node) -> Options::PostProcessing {
        constexpr std::array QN_OPTIONS_POSTPROCESSING{
            "run",
            "resolution",
            "save_aligned_stack",
            "reconstruct_tomogram",
            "reconstruct_mode",
            "reconstruct_weighting",
            "reconstruct_z_padding",
            "save_aligned_stack"
        };
        sanitize_node_(postprocessing_node, QN_OPTIONS_POSTPROCESSING);

        const auto parse_bool_parameter = [&](
            const std::string& parameter_name, bool fallback,
            std::source_location location = std::source_location::current()
        ) -> bool {
            const auto parameter_node = postprocessing_node[parameter_name];
            if (parameter_node.IsScalar())
                return parameter_node.as<bool>(fallback);
            else if (parameter_node.IsNull())
                return fallback;
            else {
                panic_at_location(location,
                                  "postprocessing:{} has an invalid type ({}). Should be a boolean",
                                  parameter_name, parameter_node.Type());
            }
        };

        Options::PostProcessing postprocessing;
        postprocessing.run = parse_bool_parameter("run", true);
        postprocessing.save_aligned_stack = parse_bool_parameter("save_aligned_stack", false);
        postprocessing.reconstruct_tomogram = parse_bool_parameter("reconstruct_tomogram", true);

        const auto resolution_node = postprocessing_node["resolution"];
        if (resolution_node.IsScalar()) {
            postprocessing.resolution = resolution_node.as<f64>();
        } else if (resolution_node.IsNull()) {
            postprocessing.resolution = -1;
        } else {
            panic("postprocessing:resolution has an invalid type ({}). "
                  "Should be a scalar or be left emtpy", resolution_node.Type());
        }

        const auto reconstruct_mode_node = postprocessing_node["reconstruct_mode"];
        if (reconstruct_mode_node.IsScalar()) {
            using namespace noa::string;
            postprocessing.reconstruct_mode = to_lower(trim(reconstruct_mode_node.as<std::string>()));
            check(postprocessing.reconstruct_mode == "fourier" or postprocessing.reconstruct_mode == "real",
                  R"(postprocessing:reconstruct_mode should be "fourier" or "real", but got {})",
                  postprocessing.reconstruct_mode);
        } else if (reconstruct_mode_node.IsNull()) {
            postprocessing.reconstruct_mode = "real";
        } else {
            panic("postprocessing:reconstruct_mode has an invalid type ({}). "
                  "Should be a string or be left emtpy", reconstruct_mode_node.Type());
        }

        const auto reconstruct_weighting_node = postprocessing_node["reconstruct_weighting"];
        if (reconstruct_weighting_node.IsScalar()) {
            using namespace noa::string;
            postprocessing.reconstruct_weighting = to_lower(trim(reconstruct_weighting_node.as<std::string>()));
            check(postprocessing.reconstruct_weighting == "fourier" or
                  postprocessing.reconstruct_weighting == "radial" or
                  starts_with(postprocessing.reconstruct_weighting, "sirt-"),
                  R"(postprocessing:reconstruct_weighting should be "fourier", "radial" or "sirt-n" (where n is a positive integer), but got {})",
                  postprocessing.reconstruct_weighting);
            check(postprocessing.reconstruct_mode == "fourier" or postprocessing.reconstruct_weighting != "fourier",
                  "Fourier weighting is currently only supported with reconstruct_mode=fourier");
        } else if (reconstruct_weighting_node.IsNull()) {
            postprocessing.reconstruct_weighting = postprocessing.reconstruct_mode == "fourier" ? "fourier" : "radial";
        } else {
            panic("postprocessing:reconstruct_weighting has an invalid type ({}). "
                  "Should be a string or be left emtpy", reconstruct_weighting_node.Type());
        }

        const auto reconstruct_z_padding_node = postprocessing_node["reconstruct_z_padding"];
        if (reconstruct_z_padding_node.IsScalar()) {
            postprocessing.reconstruct_z_padding = reconstruct_z_padding_node.as<f64>();
            check(postprocessing.reconstruct_z_padding >= 0 and postprocessing.reconstruct_z_padding <= 100,
                  "postprocessing:reconstruct_z_padding should be between 0 and 100, but got {}",
                  postprocessing.reconstruct_z_padding);
        } else if (reconstruct_z_padding_node.IsNull()) {
            postprocessing.reconstruct_z_padding = 10;
        } else {
            panic("postprocessing:reconstruct_z_padding has an invalid type ({}). "
                  "Should be a scalar or be left emtpy", reconstruct_z_padding_node.Type());
        }

        return postprocessing;
    }

    auto parse_compute_(const YAML::Node& compute_node) -> Options::Compute {
        constexpr std::array COMPONENTS{
            "device",
            "n_cpu_threads",
            "register_input_stack",
            "log_level",
        };
        sanitize_node_(compute_node, COMPONENTS);

        Options::Compute compute;

        // device
        YAML::Node device_node = compute_node["device"];
        std::string device_name;
        if (device_node.IsNull()) {
            device_name = "auto";
        } else if (device_node.IsScalar()) {
            device_name = noa::string::to_lower(noa::string::trim(device_node.as<std::string>()));
        } else {
            panic("compute:device has an invalid type ({}). "
                  "Should be a scalar or be left emtpy (defaulting to \"auto\")",
                  device_node.Type());
        }

        if (device_name == "auto") {
            compute.device = Device::is_any_gpu() ? Device::most_free_gpu() : "cpu";
        } else {
            // Let the device parsing do its thing and possibly throw...
            compute.device = Device(device_name);
        }

        // n_cpu_threads
        const auto n_cpu_threads_node = compute_node["n_cpu_threads"];
        if (n_cpu_threads_node.IsScalar()) {
            compute.n_cpu_threads = n_cpu_threads_node.as<i64>();
        } else if (n_cpu_threads_node.IsNull()) {
            compute.n_cpu_threads = std::min(static_cast<i64>(noa::cpu::Device::cores().logical), i64{16});
        } else {
            panic("compute:n_cpu_threads has an invalid type ({}). Should be a scalar or be left emtpy",
                  device_node.Type());
        }

        // register_input_stack
        const auto register_input_stack_node = compute_node["register_input_stack"];
        if (register_input_stack_node.IsNull()) {
            compute.register_input_stack = true;
        } else if (register_input_stack_node.IsScalar()) {
            compute.register_input_stack = register_input_stack_node.as<bool>(true);
        } else {
            panic("compute:register_input_stack has an invalid type ({}). "
                  "Should be a boolean or be left emtpy (defaulting to \"true\")",
                  device_node.Type());
        }

        // log_level
        const auto log_level_node = compute_node["log_level"];
        if (log_level_node.IsNull()) {
            compute.log_level = "info";
        } else if (log_level_node.IsScalar()) {
            const auto level = noa::string::to_lower(noa::string::trim(compute_node["log_level"].as<std::string>()));
            constexpr std::array valid_levels{"off", "error", "warn", "status", "info", "trace", "debug"};
            check(std::ranges::find(valid_levels, level) != valid_levels.end(),
                  "compute:log_level={} is not valid. Should be {}",
                  level, valid_levels);
            compute.log_level = level;
        } else {
            panic("compute:log_level has an invalid type ({}). "
                  "Should be a scalar or be left emtpy (defaulting to \"info\")",
                  device_node.Type());
        }
        return compute;
    }
}

namespace qn {
    Options::Options(int argc, char** argv) {
        check(argc == 2, "Incorrect number of arguments. Enter the filename of the YAML parameter file");

        try {
            YAML::Node node = YAML::LoadFile(argv[1]);

            constexpr std::array COMPONENTS{
                "files",
                "experiment",
                "preprocessing",
                "alignment",
                "postprocessing",
                "compute"
            };
            sanitize_node_(node, COMPONENTS);

            files = parse_files_(node["files"]);
            experiment = parse_experiment_(node["experiment"]);
            preprocessing = parse_preprocessing_(node["preprocessing"]);
            alignment = parse_alignment_(node["alignment"]);
            postprocessing = parse_postprocessing_(node["postprocessing"]);
            compute = parse_compute_(node["compute"]);
        } catch (...) {
            panic("Failed to parse the parameters");
        }

        // Addition program checks.
        constexpr auto MAX = std::numeric_limits<f64>::max();
        check(not noa::allclose(experiment.rotation_offset, MAX) or alignment.fit_rotation_offset,
              "experiment.rotation_offset: no initial estimate is provided and the rotation offset alignment is turned off");
        check(not noa::allclose(experiment.thickness, MAX) or alignment.fit_thickness,
              "experiment.thickness: no initial estimate is provided and the thickness estimate is turned off");
    }
}
