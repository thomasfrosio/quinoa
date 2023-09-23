#include <filesystem>
#include <optional>
#include <string_view>
#include <noa/unified/Device.hpp>

#include "quinoa/Exception.h"
#include "quinoa/io/Options.h"
#include "quinoa/io/YAML.h"
#include "quinoa/core/Metadata.h"

namespace {
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
                QN_THROW("Invalid parameter: {}", name);
        }

        // Then, add the missing entries.
        for (const auto& e: expected)
            if (!node[e].IsDefined())
                node[e] = YAML::Null;
    }

    using namespace qn;

    auto parse_files_(YAML::Node files_node) -> Options::Files {
        constexpr std::array QN_OPTIONS_FILES{
                "input_directory",
                "input_stack",
                "input_tlt",
                "input_exposure",
                "output_directory",
        };
        sanitize_node_(files_node, QN_OPTIONS_FILES);

        Options::Files files;

        // input_directory
        Path input_directory;
        const auto input_directory_node = files_node["input_directory"];
        if (input_directory_node.IsScalar()) {
            input_directory = files_node["input_directory"].as<Path>();
            QN_CHECK(fs::is_directory(input_directory),
                     "files:input_directory={}. Does not refer to an existing directory",
                     input_directory);

        } else if (input_directory_node.IsNull()) {
            input_directory = fs::current_path();

        } else {
            QN_THROW("files:input_directory has an invalid type ({}). "
                     "It should be a path to an existing directory or be left empty (defaulting on the cwd)",
                     input_directory_node.Type());
        }
        const std::string basename = *(--(input_directory / "").parent_path().end());

        // input_stack
        const auto input_stack_node = files_node["input_stack"];
        if (input_stack_node.IsScalar()) {
            files.input_stack = input_stack_node.as<Path>();
            QN_CHECK(fs::is_regular_file(files.input_stack),
                     "files:input_stack={}. File does not exist or permissions to read are not granted",
                     files.input_stack);

        } else if (input_stack_node.IsNull()) { // search within the input directory
            const auto get_if_file_exists = [&](const Path& filename) {
                Path stack_filename = filename;
                if (fs::is_regular_file(stack_filename))
                    return stack_filename;
                return Path();
            };

            // Try to find the input stack.
            for (auto& e: std::array<std::string, 3>{".st", ".mrc", ".mrcs"}) {
                auto input_stack = get_if_file_exists(input_directory / (basename + e));
                if (!input_stack.empty()) {
                    QN_CHECK(files.input_stack.empty(),
                             "Multiple files within the input directory fits the pattern of the input stack. "
                             "Found: {} and {}", files.input_stack.filename(), input_stack.filename());
                    files.input_stack = input_stack;
                }
            }
            QN_CHECK(!files.input_stack.empty(),
                     "Could not find the input stack {}.(st|mrc|mrcs) in the input directory {}",
                     basename, input_directory);

        } else {
            QN_THROW("files:input_stack has an invalid type ({}). "
                     "It should be a path to an existing file or be left empty",
                     input_stack_node.Type());
        }

        // Optional input files.
        const auto get_optional_file = [&](const std::string& parameter_name, std::string_view extension) {
            Path file;
            const auto parameter_node = files_node[parameter_name];
            if (parameter_node.IsScalar()) {
                // If a value was passed, check that it's a valid path.
                file = parameter_node.as<Path>();
                if (!fs::is_regular_file(file)) {
                    QN_THROW_FUNC("parse_files_",
                                  "files:{}={}. File does not exist or permissions to read are not granted",
                                  parameter_name, file);
                }
            } else if (parameter_node.IsNull()) {
                // Otherwise, fallback to the input directory.
                const Path filename = input_directory / noa::string::format("{}.{}", basename, extension);
                if (fs::is_regular_file(filename))
                    file = filename;
            } else {
                QN_THROW("files:{} has an invalid type ({}). "
                         "It should be a path to an existing file or be left empty",
                         parameter_name, parameter_node.Type());
            }
            return file;
        };
        files.input_tlt = get_optional_file("input_tlt", "tlt");
        files.input_exposure = get_optional_file("input_exposure", "exposure");

        // output_directory
        const auto output_directory_node = files_node["output_directory"];
        if (output_directory_node.IsScalar()) {
            files.output_directory = output_directory_node.as<Path>();
        } else if (output_directory_node.IsNull()) {
            files.output_directory = fs::current_path();
        } else {
            QN_THROW("files:output_directory has an invalid type ({}). "
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
                "elevation_offset",
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

        const auto parse_collection_order_to_optional = [&](const YAML::Node& order_node)
                -> std::optional<Options::Experiment::CollectionOrder> {
            if (order_node.IsMap()) {
                // Every parameter should be specified.
                for (auto parameter: QN_OPTIONS_EXPERIMENT_COLLECTION_ORDER) {
                    if (!order_node[parameter].IsScalar())
                        QN_THROW_FUNC("parse_experiment_", "experiment:collection_order:{} has an invalid type ({})",
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
            } else if (!order_node.IsNull()) {
                QN_THROW_FUNC("parse_experiment_",
                              "experiment:collection_order has an invalid type ({}). Should be a map or be left empty",
                              order_node.Type());
            } else {
                return std::nullopt;
            }
        };

        const auto parse_parameter = [&](const std::string& parameter_name, f64 fallback) -> f64 {
            const auto parameter_node = experiment_node[parameter_name];
            if (parameter_node.IsScalar())
                return parameter_node.as<f64>(fallback);
            else if (parameter_node.IsNull())
                return fallback;
            else {
                QN_THROW_FUNC("parse_tilt_scheme_",
                              "tilt_scheme:{} has an invalid type ({}). Should be a scalar",
                              parameter_name, parameter_node.Type());
            }
        };

        // Use max() to say that the angle offsets are not specified.
        constexpr f64 UNSPECIFIED_VALUE = std::numeric_limits<f64>::max();
        Options::Experiment experiment;
        experiment.collection_order = parse_collection_order_to_optional(experiment_node["collection_order"]);
        experiment.rotation_offset = parse_parameter("rotation_offset", UNSPECIFIED_VALUE);
        experiment.tilt_offset = parse_parameter("tilt_offset", UNSPECIFIED_VALUE);
        experiment.elevation_offset = parse_parameter("elevation_offset", UNSPECIFIED_VALUE);
        experiment.voltage = parse_parameter("voltage", 300.);
        experiment.amplitude = parse_parameter("amplitude", 0.07);
        experiment.cs = parse_parameter("cs", 2.7);
        experiment.phase_shift = parse_parameter("phase_shift", 0.);
        experiment.astigmatism_value = parse_parameter("astigmatism_value", 0.);
        experiment.astigmatism_angle = parse_parameter("astigmatism_angle", 0.);
        experiment.thickness = parse_parameter("thickness", UNSPECIFIED_VALUE);

        // Angle range (this is optional).
        experiment.phase_shift = MetadataSlice::to_angle_range(experiment.phase_shift);
        experiment.astigmatism_angle = MetadataSlice::to_angle_range(experiment.astigmatism_angle);

        // Sanitize.
        QN_CHECK(experiment.voltage >= 50 && experiment.voltage <= 450,
                 "experiment::voltage={} (kV). Value is not supported",
                 experiment.voltage);
        QN_CHECK(experiment.amplitude >= 0 && experiment.amplitude <= 0.2,
                 "experiment::amplitude={} (fraction). Value is not supported",
                 experiment.amplitude);
        QN_CHECK(experiment.cs >= 0 && experiment.cs <= 4,
                 "experiment::cs={} (micrometers). Value is not supported",
                 experiment.cs);
        QN_CHECK(experiment.phase_shift >= 0 && experiment.phase_shift <= 45,
                 "experiment::phase_shift={} (degrees). Value is not supported",
                 experiment.phase_shift);
        QN_CHECK(experiment.astigmatism_value >= 0 && experiment.astigmatism_value <= 0.5,
                 "experiment::astigmatism_value={} (micrometers). Value is not supported",
                 experiment.astigmatism_value);
        QN_CHECK(experiment.thickness == UNSPECIFIED_VALUE ||
                 (experiment.thickness > 20. && experiment.thickness < 400.),
                 "experiment.thickness={}nm. Value is not supported");

        return experiment;
    }

    auto parse_preprocessing_(YAML::Node preprocessing_node) -> Options::Preprocessing {
        constexpr std::array QN_OPTIONS_PREPROCESSING{
                "run",
                "exclude_blank_views",
                "exclude_view_indexes",
        };
        sanitize_node_(preprocessing_node, QN_OPTIONS_PREPROCESSING);

        const auto parse_parameter = [&](const std::string& parameter_name, bool fallback) -> bool {
            const auto parameter_node = preprocessing_node[parameter_name];
            if (parameter_node.IsScalar())
                return parameter_node.as<bool>(fallback);
            else if (parameter_node.IsNull())
                return fallback;
            else {
                QN_THROW_FUNC("parse_preprocessing_",
                              "preprocessing:{} has an invalid type ({}). Should be a boolean",
                              parameter_name, parameter_node.Type());
            }
        };

        Options::Preprocessing preprocessing;
        preprocessing.run = parse_parameter("run", true);
        preprocessing.exclude_blank_views = parse_parameter("exclude_blank_views", true);

        const YAML::Node exclude_view_indexes_node = preprocessing_node["exclude_view_indexes"];
        if (exclude_view_indexes_node.IsSequence())
            preprocessing.exclude_view_indexes = exclude_view_indexes_node.as<std::vector<i64>>();
        else if (exclude_view_indexes_node.IsScalar())
            preprocessing.exclude_view_indexes.emplace_back(exclude_view_indexes_node.as<i64>());
        else if (!exclude_view_indexes_node.IsNull()) {
            QN_THROW("preprocessing:exclude_view_indexes has an invalid type ({}). Should be a scalar or a sequence",
                     exclude_view_indexes_node.Type());
        }

        return preprocessing;
    }

    auto parse_alignment_(YAML::Node alignment_node) -> Options::Alignment {
        constexpr std::array QN_OPTIONS_ALIGNMENT{
                "run",
                "fit_rotation_offset",
                "fit_tilt_offset",
                "fit_elevation_offset",
                "fit_phase_shift",
                "fit_astigmatism",
                "use_initial_pairwise_alignment",
                "use_ctf_estimate",
                "use_thickness_estimate",
                "use_projection_matching",
        };
        sanitize_node_(alignment_node, QN_OPTIONS_ALIGNMENT);

        const auto parse_parameter = [&](const std::string& parameter_name, bool fallback) -> bool {
            const auto parameter_node = alignment_node[parameter_name];
            if (parameter_node.IsScalar())
                return parameter_node.as<bool>(fallback);
            else if (parameter_node.IsNull())
                return fallback;
            else {
                QN_THROW_FUNC("parse_alignment_",
                              "alignment:{} has an invalid type ({}). Should be a boolean",
                              parameter_name, parameter_node.Type());
            }
        };

        Options::Alignment alignment;
        alignment.run = parse_parameter("run", true);
        alignment.fit_rotation_offset = parse_parameter("fit_rotation_offset", true);
        alignment.fit_tilt_offset = parse_parameter("fit_tilt_offset", true);
        alignment.fit_elevation_offset = parse_parameter("fit_elevation_offset", true);
        alignment.fit_phase_shift = parse_parameter("fit_phase_shift", false);
        alignment.fit_astigmatism = parse_parameter("fit_astigmatism", false);
        alignment.use_initial_pairwise_alignment = parse_parameter("use_initial_pairwise_alignment", true);
        alignment.use_ctf_estimate = parse_parameter("use_ctf_estimate", true);
        alignment.use_thickness_estimate = parse_parameter("use_thickness_estimate", true);
        alignment.use_projection_matching = parse_parameter("use_projection_matching", true);
        return alignment;
    }

    auto parse_compute_(const YAML::Node& compute_node) -> Options::Compute {
        constexpr std::array QN_OPTIONS_COMPUTE{
                "device",
                "n_cpu_threads",
                "register_input_stack",
                "log_level",
        };
        sanitize_node_(compute_node, QN_OPTIONS_COMPUTE);

        Options::Compute compute;

        // device
        YAML::Node device_node = compute_node["device"];
        std::string device_name;
        if (device_node.IsNull()) {
            device_name = "auto";
        } else if (device_node.IsScalar()) {
            device_name = string::lower(string::trim(device_node.as<std::string>()));
        } else {
            QN_THROW("compute:device has an invalid type ({}). "
                     "Should be a scalar or be left emtpy (defaulting to \"auto\")",
                     device_node.Type());
        }

        if (device_name == "auto") {
            compute.device =
                    noa::Device::is_any(noa::DeviceType::GPU) ?
                    noa::Device::most_free(noa::DeviceType::GPU) :
                    noa::Device("cpu");

        } else if (string::starts_with(device_name, "gpu")){
            // TODO noa::Device::has_id(std::string_view)? cpu:1
            const size_t pos = device_name.find(':');
            if (pos != std::string::npos && pos != device_name.size() - 1)
                compute.device = noa::Device::most_free(noa::DeviceType::GPU);
            else
                compute.device = noa::Device(device_name);

        } else {
            // Let the device parsing do its thing and possibly throw...
            compute.device = noa::Device(device_name);
        }

        // n_cpu_threads
        const auto n_cpu_threads_node = compute_node["n_cpu_threads"];
        if (n_cpu_threads_node.IsScalar()) {
            compute.n_cpu_threads = n_cpu_threads_node.as<i64>();
        } else if (n_cpu_threads_node.IsNull()){
            compute.n_cpu_threads = std::min(static_cast<i64>(noa::cpu::Device::cores().logical), i64{16});
        } else {
            QN_THROW("compute:n_cpu_threads has an invalid type ({}). Should be a scalar or be left emtpy",
                     device_node.Type());
        }

        // register_input_stack
        const auto register_input_stack_node = compute_node["register_input_stack"];
        if (register_input_stack_node.IsNull()) {
            compute.register_input_stack = true;
        } else if (register_input_stack_node.IsScalar()) {
            compute.register_input_stack = register_input_stack_node.as<bool>(true);
        } else {
            QN_THROW("compute:register_input_stack has an invalid type ({}). "
                     "Should be a boolean or be left emtpy (defaulting to \"true\")",
                     device_node.Type());
        }

        // log_level
        const auto log_level_node = compute_node["log_level"];
        if (log_level_node.IsNull()) {
            compute.log_level = "info";
        } else if (log_level_node.IsScalar()) {
            const auto level = string::lower(string::trim(compute_node["log_level"].as<std::string>()));
            constexpr std::array valid_levels{"off", "error", "warn", "status", "info", "trace", "debug"};
            QN_CHECK(std::find(valid_levels.begin(), valid_levels.end(), level) != valid_levels.end(),
                     "compute:log_level={} is not valid. Should be {}",
                     level, valid_levels);
            compute.log_level = level;
        } else {
            QN_THROW("compute:log_level has an invalid type ({}). "
                     "Should be a scalar or be left emtpy (defaulting to \"info\")",
                     device_node.Type());
        }
        return compute;
    }
}

namespace qn {
    Options::Options(int argc, char** argv) {
        if (argc != 2)
            QN_THROW("Incorrect number of arguments. Enter the filename of the YAML parameter file");

        try {
            YAML::Node node = YAML::LoadFile(argv[1]);

            constexpr std::array QN_OPTIONS_{
                    "files",
                    "experiment",
                    "preprocessing",
                    "alignment",
                    "reconstruction",
                    "compute"
            };
            sanitize_node_(node, QN_OPTIONS_);

            files = parse_files_(node["files"]);
            experiment = parse_experiment_(node["experiment"]);
            preprocessing = parse_preprocessing_(node["preprocessing"]);
            alignment = parse_alignment_(node["alignment"]);
            // TODO reconstruction
            compute = parse_compute_(node["compute"]);

        } catch (...) {
            QN_THROW("Failed to parse the parameters");
        }
    }
}
