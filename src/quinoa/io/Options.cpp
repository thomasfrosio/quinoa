#include <noa/IO.hpp>

#include "quinoa/io/Options.h"
#include "quinoa/Exception.h"
#include "quinoa/io/YAML.h"

namespace {
    constexpr std::array<std::string_view, 35> QN_OPTIONS_ = {
            "preprocessing_run",
            "preprocessing_exclude_blank_views",
            "preprocessing_exclude_view_indexes",

            "alignment_run",
            "alignment_resolution",
            "alignment_rotation_offset",
            "alignment_tilt_offset",
            "alignment_elevation_offset",
            "alignment_pairwise_shift",
            "alignment_projection_matching",
            "alignment_projection_matching_shift_only",
            "alignment_save_input_stack",
            "alignment_save_aligned_stack",
            "alignment_save_aligned_stack_resolution",

            "refinement_run",

            "reconstruction_run",
            "reconstruction_resolution",
            "reconstruction_thickness",
            "reconstruction_cube_size",

            "output_directory",

            "stack_directory",
            "stack_file",
            "stack_mdoc",
            "stack_tlt",
            "stack_exposure",

            "rotation_angle",

            "order_starting_angle",
            "order_starting_direction",
            "order_angle_increment",
            "order_group",
            "order_exclude_start",
            "order_per_view_exposure",

            "compute_cpu_threads",
            "compute_device",
            "logging"
    };

    void check_option_is_valid_(const YAML::Node& node) {
        std::string name;
        for (const auto& e: node) {
            name = e.first.as<std::string>();
            if (std::find(QN_OPTIONS_.begin(), QN_OPTIONS_.end(), name) == QN_OPTIONS_.end())
                QN_THROW("Invalid option: {}", name);
        }
    }

    void add_missing_options_(YAML::Node node) {
        std::string name;
        for (const auto& e: QN_OPTIONS_) {
            name = e; // YAML::Node::operator[] doesn't support std::string_view.
            if (!node[name].IsDefined())
                node[name] = YAML::Null;
        }
    }

    void find_input_filenames_(YAML::Node& node) {
        using namespace qn;

        // Extract the input directory if specified.
        const Path directory_path = node["stack_directory"].IsScalar() ?
                                    node["stack_directory"].as<Path>() : "";
        std::string basename;
        if (!directory_path.empty()) {
            QN_CHECK(fs::is_directory(directory_path),
                     "The values of \"stack_directory\" does not refer to an existing directory");
            basename = *(--directory_path.parent_path().end()); // FIXME
        }

        // Stack file.
        if (node["stack_file"].IsScalar()) {
            const auto stack_filename = node["stack_file"].as<Path>();
            QN_CHECK(fs::is_regular_file(stack_filename),
                     "A stack filename was provided, but the file doesn't exist or permissions to read are not granted");

        } else if (!directory_path.empty()) {
            i32 count = 0;
            for (auto& e: std::array<std::string, 3>{".st", ".mrc", ".mrcs"}) {
                const Path stack_filename = directory_path / (basename + e);
                if (fs::is_regular_file(stack_filename)) {
                    node["stack_file"] = stack_filename;
                    break;
                }
                ++count;
            }
            if (count == 3)
                QN_THROW("Could not find the stack {}.(st|mrc|mrcs) in directory {}", basename, directory_path);

        } else {
            QN_THROW("The stack could not be found. Check the options \"stack_file\" or "
                     "\"stack_directory\" to make sure they are correctly specified");
        }

        // Optional files.
        for (auto& extension: std::array<std::string, 3>{"mdoc", "tlt", "exposure"}) {
            const auto node_name = noa::string::format("stack_{}", extension);
            if (node[node_name].IsScalar()) {
                const auto filename = node[node_name].as<Path>();
                QN_CHECK(fs::is_regular_file(filename),
                         "The \"{}\" option was provided, but the corresponding filename doesn't exist or "
                         "permissions to read are not granted", node_name);

            } else if (!directory_path.empty()) {
                const Path filename = directory_path / noa::string::format("{}.{}", basename, extension);
                if (fs::is_regular_file(filename))
                    node[node_name] = filename;
            }
        }
    }
}

namespace qn {
    Options::Options(int argc, char** argv) {
        if (argc != 2)
            QN_THROW("Incorrect number of arguments. Enter the filename of the YAML parameter file");

        try {
            m_options = YAML::LoadFile(argv[1]);
        } catch (...) {
            QN_THROW("Failed to load the parameter file");
        }

        // Make sure this is our parameter file.
        if (m_options["quinoa"].Type() != YAML::NodeType::Map)
            QN_THROW("Invalid parameter file. Main entry \"quinoa\" is missing");
        m_options = m_options["quinoa"];

        // Check the options are all valid options.
        // This is mostly to prevent options being ignored because of a typo.
        check_option_is_valid_(m_options);

        // Add missing entries (as null), so that we don't need to check
        // whether a node with a valid name is defined.
        add_missing_options_(m_options);

        // This should be done at the end, when all nodes are defined.
        find_input_filenames_(m_options);
    }
}
