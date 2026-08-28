#include "quinoa/Logger.hpp"
#include "quinoa/preprocessing/ExcludeViews.hpp"
#include "quinoa/preprocessing/Run.hpp"

namespace qn {
    void preprocess(
        const Path& stack_filename,
        Metadata& metadata,
        Device device,
        const Settings::Preprocessing& settings,
        const Path& diagnostics_directory
    ) {
        if (not settings.exclude_stack_indices.empty()) {
            metadata.stack.exclude_if([&](const auto& image) {
                for (isize e: settings.exclude_stack_indices)
                    if (e == image.index) {
                        Logger::info("Excluding view: index={} (tilt={:+.2f})", image.index, image.angles[1]);
                        return true;
                    }
                return false;
            });
        }

        // TODO Hot pixels correction
        // TODO Frame alignment

        if (settings.exclude_blank_views) {
            detect_and_exclude_blank_views(
                stack_filename, metadata.stack, {
                    .compute_device = device,
                    .output_directory = diagnostics_directory / "preprocessing",
                });
        }
    }
}
