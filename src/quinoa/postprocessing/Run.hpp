#pragma once

#include "quinoa/Stack.hpp"

namespace qn {
    void save_stack(
        StackLoader& stack,
        const Path& filename,
        const Metadata::Stack& metadata,
        bool cache_loader,
        const Settings::PostProcessing::Stack& saving_parameters
    );

    auto reconstruct_tomogram(
        StackLoader&& stack,
        Metadata& metadata,
        const Settings::PostProcessing::Tomogram& settings
    ) -> Array<f32>;

    void postprocess(
        const Path& input_stack,
        const Metadata& metadata,
        Device device,
        const Settings::PostProcessing& settings,
        const Path& output_directory
    );
}
