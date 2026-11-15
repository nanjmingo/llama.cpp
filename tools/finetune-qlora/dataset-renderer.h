#pragma once

#include "dataset-types.h"

#include <string>
#include <vector>

#include "chat.h"

struct dataset_render_result {
    std::vector<rendered_example> examples;
    std::vector<std::string> diagnostics;
};

dataset_render_result render_dataset_example(const dataset_example & example,
                                                common_chat_templates * templates);
