#pragma once

#include "dataset-reader.h"
#include "dataset-types.h"
#include "dataset-window.h"

#include <string>
#include <vector>

#include "chat.h"
#include "llama.h"

struct dataset_load_result {
    std::vector<training_sample> samples;
    dataset_stats stats;
    bool success = false;
};

dataset_load_result load_training_samples(const dataset_options & options,
                                          const llama_vocab * vocab,
                                          common_chat_templates * templates);
