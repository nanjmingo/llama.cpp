#pragma once

#include "dataset-types.h"

#include <vector>

#include "ggml-opt.h"

struct window_options {
    int32_t context_length = 0;
    llama_token bos_token = 0;
    int32_t overlap_tokens = -1;
};

ggml_opt_dataset_t build_training_dataset(const std::vector<training_sample> & samples,
                                          const window_options & options);
