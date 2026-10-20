#pragma once

#include "chat.h"
#include "common.h"
#include "llama.h"

#include <string>
#include <vector>

// Tokenized sample with a mask indicating which tokens contribute to the loss.
struct training_sample {
    std::vector<llama_token> tokens;
    std::vector<bool>        is_label;
};

// Load JSONL input and tokenize prompt/response pairs. The prompt tokens are
// retained in the sample but masked out from the training loss.
std::vector<training_sample> load_jsonl(const std::string & path,
                                        const llama_vocab * vocab,
                                        common_chat_templates * tmpls);

// Pack variable-length samples into context-sized windows for ggml training.
ggml_opt_dataset_t build_dataset(const std::vector<training_sample> & samples,
                                 int32_t n_ctx,
                                 llama_token bos_token);
