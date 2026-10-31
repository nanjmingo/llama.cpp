#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

struct dataset_options {
    std::vector<std::string> input_paths;
    size_t threads = 0;
    bool skip_invalid = true;
};

struct dataset_message {
    std::string role;
    std::string content;
};

struct chat_example {
    std::vector<dataset_message> messages;
};

struct prompt_completion_example {
    std::string prompt;
    std::string completion;
};

struct text_example {
    std::string text;
};

using dataset_example = std::variant<chat_example, prompt_completion_example, text_example>;

struct sourced_dataset_example {
    dataset_example example;
    std::string source_path;
    size_t record_number = 0;
};

struct rendered_piece {
    std::string text;
    bool parse_special = true;
    float loss_weight = 0.0f;
};

struct rendered_example {
    std::vector<rendered_piece> pieces;
    bool add_special = false;
    bool text_continuation = false;
};

struct training_sample {
    std::vector<llama_token> tokens;
    std::vector<float> weights;
    bool text_continuation = false;
};

struct training_window {
    std::vector<int32_t> data;
    std::vector<int32_t> labels;
    std::vector<float> weights;
};

struct dataset_stats {
    size_t records_read = 0;
    size_t records_accepted = 0;
    size_t invalid_records = 0;
    size_t rendered_samples = 0;
    size_t training_windows = 0;
};

struct dataset_diagnostic {
    std::string source_path;
    size_t record_number = 0;
    std::string message;
};
