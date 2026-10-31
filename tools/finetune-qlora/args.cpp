#include "args.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

std::string normalize_option(std::string option) {
    for (char & c : option) {
        if (c == '_') {
            c = '-';
        }
    }
    return option;
}

int32_t parse_non_negative_int(const std::string & value, const char * name) {
    size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed < 0 || parsed > INT32_MAX) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    return static_cast<int32_t>(parsed);
}

int32_t parse_positive_int(const std::string & value, const char * name) {
    const int32_t parsed = parse_non_negative_int(value, name);
    if (parsed == 0) {
        throw std::invalid_argument(std::string(name) + " must be greater than zero");
    }
    return parsed;
}

float parse_float(const std::string & value, const char * name) {
    size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + value);
    }
    return parsed;
}

std::vector<std::string> break_str_into_lines(const std::string & input, size_t max_chars_per_line) {
    std::vector<std::string> result;
    std::istringstream stream(input);
    std::string line;

    while (std::getline(stream, line)) {
        std::istringstream words(line);
        std::string word;
        std::string current;
        while (words >> word) {
            if (!current.empty() && current.size() + 1 + word.size() > max_chars_per_line) {
                result.push_back(current);
                current.clear();
            }
            if (!current.empty()) {
                current += ' ';
            }
            current += word;
        }
        result.push_back(current);
    }
    return result;
}

const std::vector<qlora_arg> & get_qlora_arg_defs() {
    static const std::vector<qlora_arg> options = [] {
        std::vector<qlora_arg> result;
        auto add_opt = [&result](const qlora_arg & opt) {
            result.push_back(opt);
        };

        // Keep each option's spelling, help text and behavior together, using
        // the same registration style as common/arg.cpp.
        add_opt(qlora_arg(
            {"--train-file"}, "FNAME",
            "Training dataset path; may be repeated [required]",
            [](qlora_params & params, const std::string & value) {
                params.dataset.input_paths.push_back(value);
            }
        ));
        add_opt(qlora_arg(
            {"--fail-on-invalid-records"},
            "Fail instead of skipping an invalid dataset record",
            [](qlora_params & params) {
                params.dataset.skip_invalid = false;
            }
        ));
        add_opt(qlora_arg(
            {"--lora-rank"}, "N",
            "LoRA rank (default: 16; existing adapters are converted to this rank; rank reduction is rejected)",
            [](qlora_params & params, const std::string & value) {
                params.lora_rank = parse_positive_int(value, "--lora-rank");
            }
        ));
        add_opt(qlora_arg(
            {"--lora-alpha"}, "F",
            "LoRA alpha (default: rank; existing adapter weights are rescaled to preserve their initial effect)",
            [](qlora_params & params, const std::string & value) {
                params.lora_alpha = parse_float(value, "--lora-alpha");
                if (params.lora_alpha < 0.0f) {
                    throw std::invalid_argument("--lora-alpha must be non-negative");
                }
            }
        ));
        add_opt(qlora_arg(
            {"--lora-type"}, "TYPE",
            "LoRA tensor type: f32, f16, or bf16",
            [](qlora_params & params, const std::string & value) {
                if (value != "f32" && value != "f16" && value != "bf16") {
                    throw std::invalid_argument("--lora-type must be f32, f16, or bf16");
                }
                params.lora_type = value;
            }
        ));
        add_opt(qlora_arg(
            {"--optimizer-state-type"}, "TYPE",
            "AdamW moment type: f32, f16, or bf16 (default: f32)",
            [](qlora_params & params, const std::string & value) {
                if (value != "f32" && value != "f16" && value != "bf16") {
                    throw std::invalid_argument("--optimizer-state-type must be f32, f16, or bf16");
                }
                params.optimizer_state_type = value;
            }
        ));
        add_opt(qlora_arg(
            {"--lora-targets"}, "CSV",
            "Comma-separated trainable tensor-name substrings; use explicit *_exps names to train MoE experts",
            [](qlora_params & params, const std::string & value) {
                params.lora_targets = value;
            }
        ));
        add_opt(qlora_arg(
            {"--output-dir"}, "DIR",
            "Output directory for checkpoints, adapter snapshots and the final adapter [required]",
            [](qlora_params & params, const std::string & value) {
                params.output_dir = value;
            }
        ));
        add_opt(qlora_arg(
            {"--save-checkpoint-steps"}, "N",
            "Save a resumable checkpoint every N optimizer steps; 0 saves only the final checkpoint when enabled",
            [](qlora_params & params, const std::string & value) {
                params.save_checkpoint_steps = parse_non_negative_int(value, "--save-checkpoint-steps");
            }
        ));
        add_opt(qlora_arg(
            {"--save-checkpoint-limit"}, "N",
            "Enable checkpoint saving and retain at most N checkpoints; 0 disables checkpoint saving",
            [](qlora_params & params, const std::string & value) {
                params.save_checkpoint_limit = parse_non_negative_int(value, "--save-checkpoint-limit");
            }
        ));
        add_opt(qlora_arg(
            {"--save-snapshot-steps"}, "N",
            "Save an adapter-only snapshot every N optimizer steps; 0 disables intermediate snapshots",
            [](qlora_params & params, const std::string & value) {
                params.save_snapshot_steps = parse_non_negative_int(value, "--save-snapshot-steps");
            }
        ));
        add_opt(qlora_arg(
            {"--save-snapshot-limit"}, "N",
            "Enable snapshot saving and retain at most N snapshots; 0 disables snapshot saving",
            [](qlora_params & params, const std::string & value) {
                params.save_snapshot_limit = parse_non_negative_int(value, "--save-snapshot-limit");
            }
        ));
        add_opt(qlora_arg(
            {"--resume-from"}, "DIR",
            "Resume from a checkpoint directory or an output directory containing checkpoints",
            [](qlora_params & params, const std::string & value) {
                params.resume_from = value;
            }
        ));
        add_opt(qlora_arg(
            {"--gradient-accumulation-steps"}, "N",
            "Accumulate gradients across up to N full-context windows before each optimizer update (default: 1)",
            [](qlora_params & params, const std::string & value) {
                params.gradient_accumulation_steps = parse_positive_int(value, "--gradient-accumulation-steps");
            }
        ));
        add_opt(qlora_arg(
            {"--max-steps"}, "N",
            "Stop after N optimizer updates and repeat the dataset as needed; overrides --epochs (default: disabled)",
            [](qlora_params & params, const std::string & value) {
                params.max_steps = parse_positive_int(value, "--max-steps");
            }
        ));
        add_opt(qlora_arg(
            {"--val-windows"}, "N",
            "Evaluate at most N dataset windows after the training range; 0 disables validation (default: all remaining windows)",
            [](qlora_params & params, const std::string & value) {
                params.val_windows = parse_non_negative_int(value, "--val-windows");
            }
        ));
        add_opt(qlora_arg(
            {"--freeze-layers"}, "N",
            "Do not allocate LoRA for the first N transformer layers",
            [](qlora_params & params, const std::string & value) {
                params.freeze_layers = parse_non_negative_int(value, "--freeze-layers");
            }
        ));
        add_opt(qlora_arg(
            {"--lr-scheduler"}, "TYPE",
            "constant or cosine (default: constant)",
            [](qlora_params & params, const std::string & value) {
                if (value != "constant" && value != "cosine") {
                    throw std::invalid_argument("--lr-scheduler must be constant or cosine");
                }
                params.lr_scheduler = value;
            }
        ));
        add_opt(qlora_arg(
            {"--warmup-steps"}, "N",
            "Linear LR warmup steps (default: 0)",
            [](qlora_params & params, const std::string & value) {
                params.warmup_steps = parse_non_negative_int(value, "--warmup-steps");
            }
        ));
        add_opt(qlora_arg(
            {"--warmup-init-ratio"}, "RATIO",
            "Initial warmup LR as a fraction of peak LR (default: 0.1)",
            [](qlora_params & params, const std::string & value) {
                params.warmup_init_ratio = parse_float(value, "--warmup-init-ratio");
                if (!(params.warmup_init_ratio > 0.0f && params.warmup_init_ratio <= 1.0f)) {
                    throw std::invalid_argument("--warmup-init-ratio must be greater than 0 and at most 1");
                }
            }
        ));
        add_opt(qlora_arg(
            {"--shuffle-dataset"},
            "Shuffle dataset windows each epoch",
            [](qlora_params & params) {
                params.shuffle_dataset = true;
            }
        ));

        return result;
    }();
    return options;
}

}

std::string qlora_arg::to_string() const {
    const size_t n_leading_spaces = 40;
    const size_t n_chars_per_line_help = 70;
    const std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream stream;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << args[i];
    }
    if (value_hint != nullptr) {
        stream << " " << value_hint;
    }

    const auto current_width = static_cast<size_t>(stream.tellp());
    if (current_width > n_leading_spaces - 3) {
        stream << "\n" << leading_spaces;
    } else {
        stream << std::string(n_leading_spaces - current_width, ' ');
    }

    const auto help_lines = break_str_into_lines(help, n_chars_per_line_help);
    for (size_t i = 0; i < help_lines.size(); ++i) {
        if (i > 0) {
            stream << leading_spaces;
        }
        stream << help_lines[i] << "\n";
    }
    return stream.str();
}

bool qlora_parse_args(int argc,
                      char ** argv,
                      qlora_params & params,
                      std::vector<std::string> & common_argv,
                      std::string & error) {
    common_argv.clear();
    if (argc > 0) {
        common_argv.emplace_back(argv[0]);
    }

    std::unordered_map<std::string, const qlora_arg *> arg_to_option;
    for (const qlora_arg & option : get_qlora_arg_defs()) {
        for (const char * name : option.args) {
            arg_to_option.emplace(name, &option);
        }
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string original_arg = argv[i];
            const size_t equals = original_arg.find('=');
            const bool has_inline_value = equals != std::string::npos;
            const std::string option_name = normalize_option(original_arg.substr(0, equals));
            std::string value = has_inline_value ? original_arg.substr(equals + 1) : std::string();

            const auto found = arg_to_option.find(option_name);
            if (found == arg_to_option.end()) {
                // Leave common options and their values untouched for the
                // shared llama.cpp parser.
                common_argv.push_back(original_arg);
                continue;
            }

            const qlora_arg & option = *found->second;
            if (option.handler_void != nullptr) {
                if (has_inline_value) {
                    throw std::invalid_argument(option_name + " does not take a value");
                }
                option.handler_void(params);
                continue;
            }

            if (option.handler_string == nullptr) {
                throw std::logic_error("no handler registered for " + option_name);
            }
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("missing value for " + option_name);
                }
                value = argv[++i];
            }
            if (value.empty()) {
                throw std::invalid_argument("empty value for " + option_name);
            }

            try {
                option.handler_string(params, value);
            } catch (const std::exception & ex) {
                throw std::invalid_argument("error while handling argument \"" + option_name + "\": " + ex.what());
            }
        }
    } catch (const std::exception & ex) {
        error = ex.what();
        return false;
    }

    const bool help_requested = std::find(common_argv.begin(), common_argv.end(), "--help") != common_argv.end() ||
                                std::find(common_argv.begin(), common_argv.end(), "-h") != common_argv.end();
    if (params.output_dir.empty() && !help_requested) {
        error = "--output-dir is required";
        return false;
    }

    return true;
}

void qlora_print_usage(int /*argc*/, char ** /*argv*/) {
    std::printf("\n----- QLoRA-specific params -----\n\n");
    for (const qlora_arg & option : get_qlora_arg_defs()) {
        std::printf("%s", option.to_string().c_str());
    }
    std::printf("\nQLoRA accepts one input --lora adapter. Existing adapters are converted to --lora-rank/--lora-alpha; rank reduction is rejected.\n");
    std::printf("Tensors matching --lora-targets are trained, missing targets are added, and other existing tensors are retained frozen.\n");
    std::printf("MoE expert LoRA is opt-in via ffn_gate_exps, ffn_up_exps, ffn_down_exps, or ffn_gate_up_exps.\n");
    std::printf("Checkpoints store adapter weights, AdamW state, dataset order and training progress.\n");
}
