#include "dataset-parser.h"
#include "dataset-reader.h"
#include "dataset-renderer.h"
#include "dataset-tokenize.h"

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model_path;
    std::string output_path;
    std::vector<std::string> input_paths;
    int32_t context_size = 0;
    bool token_details = false;
};

struct backend_scope {
    backend_scope() { llama_backend_init(); }
    ~backend_scope() { llama_backend_free(); }
};

void print_usage(const char * program) {
    std::cerr << "Usage: " << program
              << " --model MODEL.gguf --input DATA.jsonl [--input DATA2.jsonl]"
              << " --output OUTPUT.txt --context-size N [--token-details]\n";
}

bool parse_positive_int(const std::string & value, int32_t & output) {
    try {
        size_t consumed = 0;
        const long parsed = std::stol(value, &consumed);
        if (consumed != value.size() || parsed <= 0 || parsed > INT32_MAX) {
            return false;
        }
        output = static_cast<int32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--model" && i + 1 < argc) {
            result.model_path = argv[++i];
        } else if (argument == "--input" && i + 1 < argc) {
            result.input_paths.emplace_back(argv[++i]);
        } else if (argument == "--output" && i + 1 < argc) {
            result.output_path = argv[++i];
        } else if (argument == "--context-size" && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], result.context_size)) {
                throw std::runtime_error("--context-size must be a positive 32-bit integer");
            }
        } else if (argument == "--token-details") {
            result.token_details = true;
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("unknown or incomplete argument: " + argument);
        }
    }
    if (result.model_path.empty() || result.input_paths.empty() ||
        result.output_path.empty() || result.context_size <= 0) {
        print_usage(argv[0]);
        throw std::runtime_error("--model, --input, --output, and --context-size are required");
    }
    return result;
}

const char * format_name(const dataset_example & example) {
    if (std::holds_alternative<text_example>(example)) {
        return "text";
    }
    if (std::holds_alternative<prompt_completion_example>(example)) {
        return "prompt/completion";
    }
    return "messages";
}

std::string special_token_display(const llama_vocab * vocab, llama_token token) {
    if (token == llama_vocab_bos(vocab)) {
        return "<BOS>";
    }
    if (token == llama_vocab_eos(vocab)) {
        return "<EOS>";
    }
    if (token == llama_vocab_eot(vocab)) {
        return "<EOT>";
    }
    if (token == llama_vocab_sep(vocab)) {
        return "<SEP>";
    }
    return common_token_to_piece(vocab, token, true);
}

bool is_displayed_as_special(const llama_vocab * vocab, llama_token token) {
    return token == llama_vocab_bos(vocab) ||
           token == llama_vocab_eos(vocab) ||
           token == llama_vocab_eot(vocab) ||
           token == llama_vocab_sep(vocab) ||
           llama_vocab_is_control(vocab, token);
}

std::string detokenize_sample(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string result;
    std::vector<llama_token> text_tokens;
    const auto flush_text_tokens = [&]() {
        if (text_tokens.empty()) {
            return;
        }
        result += common_detokenize(vocab, text_tokens, true);
        text_tokens.clear();
    };

    for (const llama_token token : tokens) {
        if (is_displayed_as_special(vocab, token)) {
            flush_text_tokens();
            result += special_token_display(vocab, token);
        } else {
            text_tokens.push_back(token);
        }
    }
    flush_text_tokens();
    return result;
}

std::string token_bytes_hex(const llama_vocab * vocab, llama_token token) {
    const std::string piece = common_token_to_piece(vocab, token, false);
    std::ostringstream output;
    output << std::uppercase << std::hex << std::setfill('0');
    for (size_t i = 0; i < piece.size(); ++i) {
        if (i > 0) {
            output << ' ';
        }
        output << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(piece[i]));
    }
    return output.str();
}

float effective_sample_weight(const training_sample & sample, size_t index, int32_t context_size) {
    if (sample.text_continuation && index < static_cast<size_t>(context_size / 2)) {
        return 0.0f;
    }
    return sample.weights[index];
}

bool token_is_target_candidate(const training_sample & sample, size_t index, int32_t context_size) {
    return effective_sample_weight(sample, index, context_size) > 0.0f;
}

std::string format_weight(float weight) {
    std::ostringstream output;
    output << std::setprecision(4) << std::defaultfloat << weight;
    return output.str();
}

void write_weight_ranges(std::ofstream & output,
                         const training_sample & sample,
                         int32_t context_size) {
    if (sample.tokens.empty()) {
        return;
    }
    size_t range_start = 0;
    float range_weight = effective_sample_weight(sample, 0, context_size);
    bool first = true;
    for (size_t i = 1; i <= sample.tokens.size(); ++i) {
        if (i < sample.tokens.size() &&
            effective_sample_weight(sample, i, context_size) == range_weight) {
            continue;
        }
        if (!first) {
            output << ',';
        }
        output << range_start << ".." << i << ':' << format_weight(range_weight);
        first = false;
        if (i < sample.tokens.size()) {
            range_start = i;
            range_weight = effective_sample_weight(sample, i, context_size);
        }
    }
}

void write_token_details(std::ofstream & output,
                         const training_sample & sample,
                         const llama_vocab * vocab,
                         int32_t context_size) {
    output << "[TOKEN DETAILS] index=id:rendered_weight:effective_weight:piece_bytes_hex\n";
    for (size_t i = 0; i < sample.tokens.size(); ++i) {
        const float effective_weight = effective_sample_weight(sample, i, context_size);
        output << i << '=' << sample.tokens[i]
               << ':' << format_weight(sample.weights[i])
               << ':' << format_weight(effective_weight) << ':';
        if (is_displayed_as_special(vocab, sample.tokens[i])) {
            output << special_token_display(vocab, sample.tokens[i]);
        } else {
            output << token_bytes_hex(vocab, sample.tokens[i]);
        }
        output << '\n';
    }
}

void write_sample(std::ofstream & output,
                  const training_sample & sample,
                  const llama_vocab * vocab,
                  int32_t context_size,
                  size_t sample_index,
                  size_t sample_count,
                  bool token_details) {
    const size_t token_count = sample.tokens.size();
    size_t supervised_count = 0;
    for (size_t i = 0; i < token_count; ++i) {
        supervised_count += token_is_target_candidate(sample, i, context_size) ? 1 : 0;
    }

    output << "\n# sample " << sample_index + 1 << '/' << sample_count
           << " tokens=" << token_count
           << " target_candidates=" << supervised_count;
    if (sample.text_continuation) {
        output << " text_initial_context=" << std::min(token_count, static_cast<size_t>(context_size / 2));
    }
    output << " effective_weight_ranges=";
    write_weight_ranges(output, sample, context_size);
    output << '\n';

    const std::string decoded_text = detokenize_sample(vocab, sample.tokens);
    output << decoded_text;
    if (decoded_text.empty() || decoded_text.back() != '\n') {
        output << '\n';
    }

    if (token_details) {
        write_token_details(output, sample, vocab, context_size);
    }
}

void write_record(std::ofstream & output,
                  const sourced_dataset_example & sourced,
                  common_chat_templates * templates,
                  const llama_vocab * vocab,
                  int32_t context_size,
                  bool token_details) {
    output << "# record=" << sourced.record_number
           << " format=" << format_name(sourced.example) << '\n';

    const dataset_render_result rendered = render_dataset_example(sourced.example, templates);
    for (const std::string & diagnostic : rendered.diagnostics) {
        output << "[RENDER DIAGNOSTIC] " << diagnostic << '\n';
    }
    if (rendered.examples.empty()) {
        output << "# no_training_sample\n";
        return;
    }

    for (size_t i = 0; i < rendered.examples.size(); ++i) {
        training_sample sample;
        std::string error;
        if (!tokenize_rendered_example(vocab, rendered.examples[i], sample, error)) {
            output << "# tokenization_error=" << error << '\n';
            continue;
        }
        write_sample(output, sample, vocab, context_size, i, rendered.examples.size(), token_details);
    }
}

}

int main(int argc, char ** argv) {
    try {
        const options params = parse_options(argc, argv);
        std::ofstream output(params.output_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("failed to open output file: " + params.output_path);
        }

        backend_scope backend;
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.vocab_only = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(params.model_path.c_str(), model_params), llama_model_free);
        if (!model) {
            throw std::runtime_error("failed to load model vocabulary and metadata: " + params.model_path);
        }

        common_chat_templates_ptr templates;
        const char * model_template = llama_model_chat_template(model.get(), nullptr);
        if (model_template && model_template[0] != '\0') {
            templates = common_chat_templates_init(model.get(), "");
        }

        output << "# QLoRA dataset rendering\n"
               << "# model=" << params.model_path << '\n'
               << "# context_size=" << params.context_size << '\n'
               << "# chat_template=" << (templates ? "model default" : "unavailable") << '\n'
               << "# Special tokens use readable markers. Effective weights include the text initial-context mask; window overlap and trailing partial-window handling are applied later.\n"
               << "# Token-by-token details are omitted by default; pass --token-details to include them.\n";

        dataset_options dataset;
        dataset.input_paths = params.input_paths;
        std::string last_source_path;
        const dataset_read_summary summary = read_dataset_records_batched(
            dataset,
            [&](dataset_read_result && batch) {
                for (const dataset_diagnostic & diagnostic : batch.diagnostics) {
                    output << "\n[DATASET DIAGNOSTIC] " << diagnostic.source_path;
                    if (diagnostic.record_number > 0) {
                        output << ':' << diagnostic.record_number;
                    }
                    output << " " << diagnostic.message << '\n';
                }
                for (const sourced_dataset_example & sourced : batch.examples) {
                    if (sourced.source_path != last_source_path) {
                        if (!last_source_path.empty()) {
                            output << '\n';
                        }
                        output << "# source=" << sourced.source_path << '\n';
                        last_source_path = sourced.source_path;
                    }
                    write_record(output, sourced, templates.get(), llama_model_get_vocab(model.get()),
                                 params.context_size, params.token_details);
                }
            });

        output << "\n# records_read=" << summary.stats.records_read << '\n'
               << "# records_accepted=" << summary.stats.records_accepted << '\n'
               << "# invalid_records=" << summary.stats.invalid_records << '\n';
        output.flush();
        if (!output) {
            throw std::runtime_error("failed while writing output file: " + params.output_path);
        }
        if (!summary.success) {
            throw std::runtime_error("failed to read one or more dataset files");
        }

        std::cout << "Wrote " << summary.stats.records_accepted << " dataset records to "
                  << params.output_path << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
