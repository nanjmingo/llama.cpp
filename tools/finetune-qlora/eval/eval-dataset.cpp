#include "dataset-format.h"
#include "dataset-reader.h"
#include "dataset-template.h"
#include "dataset-tokenize.h"

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model_path;
    std::string output_path;
    std::vector<std::string> input_paths;
    int32_t context_size = 0;
};

struct backend_scope {
    backend_scope() { llama_backend_init(); }
    ~backend_scope() { llama_backend_free(); }
};

void print_usage(const char * program) {
    std::cerr << "Usage: " << program
              << " --model MODEL.gguf --input DATA.jsonl [--input DATA2.jsonl]"
              << " --output OUTPUT.txt --context-size N\n";
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

std::string escape_piece(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            case '"': result += "\\\""; break;
            default: result += ch; break;
        }
    }
    return result;
}

std::string token_display(const llama_vocab * vocab, llama_token token) {
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
    return common_token_to_piece(vocab, token, llama_vocab_is_control(vocab, token));
}

bool token_is_target_candidate(const training_sample & sample, size_t index, int32_t context_size) {
    if (sample.text_continuation && index < static_cast<size_t>(context_size / 2)) {
        return false;
    }
    return sample.weights[index] > 0.0f;
}

void write_sample(std::ofstream & output,
                  const training_sample & sample,
                  const llama_vocab * vocab,
                  int32_t context_size,
                  size_t sample_index,
                  size_t sample_count) {
    const size_t token_count = sample.tokens.size();
    size_t supervised_count = 0;
    for (size_t i = 0; i < token_count; ++i) {
        supervised_count += token_is_target_candidate(sample, i, context_size) ? 1 : 0;
    }

    output << "\n--- rendered sample " << sample_index + 1 << '/' << sample_count
           << " tokens=" << token_count
           << " target_candidate_tokens=" << supervised_count;
    if (sample.text_continuation) {
        output << " initial_context_tokens=" << std::min(token_count, static_cast<size_t>(context_size / 2));
    }
    output << " ---\n";

    bool has_segment = false;
    bool current_supervised = false;
    std::string segment;
    size_t segment_start = 0;
    size_t segment_end = 0;
    const auto flush_segment = [&]() {
        if (!has_segment) {
            return;
        }
        if (sample.text_continuation) {
            output << (current_supervised ? "[PREDICTION CANDIDATES tokens=" : "[INITIAL CONTEXT tokens=");
        } else {
            output << (current_supervised ? "[RESPONSE TARGET CANDIDATES tokens=" : "[PROMPT tokens=");
        }
        output << segment_start << ".." << segment_end << "]\n";
        output << segment << "\n";
    };

    for (size_t i = 0; i < token_count; ++i) {
        const bool supervised = token_is_target_candidate(sample, i, context_size);
        if (!has_segment || supervised != current_supervised) {
            flush_segment();
            has_segment = true;
            current_supervised = supervised;
            segment.clear();
            segment_start = i;
        }
        segment += token_display(vocab, sample.tokens[i]);
        segment_end = i + 1;
    }
    flush_segment();

    output << "[TOKEN DETAILS]\n";
    for (size_t i = 0; i < token_count; ++i) {
        const bool supervised = token_is_target_candidate(sample, i, context_size);
        output << "  index=" << i
               << " id=" << sample.tokens[i]
               << " sample_weight=" << std::fixed << std::setprecision(4)
               << sample.weights[i]
               << " role=" << (supervised ? "target_candidate" : "context")
               << " piece=\"" << escape_piece(token_display(vocab, sample.tokens[i])) << "\"\n";
    }
}

void write_record(std::ofstream & output,
                  const sourced_dataset_example & sourced,
                  common_chat_templates * templates,
                  const llama_vocab * vocab,
                  int32_t context_size) {
    output << "\n============================================================\n"
           << "source=" << sourced.source_path << '\n'
           << "record=" << sourced.record_number << '\n'
           << "format=" << format_name(sourced.example) << '\n';

    const dataset_template_result rendered = render_dataset_example(sourced.example, templates);
    for (const std::string & diagnostic : rendered.diagnostics) {
        output << "[RENDER DIAGNOSTIC] " << diagnostic << '\n';
    }
    if (rendered.examples.empty()) {
        output << "[NO TRAINING SAMPLE]\n";
        return;
    }

    for (size_t i = 0; i < rendered.examples.size(); ++i) {
        training_sample sample;
        std::string error;
        if (!tokenize_rendered_example(vocab, rendered.examples[i], sample, error)) {
            output << "[TOKENIZATION ERROR] " << error << '\n';
            continue;
        }
        write_sample(output, sample, vocab, context_size, i, rendered.examples.size());
    }
}

} // namespace

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

        output << "QLoRA dataset rendering and tokenization\n"
               << "model=" << params.model_path << '\n'
               << "context_size=" << params.context_size << '\n'
               << "chat_template=" << (templates ? "model default template" : "unavailable") << '\n'
               << "Special tokens are rendered as <BOS>, <EOS>, <EOT>, <SEP>, or their vocabulary spelling.\n"
               << "Roles and sample weights are shown before window overlap masking and per-window loss normalization.\n";

        dataset_options dataset;
        dataset.input_paths = params.input_paths;
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
                    write_record(output, sourced, templates.get(), llama_model_get_vocab(model.get()),
                                 params.context_size);
                }
            });

        output << "\n============================================================\n"
               << "records_read=" << summary.stats.records_read << '\n'
               << "records_accepted=" << summary.stats.records_accepted << '\n'
               << "invalid_records=" << summary.stats.invalid_records << '\n';
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
