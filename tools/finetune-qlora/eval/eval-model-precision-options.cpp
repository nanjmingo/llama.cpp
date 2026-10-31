#include "eval-model-precision-options.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qlora_model_eval {
namespace {

ggml_type parse_cache_type(const std::string & value) {
    const ggml_type types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    };
    for (const ggml_type type : types) {
        if (value == ggml_type_name(type)) {
            return type;
        }
    }
    throw std::runtime_error("unsupported KV cache type: " + value);
}

int parse_positive_integer(const std::string & value, const std::string & option) {
    size_t parsed = 0;
    long long number = 0;
    try {
        number = std::stoll(value, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(option + " must be a positive integer");
    }
    if (parsed != value.size() || number < 1 || number > std::numeric_limits<int>::max()) {
        throw std::runtime_error(option + " must be a positive integer");
    }
    return static_cast<int>(number);
}

int parse_gpu_layers(const std::string & value, const std::string & option) {
    size_t parsed = 0;
    long long number = 0;
    try {
        number = std::stoll(value, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(option + " must be -1 or a non-negative integer");
    }
    if (parsed != value.size() || number < -1 || number > std::numeric_limits<int>::max()) {
        throw std::runtime_error(option + " must be -1 or a non-negative integer");
    }
    return static_cast<int>(number);
}

std::vector<std::string> parse_string_list(const std::string & value, const std::string & option) {
    if (value.empty() || value.front() == ',' || value.back() == ',') {
        throw std::runtime_error(option + " contains an empty target");
    }

    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t first = item.find_first_not_of(" \t\r\n");
        const size_t last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::runtime_error(option + " contains an empty target");
        }
        result.push_back(item.substr(first, last - first + 1));
    }
    if (result.empty()) {
        throw std::runtime_error(option + " requires at least one target");
    }
    return result;
}

std::vector<float> parse_float_list(const std::string & value, const std::string & option) {
    if (value.empty() || value.front() == ',' || value.back() == ',') {
        throw std::runtime_error(option + " contains an empty value");
    }

    std::vector<float> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t first = item.find_first_not_of(" \t");
        const size_t last = item.find_last_not_of(" \t");
        if (first == std::string::npos) {
            throw std::runtime_error(option + " contains an empty value");
        }
        item = item.substr(first, last - first + 1);
        size_t parsed = 0;
        const float number = std::stof(item, &parsed);
        if (parsed != item.size() || !std::isfinite(number) || number <= 0.0f) {
            throw std::runtime_error(option + " values must be positive finite numbers");
        }
        result.push_back(number);
    }
    if (result.empty()) {
        throw std::runtime_error(option + " requires at least one value");
    }
    return result;
}

} // namespace

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            options.model.model_path = argv[++i];
        } else if (arg == "--lora-targets" && i + 1 < argc) {
            options.model.lora_targets = parse_string_list(argv[++i], arg);
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            options.model.n_gpu_layers = parse_gpu_layers(argv[++i], arg);
        } else if (arg == "--flash-attn" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "on") {
                options.model.flash_attention = true;
            } else if (value == "off") {
                options.model.flash_attention = false;
            } else {
                throw std::runtime_error("--flash-attn must be on or off");
            }
        } else if (arg == "--activation-precision" && i + 1 < argc) {
            options.precision.activation_precision_name = argv[++i];
            if (options.precision.activation_precision_name == "f32") {
                options.precision.activation_precision = GGML_PREC_F32;
            } else if (options.precision.activation_precision_name == "f16") {
                options.precision.activation_precision = GGML_PREC_F16;
            } else if (options.precision.activation_precision_name == "bf16") {
                options.precision.activation_precision = GGML_PREC_BF16;
            } else if (options.precision.activation_precision_name == "q8") {
                options.precision.activation_precision = GGML_PREC_Q8;
            } else {
                throw std::runtime_error("invalid activation precision: " + options.precision.activation_precision_name);
            }
        } else if ((arg == "-ctk" || arg == "--cache-type-k") && i + 1 < argc) {
            options.precision.inference_cache_type_k = parse_cache_type(argv[++i]);
        } else if ((arg == "-ctv" || arg == "--cache-type-v") && i + 1 < argc) {
            options.precision.inference_cache_type_v = parse_cache_type(argv[++i]);
        } else if (arg == "--learning-rates" && i + 1 < argc) {
            options.learning_rates = parse_float_list(argv[++i], arg);
        } else if (arg == "--epoch" && i + 1 < argc) {
            options.epochs = parse_positive_integer(argv[++i], arg);
        } else if (arg == "--verbose-logs") {
            options.model.verbose_library_logs = true;
        } else {
            throw std::runtime_error(std::string("usage: ") + argv[0] +
                " --model MODEL.gguf [--lora-targets CSV] [--n-gpu-layers N] [--flash-attn on|off]"
                " [--activation-precision f32|f16|bf16|q8]"
                " [-ctk|--cache-type-k TYPE] [-ctv|--cache-type-v TYPE]"
                " [--learning-rates LIST] [--epoch N] [--verbose-logs]");
        }
    }
    if (options.model.model_path.empty()) {
        throw std::runtime_error(std::string("usage: ") + argv[0] +
            " --model MODEL.gguf [--lora-targets CSV] [--n-gpu-layers N] [--flash-attn on|off]"
            " [--activation-precision f32|f16|bf16|q8]"
            " [-ctk|--cache-type-k TYPE] [-ctv|--cache-type-v TYPE]"
            " [--learning-rates LIST] [--epoch N] [--verbose-logs]");
    }
    return options;
}

} // namespace qlora_model_eval
