#include "eval-model-precision-options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qlora_model_eval;

namespace {

Options parse(const std::vector<std::string> & args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (const std::string & arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    return parse_options(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}


void test_precision_defaults() {
    const Options options = parse({ "eval", "--model", "model.gguf" });
    require(options.precision.forward_precision_name == "f32", "default forward precision should be f32");
    require(options.precision.forward_precision == GGML_PREC_F32, "default forward precision enum changed");
    require(options.precision.backward_precision_name == "f32", "default backward precision should be f32");
    require(options.precision.backward_precision == GGML_PREC_F32, "default backward precision enum changed");
}

void test_forward_precision_values() {
    const struct {
        const char * name;
        ggml_prec precision;
    } cases[] = {
        { "f32", GGML_PREC_F32 },
        { "f16", GGML_PREC_F16 },
        { "bf16", GGML_PREC_BF16 },
        { "q8", GGML_PREC_Q8 },
    };
    for (const auto & test_case : cases) {
        const Options options = parse({ "eval", "--model", "model.gguf", "--forward-precision", test_case.name });
        require(options.precision.forward_precision == test_case.precision, "forward precision was not parsed");
    }
}

void test_backward_precision_values() {
    const struct {
        const char * name;
        ggml_prec precision;
    } cases[] = {
        { "f32", GGML_PREC_F32 },
        { "f16", GGML_PREC_F16 },
        { "bf16", GGML_PREC_BF16 },
    };
    for (const auto & test_case : cases) {
        const Options options = parse({ "eval", "--model", "model.gguf", "--backward-precision", test_case.name });
        require(options.precision.backward_precision == test_case.precision, "backward precision was not parsed");
    }
}

void test_unknown_precision_values_are_rejected() {
    for (const auto & args : {
            std::vector<std::string>{ "eval", "--model", "model.gguf", "--forward-precision", "fp8" },
            std::vector<std::string>{ "eval", "--model", "model.gguf", "--backward-precision", "fp8" },
            std::vector<std::string>{ "eval", "--model", "model.gguf", "--backward-precision", "auto" } }) {
        bool rejected = false;
        try {
            parse(args);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "unknown precision value should be rejected");
    }
}

void test_default_targets() {
    const Options options = parse({ "eval", "--model", "model.gguf" });
    const std::vector<std::string> expected = {
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"
    };
    require(options.model.lora_targets == expected, "default LoRA targets changed");
}

void test_flash_attention_defaults_to_off() {
    const Options options = parse({ "eval", "--model", "model.gguf" });
    require(!options.model.flash_attention, "Flash Attention should default to off");
}

void test_flash_attention_on_and_off() {
    const Options enabled = parse({ "eval", "--model", "model.gguf", "--flash-attn", "on" });
    const Options disabled = parse({ "eval", "--model", "model.gguf", "--flash-attn", "off" });
    require(enabled.model.flash_attention, "Flash Attention on was not parsed");
    require(!disabled.model.flash_attention, "Flash Attention off was not parsed");
}

void test_invalid_flash_attention_value_is_rejected() {
    bool rejected = false;
    try {
        parse({ "eval", "--model", "model.gguf", "--flash-attn", "auto" });
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "invalid Flash Attention value should be rejected");
}

void test_gpu_layers_default_to_cpu() {
    const Options options = parse({ "eval", "--model", "model.gguf" });
    require(options.model.n_gpu_layers == 0, "GPU layers should default to zero");
}

void test_gpu_layers_option() {
    const Options partial = parse({ "eval", "--model", "model.gguf", "--n-gpu-layers", "12" });
    const Options all = parse({ "eval", "--model", "model.gguf", "--n-gpu-layers", "-1" });
    require(partial.model.n_gpu_layers == 12, "GPU layer count was not parsed");
    require(all.model.n_gpu_layers == -1, "all-GPU-layers option was not parsed");
}

void test_invalid_gpu_layer_count_is_rejected() {
    bool rejected = false;
    try {
        parse({ "eval", "--model", "model.gguf", "--n-gpu-layers", "-2" });
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "invalid GPU layer count should be rejected");
}

void test_custom_targets() {
    const Options options = parse({
        "eval", "--model", "model.gguf", "--lora-targets", " ffn_gate_up_exps, ffn_down_exps "
    });
    const std::vector<std::string> expected = { "ffn_gate_up_exps", "ffn_down_exps" };
    require(options.model.lora_targets == expected, "custom LoRA targets were not parsed correctly");
}

void test_empty_target_is_rejected() {
    bool rejected = false;
    try {
        parse({ "eval", "--model", "model.gguf", "--lora-targets", "ffn_gate_exps,,ffn_down_exps" });
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "empty LoRA target should be rejected");
}

} // namespace

int main() {
    try {
        test_default_targets();
        test_precision_defaults();
        test_forward_precision_values();
        test_backward_precision_values();
        test_unknown_precision_values_are_rejected();
        test_flash_attention_defaults_to_off();
        test_flash_attention_on_and_off();
        test_invalid_flash_attention_value_is_rejected();
        test_gpu_layers_default_to_cpu();
        test_gpu_layers_option();
        test_invalid_gpu_layer_count_is_rejected();
        test_custom_targets();
        test_empty_target_is_rejected();
        std::cout << "[PASS] QLoRA evaluation options\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
