#include "args.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_default_lora_targets() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir=out";
    char * argv[] = { arg0, arg1 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(2, argv, params, common_args, error), "default LoRA targets were rejected");
    require(params.lora_targets == "attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down",
            "default LoRA targets changed");
}

void test_kv_cache_type_arguments_are_forwarded() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir=out";
    char arg2[] = "--cache-type-k";
    char arg3[] = "q8_0";
    char arg4[] = "--cache-type-v=q8_0";
    char * argv[] = { arg0, arg1, arg2, arg3, arg4 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(5, argv, params, common_args, error), "KV cache type arguments were rejected");
    require(common_args.size() == 4 &&
            common_args[1] == "--cache-type-k" && common_args[2] == "q8_0" &&
            common_args[3] == "--cache-type-v=q8_0",
            "KV cache type arguments were not forwarded to the common parser");
}

void test_gradient_accumulation_argument() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir";
    char arg2[] = "out";
    char arg3[] = "--gradient-accumulation-steps";
    char arg4[] = "4";
    char arg5[] = "-c";
    char arg6[] = "4096";
    char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(7, argv, params, common_args, error), "valid accumulation argument was rejected");
    require(params.gradient_accumulation_steps == 4, "accumulation argument was not parsed");
    require(common_args.size() == 3 && common_args[1] == "-c" && common_args[2] == "4096",
            "common arguments were not preserved during QLoRA parsing");
}

void test_gradient_accumulation_argument_rejects_zero() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--gradient-accumulation-steps=0";
    char * argv[] = { arg0, arg1 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(!qlora_parse_args(2, argv, params, common_args, error), "zero accumulation argument was accepted");
    require(error.find("greater than zero") != std::string::npos, "zero accumulation error message is unclear");
}

void test_max_steps_argument() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir=out";
    char arg2[] = "--max-steps";
    char arg3[] = "250";
    char * argv[] = { arg0, arg1, arg2, arg3 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(4, argv, params, common_args, error), "valid max steps argument was rejected");
    require(params.max_steps == 250, "max steps argument was not parsed");
    require(common_args.size() == 1, "QLoRA max steps argument was forwarded to common argument parsing");
}

void test_max_steps_argument_rejects_non_positive_values() {
    char arg1[] = "--max-steps=0";
    char arg2[] = "--max-steps=-1";
    for (char * value : { arg1, arg2 }) {
        char arg0[] = "llama-finetune-qlora";
        char * argv[] = { arg0, value };

        qlora_params params;
        std::vector<std::string> common_args;
        std::string error;
        require(!qlora_parse_args(2, argv, params, common_args, error),
                "non-positive max steps argument was accepted");
        require(!error.empty(), "invalid max steps error message is empty");
    }
}


void test_val_windows_argument() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir=out";
    char arg2[] = "--val-windows";
    char arg3[] = "0";
    char * argv[] = { arg0, arg1, arg2, arg3 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(4, argv, params, common_args, error), "zero validation-window limit was rejected");
    require(params.val_windows == 0, "zero validation-window limit was not parsed");
}

void test_val_windows_positive_value() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir=out";
    char arg2[] = "--val-windows=7";
    char * argv[] = { arg0, arg1, arg2 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(3, argv, params, common_args, error), "positive validation-window limit was rejected");
    require(params.val_windows == 7, "positive validation-window limit was not parsed");
}

void test_val_windows_rejects_negative_value() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--val-windows=-1";
    char * argv[] = { arg0, arg1 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(!qlora_parse_args(2, argv, params, common_args, error), "negative validation-window limit was accepted");
}

void test_precision_type_arguments() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir=out";
    char arg2[] = "--lora-type";
    char arg3[] = "bf16";
    char arg4[] = "--optimizer-state-type=bf16";
    char * argv[] = { arg0, arg1, arg2, arg3, arg4 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(5, argv, params, common_args, error), "valid precision arguments were rejected");
    require(params.lora_type == "bf16", "LoRA storage type was not parsed");
    require(params.optimizer_state_type == "bf16", "optimizer-state type was not parsed");
}

void test_precision_type_arguments_reject_unknown_values() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--lora-type=fp8";
    char * argv[] = { arg0, arg1 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(!qlora_parse_args(2, argv, params, common_args, error), "unknown LoRA type was accepted");
    require(error.find("f32, f16, or bf16") != std::string::npos,
            "invalid LoRA type error message is unclear");
}

void test_precision_type_arguments_reject_q8_0() {
    {
        char arg0[] = "llama-finetune-qlora";
        char arg1[] = "--lora-type=q8_0";
        char * argv[] = { arg0, arg1 };
        qlora_params params;
        std::vector<std::string> common_args;
        std::string error;
        require(!qlora_parse_args(2, argv, params, common_args, error), "Q8_0 LoRA storage type was accepted");
    }
    {
        char arg0[] = "llama-finetune-qlora";
        char arg1[] = "--optimizer-state-type=q8_0";
        char * argv[] = { arg0, arg1 };
        qlora_params params;
        std::vector<std::string> common_args;
        std::string error;
        require(!qlora_parse_args(2, argv, params, common_args, error), "Q8_0 AdamW state type was accepted");
    }
}

void test_output_arguments() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir";
    char arg2[] = "output";
    char arg3[] = "--save-checkpoint-steps=100";
    char arg4[] = "--save-checkpoint-limit=3";
    char arg5[] = "--save-snapshot-steps";
    char arg6[] = "200";
    char arg7[] = "--save-snapshot-limit=10";
    char arg8[] = "--resume-from";
    char arg9[] = "checkpoints/checkpoint-000200";
    char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(10, argv, params, common_args, error), "valid output arguments were rejected");
    require(params.output_dir == "output", "output directory was not parsed");
    require(params.save_checkpoint_steps == 100, "checkpoint interval was not parsed");
    require(params.save_checkpoint_limit == 3, "checkpoint retention limit was not parsed");
    require(params.save_snapshot_steps == 200, "snapshot interval was not parsed");
    require(params.save_snapshot_limit == 10, "snapshot retention limit was not parsed");
    require(params.resume_from == "checkpoints/checkpoint-000200", "resume directory was not parsed");
}

void test_output_dir_is_required() {
    {
        char arg0[] = "llama-finetune-qlora";
        char * argv[] = { arg0 };

        qlora_params params;
        std::vector<std::string> common_args;
        std::string error;
        require(!qlora_parse_args(1, argv, params, common_args, error), "missing output directory was accepted");
        require(error.find("--output-dir is required") != std::string::npos,
                "missing output directory error message is unclear");
    }
    {
        char arg0[] = "llama-finetune-qlora";
        char arg1[] = "--resume-from=checkpoint-000100";
        char * argv[] = { arg0, arg1 };

        qlora_params params;
        std::vector<std::string> common_args;
        std::string error;
        require(!qlora_parse_args(2, argv, params, common_args, error),
                "resume-from without output-dir was accepted");
    }
}

void test_help_does_not_require_output_dir() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--help";
    char * argv[] = { arg0, arg1 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(2, argv, params, common_args, error), "help option required an output directory");
    require(common_args.size() == 2 && common_args[1] == "--help", "help option was not forwarded to common parser");
}

void test_output_defaults_are_zero() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir=out";
    char * argv[] = { arg0, arg1 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(2, argv, params, common_args, error), "default output arguments were rejected");
    require(params.save_checkpoint_steps == 0 && params.save_checkpoint_limit == 0 &&
            params.save_snapshot_steps == 0 && params.save_snapshot_limit == 0,
            "save intervals and limits should default to zero");
}

void test_zero_limits_disable_save_options() {
    char arg0[] = "llama-finetune-qlora";
    char arg1[] = "--output-dir=out";
    char arg2[] = "--save-checkpoint-steps=100";
    char arg3[] = "--save-checkpoint-limit=0";
    char arg4[] = "--save-snapshot-steps=200";
    char arg5[] = "--save-snapshot-limit=0";
    char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5 };

    qlora_params params;
    std::vector<std::string> common_args;
    std::string error;
    require(qlora_parse_args(6, argv, params, common_args, error), "zero save limits were rejected");
    require(params.save_checkpoint_steps == 100 && params.save_checkpoint_limit == 0,
            "zero checkpoint limit was not parsed as disabled");
    require(params.save_snapshot_steps == 200 && params.save_snapshot_limit == 0,
            "zero snapshot limit was not parsed as disabled");
}

void test_output_options_reject_negative_values() {
    const char * options[] = {
        "--save-checkpoint-steps=-1",
        "--save-checkpoint-limit=-1",
        "--save-snapshot-steps=-1",
        "--save-snapshot-limit=-1",
    };
    for (const char * option : options) {
        char arg0[] = "llama-finetune-qlora";
        char arg1[] = "--output-dir=out";
        std::vector<std::string> option_storage = { option };
        std::vector<char *> argv = { arg0, arg1, option_storage[0].data() };
        qlora_params params;
        std::vector<std::string> common_args;
        std::string error;
        require(!qlora_parse_args(static_cast<int>(argv.size()), argv.data(), params, common_args, error),
                "negative output option value was accepted");
    }
}

} // namespace

int main() {
    test_default_lora_targets();
    test_kv_cache_type_arguments_are_forwarded();
    test_gradient_accumulation_argument();
    test_gradient_accumulation_argument_rejects_zero();
    test_max_steps_argument();
    test_max_steps_argument_rejects_non_positive_values();
    test_val_windows_argument();
    test_val_windows_positive_value();
    test_val_windows_rejects_negative_value();
    test_precision_type_arguments();
    test_precision_type_arguments_reject_unknown_values();
    test_precision_type_arguments_reject_q8_0();
    test_output_arguments();
    test_output_dir_is_required();
    test_help_does_not_require_output_dir();
    test_output_defaults_are_zero();
    test_zero_limits_disable_save_options();
    test_output_options_reject_negative_values();
    return 0;
}
