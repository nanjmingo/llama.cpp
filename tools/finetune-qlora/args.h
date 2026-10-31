#pragma once

#include "dataset-types.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

struct qlora_params {
    int32_t     lora_rank          = 16;
    float       lora_alpha         = 0.0f; // 0 = use rank
    std::string lora_type          = "f32";
    std::string optimizer_state_type = "f32";
    std::string lora_targets       = "attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down";
    std::string output_dir;
    std::string resume_from;
    dataset_options dataset;
    int32_t     save_checkpoint_steps = 0;
    int32_t     save_checkpoint_limit = 0;
    int32_t     save_snapshot_steps = 0;
    int32_t     save_snapshot_limit = 0;
    int32_t     gradient_accumulation_steps = 1;
    int32_t     max_steps          = 0; // 0 = use --epochs; otherwise stop after N optimizer updates
    int32_t     val_windows        = -1; // -1 = all remaining windows; 0 = no validation
    int32_t     freeze_layers      = 0;
    std::string lr_scheduler       = "constant"; // constant, cosine
    int32_t     warmup_steps       = 0;
    float       warmup_init_ratio  = 0.1f;
    bool        shuffle_dataset    = false;
};

// QLoRA-specific counterpart to common_arg. Keep option metadata and its
// handler together so adding an option only requires changing one definition.
struct qlora_arg {
    std::vector<const char *> args;
    const char * value_hint = nullptr;
    std::string help;
    void (*handler_void)(qlora_params & params) = nullptr;
    void (*handler_string)(qlora_params & params, const std::string & value) = nullptr;

    qlora_arg() = default;

    qlora_arg(
        const std::initializer_list<const char *> & args,
        const std::string & help,
        void (*handler)(qlora_params & params)
    ) : args(args), help(help), handler_void(handler) {}

    qlora_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(qlora_params & params, const std::string & value)
    ) : args(args), value_hint(value_hint), help(help), handler_string(handler) {}

    std::string to_string() const;
};

// Parse QLoRA-only arguments and copy all remaining arguments to common_argv.
// This keeps QLoRA options out of common/arg.cpp while following its
// common_arg-style option-definition format.
bool qlora_parse_args(int argc,
                      char ** argv,
                      qlora_params & params,
                      std::vector<std::string> & common_argv,
                      std::string & error);

// Appended to the common finetune usage output when --help is requested.
void qlora_print_usage(int argc, char ** argv);
