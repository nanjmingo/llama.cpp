#pragma once

#include "dataset-types.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

struct qlora_params {
    int32_t     lora_rank          = 16;
    float       lora_alpha         = 0.0f;
    std::string lora_type          = "f32";
    std::string optimizer_state_type = "f32";
    std::string forward_precision_name = "f32";
    ggml_prec forward_precision = GGML_PREC_F32;
    std::string backward_precision_name = "f32";
    ggml_prec backward_precision = GGML_PREC_F32;
    std::string lora_targets       = "attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down";
    std::string output_dir;
    std::string resume_from;
    dataset_options dataset;
    int32_t     save_checkpoint_steps = 0;
    int32_t     save_checkpoint_limit = 0;
    int32_t     save_snapshot_steps = 0;
    int32_t     save_snapshot_limit = 0;
    int32_t     gradient_accumulation_steps = 1;
    int32_t     grad_stats_interval = 0;
    int32_t     max_steps          = 0;
    int32_t     val_windows        = -1;
    int32_t     freeze_layers      = 0;
    std::string lr_scheduler       = "constant";
    int32_t     warmup_steps       = 0;
    float       warmup_init_ratio  = 0.1f;
    bool        shuffle_dataset    = false;
};

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

bool qlora_parse_args(int argc,
                      char ** argv,
                      qlora_params & params,
                      std::vector<std::string> & common_argv,
                      std::string & error);

void qlora_print_usage(int argc, char ** argv);
