#pragma once

#include "common.h"

#include <cstdint>
#include <string>

// Keep schedule state and LR calculation independent from the training callback.
struct qlora_lr_schedule {
    const lr_opt * lr = nullptr;
    std::string    type = "constant";
    int64_t        warmup_steps = 0;
    float          warmup_init_ratio = 0.1f;
    int64_t        total_steps = 0;
    int64_t        step = 0;
    float          current_lr = 0.0f;

    float get_lr() const;
    void advance_step() { ++step; }
};

ggml_opt_optimizer_params qlora_opt_lr_pars(void * userdata);
