#pragma once

#include "common.h"

#include <cstdint>
#include <string>

struct qlora_lr_schedule {
    const lr_opt * lr = nullptr;
    std::string    type = "constant";
    int64_t        warmup_steps = 0;
    int64_t        total_steps = 0;
    int64_t        step = 0; // completed optimizer updates
    float          current_lr = 0.0f;

    float get_lr(int64_t step) const; // upcoming one-based update; step 0 is the initial point
    void advance_step() { ++step; }
};

ggml_opt_optimizer_params qlora_opt_lr_pars(void * userdata);
