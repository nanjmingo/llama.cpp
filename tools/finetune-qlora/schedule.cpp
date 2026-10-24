#include "schedule.h"

#include <algorithm>
#include <cmath>

float qlora_lr_schedule::get_lr() const {
    const float base_lr = lr->lr0;

    if (step < warmup_steps) {
        const float warmup_start_lr = base_lr * warmup_init_ratio;
        const float warmup_progress = static_cast<float>(step) / static_cast<float>(warmup_steps);
        return warmup_start_lr + (base_lr - warmup_start_lr) * warmup_progress;
    }

    if (type != "cosine" || total_steps <= warmup_steps) {
        return lr->get_lr();
    }

    const float min_lr = std::max(0.0f, lr->lr_min);
    const double decay_progress =
        static_cast<double>(step - warmup_steps) / static_cast<double>(total_steps - warmup_steps);
    const double cosine_weight =
        0.5 * (1.0 + std::cos(std::acos(-1.0) * std::min(1.0, decay_progress)));
    return min_lr + (base_lr - min_lr) * static_cast<float>(cosine_weight);
}

ggml_opt_optimizer_params qlora_opt_lr_pars(void * userdata) {
    qlora_lr_schedule & schedule = *static_cast<qlora_lr_schedule *>(userdata);
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    schedule.current_lr = schedule.get_lr();
    result.adamw.alpha = result.sgd.alpha = schedule.current_lr;
    result.sgd.wd = result.adamw.wd = schedule.lr->wd;
    return result;
}
