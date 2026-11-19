#include "schedule.h"

#include <algorithm>
#include <cmath>

float qlora_lr_schedule::get_lr(int64_t step) const {
    const float base_lr = lr->lr0;

    if (warmup_steps > 0 && step == 0) {
        return 0.0f;
    }
    if (warmup_steps > 0 && step <= warmup_steps) {
        return base_lr * static_cast<float>(step) / static_cast<float>(warmup_steps);
    }

    // `step` is the upcoming one-based optimizer update. Keep the decay schedule
    // indexed by completed updates, as it was before the warmup change.
    if (type == "linear") {
        const float min_lr = std::max(0.0f, lr->lr_min);
        const int64_t first_decay_step = warmup_steps + 1;
        // For a zero floor, make the step after the final training update the
        // zero endpoint so the final update itself still has a non-zero LR.
        const int64_t decay_steps = min_lr == 0.0f
            ? total_steps - warmup_steps
            : total_steps - first_decay_step;
        if (decay_steps <= 0) {
            return step <= total_steps ? base_lr : min_lr;
        }
        const double decay_progress = std::clamp(
            static_cast<double>(step - first_decay_step) / static_cast<double>(decay_steps), 0.0, 1.0);
        return base_lr + (min_lr - base_lr) * static_cast<float>(decay_progress);
    }

    const int64_t completed_steps = std::max<int64_t>(0, step - 1);
    if (type != "cosine" || total_steps <= warmup_steps) {
        const float epoch = static_cast<float>(lr->epoch);
        return lr->lr_min <= 0.0f ? lr->lr0 :
            epoch >= lr->decay_epochs ? lr->lr_min :
            lr->lr0 * std::pow(0.5f, epoch * lr->scale_epoch);
    }

    const float min_lr = std::max(0.0f, lr->lr_min);
    const double decay_progress =
        static_cast<double>(completed_steps - warmup_steps) / static_cast<double>(total_steps - warmup_steps);
    const double cosine_weight =
        0.5 * (1.0 + std::cos(std::acos(-1.0) * std::min(1.0, decay_progress)));
    return min_lr + (base_lr - min_lr) * static_cast<float>(cosine_weight);
}

ggml_opt_optimizer_params qlora_opt_lr_pars(void * userdata) {
    qlora_lr_schedule & schedule = *static_cast<qlora_lr_schedule *>(userdata);
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    schedule.current_lr = schedule.get_lr(schedule.step + 1);
    result.adamw.alpha = result.sgd.alpha = schedule.current_lr;
    result.sgd.wd = result.adamw.wd = schedule.lr->wd;
    return result;
}
