#include "schedule.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require_close(float actual, float expected, const std::string & message) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-7f * std::max(1.0f, std::fabs(expected))) {
        throw std::runtime_error(message + ": expected=" + std::to_string(expected) +
                                 " actual=" + std::to_string(actual));
    }
}

void test_constant_schedule_warmup() {
    lr_opt lr;
    lr.lr0 = 1e-3f;
    lr.lr_min = 0.0f;
    lr.epoch = 0;
    qlora_lr_schedule schedule{ &lr, "constant", 4, 0.1f, 8, 0, 0.0f };
    const float expected[] = { 1e-4f, 3.25e-4f, 5.5e-4f, 7.75e-4f, 1e-3f };
    for (int step = 0; step <= 4; ++step) {
        schedule.step = step;
        require_close(schedule.get_lr(), expected[step], "constant scheduler warmup value is incorrect");
    }
}

void test_cosine_schedule_boundaries() {
    lr_opt lr;
    lr.lr0 = 1e-3f;
    lr.lr_min = 1e-4f;
    lr.epoch = 0;
    qlora_lr_schedule schedule{ &lr, "cosine", 2, 0.1f, 8, 0, 0.0f };
    schedule.step = 0;
    require_close(schedule.get_lr(), 1e-4f, "cosine scheduler warmup start is incorrect");
    schedule.step = 1;
    require_close(schedule.get_lr(), 5.5e-4f, "cosine scheduler warmup midpoint is incorrect");
    schedule.step = 2;
    require_close(schedule.get_lr(), 1e-3f, "cosine scheduler decay start is incorrect");
    schedule.step = 5;
    require_close(schedule.get_lr(), 5.5e-4f, "cosine scheduler midpoint is incorrect");
    schedule.step = 8;
    require_close(schedule.get_lr(), 1e-4f, "cosine scheduler final value is incorrect");
    schedule.step = 100;
    require_close(schedule.get_lr(), 1e-4f, "cosine scheduler should remain at lr_min after total_steps");
}

void test_optimizer_callback_uses_schedule_value() {
    lr_opt lr;
    lr.lr0 = 2e-4f;
    lr.lr_min = 0.0f;
    lr.wd = 0.03f;
    lr.epoch = 0;
    qlora_lr_schedule schedule{ &lr, "constant", 2, 0.25f, 4, 0, 0.0f };
    ggml_opt_optimizer_params params = qlora_opt_lr_pars(&schedule);
    require_close(params.adamw.alpha, 5e-5f, "AdamW must use scheduled learning rate");
    require_close(params.sgd.alpha, 5e-5f, "SGD must use scheduled learning rate");
    require_close(params.adamw.wd, lr.wd, "AdamW weight decay must match configured weight decay");
    require_close(params.sgd.wd, lr.wd, "SGD weight decay must match configured weight decay");
    require_close(schedule.current_lr, 5e-5f, "schedule should record the learning rate passed to optimizer");
}

} // namespace

int main() {
    try {
        test_constant_schedule_warmup();
        test_cosine_schedule_boundaries();
        test_optimizer_callback_uses_schedule_value();
        std::cout << "schedule tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
